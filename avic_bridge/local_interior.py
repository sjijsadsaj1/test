"""Request-local opening proposals from ApexNav's existing GroundingDINO service.

Not frontier segmentation or proof of visibility/traversability. No search-target
prompt, remote credentials, new weights, cross-observation cache, or GPU imports.
"""
import base64
import hashlib
import io
import json
import math
import os
from urllib.request import Request,build_opener,ProxyHandler
from urllib.parse import urlsplit
from PIL import Image
from .interior_reference import validate_box,refine_box

CAPTION='open doorway . room entrance . hallway .'


def opening_core(image,box):
    """Detector proposes the OUTER doorway, unlike GLM's requested inner patch.

    Remove long side-edge bands (up to outer40%) and conservative lintel/threshold
    strips (top15%, bottom5%). Heuristic only; no claim of semantic masks.
    """
    import numpy as np
    x1,y1,x2,y2=box
    patch=np.asarray(image.crop(box).convert('RGB')).astype(np.int16)
    support=(np.max(np.abs(patch[:,1:]-patch[:,:-1]),axis=2)>18).mean(0)
    edges=np.flatnonzero(support>=.30)+x1+1
    left=edges[edges<=x1+.40*(x2-x1)]
    right=edges[edges>=x2-.40*(x2-x1)]
    a=int(left[-1])+2 if len(left) else x1+2
    b=int(right[0])-2 if len(right) else x2-2
    return (a,y1+round(.15*(y2-y1)),b,y2-round(.05*(y2-y1)))


def encode_for_apex_detector(image):
    # Existing server cv2.imdecode returns BGR but predict treats that array as RGB.
    # Swap on the wire (like the upstream client), losslessly, not in the real crop.
    r,g,b=image.convert('RGB').split()
    data=io.BytesIO();Image.merge('RGB',(b,g,r)).save(data,format='PNG')
    return base64.b64encode(data.getvalue()).decode()


class LocalInteriorLocator:
    localization_source='local_grounding_dino_live'
    last_usage={}
    def __init__(self,real,timeout=30):
        self.raw=base64.b64decode(real,validate=True)
        self.image=Image.open(io.BytesIO(self.raw)).convert('RGB')
        self.sha=hashlib.sha256(self.raw).hexdigest()
        self.url=os.environ.get('AVIC_INTERIOR_DINO_URL','http://127.0.0.1:12181/gdino')
        parsed=urlsplit(self.url)
        if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or parsed.username or parsed.password:
            raise ValueError('local_roi_requires_loopback_detector')
        self.timeout=timeout;self.detections=None;self.error=None
        self.last_attempts=0;self.total_requests=0;self.last_diagnostic={}

    def _detect(self):
        self.last_attempts=1;self.total_requests+=1
        body=dict(image=encode_for_apex_detector(self.image),caption=CAPTION,box_threshold=.20,text_threshold=.20)
        request=Request(self.url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        try:
            with build_opener(ProxyHandler({})).open(request,timeout=self.timeout) as response:
                data=json.loads(response.read())
        except Exception as exc:
            raise ValueError('local_detector_transport_'+type(exc).__name__) from None
        if not isinstance(data,dict) or not all(isinstance(data.get(k),list) for k in ('boxes','logits','phrases')):
            raise ValueError('local_detector_invalid_schema')
        if not len(data['boxes'])==len(data['logits'])==len(data['phrases']):
            raise ValueError('local_detector_mismatched_lengths')
        result=[];w,h=self.image.size
        for box,confidence,phrase in zip(data['boxes'],data['logits'],data['phrases']):
            if (not isinstance(box,list) or len(box)!=4 or
                any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not -.05<=v<=1.05 for v in box)
                or isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence)
                or not 0<=confidence<=1 or not isinstance(phrase,str)):
                raise ValueError('local_detector_invalid_normalized_xyxy')
            pixels=[max(0,min(w if i%2==0 else h,round(v*(w if i%2==0 else h)))) for i,v in enumerate(box)]
            result.append(dict(box=pixels,normalized_xyxy=box,confidence=confidence,phrase=phrase))
        return result

    def locate_interior(self,real,hint):
        if hashlib.sha256(base64.b64decode(real,validate=True)).hexdigest()!=self.sha:
            raise ValueError('local_roi_observation_mismatch')
        if (hint['width'],hint['height'])!=self.image.size:raise ValueError('local_roi_size_mismatch')
        self.last_attempts=0
        if self.error:raise ValueError(self.error)
        if self.detections is None:
            try:self.detections=self._detect()
            except ValueError as exc:self.error=str(exc);raise
        rejected=[];valid=[]
        for detection in self.detections:
            try:
                box=validate_box(dict(visible=True,box=detection['box']),hint)
                core=opening_core(self.image,box)
                validate_box(dict(visible=True,box=list(core)),hint)
                refined,quality=refine_box(self.image,core,hint)
                # Directional overlap first; among valid proposals prefer confidence.
                distance=abs((refined[0]+refined[2])/2-hint['column_px'])/hint['width']
                valid.append((detection['confidence']-.10*distance,detection,core,refined,quality))
            except ValueError as exc:rejected.append(dict(box=detection['box'],reason=str(exc)))
        self.last_diagnostic=dict(anchor_sha256=self.sha,caption=CAPTION,box_threshold=.20,text_threshold=.20,
            detections=self.detections,rejected=rejected,request_local_reuse=self.last_attempts==0,
            scope='detected opening + direction + RGB quality, NOT measured frontier segmentation')
        if not valid:return dict(visible=False)
        _,chosen,core,refined,quality=max(valid,key=lambda x:x[0])
        self.last_diagnostic.update(chosen=chosen,opening_core_box=list(core),refined_box=list(refined),quality=quality,
            opening_core_policy='long side edges support30% in outer40%; top15%/bottom5% strips')
        return dict(visible=True,box=list(core))
