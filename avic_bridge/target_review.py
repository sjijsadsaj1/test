"""Review a marked REAL map object before geometry-authorized STOP. No GT."""
import base64
from collections import defaultdict,deque
import hashlib
import io
import json
import os
from pathlib import Path
import time
import uuid

CONFUSIONS={
    'chair':['sofa','stool','table'], 'cabinet':['door','refrigerator','wall panel'],
    'table':['counter','cabinet','bed'], 'sink':['bathtub','toilet','counter'],
    'toilet':['sink','chair','bathtub'], 'picture':['window','mirror','television'],
    'chest_of_drawers':['cabinet without drawers','table','bed'],
    'seating':['table','bed','cabinet'], 'plant':['floral picture','lamp','sculpture'],
}
PROMPT='''Verify the identity of ONE marked real object instance for ObjectNav.
The request names the DATASET target, detector alias, and common confusing categories.
Decide target / confusable / other / uncertain. Judge visible shape, parts and material;
room type, contextual likelihood, predicted objects and imagined rooms are NOT evidence.
A cabinet may be built-in or free-standing; seating includes chairs and sofas; a drawer unit
may be small or bedside. Do not silently replace the dataset category with a detector alias.
Images are real full camera views with a red object mask bounding box, an exact crop, and
when available older real views associated with the SAME map instance. These are not a
360-degree panorama. Gray padding, if present, is an input adapter and is NOT scene content.
If the marked object cannot be visually identified, choose uncertain, not other or confusable.
Absence in the current view is not evidence that a previously observed object was wrong.
Return JSON only: {"verdict":"target|confusable|other|uncertain",
"identified_category":"visible category or unknown","visible_evidence":"specific visible parts",
"reason":"brief explanation"}. Do not judge arrival, distance, reachability, or task success.'''


def review_request(client,payload,root):
    from .vlm_transport import image_part
    from PIL import Image,ImageDraw
    folder=Path(root)/payload['request_id'];folder.mkdir(parents=True,exist_ok=False)
    content=[dict(type='text',text=PROMPT+'\n'+json.dumps(dict(target=payload['target'],
        detector_alias=payload['detector_alias'],confusions=CONFUSIONS.get(payload['target'],[]))))]
    images=[];crop_sent=False
    for i,view in enumerate(payload['views'][:3]):
        image=Image.open(io.BytesIO(base64.b64decode(view['rgb_b64'],validate=True))).convert('RGB')
        box=view.get('box');marked=image.copy()
        if box is not None:
            if len(box)!=4 or not 0<=box[0]<box[2]<=image.width or not 0<=box[1]<box[3]<=image.height:
                raise ValueError('invalid marked object box')
            ImageDraw.Draw(marked).rectangle([box[0],box[1],box[2]-1,box[3]-1],outline='red',width=3)
        image.save(folder/f'view_{i}_real.png');marked.save(folder/f'view_{i}_marked.png')
        content.extend([dict(type='text',text=json.dumps({k:v for k,v in view.items() if k!='rgb_b64'})),image_part(marked)])
        if not crop_sent and box is not None:
            crop=image.crop(box);crop.save(folder/'object_crop.png');content.append(image_part(crop));crop_sent=True
        images.append(dict(index=i,step=view['step'],box=box))
    (folder/'input.json').write_text(json.dumps({k:v for k,v in payload.items() if k!='views'},indent=2)+'\n')
    (folder/'prompt.txt').write_text(content[0]['text'])
    start=time.monotonic()
    try:
        result=client.request(content,max_tokens=400)
        if not isinstance(result,dict) or result.get('verdict') not in ('target','confusable','other','uncertain'):
            raise ValueError('invalid review verdict')
        if any(not isinstance(result.get(k),str) or not result[k].strip() for k in ('identified_category','visible_evidence','reason')):
            raise ValueError('missing visual evidence')
    except Exception as exc:
        result=dict(verdict='uncertain',identified_category='unknown',visible_evidence='unavailable',
                    reason='review unavailable; use original rules',error=type(exc).__name__)
    result.update(request_id=payload['request_id'],instance_id=payload['instance_id'],
                  seconds=time.monotonic()-start,views=images)
    (folder/'raw.txt').write_text(getattr(client,'last_answer',''))
    (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


class TargetReviewer:
    """Small per-episode budget; all turns are ordinary counted Habitat actions."""
    def __init__(self,url):
        import rospy
        from plan_env.srv import ReviewTarget
        self.proxy=rospy.ServiceProxy('/apexnav/review_target',ReviewTarget)
        self.url=url.rstrip('/')+'/v1/review';self.reset()

    def reset(self):
        self.history=defaultdict(lambda:deque(maxlen=8));self.done=set();self.calls=0
        self.pending=None;self.seen_inputs=set();self.scanned=set();self.extra_actions=0

    def log(self,**data):print('[TARGET_REVIEW] '+json.dumps(data,ensure_ascii=False),flush=True)

    def process(self,*,episode,step,action,final_state,target,alias,rgb,masks,labels,pose):
        import numpy as np
        from PIL import Image
        if step>=499:return action,final_state
        # This read-only snapshot also binds current detection masks to original
        # object-map cluster IDs. No nearest-looking object or room substitution.
        try:snapshot=self.proxy(episode,step,0,-1)
        except Exception as exc:
            self.log(step=step,event='fallback',reason='instance service unavailable',error=type(exc).__name__)
            return action,final_state
        if not snapshot.accepted:return action,final_state
        buf=io.BytesIO();Image.fromarray(np.asarray(rgb,dtype=np.uint8)).save(buf,format='JPEG',quality=90)
        raw=buf.getvalue();sha=hashlib.sha256(raw).hexdigest();current={}
        for i,(mask,label) in enumerate(zip(masks,labels)):
            if int(label)!=0 or i>=len(snapshot.detection_instances):continue
            ident=int(snapshot.detection_instances[i])
            if ident<0:continue
            mask=np.asarray(mask).squeeze();ys,xs=np.where(mask>0)
            if not len(xs):continue
            entry=dict(step=step,box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
                       rgb_b64=base64.b64encode(raw).decode(),sha256=sha,pose=list(pose),instance_id=ident)
            current[ident]=entry
            history=self.history[ident]
            if not history or (history[-1]['sha256']!=sha and
                (step-history[-1]['step']>=6 or abs(pose[2]-history[-1]['pose'][2])>.25 or
                 np.linalg.norm(np.array(pose[:2])-history[-1]['pose'][:2])>.15)):
                history.append(entry)
        ident=int(snapshot.instance_id)
        if self.pending is not None and self.pending['id']!=ident:
            self.log(step=step,event='scan_cancelled',reason='planner changed instance')
            self.pending=None
        eligible=(action==0 and final_state==4 and snapshot.exploration_result==1 and ident>=0)
        if action==0 and final_state==4:
            self.log(step=step,event='eligibility',instance=ident,exploration_result=snapshot.exploration_result,
                     eligible=eligible,history_frames=len(self.history[ident]),current_mask=ident in current)
        if self.pending is not None:
            turns=self.pending['turns']
            if turns:return self.command(snapshot,episode,step,turns.pop(0),action,final_state)
            self.pending=None
            # Recheck after the bounded scan; never manufacture a STOP.
            if not eligible:self.done.add(ident);return action,final_state
        elif not eligible or ident in self.done:return action,final_state
        if self.calls>=4:
            self.done.add(ident);self.log(step=step,event='fallback',reason='episode review budget')
            return action,final_state
        history=list(self.history[ident]);views=[]
        if ident in current:views.append(current[ident])
        elif history:
            views.extend([dict(step=step,box=None,rgb_b64=base64.b64encode(raw).decode(),sha256=sha,
                pose=list(pose),instance_id=ident,note='current real view: no associated object mask; do not infer absence'),history[-1]])
        else:
            self.done.add(ident);self.log(step=step,event='fallback',reason='no real instance-associated image',instance=ident)
            return action,final_state
        # At most one earlier SAME-instance view, separated in camera pose.
        earlier=next((v for v in history if v['sha256']!=views[0]['sha256'] and v['step']<views[0]['step']),None)
        if earlier and len(views)==1:views.append(earlier)
        fingerprint=(ident,tuple((v['sha256'],tuple(v['box']) if v['box'] else None) for v in views))
        if fingerprint in self.seen_inputs:
            self.done.add(ident);self.log(step=step,event='fallback',reason='identical observation already checked')
            return action,final_state
        self.seen_inputs.add(fingerprint);self.calls+=1
        payload=dict(request_id=uuid.uuid4().hex,episode=episode,step=step,instance_id=ident,
                     target=target,detector_alias=alias,views=views)
        from urllib.request import Request,urlopen
        start=time.monotonic()
        try:
            with urlopen(Request(self.url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=90) as response:
                verdict=json.loads(response.read())
            if verdict.get('request_id')!=payload['request_id'] or verdict.get('instance_id')!=ident:
                raise ValueError('review identity mismatch')
            if verdict.get('verdict') not in ('target','confusable','other','uncertain'):
                raise ValueError('invalid review response')
        except Exception as exc:
            self.done.add(ident);self.log(step=step,event='fallback',reason='VLM call failed',error=type(exc).__name__)
            return action,final_state
        self.log(step=step,instance=ident,event='verdict',result=verdict,elapsed=time.monotonic()-start)
        if verdict.get('error'):
            self.done.add(ident);return action,final_state
        if verdict['verdict'] in ('confusable','other'):
            self.done.add(ident)
            return self.command(snapshot,episode,step,1,action,final_state)
        if verdict['verdict']=='uncertain' and ident not in self.scanned and self.calls<4:
            # Mark this instance so an uncertain second check cannot start another scan.
            self.scanned.add(ident)
            self.pending=dict(id=ident,turns=[3,3,2])
            return self.command(snapshot,episode,step,2,action,final_state)
        self.done.add(ident)
        return action,final_state

    def command(self,snapshot,episode,step,operation,original_action,original_state):
        result=self.proxy(episode,step,operation,snapshot.instance_id)
        if not result.accepted:
            self.pending=None;self.done.add(snapshot.instance_id)
            self.log(step=step,event='fallback',reason=result.reason)
            return original_action,original_state
        self.extra_actions+=1
        self.log(step=step,event='reject' if operation==1 else 'observe',instance=snapshot.instance_id,
                 action=result.action,extra_actions=self.extra_actions)
        return int(result.action),int(result.final_state)
