"""Target-independent visible-space proposals, not frontier segmentation.

Frame-local proposals and patch verdicts; directions select visible entrances.
No generated image, navigation target category or success label enters the VLM.
"""
import base64
import hashlib
import io
from PIL import Image
from .interior_reference import validate_box, refine_box

PROMPT = '''Identify useful visible interior reference patches in this real indoor photograph.
We will expand each patch into a hypothetical room photograph. Identify the SPACE
seen through an open doorway or a visible continuation of a room, NOT the door object.
An open door leaf is still a solid foreground panel: select the visible space BESIDE
or BEHIND it, never the leaf itself. Exclude door frames, near walls, fireplaces,
furniture close-ups, and floor-only/rug-only patches. A useful patch must contain
some visible room context above the floor (surfaces or contents beyond the opening).
Do not guess what is hidden behind a solid surface. Do not invent an opening.
Find all usable regions in the image, at most 6. It is valid to find none.
For each return a tight INNER rectangular patch excluding foreground occluders,
but preserving enough room context. Do not return the outer doorway bounding box.
Coordinates are integers 0..1000 normalized independently to image width and height.
Return JSON: {"regions":[{"kind":"opening_interior" or "open_space_continuation",
"box_1000":[left,top,right,bottom],"evidence":"brief visible contents beyond opening"}]}.
No search target is provided; do not optimize for any particular object.'''

def branch_hint(region,hint):
    """Match direction to entrance, not to the de-framed reference fragment."""
    outer=region.get('opening_box')
    if outer is None:return hint
    validate_box(dict(visible=True,box=outer),hint)
    x1,y1,x2,y2=region['box']
    if not (outer[0]<=x1<x2<=outer[2] and outer[1]<=y1<y2<=outer[3]):
        raise ValueError('reference_not_inside_entrance')
    return dict(hint,column_px=(x1+x2)/2,branch_column_px=hint['column_px'],
                branch_opening_box=outer,association='direction_to_opening_then_inner_reference')


def check_crop(image, box, hint):
    box = validate_box(dict(visible=True, box=list(box)), hint)
    # Real replay showed that even the VLM's inner box can retain a door-frame
    # strip. Trim strong side edges INSIDE the semantic region, never search the
    # full detector box for an unrelated floor-only replacement.
    box,quality=refine_box(image,box,hint)
    quality.update(policy='semantic_inner_patch_v1',
                   scope='VLM context + inner edge refinement; NOT measured frontier segmentation')
    return box,quality


class SemanticInteriorLocator:
    localization_source = 'glm_visible_space_v1'

    def __init__(self, client):
        self.client=client
        self.bundled=getattr(client,'model','')=='bundled-qwen3-vl-visible-space'
        if self.bundled:self.localization_source='bundled_qwen3_vl_visible_space'
        self.sha=None
        self.regions=[]
        self.last_attempts=0
        self.last_usage={}
        self.last_diagnostic={}
        self.error=None
        self.refinements={}
        self.total_requests=0

    def locate_interior(self, real, hint):
        raw=base64.b64decode(real, validate=True)
        image=Image.open(io.BytesIO(raw)).convert('RGB')
        if image.size != (hint['width'],hint['height']):
            raise ValueError('semantic_roi_size_mismatch')
        sha=hashlib.sha256(raw).hexdigest()
        self.last_attempts=0;self.last_usage={}
        if sha != self.sha:
            self.sha=sha;self.regions=[];self.error=None;self.refinements={}
            body=dict(model=self.client.model,messages=[dict(role='user',content=[
                dict(type='text',text=PROMPT),
                dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+real))])],
                thinking=dict(type='disabled'),temperature=0,max_tokens=1200,
                response_format=dict(type='json_object'))
            try:
                reply=self.client.propose_regions(real,image) if self.bundled else self.client._request(body,retries=0)
                if self.bundled:self.total_requests+=1
                regions=reply.get('regions') if isinstance(reply,dict) else None
                if not isinstance(regions,list) or len(regions)>6:
                    raise ValueError('invalid_semantic_regions')
                parsed=[]
                for region in regions:
                    if not isinstance(region,dict): raise ValueError('invalid_semantic_region')
                    box=region.get('box_1000')
                    if (region.get('kind') not in ('opening_interior','open_space_continuation')
                        or not isinstance(box,list) or len(box)!=4
                        or any(type(v) is not int or not 0<=v<=1000 for v in box)
                        or not box[0]<box[2] or not box[1]<box[3]):
                        raise ValueError('invalid_semantic_region')
                    entry=dict(region,box=[round(v*image.size[i%2]/1000) for i,v in enumerate(box)])
                    if self.bundled:
                        outer=region.get('opening_box_1000')
                        if (not isinstance(outer,list) or len(outer)!=4 or
                            any(type(v) is not int or not 0<=v<=1000 for v in outer)):
                            raise ValueError('invalid_semantic_opening_box')
                        entry['opening_box']=[round(v*image.size[i%2]/1000) for i,v in enumerate(outer)]
                    parsed.append(entry)
                self.regions=parsed
            except Exception as exc:
                # Remember failure only for this frame; do not silently retry per candidate.
                self.error=exc
            finally:
                self.last_attempts=self.client.last_attempts
                self.last_usage=dict(self.client.last_usage)
        self.last_diagnostic=dict(anchor_sha256=sha,regions=self.regions,
            request_local_reuse=self.last_attempts==0,policy='semantic_inner_patch_v1',rejected=[],refinements=self.refinements)
        if self.bundled:
            self.last_diagnostic['detector']=getattr(self.client,'proposal_diagnostic',{})
        if self.bundled and self.error:self.last_diagnostic['model_reply']=getattr(self.client,'last_response',None)
        if self.error is not None: raise self.error
        valid=[]
        regions=self.regions
        if self.bundled:
            # Lock association BEFORE any semantic crop assessment. A failed
            # de-framing must never cause a switch to another detected object.
            eligible=[]
            for index,region in enumerate(regions):
                outer=region['opening_box']
                try:
                    validate_box(dict(visible=True,box=outer),hint)
                    distance=abs((outer[0]+outer[2])/2-hint['column_px'])
                    eligible.append((distance,index,region))
                except ValueError as exc:
                    self.last_diagnostic['rejected'].append(dict(box=outer,reason=str(exc)))
            self.last_diagnostic['association_policy']='lock_directional_outer_then_deframe_v1'
            if not eligible:return dict(visible=False)
            _,index,locked=min(eligible,key=lambda item:(item[0],item[1]))
            locked=dict(locked,box=list(locked['opening_box']))
            self.last_diagnostic['locked_outer']=dict(
                proposal_index=index,box=locked['box'],
                column_px=hint['column_px'],allow_other_region_on_failure=False)
            regions=[locked]
        for region in regions:
            try:
                effective_hint=branch_hint(region,hint)
                if self.bundled:
                    validate_box(dict(visible=True,box=region['box']),effective_hint)
                    key=','.join(map(str,region['box']))
                    if key not in self.refinements:
                        self.refinements[key]=self.client.refine_region(real,image,region['box'])
                        self.last_attempts+=self.client.last_attempts
                        for name,value in self.client.last_usage.items():
                            if isinstance(value,(int,float)):self.last_usage[name]=self.last_usage.get(name,0)+value
                    refined=self.refinements[key]
                    if not refined['usable']:raise ValueError('local_vlm_occluder_or_insufficient')
                    region=dict(region,box=refined['box'],patch_verdict=refined)
                    effective_hint=branch_hint(region,hint)
                box,quality=check_crop(image,region['box'],effective_hint)
                distance=abs((box[0]+box[2])/2-hint['column_px'])/image.width
                valid.append((distance,region,quality))
            except ValueError as exc:
                self.last_diagnostic['rejected'].append(dict(box=region['box'],reason=str(exc)))
        if not valid: return dict(visible=False)
        _,chosen,quality=min(valid,key=lambda item:item[0])
        self.last_diagnostic.update(chosen=chosen,quality=quality)
        return dict(visible=True,box=chosen['box'],
            box_1000=[round(v/image.size[i%2]*1000) for i,v in enumerate(chosen['box'])],
            proposal_box_1000=chosen['box_1000'],crop_hint=branch_hint(chosen,hint))
