"""Single-view Qwen Image route imagination plus joint GLM diagnostics.

Heavy model imports are lazy. No prediction cache, target-conditioned generation,
prompt enhancer, or hand-written diffusion sigma schedule is used.
"""
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
from PIL import Image
from .protocol import CURRENT_ROUTE_CONTRACT
from . import alignment_audit
from . import interior_reference

CONTRACT='image_glm_single_view_v1'

def review_nonblocking():
    return os.environ.get('AVIC_GLM_NONBLOCKING','0')=='1'

def review_enabled():
    return os.environ.get('AVIC_GLM_REVIEW','1')!='0'

def glm_required():
    return review_enabled() or (os.environ.get('AVIC_IMAGE_INTERIOR')=='1' and
                               os.environ.get('AVIC_IMAGE_ROI_LOCATOR','glm')=='glm')

class GLMAPIError(RuntimeError):
    """Structured diagnostics with no credentials, URLs or provider message text."""
    def __init__(self,kind,http_status=None,provider_code=None):
        self.diagnostic={'kind':kind}
        if isinstance(http_status,int):self.diagnostic['http_status']=http_status
        if isinstance(provider_code,(str,int)) and re.fullmatch(r'[A-Za-z0-9_-]{1,32}',str(provider_code)):
            self.diagnostic['provider_code']=str(provider_code)
        super().__init__('GLM API technical failure: '+kind)

def number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def segment(value):
    value=str(value)
    if not re.fullmatch(r'[A-Za-z0-9_.:-]+',value) or value in ('.','..'):
        raise ValueError('unsafe artifact identity')
    return value

def jpeg(encoded):
    raw=base64.b64decode(encoded,validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.format!='JPEG':raise ValueError('real anchor must be JPEG')
        image.load();return raw,image.convert('RGB')

def motion(actions,camera):
    forward=camera.get('forward_step_m');turn=camera.get('turn_angle_deg')
    if not number(forward) or forward<=0 or not number(turn) or not 0<turn<=180:
        raise ValueError('invalid action geometry')
    x=y=yaw=0.;words=[]
    for action in actions:
        if type(action) is not int:raise ValueError('route actions must be integer IDs')
        if action==1:
            x+=forward*math.cos(yaw);y+=forward*math.sin(yaw);words.append('move forward %.3f meters'%forward)
        elif action==2:yaw+=math.radians(turn);words.append('turn left %.1f degrees'%turn)
        elif action==3:yaw-=math.radians(turn);words.append('turn right %.1f degrees'%turn)
        else:raise ValueError('unsupported route action')
    return dict(dx_m=x,dy_m=y,yaw_rad=yaw,frame='relative robot planar: +x initial forward, +y initial left; expected, NOT measured'),'; '.join(words)

def route_prompt(actions,camera):
    expected,text=motion(actions,camera)
    prompt=('Predict one plausible first-person camera view after this complete motion sequence from the reference photograph: '+text+'. '
        'Expected relative displacement: forward-axis %.3f m, left-axis %.3f m, yaw %.1f degrees (left positive). '
        'These are requested motion conditions, not measured camera poses. Keep known walls, doors, furniture and object identities consistent. '
        'Do not add a requested search object or fabricate certainty in unseen space. Preserve occlusions and uncertain regions naturally. '
        'Produce a realistic navigation camera photograph, not a diagram, text overlay, map or montage.')%(expected['dx_m'],expected['dy_m'],math.degrees(expected['yaw_rad']))
    return prompt,expected

def validate_plans(payload):
    policy=payload.get('score_policy',{})
    if policy.get('mode')!='image_glm' or policy.get('contract')!=CONTRACT:
        raise ValueError('image_glm requires explicit image_glm_single_view_v1 contract')
    nav=payload.get('navigation',{})
    if nav.get('route_contract')!=CURRENT_ROUTE_CONTRACT:raise ValueError('invalid current route contract')
    plans=nav.get('candidate_plans',[]);ids=nav.get('imagination_plan_ids',[])
    selected=[p for p in plans if p.get('plan_id') in ids]
    if not 1<=len(selected)<=3 or len(ids)!=len(set(ids)) or len({p['plan_id'] for p in selected})!=len(selected) or set(ids)!={p['plan_id'] for p in selected}:
        raise ValueError('requires one to three uniquely identified requested plans')
    if payload.get('signals',{}).get('frontier_query_active') is not True:raise ValueError('no active real frontier query')
    step=payload.get('episode',{}).get('step_index');image=payload.get('observation',{}).get('rgb_jpeg_b64');pitch=payload.get('camera',{}).get('pitch_rad')
    if type(step) is not int or not number(pitch):raise ValueError('invalid observation step/pitch')
    jpeg(image)
    routes=[]
    for plan in selected:
        segment(plan['plan_id']);anchor=plan.get('frontier_anchor',{});frontier=plan.get('source_frontier',{})
        actions=[a.get('id') for a in plan.get('actions',[])];minimum=plan.get('min_prefix_length')
        if (plan.get('route_contract')!=CURRENT_ROUTE_CONTRACT or anchor.get('binding_source')!='current_observation'
            or anchor.get('step_index')!=step or anchor.get('rgb_jpeg_b64')!=image or anchor.get('camera_pitch_rad')!=pitch
            or actions!=frontier.get('path_action_prefix') or type(minimum) is not int or not 1<=minimum<=len(actions)):
            raise ValueError('mismatched anchor or complete A* route identity')
        if any(not number(anchor.get(key)) for key in ['robot_x','robot_y','robot_yaw_rad','camera_pitch_rad']):raise ValueError('invalid real anchor pose')
        if any(not number(frontier.get(key)) for key in ['x','y','semantic_score']):raise ValueError('invalid frontier coordinates/map score')
        motion(actions,payload['camera']);routes.append(tuple(actions))
    if len(set(routes))!=len(routes):raise ValueError('indistinguishable full route prefixes')
    for index,route in enumerate(routes):
        for other in routes[index+1:]:
            if route[:min(len(route),len(other))]==other[:min(len(route),len(other))]:
                raise ValueError('route pair does not expose branch divergence')
    return selected

class GLMReviewClient:
    def __init__(self):
        self.key=os.environ.get('AVIC_GLM_API_KEY','')
        self.base=os.environ.get('AVIC_GLM_BASE_URL','https://open.bigmodel.cn/api/paas/v4').rstrip('/')
        self.model=os.environ.get('AVIC_GLM_MODEL','glm-4.6v-flash')
        self.timeout=min(10. if review_nonblocking() else 60.,float(os.environ.get('AVIC_GLM_TIMEOUT','45')))
        self.max_tokens=max(128,min(1200,int(os.environ.get('AVIC_GLM_MAX_TOKENS','1200'))))
        self.retries=0 if review_nonblocking() else max(0,min(3,int(os.environ.get('AVIC_GLM_RETRIES','3'))))
        self.last_attempts=0;self.last_usage={}
        if not self.key:raise RuntimeError('GLM API key is not configured')
    def review(self,real,candidates,target):
        prompt=('Review the real navigation view and each predicted route view. An interior_crop candidate is a freely recomposed room hypothesis from its selected image region, NOT a measured route endpoint. '
            'Unknown/unseen space is uncertain, not automatically inconsistent. '
            'An imagined target is not by itself proof or an automatic rejection. Report only diagnostics. Return a JSON object with evaluations, '
            'exactly one entry per plan_id, fields plan_id, consistency (consistent/inconsistent/uncertain), branch_alignment '
            '(aligned/misaligned/uncertain), target_context_support (plausible/unsupported/uncertain), reason (short). '
            'Search context: '+json.dumps(target,ensure_ascii=False)+'. Real view follows.')
        content=[dict(type='text',text=prompt),dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+real))]
        for candidate in candidates:
            content.extend([dict(type='text',text=json.dumps(dict(plan_id=candidate['plan_id'],expected_motion=candidate['relative_pose'],
                conditioning=candidate.get('selection')),ensure_ascii=False)),
                dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+candidate['jpeg_b64']))])
        body=dict(model=self.model,messages=[dict(role='user',content=content)],thinking=dict(type='disabled'),
                  max_tokens=self.max_tokens,temperature=0,response_format=dict(type='json_object'))
        return self._request(body)

    def locate_interior(self,real,hint):
        prompt=(
            'Locate a visible indoor fragment in the candidate exploration direction in this image. '
            'The horizontal bearing column is %.1f on a 0..1000 normalized horizontal axis. '
            'It is a direction hint, NOT proof that the frontier is visible. '
            'Choose a tight rectangular patch containing ONLY the interior beyond an opening, or the visible continuation of open space, '
            'near this column. Exclude foreground door frames, door panels and occluding corridor walls; '
            'keep useful visible interior contents. Shrink inward past frame edges; do not include the entire doorway. '
            'Do not choose another opening elsewhere to obtain a crop. '
            'If this direction is blocked or no usable interior is visible, return {"visible":false}. '
            'Otherwise return {"visible":true,"box_1000":[left,top,right,bottom]}. '
            'All four coordinates must be integers in [0,1000], normalized independently to image width and height. '
            'Return JSON only. Do not infer a search target.'
        )%(hint['column_px']/hint['width']*1000)
        if hint.get('roi_policy')=='content_roi_v2':
            prompt += (' Black or missing-render regions contain no useful reference; do not select them. '
                       'Retain enough actual visible interior context, not a tiny strip or only a door frame. '
                       'Prefer a patch showing the room beyond the opening, including visible interior surfaces and contents; '
                       'do not choose just the floor immediately in front of the camera. '
                       'The pixel crop must be at least 52 pixels wide, 100 pixels tall, and about 2 percent of the image area. '
                       'The input image is %d by %d pixels; still return normalized 0..1000 coordinates. '
                       'If no such interior patch is visible in this direction, return {"visible":false}.')
            prompt=prompt%(hint['width'],hint['height'])
        body=dict(model=self.model, messages=[dict(role='user',content=[dict(type='text',text=prompt),
            dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+real))])],
            thinking=dict(type='disabled'),max_tokens=self.max_tokens,temperature=0,response_format=dict(type='json_object'))
        result=self._request(body,retries=2)
        if isinstance(result,dict) and result.get('visible') is True:
            box=result.get('box_1000')
            if not isinstance(box,list) or len(box)!=4 or any(type(v) is not int or not 0<=v<=1000 for v in box):
                raise ValueError('invalid_normalized_roi')
            result['box']=[round(v*hint['width' if i%2==0 else 'height']/1000) for i,v in enumerate(box)]
        return result

    def _request(self,body,retries=None):
        retries=self.retries if retries is None else retries
        self.last_attempts=0;self.last_usage={}
        for attempt in range(retries+1):
            self.last_attempts=attempt+1
            try:
                request=Request(self.base+'/chat/completions',data=json.dumps(body).encode(),
                    headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key},method='POST')
                with urlopen(request,timeout=self.timeout) as response:result=json.loads(response.read())
                usage=result.get('usage',{})
                self.last_usage={key:value for key,value in usage.items()
                    if key in ('prompt_tokens','completion_tokens','total_tokens') and type(value) is int and value>=0}
                choice=result['choices'][0]
                if choice.get('finish_reason')=='length':raise GLMAPIError('TruncatedResponse')
                text=choice['message']['content'];text=re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip())
                return json.loads(text)
            except HTTPError as exc:
                code=None
                try:code=json.loads(exc.read(8192)).get('error',{}).get('code')
                except (ValueError,TypeError,AttributeError):pass
                transient=exc.code in (500,502,503,504) or (exc.code==429 and str(code) in ('1302','1305'))
                if transient and attempt<retries:
                    time.sleep(2**(attempt+1));continue
                raise GLMAPIError('HTTPError',exc.code,code) from None
            except (URLError,TimeoutError) as exc:
                if attempt<retries:
                    time.sleep(2**(attempt+1));continue
                raise GLMAPIError(type(exc).__name__) from None
            except GLMAPIError:raise
            except Exception as exc:raise GLMAPIError(type(exc).__name__) from None

def reviews_by_id(review,ids):
    values=review.get('evaluations') if isinstance(review,dict) else None
    if not isinstance(values,list) or len(values)!=len(ids):raise ValueError('GLM missing candidate evaluations')
    result={}
    allowed=dict(consistency={'consistent','inconsistent','uncertain'},branch_alignment={'aligned','misaligned','uncertain'},
                 target_context_support={'plausible','unsupported','uncertain'})
    for item in values:
        if not isinstance(item,dict) or item.get('plan_id') not in ids or item['plan_id'] in result:raise ValueError('GLM duplicate/unknown plan identity')
        if any(item.get(key) not in choices for key,choices in allowed.items()) or not isinstance(item.get('reason'),str):raise ValueError('GLM invalid structured review')
        clean={key:item[key] for key in ['plan_id','reason',*allowed]};secret=os.environ.get('AVIC_GLM_API_KEY','')
        if secret:clean['reason']=clean['reason'].replace(secret,'[REDACTED]')
        result[item['plan_id']]=clean
    return result

class QwenImageGLMBackend:
    name='qwen_image_glm'
    def __init__(self,pipe=None,glm=None,itm=None):
        self._pipe=pipe;self._glm=glm;self._itm=itm;self._lock=threading.RLock()
        self._injected=pipe is not None and glm is not None
        self.model_dir=os.environ.get('AVIC_IMAGE_MODEL_DIR','/data/wzy/Qwen-Image-2.1-Turbo')
        self.device=os.environ.get('AVIC_IMAGE_DEVICE','cuda:0')
        self.width=int(os.environ.get('AVIC_IMAGE_WIDTH','768'));self.height=int(os.environ.get('AVIC_IMAGE_HEIGHT','768'))
        self.seed=int(os.environ.get('AVIC_IMAGE_SEED','23'))
        self.interior=os.environ.get('AVIC_IMAGE_INTERIOR','0')=='1'
        self.roi_v2=os.environ.get('AVIC_IMAGE_ROI_V2','0')=='1'
        self.review_enabled=review_enabled()
        self.roi_locator=os.environ.get('AVIC_IMAGE_ROI_LOCATOR','glm')
        self.geometry_roi=self.roi_locator=='frontier_geometry'
        self.dual_context=os.environ.get('AVIC_IMAGE_DUAL_CONTEXT')=='1'
        if self.dual_context and not self.geometry_roi:raise ValueError('dual context requires original frontier geometry mode')
        self.semantic_roi=os.environ.get('AVIC_IMAGE_SEMANTIC_ROI','0')=='1' or self.roi_locator=='bundled'
        if self.semantic_roi and not (self.interior and self.roi_locator in ('glm','bundled')):
            raise ValueError('semantic ROI requires interior mode and semantic locator')
        if self.roi_locator not in ('glm','gdino','bundled','frontier_geometry'):raise ValueError('unknown interior locator')
        if self.geometry_roi and (not self.interior or self.semantic_roi or self.review_enabled):
            raise ValueError('geometry requires interior, semantic ROI off and post-review off')
        if self.roi_locator=='gdino' and not (self.interior and self.roi_v2):
            raise ValueError('local interior locator requires interior + ROI v2')
        self.glm_required=self.review_enabled or (self.interior and self.roi_locator=='glm')
        self.output_root=Path(os.environ.get('AVIC_IMAGE_OUTPUT_ROOT',str(Path(__file__).resolve().parents[1]/'experiments/runs/image_glm')))
        if min(self.width,self.height)<=0 or self.width%16 or self.height%16:raise ValueError('image dimensions must be positive multiples of16')
    def health(self):
        root=Path(self.model_dir)
        try:
            index=json.loads((root/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())
            files=[root/'transformer'/name for name in set(index['weight_map'].values())]
            files += [root/'text_encoder/model.safetensors',root/'vae/diffusion_pytorch_model.safetensors']
            weights_ready=all(p.is_file() for p in files)
        except (OSError,ValueError,KeyError):files=[];weights_ready=False
        for path in files:
            if not path.is_file():weights_ready=False;break
            if path.stat().st_size<1024:
                weights_ready=False;break
            with path.open('rb') as handle:
                if handle.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
                    weights_ready=False;break
        return dict(ok=self._injected or (weights_ready and (not self.glm_required or bool(os.environ.get('AVIC_GLM_API_KEY')))),backend=self.name,
            interior_reference_enabled=self.interior,
            interior_roi_locator=self.roi_locator,glm_required=self.glm_required,
            frontier_geometry_contract='original_cluster_projection_v1' if self.geometry_roi else None,
            dual_context=self.dual_context,
            semantic_roi=self.semantic_roi,
            interior_roi_policy='original_cluster_projection_v1' if self.geometry_roi else ('semantic_inner_patch_v1' if self.semantic_roi else ('content_roi_v2' if self.roi_v2 else 'legacy_inset')),glm_review_enabled=self.review_enabled,
            image_glm_contract=CONTRACT,model_loaded=self._pipe is not None,model_weights_ready=weights_ready,
            model_path=self.model_dir,model_path_readable=os.path.isdir(self.model_dir) and os.access(self.model_dir,os.R_OK),
            glm_key_configured=bool(os.environ.get('AVIC_GLM_API_KEY')),prediction_cache=False,
            glm_nonblocking=review_nonblocking(),
            image_alignment_audit_contract=dict(schema_version=CONTRACT,enabled=alignment_audit.enabled(),
                generated_camera_pose='expected textual motion only; NOT measured'))
    def _get_pipe(self):
        if self._pipe is None:
            import torch
            from diffusers import QwenImage21Pipeline
            self._pipe=QwenImage21Pipeline.from_pretrained(self.model_dir,local_files_only=True,torch_dtype=torch.bfloat16).to(self.device)
        return self._pipe
    def _get_glm(self):
        if self._glm is None:self._glm=GLMReviewClient()
        return self._glm
    def _get_itm(self):
        if self._itm is None:
            from .service import ApexNavITMClient
            self._itm=ApexNavITMClient()
        return self._itm
    def decide(self,payload):
        plans=validate_plans(payload);request_id=segment(payload['request_id']);started=time.monotonic()
        policy=payload['score_policy'];rw=policy.get('real_weight',.25);pw=policy.get('imagination_weight',.75)
        if not number(rw) or not number(pw) or rw<0 or pw<0 or rw+pw<=0:raise ValueError('invalid image score weights')
        with self._lock:
            directory=None;generated=[];stage='credentials'
            try:
                glm=self._get_glm() if self.glm_required else None
                directory=self.output_root/request_id;directory.mkdir(parents=True,exist_ok=False)
                raw,real_image=jpeg(payload['observation']['rgb_jpeg_b64']);(directory/'anchor_original.jpg').write_bytes(raw)
                locator=glm
                if self.geometry_roi:
                    from .frontier_geometry import GeometryLocator
                    from .local_semantic_client import LocalSemanticClient
                    locator=GeometryLocator(lambda:LocalSemanticClient(model_path=self.model_dir,pipe=self._get_pipe()))
                if self.semantic_roi:
                    from .semantic_interior import SemanticInteriorLocator
                    if self.roi_locator=='bundled':
                        from .local_semantic_client import LocalSemanticClient
                        # Share the already-resident image encoder; no second model copy.
                        locator=SemanticInteriorLocator(LocalSemanticClient(model_path=self.model_dir,pipe=self._get_pipe()))
                    else:locator=SemanticInteriorLocator(glm)
                if self.interior and self.roi_locator=='gdino':
                    from .local_interior import LocalInteriorLocator
                    # Fresh instance per request; detector results never cross RGB/pose boundaries.
                    locator=LocalInteriorLocator(payload['observation']['rgb_jpeg_b64'])
                generated=[]
                stage='image_generation'
                for plan in plans:
                    prompt,relative=route_prompt([a['id'] for a in plan['actions']],payload['camera']);tick=time.monotonic()
                    condition=real_image;image_kwargs={};width,height=self.width,self.height
                    selection=dict(mode='route_full_image',reason='interior_mode_disabled')
                    if self.interior:
                        hint=interior_reference.direction_hint(plan,payload['camera'],real_image.size)
                        if self.geometry_roi:
                            roi=plan['source_frontier'].get('geometry_roi',{})
                            if (roi.get('contract')!='original_cluster_projection_v1' or
                                roi.get('step_index')!=payload['episode']['step_index'] or
                                roi.get('query_id')!=plan['source_frontier'].get('query_id') or
                                roi.get('frontier_xy')!=[plan['source_frontier']['x'],plan['source_frontier']['y']]):
                                raise ValueError('missing_or_stale_original_cluster_projection')
                            # A cluster may intersect the FOV even when its centroid does not.
                            hint.update(eligible=bool(roi.get('visible')),geometry_roi=roi,
                                width=real_image.width,height=real_image.height,reason=roi['reason'])
                        if self.roi_v2:hint['roi_policy']='content_roi_v2'
                        selection=dict(mode='route_full_image',hint=hint,reason=hint['reason'])
                        if hint['eligible']:
                            roi_start=time.monotonic()
                            detected_box=None
                            try:
                                located=locator.locate_interior(payload['observation']['rgb_jpeg_b64'],hint)
                                crop_hint=located.get('crop_hint',hint) if (self.semantic_roi or self.geometry_roi) else hint
                                detected_box=interior_reference.validate_box(located,crop_hint)
                                if self.geometry_roi:
                                    box=detected_box;quality=dict(policy='original_cluster_projection_v1')
                                elif self.semantic_roi:
                                    from .semantic_interior import check_crop
                                    box,quality=check_crop(real_image,detected_box,crop_hint)
                                elif self.roi_v2:
                                    box,quality=interior_reference.refine_box(real_image,detected_box,hint)
                                else:
                                    box=interior_reference.inset_box(detected_box);quality=None
                                condition=real_image.crop(box);prompt=interior_reference.PROMPT
                                width,height=768,576
                                image_kwargs=dict(true_cfg_scale=3.,negative_prompt='')
                                selection.update(mode='interior_crop',reason='automatic_directional_interior_roi',box=list(box),
                                    detected_box=list(detected_box),box_1000=located.get('box_1000'),
                                    inset_policy='content_roi_v2' if self.roi_v2 else '25 percent each horizontal edge, 5 percent vertical; min24x32; heuristic not segmentation',
                                    quality=quality,
                                    camera_pose_claim='unconstrained room hypothesis; route endpoint NOT imposed')
                                if self.semantic_roi:selection['inset_policy']='semantic_inner_patch_v1; side-edge refinement inside semantic region'
                                if self.semantic_roi:selection['crop_hint']=crop_hint
                                if self.geometry_roi:
                                    selection['inset_policy']='original_cluster_projection_v1; VLM trims only the fixed projected region'
                                    selection['crop_hint']=crop_hint
                            except (GLMAPIError,ValueError) as exc:
                                selection['reason']='roi_unavailable:'+ (str(exc) if type(exc) is ValueError else exc.diagnostic['kind'])
                                if self.roi_v2 and detected_box is not None:
                                    selection['rejected_box']=list(detected_box)
                                    selection['rejected_quality']=interior_reference.crop_quality(real_image.crop(detected_box))
                                if isinstance(exc,GLMAPIError):selection['api_error']=exc.diagnostic
                            selection.update(localization_ms=(time.monotonic()-roi_start)*1000,
                                localization_attempts=locator.last_attempts,localization_usage=locator.last_usage,
                                localization_source=getattr(locator,'localization_source','live_glm_api'))
                            if self.roi_locator=='gdino':selection['local_detector']=locator.last_diagnostic
                            if self.semantic_roi:selection['semantic_locator']=locator.last_diagnostic
                            if self.geometry_roi:selection['geometry_locator']=dict(locator.last_diagnostic)
                    if self.geometry_roi and selection['mode']!='interior_crop':
                        selection['mode']='skip_keep_real_score'
                        generated.append(dict(plan_id=plan['plan_id'],path=None,sha256=None,generation_ms=0.,
                            relative_pose=relative,actual_width=0,actual_height=0,
                            actions=[a['id'] for a in plan['actions']],source_frontier=plan['source_frontier'],
                            selection=selection,input_path=None,prompt=None,geometry_skipped=True))
                        continue
                    input_path=directory/(segment(plan['plan_id'])+'_input.png');condition.save(input_path)
                    model_images=condition.convert('RGBA')
                    input_paths=[str(input_path)]
                    if self.dual_context:
                        model_images=[real_image.convert('RGBA'),condition.convert('RGBA')]
                        input_paths=[str(directory/'anchor_original.jpg'),str(input_path)]
                        prompt=('Image 1 is the complete current observation, used only for environmental context. '
                            'Image 2 is the local reference for ONE exploration branch, cropped from image 1 '
                            'at pixel box '+str(selection['box'])+' in image size '+str(real_image.size)+'. '
                            'Imagine the unseen interior associated with image 2. Use image 1 for spatial and visual '
                            'context, but do not switch to another region in image 1, reproduce the full observation, '
                            'or output a collage. Produce one interior photograph. '+prompt)
                    selection.update(conditioning_image_count=len(input_paths),conditioning_input_paths=input_paths,
                        conditioning_sha256=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in input_paths])
                    request_path=directory/(segment(plan['plan_id'])+'_request.json')
                    request_path.write_text(json.dumps(dict(plan_id=plan['plan_id'],prompt=prompt,selection=selection,
                        source_frontier=plan['source_frontier'],anchor_sha256=hashlib.sha256(raw).hexdigest(),
                        input_path=str(input_path),input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                        width=width,height=height,seed=self.seed,steps=8,**image_kwargs),ensure_ascii=False,indent=2)+'\n')
                    import torch
                    generator=torch.Generator('cpu').manual_seed(self.seed)
                    image=self._get_pipe()(prompt=prompt,image=model_images,width=width,height=height,
                        output_resolution=max(width,height),generator=generator,num_inference_steps=8,use_kv_cache=True,**image_kwargs).images[0]
                    if not isinstance(image,Image.Image):raise ValueError('pipeline returned no valid image')
                    buffer=io.BytesIO();image.convert('RGB').save(buffer,format='JPEG',quality=95);data=buffer.getvalue()
                    path=directory/(segment(plan['plan_id'])+'.jpg');path.write_bytes(data)
                    generated.append(dict(plan_id=plan['plan_id'],jpeg_b64=base64.b64encode(data).decode(),relative_pose=relative,
                        path=str(path),sha256=hashlib.sha256(data).hexdigest(),generation_ms=(time.monotonic()-tick)*1000,
                        actual_width=image.width,actual_height=image.height,
                        actions=[a['id'] for a in plan['actions']],source_frontier=plan['source_frontier'],selection=selection,
                        input_path=str(input_path),prompt=prompt))
                stage='glm_review';tick=time.monotonic()
                review_status='reviewed';review_error=None
                try:
                    if self.review_enabled:
                        reviews=reviews_by_id(glm.review(payload['observation']['rgb_jpeg_b64'],generated,payload['target']),{p['plan_id'] for p in plans})
                        reviews={pid:dict(review,status='reviewed') for pid,review in reviews.items()}
                    else:
                        review_status='disabled'
                        reviews={p['plan_id']:dict(plan_id=p['plan_id'],status='disabled',
                            reason='Post-generation GLM diagnostics disabled; NOT reviewed') for p in plans}
                except (GLMAPIError,ValueError) as exc:
                    if not review_nonblocking():raise
                    review_status='unreviewed'
                    review_error=exc.diagnostic if isinstance(exc,GLMAPIError) else {'kind':type(exc).__name__}
                    reviews={p['plan_id']:dict(plan_id=p['plan_id'],status='unreviewed',
                        reason='GLM unavailable or invalid review; NOT reviewed',error=review_error) for p in plans}
                glm_ms=(time.monotonic()-tick)*1000;tick=time.monotonic()
                stage='blip_scoring';scores=[]
                target=payload['target'];itm=self._get_itm()
                anchor_score=None
                if any(not c.get('geometry_skipped') for c in generated):
                    anchor_score=itm.cosine(payload['observation']['rgb_jpeg_b64'],target['category'],target.get('related_room','everywhere'))
                    if not number(anchor_score) or not -1<=anchor_score<=1:raise ValueError('invalid anchor BLIP score')
                for candidate,plan in zip(generated,plans):
                    if candidate.get('geometry_skipped'):
                        scores.append(dict(plan_id=plan['plan_id'],frontier_rank=plan['source_frontier'].get('rank'),
                            score_mode='image_glm',image_glm_contract=CONTRACT,geometry_skipped=True,
                            geometry_contract='original_cluster_projection_v1',imagination_evaluated=False,
                            imagination_bonus=0.,
                            real_semantic_score=plan['source_frontier']['semantic_score'],
                            decision_semantic_score=plan['source_frontier']['semantic_score'],
                            anchor_id=plan['frontier_anchor']['observation_id']))
                        continue
                    predicted=itm.cosine(candidate['jpeg_b64'],target['category'],target.get('related_room','everywhere'))
                    if not number(predicted) or not -1<=predicted<=1:raise ValueError('invalid predicted BLIP score')
                    real=float(plan['source_frontier']['semantic_score']);final=(rw*real+pw*predicted)/(rw+pw)
                    scores.append(dict(plan_id=plan['plan_id'],frontier_rank=plan['source_frontier'].get('rank'),score_mode='image_glm',
                        image_glm_contract=CONTRACT,image_score=predicted,anchor_itm_score=anchor_score,real_semantic_score=real,
                        imagined_itm_score=predicted,imagination_bonus=final-real,decision_semantic_score=final,
                        imagination_reliability=1.,imagination_evaluated=True,evidence_gain=0.,
                        observation_evidence=plan['source_frontier'].get('observation_evidence',0),
                        anchor_id=plan['frontier_anchor'].get('observation_id',''),glm_review=reviews[plan['plan_id']]))
                audit=dict(schema_version=CONTRACT,request_id=request_id,episode=payload.get('episode'),contract=CONTRACT,
                    anchor_sha256=hashlib.sha256(raw).hexdigest(),anchor_path=str(directory/'anchor_original.jpg'),
                    anchor_id=plans[0]['frontier_anchor'].get('observation_id'),
                    anchor_pose={key:plans[0]['frontier_anchor'][key] for key in ('robot_x','robot_y','robot_yaw_rad','camera_pitch_rad','step_index')},
                    route_contract=CURRENT_ROUTE_CONTRACT,alignment_audit=payload.get('alignment_audit'),
                    seed=self.seed,width=self.width,height=self.height,interior_reference_enabled=self.interior,
                    camera=payload['camera'],target=payload['target'],candidates=[dict(plan_id=c['plan_id'],image_path=c['path'],
                        image_sha256=c['sha256'],actions=c['actions'],relative_pose=c['relative_pose'],generation_ms=c['generation_ms'],
                        actual_width=c['actual_width'],actual_height=c['actual_height'],selection=c['selection'],input_path=c['input_path'],prompt=c['prompt'],
                        min_prefix_length=plans[i]['min_prefix_length'],source_frontier=c['source_frontier'],review=reviews[c['plan_id']],score=scores[i]) for i,c in enumerate(generated)],
                    interior_roi_policy='original_cluster_projection_v1' if self.geometry_roi else ('semantic_inner_patch_v1' if self.semantic_roi else ('content_roi_v2' if self.roi_v2 else 'legacy_inset')),
                    semantic_roi=self.semantic_roi,
                    interior_roi_locator=self.roi_locator,glm_required=self.glm_required,
                    frontier_geometry_contract='original_cluster_projection_v1' if self.geometry_roi else None,
                    dual_context=self.dual_context,
                    local_detector_requests=0 if self.geometry_roi else getattr(locator,'total_requests',0),
                    geometry_deframe_requests=getattr(locator,'total_requests',0) if self.geometry_roi else 0,
                    glm_review_enabled=self.review_enabled,
                    glm_latency_ms=glm_ms,glm_attempts=getattr(glm,'last_attempts',1) if self.review_enabled else 0,
                    glm_review_status=review_status,glm_error=review_error,glm_nonblocking=review_nonblocking(),
                    glm_usage=getattr(glm,'last_usage',{}) if self.review_enabled else {},blip_latency_ms=(time.monotonic()-tick)*1000,
                    latency_ms=(time.monotonic()-started)*1000,sampling='saved8-step; interior crop CFG3/768x576; legacy route CFG1/configured resolution; no sigma override')
                if not self._injected:
                    try:
                        if torch.cuda.is_available():audit['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated(self.device)
                    except Exception:audit['cuda_peak_allocated_bytes']=None
                (directory/'verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
                return dict(decision='call_wm',score_mode='image_glm',image_glm_contract=CONTRACT,candidate_scores=scores,
                    selected_action=payload['navigation'].get('baseline_action'),selected_plan_id=None,
                    artifacts=[c['path'] for c in generated if c['path']],latency_ms=audit['latency_ms'],
                    glm_review_status=review_status,glm_error=review_error,
                    workload=dict(generated_images=sum(not c.get('geometry_skipped',False) for c in generated),
                        geometry_skipped=sum(c.get('geometry_skipped',False) for c in generated)),
                    reason=('original cluster projection; generated=%d, skipped_keep_real=%d' %
                        (sum(not c.get('geometry_skipped',False) for c in generated),sum(c.get('geometry_skipped',False) for c in generated))
                        if self.geometry_roi else 'single-view generated/BLIP scored; GLM '+review_status+'; diagnostics do not gate'))
            except Exception as exc:
                # Only the exception class is exposed: HTTP URLs/auth/provider bodies
                # and credential-containing exception strings never enter artifacts.
                diagnostic=exc.diagnostic if isinstance(exc,GLMAPIError) else {}
                if directory is not None:
                    try:
                        (directory/'verification.json').write_text(json.dumps(dict(schema_version=CONTRACT,request_id=request_id,
                            episode=payload.get('episode'),technical_failure=True,error_type=type(exc).__name__,failure_stage=stage,
                            api_error=diagnostic,
                            candidates=[dict(plan_id=c['plan_id'],image_path=c['path'],image_sha256=c['sha256'],
                                actions=c['actions'],relative_pose=c['relative_pose'],generation_ms=c['generation_ms']) for c in generated]),indent=2)+'\n')
                    except OSError:pass
                return dict(decision='fallback',score_mode='image_glm',image_glm_contract=CONTRACT,candidate_scores=[],
                    technical_failure=True,failure_stage=stage,api_error=diagnostic,
                    reason='image_glm technical failure: '+type(exc).__name__,artifacts=[],
                    latency_ms=(time.monotonic()-started)*1000)
