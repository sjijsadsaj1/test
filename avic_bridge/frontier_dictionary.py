"""Real original-frontier crop -> independent dictionary object probabilities."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import time
import uuid
from .probability_contract import binding, decimal_probability, probability

CONTRACT='frontier_object_marginals_v1'
PROMPT='''Estimate what objects may be found in the immediate unexplored space reached through ONE navigation frontier.
Image 1 is a REAL full observation with a red rectangle marking the projected frontier region.
Image 2, when supplied, is the exact unmodified crop inside that rectangle.
The rectangle comes from the ORIGINAL map frontier cluster and registered camera geometry;
it is not an object detection or a guaranteed door opening. A frontier can be a hallway continuation.
Use the full observation to understand the crop's context, but do not borrow objects elsewhere in
the full image as evidence that they are behind this frontier. Retain door frames as context.
Describe visible evidence separately from your conjecture about the space beyond this region.
For EVERY dictionary object independently estimate the probability that at least one instance
would be found in that space. Several objects can co-occur. These are separate marginal events,
NOT mutually exclusive classes: DO NOT normalize, softmax, divide by their sum, or force the sum
to 1 or 100%. The same object may also occur behind several different frontiers.
If no crop is geometrically available, acknowledge that the branch is unobserved; use only
context-based conjecture, not invented visible evidence. Occlusion is not proof of absence.
There is no designated search target in this request. Treat all dictionary entries equally.
Return JSON only, with ALL dictionary keys once. The program binds the result to
the single supplied frontier; do not output or invent candidate IDs:
{"visible_evidence":"brief actual evidence",
"predicted_space":"brief conjecture beyond this frontier, not a factual observation",
"probabilities":{"<exact_dictionary_key>":"0.20"}}.
The example is a schema illustration, not an exhaustive dictionary or prior. Encode every
probability as a quoted decimal in [0,1]. Do not choose navigation actions or declare success.'''


def dictionary():
    path=Path(os.environ.get('AVIC_OBJECT_DICTIONARY',str(Path(__file__).resolve().parents[1]/'config/frontier_object_dictionary.json')))
    raw=path.read_bytes();d=json.loads(raw);items=d['objects']
    if not isinstance(items,list) or not items or len(set(items))!=len(items) or any(not isinstance(x,str) or not x for x in items):
        raise ValueError('invalid object dictionary')
    return dict(version=d['version'],objects=items,sha256=hashlib.sha256(raw).hexdigest())


def validate(row,frontier_id,objects):
    if not isinstance(row,dict):raise ValueError('invalid dictionary response')
    probs=row.get('probabilities')
    if not isinstance(probs,dict) or set(probs)!=set(objects):
        raise ValueError('missing or extra dictionary object')
    if not all(isinstance(row.get(k),str) for k in ('visible_evidence','predicted_space')):
        raise ValueError('missing evidence/conjecture')
    # Independent marginals. Deliberately NO sum check or normalization.
    return dict(frontier_id=frontier_id,visible_evidence=row['visible_evidence'],predicted_space=row['predicted_space'],
                probabilities={k:decimal_probability(probs[k]) for k in objects})


def real_inputs(candidate):
    from PIL import Image,ImageDraw
    import numpy as np
    from .frontier_crop import CONTRACT as CROP_CONTRACT
    view=candidate['view'];roi=view['roi'];f=candidate['frontier']
    cells=np.asarray(f['cluster_cells_xy'],dtype='<f8')
    if (roi.get('contract')!=CROP_CONTRACT or roi.get('source_step')!=view['step_index'] or
        roi.get('frontier_xy')!=[f['x'],f['y']] or
        roi.get('cluster_sha256')!=hashlib.sha256(cells.tobytes()).hexdigest()):
        raise ValueError('crop not bound to original frontier cluster/view')
    raw=base64.b64decode(view['rgb_jpeg_b64'],validate=True)
    image=Image.open(io.BytesIO(raw)).convert('RGB')
    if roi['image_size']!=list(image.size):raise ValueError('crop image dimensions mismatch')
    box=roi.get('box');context=image.copy();crop=None
    if box is not None:
        if (len(box)!=4 or any(type(v)is not int for v in box) or
            not 0<=box[0]<box[2]<=image.width or not 0<=box[1]<box[3]<=image.height):
            raise ValueError('crop outside true image')
        crop=image.crop(box)
        ImageDraw.Draw(context).rectangle([box[0],box[1],box[2]-1,box[3]-1],outline='red',width=3)
    return image,context,crop,hashlib.sha256(raw).hexdigest()


class DictionaryFrontierModel:
    def __init__(self,provider='local',client=None):
        from .vlm_transport import make_client
        self.client=client if client is not None else make_client(provider)
        self.provider=provider

    def complete_candidate(self,content,frontier_id,objects,directory,stem):
        """Retain valid marginals; ask only for missing/invalid entries.

        No invented defaults, renormalization or overwriting a previously valid p.
        Bounded grouped repairs are followed by small single-object requests.
        Persistent failures still surface explicitly rather than becoming p=0.
        """
        from .vlm_transport import VLMResponseError
        accepted={};evidence={};sources={};calls=0
        def request(keys,repair):
            nonlocal calls
            attempt=calls;calls+=1
            prefix=directory/(stem+'_attempt_%d'%attempt)
            prompt=content
            if repair:
                prompt=content+[dict(type='text',text=(
                    'This is a completion request for the SAME frontier and SAME real images. '
                    'Previously valid probabilities are retained by the program. Estimate ONLY the following '
                    'missing or invalid object entries independently; do not regenerate the entire dictionary. '
                    'Return visible_evidence, predicted_space and probabilities. Use ALL these exact keys: '
                    +json.dumps(keys)+'. Quoted decimal values in [0,1], no normalization. '
                    'Uncertainty requires an estimated probability, not an omitted entry.'))]
            (Path(str(prefix)+'_request.json')).write_text(json.dumps(
                dict(requested_objects=keys,repair=repair,extra_instruction=prompt[-1].get('text') if repair else None),indent=2)+'\n')
            problem=None;extra=[]
            self.client.last_answer=''
            try:
                raw=self.client.request(prompt,max_tokens=600 if len(keys)==1 else 1200)
            except VLMResponseError as exc:
                raw=None;problem=dict(kind=type(exc).__name__,detail=str(exc))
            except Exception as exc:
                Path(str(prefix)+'_fatal.json').write_text(json.dumps(dict(
                    kind=type(exc).__name__,detail=str(exc),stage='model_or_input',retry_as_dictionary=False),indent=2)+'\n')
                Path(str(prefix)+'_raw.txt').write_text(getattr(self.client,'last_answer',''))
                raise
            try:
                if raw is None:raise ValueError('no parsed response; see response error')
                Path(str(prefix)+'.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
                if not isinstance(raw,dict):raise ValueError('invalid dictionary response')
                for field in ('visible_evidence','predicted_space'):
                    if field not in evidence and isinstance(raw.get(field),str):evidence[field]=raw[field]
                probs=raw.get('probabilities')
                if not isinstance(probs,dict):raise ValueError('missing probability mapping')
                extra=sorted(set(probs)-set(keys))
                for key in keys:
                    if key in accepted or key not in probs:continue
                    try:value=decimal_probability(probs[key])
                    except (ValueError,TypeError):continue
                    accepted[key]=value;sources[key]=attempt
            except (ValueError,KeyError,TypeError) as exc:
                if problem is None:problem=dict(kind=type(exc).__name__,detail=str(exc))
            remaining=[key for key in objects if key not in accepted]
            missing_evidence=[k for k in ('visible_evidence','predicted_space') if k not in evidence]
            audit=dict(requested_objects=keys,remaining=remaining,missing_evidence=missing_evidence,
                       ignored_extra_keys=extra,accepted_sources=sources.copy(),error=problem)
            Path(str(prefix)+'_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
            Path(str(prefix)+'_raw.txt').write_text(getattr(self.client,'last_answer',''))
            return remaining,missing_evidence

        remaining,missing_evidence=request(objects,False)
        for _ in range(2):
            if not remaining and not missing_evidence:break
            remaining,missing_evidence=request(remaining,True)
        # Repeated omissions in a large object list must not discard its valid values.
        for key in list(remaining):
            for _ in range(2):
                remaining,missing_evidence=request([key],True)
                if key in accepted:break
        row=dict(evidence,probabilities=accepted)
        Path(directory/(stem+'_completion.json')).write_text(json.dumps(dict(
            calls=calls,probability_sources=sources,remaining=remaining,missing_evidence=missing_evidence,
            merged_response=row,normalization=False),ensure_ascii=False,indent=2)+'\n')
        return validate(row,frontier_id,objects),calls

    def predict(self,payload,directory):
        from .vlm_transport import image_part,model_image
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        if payload.get('contract')!=CONTRACT or payload.get('dictionary')!=dictionary():
            raise ValueError('dictionary contract/version mismatch')
        objects=payload['dictionary']['objects'];cs=payload['candidates']
        ids=[c['id'] for c in cs]
        if not 1<=len(cs)<=3 or len(set(ids))!=len(ids):raise ValueError('invalid candidate identities')
        write=lambda name,d:(directory/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
        start=time.monotonic();rows=[];inputs=[];attempts=0
        for idx,c in enumerate(cs):
            image,context,crop,sha=real_inputs(c)
            stem='candidate_%d'%idx
            image.save(directory/(stem+'_real.png'));context.save(directory/(stem+'_context.png'))
            if crop is not None:crop.save(directory/(stem+'_crop.png'))
            meta=dict(frontier_id=c['id'],dictionary=objects,crop_box=c['view']['roi']['box'],
                crop_available=crop is not None,bearing_rad=c['view']['bearing_rad'],view_age_steps=c['view']['age_steps'])
            input_layout=[model_image(im)[1] for im in ([context,crop] if crop is not None else [context])]
            if any(layout['padding'] for layout in input_layout):
                meta['image_input_layout']=input_layout
                meta['padding_note']='Neutral gray padding only fits the model input. It is NOT a wall, room, occlusion or other scene evidence. The real crop pixels inside content_box are unchanged.'
            content=[dict(type='text',text=PROMPT+'\n'+json.dumps(meta)),image_part(context)]
            if crop is not None:content.append(image_part(crop))
            actual_files=[]
            for n,item in enumerate(x for x in content if x['type']=='image_url'):
                path=stem+'_model_input_%d.jpg'%n
                (directory/path).write_bytes(base64.b64decode(item['image_url']['url'].split(',',1)[1]))
                actual_files.append(path)
            inputs.append(dict(id=c['id'],frontier=c['frontier'],view={k:v for k,v in c['view'].items() if k!='rgb_jpeg_b64'},
                image_sha256=sha,real_file=stem+'_real.png',context_file=stem+'_context.png',
                crop_file=stem+'_crop.png' if crop is not None else None,actual_model_input_files=actual_files,
                image_input_layout=input_layout))
            write('input.json',dict(contract=CONTRACT,request_id=payload['request_id'],episode=payload['episode'],
                dictionary=payload['dictionary'],target_for_navigation_only=payload.get('target'),
                model=self.client.model,provider=self.provider,prompt=PROMPT,candidates=inputs))
            # One candidate per request eliminates cross-frontier list assignment
            # and score competition; each dictionary entry still has its own p.
            row,calls=self.complete_candidate(content,c['id'],objects,directory,stem)
            attempts+=calls
            rows.append(dict(id=c['id'],**{k:row[k] for k in ('visible_evidence','predicted_space','probabilities')},crop_available=crop is not None))
        result=dict(contract=CONTRACT,request_id=payload['request_id'],binding_sha256=binding(payload),
            dictionary=payload['dictionary'],candidates=rows,provider=self.provider,model=self.client.model,
            inference_seconds=time.monotonic()-start,vlm_calls=attempts,generated_images=0,
            probability_status='independent, uncalibrated object-presence marginals; no normalization')
        write('result.json',result);return result


def decide_dictionary(c,args):
    from .controller import AvicDecision
    from .frontier_crop import select_view
    step=args['step_index'];action=args['baseline_action'];c.current_step=step
    if not c.enabled or action not in (1,2,3):return c._skip(action,'dictionary: disabled or protected stop/look')
    fs=list(args.get('frontier_candidates') or [])
    if step<c.config.warmup_steps or len(fs)<2 or args['final_state']!=0 or args['exploration_result']!=0:
        return c._skip(action,'dictionary: not a multi-frontier exploration decision')
    query=fs[0].get('query_id',-1)
    if query<0 or not fs[0].get('query_active') or query==c.last_frontier_query_id:return c._skip(action,'dictionary: no new query')
    if any(f.get('query_id')!=query for f in fs):raise ValueError('inconsistent query IDs')
    c.last_frontier_query_id=query
    if c.world_model_calls>=c.config.max_calls_per_episode or step-c.last_world_model_step<c.config.cooldown_steps:
        print('[DICTIONARY_SKIP] '+json.dumps(dict(step=step,query_id=query,reason='explicit_diagnostic_budget')),flush=True)
        return c._skip(action,'dictionary: explicit diagnostic call budget/cooldown')
    vocab=dictionary();target=args['target_category'].split('|')[0].strip().replace(' ','_')
    if target not in vocab['objects']:raise ValueError('navigation target absent from fixed dictionary')
    candidates=[dict(id='f%d'%f['rank'],frontier=f,view=select_view(c,f,step)) for f in fs[:3]]
    payload=dict(contract=CONTRACT,request_id=uuid.uuid4().hex,dictionary=vocab,target=target,
        episode=dict(scene_id=args['scene_id'],episode_id=str(args['episode_id']),step_index=step,
                     generation_epoch=c.generation_epoch,query_id=query),candidates=candidates)
    c.world_model_calls+=1;c.last_world_model_step=step;start=time.monotonic();response=c.client.decide(payload)
    if response.get('contract')!=CONTRACT or response.get('binding_sha256')!=binding(payload) or response.get('dictionary')!=vocab:
        raise ValueError('dictionary response input association mismatch')
    rs=response['candidates'];ids=[r.get('id') for r in rs]
    if len(ids)!=len(candidates) or set(ids)!={c['id'] for c in candidates}:raise ValueError('candidate response identity mismatch')
    lookup={r['id']:r for r in rs};scores=[]
    for candidate in candidates:
        f=candidate['frontier'];r=lookup[candidate['id']]
        r=validate(dict(r,frontier_id=r['id']),candidate['id'],vocab['objects']);p=r['probabilities'][target]
        scores.append(dict(plan_id=candidate['id'],query_id=query,rank=f['rank'],x=f['x'],y=f['y'],
            real_semantic_score=f['semantic_score'],decision_semantic_score=p,target_belief=p,
            # The legacy ROS wire has an imagined_itm slot; it carries p here,
            # never an image/BLIP score. No reward/delta is computed or applied.
            imagined_itm_score=p,score_mode='text_frontier',
            text_contract=CONTRACT,binding_sha256=response['binding_sha256'],dictionary=vocab,
            object_probabilities=r['probabilities'],verifier_reason=r['predicted_space']))
    chosen=max(scores,key=lambda s:(s['target_belief'],s['real_semantic_score']))['plan_id']
    print('[FRONTIER_DICTIONARY] '+json.dumps(dict(step=step,query_id=query,target=target,chosen=chosen,
        request_id=payload['request_id'],scores={s['plan_id']:s['target_belief'] for s in scores},
        dictionary_size=len(vocab['objects']),probability_sum_constraint=False,
        crops={c['id']:c['view']['roi']['box'] for c in candidates})),flush=True)
    return AvicDecision(baseline_action=action,executed_action=action,service_decision='call_wm',
        selected_plan_id=chosen,reason=CONTRACT,latency_ms=(time.monotonic()-start)*1000,used_world_model=True,
        overridden=False,frontier_scores=scores,stage_latencies_ms=dict(dictionary=response['inference_seconds']*1000),
        workload=dict(generated_images=0,vlm_calls=response['vlm_calls'],frontiers=len(scores)))
