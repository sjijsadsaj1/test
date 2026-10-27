"""Real directional views -> textual space hypotheses -> subjective goal beliefs.

No generated image, BLIP rescoring, simulator ground truth or persistent score-map
write. Two separate calls keep the search goal out of the scene hypothesis call.
The resulting numbers are uncalibrated model beliefs, not measured frequencies.
"""
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

CONTRACT = 'text_frontier_belief_v1'
SCENE_PROMPT = '''Describe plausible unexplored space reached through each supplied navigation frontier.
You see REAL camera photographs only. There is no search target in this stage.
Each candidate has its own image and a bearing measured in that image's camera
frame (positive is left). The projected column is a direction, NOT a measured
door mask; an obstacle can occlude it. Do not select another doorway or object.
An image can be historical. Never claim that an out-of-view or occluded branch
is visible. Distinguish visible evidence from conjecture about space beyond the
frontier. Do not invent evidence, assume every branch is a room, or complete a
scene merely to fill the schema. Preserve candidate IDs independent of order.
For each candidate give a short visible_evidence, visibility (visible, occluded,
or outside_view), and 1 to 3 alternative hypotheses. Each hypothesis has a short
space description and weight between 0 and 1; weights sum to 1 per candidate.
Use an 'unknown space' hypothesis for unresolved alternatives. Hypotheses should
describe room function or continuation and plausible contents without treating
them as observations. Keep descriptions concise, under 25 words per hypothesis.
Return JSON only: {"candidates":[{"id":"f0","visible_evidence":"...",
"visibility":"visible","hypotheses":[{"space":"...","weight":0.6},
{"space":"unknown space","weight":0.4}]}]}.'''

GOAL_PROMPT = '''Estimate whether at least one instance of the requested object
would be found in the immediate space represented by each textual hypothesis.
These descriptions are uncertain hypotheses, NOT observed or generated images.
Do not change or embellish the descriptions to accommodate the target. Use common
indoor spatial knowledge. Unknown space is not evidence of target absence.
Output one conditional subjective probability p_target in [0,1] and a short
reason for every supplied hypothesis ID. Probabilities across hypotheses need
not sum to 1: they are separate conditional events. Do not decide STOP or task
success. Encode every p_target as a quoted decimal string, for example "0.20".
Return JSON only: {"likelihoods":[{"id":"h0","p_target":"0.20",
"reason":"..."}]}.'''


def enabled():
    return os.environ.get('APEXNAV_TEXT_FRONTIER') == '1'


def finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def probability(v):
    if not finite(v) or not 0 <= v <= 1:
        raise ValueError('invalid subjective probability')
    return float(v)


def decimal_probability(v):
    """Explicit wire representation, not permissive JSON/error repair."""
    if isinstance(v,str):
        if not re.fullmatch(r'(?:0(?:\.\d+)?|1(?:\.0+)?)',v):
            raise ValueError('invalid decimal probability string')
        v=float(v)
    return probability(v)


def binding(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_candidates(rows, expected):
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError('missing candidate response')
    ids = [r.get('id') for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError('unknown/duplicated candidate identity')
    for r in rows:
        if r.get('visibility') not in ('visible', 'occluded', 'outside_view'):
            raise ValueError('invalid visibility')
        if not isinstance(r.get('visible_evidence'), str):
            raise ValueError('missing real evidence')
        hs = r.get('hypotheses')
        if not isinstance(hs, list) or not 1 <= len(hs) <= 3:
            raise ValueError('invalid hypotheses')
        for h in hs:
            if not isinstance(h.get('space'), str) or not h['space'].strip():
                raise ValueError('missing hypothesis text')
            probability(h.get('weight'))
        total = sum(h['weight'] for h in hs)
        if abs(total - 1) > .02:
            raise ValueError('hypothesis weights do not sum to 1')
        # Only correct decimal rounding; never repair invalid model beliefs.
        for h in hs:
            h['weight'] /= total
    return rows


class TextFrontierModel:
    def __init__(self, provider='local', client=None):
        if provider not in ('local', 'glm'):
            raise ValueError('provider must be local or glm')
        if client is None:
            from .literature_pilot import LocalProbeClient
            from .image_glm_backend import GLMReviewClient
            client = LocalProbeClient() if provider == 'local' else GLMReviewClient()
        self.client = client
        self.provider = provider

    def ask(self, content, max_tokens):
        return self.client._request(dict(model=self.client.model,
            messages=[dict(role='user', content=content)], temperature=0,
            thinking=dict(type='disabled'), max_tokens=max_tokens,
            response_format=dict(type='json_object')))

    def predict(self, payload, directory):
        from PIL import Image
        from .literature_pilot import image_part
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        if payload.get('contract') != CONTRACT:
            raise ValueError('text frontier contract mismatch')
        cs = payload.get('candidates', [])
        ids = [c.get('id') for c in cs]
        if not 1 <= len(cs) <= 3 or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('invalid frontier identities')
        content = [dict(type='text', text=SCENE_PROMPT)]
        inputs = []
        for i, c in enumerate(cs):
            view = c['view']
            raw = base64.b64decode(view['rgb_jpeg_b64'], validate=True)
            image = Image.open(io.BytesIO(raw)).convert('RGB')
            bearing = view['bearing_rad']; hfov = view['hfov_deg']
            if not finite(bearing) or not finite(hfov) or not 0 < hfov < 180:
                raise ValueError('invalid camera geometry')
            in_fov = abs(bearing) < math.radians(hfov / 2)
            meta = dict(id=c['id'], bearing_degrees=round(math.degrees(bearing), 2),
                inside_horizontal_fov=in_fov,
                projected_column_1000=round(500-500*math.tan(bearing)/math.tan(math.radians(hfov/2))) if in_fov else None,
                view_age_steps=int(view['age_steps']),
                note='bearing only; visibility/occlusion must be assessed from the REAL image')
            content.extend([dict(type='text', text=json.dumps(meta)), image_part(image)])
            image.save(directory / ('real_%d.jpg' % i))
            inputs.append(dict(id=c['id'], frontier=c['frontier'],
                view={k:v for k,v in view.items() if k!='rgb_jpeg_b64'},
                image_sha256=hashlib.sha256(raw).hexdigest(), image='real_%d.jpg'%i, prompt_metadata=meta))
        write = lambda name, data: (directory/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
        write('input.json', dict(contract=CONTRACT, request_id=payload['request_id'],
            episode=payload.get('episode'), target=payload['target'], candidates=inputs,
            binding_sha256=binding(payload), provider=self.provider, model=self.client.model,
            scene_prompt=SCENE_PROMPT, goal_prompt=GOAL_PROMPT,
            note='model sees neither map scores nor previous choices nor task outcomes'))
        start = time.monotonic()
        # Explicit final identity list and one bounded schema retry. Never drop
        # invented candidates or fabricate missing beliefs to finish a run.
        identity_instruction='Return exactly %d candidates with IDs %s, once each. No additional IDs.'%(len(ids),json.dumps(ids))
        scene_content=content+[dict(type='text',text=identity_instruction)]
        scene_attempts=0
        for attempt in range(2):
            scene_attempts+=1
            try:
                scene = self.ask(scene_content, 1500)
                write('scene_attempt_%d.json'%attempt, scene)
                rows = validate_candidates(scene.get('candidates'), ids)
                break
            except (ValueError,KeyError,TypeError) as exc:
                (directory/('scene_attempt_%d_raw.txt'%attempt)).write_text(getattr(self.client,'last_answer',''))
                write('scene_attempt_%d_error.json'%attempt,dict(error=type(exc).__name__,detail=str(exc)))
                if attempt==1:raise
                scene_content=content+[dict(type='text',text=identity_instruction+
                    ' The previous response violated the JSON schema or candidate identity list. Regenerate valid JSON for only these candidates.')]
        write('scene_raw.json', scene)
        scene_seconds = time.monotonic()-start
        # Repeated descriptions share one estimate within this query. No online
        # outcome feedback or persistent semantic cache is used.
        hypotheses = []
        by_text = {}
        for r in rows:
            for h in r['hypotheses']:
                text = h['space'].strip()
                if text not in by_text:
                    by_text[text] = 'h%d' % len(hypotheses)
                    hypotheses.append(dict(id=by_text[text], space=text))
                h['hypothesis_id'] = by_text[text]
        goal_input = dict(target=payload['target'], hypotheses=hypotheses)
        start = time.monotonic()
        expected = {h['id'] for h in hypotheses}
        goal_text=GOAL_PROMPT+'\n'+json.dumps(goal_input)
        goal_attempts=0
        for attempt in range(2):
            goal_attempts+=1
            try:
                goal=self.ask([dict(type='text',text=goal_text)],1000)
                write('goal_attempt_%d.json'%attempt,goal)
                likelihoods=goal.get('likelihoods',[])
                got=[x.get('id') for x in likelihoods]
                if len(got)!=len(expected) or set(got)!=expected:
                    raise ValueError('invalid hypothesis scoring identities')
                lookup={x['id']:decimal_probability(x.get('p_target')) for x in likelihoods}
                break
            except (ValueError,KeyError,TypeError) as exc:
                (directory/('goal_attempt_%d_raw.txt'%attempt)).write_text(getattr(self.client,'last_answer',''))
                write('goal_attempt_%d_error.json'%attempt,dict(error=type(exc).__name__,detail=str(exc)))
                if attempt==1:raise
                goal_text=GOAL_PROMPT+'\n'+json.dumps(goal_input)+'\nReturn valid JSON with exactly these IDs: '+json.dumps(sorted(expected))+'. Encode probabilities as quoted decimals.'
        write('goal_raw.json', goal)
        for r in rows:
            for h in r['hypotheses']:
                h['p_target_given_space'] = lookup[h['hypothesis_id']]
            r['target_belief'] = sum(h['weight']*h['p_target_given_space'] for h in r['hypotheses'])
        result = dict(contract=CONTRACT, request_id=payload['request_id'],
            binding_sha256=binding(payload), candidates=rows, provider=self.provider,
            model=self.client.model, scene_seconds=scene_seconds,scene_attempts=scene_attempts,
            goal_seconds=time.monotonic()-start,goal_attempts=goal_attempts,generated_images=0,
            probability_status='uncalibrated subjective belief; not success probability')
        write('result.json', result)
        return result


def select_view(controller, frontier, current_step):
    """Nearest directional real observation, without using map/ITM scores.

    Bearing is not depth visibility. Both source pose and age remain explicit.
    If no directional image exists, send current view as outside-view evidence.
    """
    hfov = float(controller.rgb_hfov_deg)
    choices = []
    for v in controller.observation_history:
        dx=frontier['x']-v['robot_x'];dy=frontier['y']-v['robot_y']
        angle=math.atan2(dy, dx)-v['robot_yaw_rad']
        angle=math.atan2(math.sin(angle), math.cos(angle))
        if abs(v['camera_pitch_rad']) > .01:
            continue
        if abs(angle) < math.radians(hfov/2) and math.hypot(dx,dy) <= 8:
            choices.append((abs(angle)+.005*(current_step-v['step_index']), v, angle))
    if choices:
        _, view, angle = min(choices, key=lambda x:x[0])
    else:
        view=controller.observation_history[-1]
        angle=math.atan2(frontier['y']-view['robot_y'],frontier['x']-view['robot_x'])-view['robot_yaw_rad']
        angle=math.atan2(math.sin(angle),math.cos(angle))
    return dict(view, bearing_rad=angle, hfov_deg=hfov, age_steps=current_step-view['step_index'])


def decide_text(controller, args):
    from .controller import AvicDecision
    c=controller; step=args['step_index']; action=args['baseline_action']
    c.current_step=step
    if not c.enabled or action not in (1,2,3):
        return c._skip(action, 'text: disabled or protected stop/look')
    fs=list(args.get('frontier_candidates') or [])
    if step<c.config.warmup_steps or len(fs)<2 or args['final_state']!=0 or args['exploration_result']!=0:
        return c._skip(action, 'text: not a multi-frontier exploration decision')
    query=fs[0].get('query_id',-1)
    if query<0 or not fs[0].get('query_active') or query==c.last_frontier_query_id:
        return c._skip(action, 'text: no new frontier query')
    if any(f.get('query_id')!=query for f in fs):
        raise ValueError('text: inconsistent query identity')
    c.last_frontier_query_id=query
    if c.world_model_calls>=c.config.max_calls_per_episode or step-c.last_world_model_step<c.config.cooldown_steps:
        return c._skip(action, 'text: explicit call budget/cooldown')
    if not c.observation_history:
        raise ValueError('text: real observation missing')
    candidates=[dict(id='f%d'%f['rank'],frontier=f,view=select_view(c,f,step)) for f in fs[:3]]
    payload=dict(contract=CONTRACT, request_id=uuid.uuid4().hex,
        episode=dict(scene_id=args['scene_id'],episode_id=str(args['episode_id']),
                     step_index=step,generation_epoch=c.generation_epoch,query_id=query),
        target=args['target_category'],candidates=candidates)
    c.world_model_calls+=1;c.last_world_model_step=step
    start=time.monotonic()
    response=c.client.decide(payload)
    if response.get('contract')!=CONTRACT or response.get('binding_sha256')!=binding(payload):
        raise ValueError('text: response does not match exact images/poses/query')
    rows=response.get('candidates',[])
    validate_candidates(rows,[x['id'] for x in candidates])
    lookup={r['id']:r for r in rows}
    scores=[]
    for candidate in candidates:
        f=candidate['frontier'];r=lookup[candidate['id']]
        p=probability(r.get('target_belief'))
        expected=sum(probability(h['weight'])*probability(h['p_target_given_space']) for h in r['hypotheses'])
        if abs(expected-p)>1e-8:
            raise ValueError('text: incorrect hypothesis mixture')
        scores.append(dict(plan_id=candidate['id'],query_id=query,rank=f['rank'],x=f['x'],y=f['y'],
            real_semantic_score=f['semantic_score'],decision_semantic_score=p,
            # ROS legacy scalar slot, explicitly NOT a BLIP similarity.
            imagined_itm_score=p,imagination_bonus=p-f['semantic_score'],
            target_belief=p,score_mode='text_frontier',text_contract=CONTRACT,
            verifier_reason=r['visible_evidence'],binding_sha256=response['binding_sha256']))
    chosen=max(scores,key=lambda x:(x['target_belief'],x['real_semantic_score']))['plan_id']
    print('[TEXT_FRONTIER] '+json.dumps(dict(query_id=query,step=step,target=args['target_category'],
        baseline_ids=['f%d'%f['rank'] for f in fs if f.get('selected_by_apexnav')],
        chosen=chosen,beliefs={s['plan_id']:s['target_belief'] for s in scores},
        request_id=payload['request_id'],generated_images=0)),flush=True)
    return AvicDecision(baseline_action=action,executed_action=action,service_decision='call_wm',
        selected_plan_id=chosen,reason=CONTRACT,latency_ms=(time.monotonic()-start)*1000,
        used_world_model=True,overridden=False,frontier_scores=scores,
        stage_latencies_ms={k:response[k]*1000 for k in ('scene_seconds','goal_seconds')},
        workload=dict(generated_images=0,vlm_calls=response.get('scene_attempts',1)+response.get('goal_attempts',1),frontiers=len(scores)))
