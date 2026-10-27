"""Single-frontier video/Q25 scoring, without evidence gates or bonus logic."""
import json
import base64
import hashlib
import math
from pathlib import Path
import time
import numpy as np
from .protocol import make_candidate_plan
from .frontier_once import CONTRACT
from . import alignment_audit


def evaluate_once(backend, payload):
    plans = payload['navigation']['candidate_plans']
    if len(plans) != 1:
        raise ValueError('frontier_once streams exactly one new frontier per request')
    started = time.monotonic()
    prefixes, reason, errors = backend._select_prefixes(payload, plans)
    if errors or plans[0]['plan_id'] not in prefixes:
        raise RuntimeError('invalid Qwen prefix: '+str(errors))
    plan = plans[0]
    bound = make_candidate_plan(plan['plan_id'], prefixes[plan['plan_id']])
    bound.update(source_frontier=plan['source_frontier'], frontier_anchor=plan['frontier_anchor'],
                 max_rerank_frames=6)
    item = backend._get_svc().generate_plan(request_id=payload['request_id'], plan=bound,
        rgb_jpeg_b64=plan['frontier_anchor']['rgb_jpeg_b64'], camera=payload['camera'],
        generation_scope=payload['episode']['generation_epoch'])
    target = payload['target']
    values = [float(backend._get_itm().cosine(f['jpeg_b64'], target['category'],
              target['related_room'])) for f in item['future_frames']]
    if not values or any(not math.isfinite(v) or not -1 <= v <= 1 for v in values):
        raise RuntimeError('invalid future BLIP scores')
    imagined = float(np.quantile(values, 0.25))
    real = float(plan['source_frontier']['semantic_score'])
    score = dict(plan_id=plan['plan_id'], score_mode='frontier_once',
        real_semantic_score=real, imagined_itm_score=imagined,
        decision_semantic_score=(real+imagined)/2, frame_itm_scores=values)
    audit = dict(contract=CONTRACT, request_id=payload['request_id'], target=target,
        frontier_refresh=payload.get('frontier_refresh'),
        source_frontier=plan['source_frontier'], requested_route_actions=plan['actions'],
        selected_actions=bound['actions'], horizon_reason=reason,
        source_video=item['source_video'], generation_cache_hit=item.get('generation_cache_hit', False),
        generation_origin_request=item.get('generation_origin_request', payload['request_id']),
        frames=[{k:v for k,v in f.items() if k!='jpeg_b64'}
            for f in item['future_frames']], score=score, latency_s=time.monotonic()-started,
        note='diagnostic ablation: no verifier, reward, reliability or score-band gate')
    path = Path(item['artifact']).parent.parent/'verification.json'
    if alignment_audit.enabled():
        anchor=plan['frontier_anchor']
        audit['anchor_id']=anchor.get('observation_id')
        audit['anchor_rgb_sha256']=hashlib.sha256(base64.b64decode(anchor['rgb_jpeg_b64'])).hexdigest()
        audit['score']=dict(score,frontier_rank=plan['source_frontier'].get('rank'))
        audit['generation_origin_plan']=item.get('generation_origin_plan')
        audit['generation_key']=item.get('generation_key')
        audit['alignment_audit']=dict(payload.get('alignment_audit') or {},
            schema_version=alignment_audit.SCHEMA,episode=payload.get('episode'),camera=payload.get('camera'),
            generated=item.get('alignment_audit'),backend='svc_qwen',score_mode='frontier_once')
        audit['anchor_pose']=plan['frontier_anchor'].get('alignment_pose')
        audit['anchor_pitch_rad']=plan['frontier_anchor'].get('camera_pitch_rad')
    path.write_text(json.dumps(audit, indent=2)+'\n')
    return dict(decision='call_wm', selected_action=payload['navigation']['baseline_action'],
        selected_plan_id=plan['plan_id'], candidate_scores=[score], reason=CONTRACT,
        artifacts=[str(path),item['source_video']])
