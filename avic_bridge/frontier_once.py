"""Opt-in exploratory ablation; never used by the standard AVIC controller.

Identity is a spatial proxy, NOT a persistent map-cluster ID. Greedy one-to-one
matching within a fixed 0.75m birth radius prevents rank reuse and chain drift.
Splits get separate identities; vanished entries remain available until reset.
"""
import json
import math
import os
import time
import uuid

from .protocol import SCHEMA_VERSION, action_dict, make_candidate_plan
from . import alignment_audit

CONTRACT = "frontier_once_spatial_q25_half_v1_20260923"
REFRESH_CONTRACT = "frontier_score_delta_refresh_v1_20260925"


def refresh_threshold():
    """Absolute real-score change since the last successful prediction; 0 disables."""
    value = float(os.environ.get('APEXNAV_FRONTIER_REFRESH_DELTA', '0'))
    if not math.isfinite(value) or not 0 <= value <= 2:
        raise ValueError('APEXNAV_FRONTIER_REFRESH_DELTA must be finite and in [0,2]')
    return value


class FrontierRegistry:
    def __init__(self, radius=0.75):
        self.radius = radius
        self.entries = []

    def associate(self, frontiers):
        pairs = []
        for i, f in enumerate(frontiers):
            for j, e in enumerate(self.entries):
                distance = math.hypot(float(f['x'])-e['x'], float(f['y'])-e['y'])
                if distance <= self.radius:
                    pairs.append((distance, j, i))
        matches, used = {}, set()
        for _, j, i in sorted(pairs):
            if i not in matches and j not in used:
                matches[i] = j
                used.add(j)
        for i, f in enumerate(frontiers):
            if i not in matches:
                matches[i] = len(self.entries)
                self.entries.append(dict(id=len(self.entries), x=float(f['x']),
                                         y=float(f['y']), score=None))
        return [self.entries[matches[i]] for i in range(len(frontiers))]


def decide_once(controller, context):
    from .controller import AvicDecision
    c = controller
    threshold = refresh_threshold()
    started = time.monotonic()
    baseline = context['baseline_action']
    frontiers = list(context.get('frontier_candidates') or [])
    # Keep STOP, initialization, object pursuit and collision handling authoritative.
    if (baseline not in (1, 2, 3) or context['step_index'] < 26 or
            context['final_state'] != 0 or context['exploration_result'] != 0 or
            not frontiers or not frontiers[0].get('query_active')):
        return c._skip(baseline, 'frontier_once: no safe exploration transaction')
    query = int(frontiers[0]['query_id'])
    if query == c.last_frontier_query_id:
        return c._skip(baseline, 'frontier_once: transaction already handled')
    c.last_frontier_query_id = query
    if not hasattr(c, 'once_registry'):
        c.once_registry = FrontierRegistry()
    entries = c.once_registry.associate(frontiers)
    observation = next((o for o in reversed(c.observation_history)
                        if o['step_index'] == context['step_index']), None)
    if observation is None:
        raise RuntimeError('frontier_once: missing current observation')
    scores, generated, refreshed = [], 0, 0
    for frontier, entry in zip(frontiers, entries):
        real = float(frontier['semantic_score'])
        if not math.isfinite(real) or not -1 <= real <= 1:
            raise RuntimeError('frontier_once: invalid real frontier score')
        previous_real = entry.get('prediction_real_score')
        delta = abs(real-previous_real) if previous_real is not None else None
        is_refresh = (entry['score'] is not None and threshold > 0 and delta is not None
                      and delta + 1e-12 >= threshold)
        if entry['score'] is None or is_refresh:
            actions = frontier.get('path_action_prefix', [])
            if (not actions or len(actions) > 6 or any(a not in (1, 2, 3) for a in actions)
                    or actions[0] != frontier.get('first_action')):
                raise RuntimeError('frontier_once: invalid reachable-frontier route')
            plan_id = 'track_%d' % entry['id']
            plan = make_candidate_plan(plan_id, actions)
            anchor = dict(observation, binding_source='current_observation',
                          anchor_quality=0.0, frontier_distance_m=math.hypot(
                              frontier['x']-observation['robot_x'], frontier['y']-observation['robot_y']),
                          relative_yaw_to_frontier_rad=float(frontier['relative_yaw_rad']))
            plan.update(source_frontier=frontier, frontier_anchor=anchor, min_prefix_length=1)
            payload = dict(schema_version=SCHEMA_VERSION,
                request_id='%s-%s-%06d-%s-%s' % (context['scene_id'].split('/')[-1],
                    context['episode_id'], context['step_index'], plan_id, uuid.uuid4().hex[:8]),
                episode=dict(generation_epoch=c.generation_epoch, scene_id=context['scene_id'],
                    episode_id=context['episode_id'], step_index=context['step_index']),
                observation=dict(rgb_jpeg_b64=observation['rgb_jpeg_b64']),
                target=dict(category=context['target_category'], related_room=context['target_room'],
                    related_objects=context.get('target_related_objects') or []),
                camera=dict(forward_step_m=0.25, turn_angle_deg=30.0,
                    pitch_rad=context['camera_pitch_rad']),
                navigation=dict(baseline_action=action_dict(baseline), candidate_plans=[plan],
                    imagination_plan_ids=[plan_id], route_contract=CONTRACT),
                signals=dict(frontier_query_active=True), score_policy=dict(mode='frontier_once'))
            if threshold > 0:
                payload['frontier_refresh'] = dict(contract=REFRESH_CONTRACT,
                    reason='real_score_delta' if is_refresh else 'first_observation',
                    threshold=threshold, previous_real_score=previous_real,
                    current_real_score=real, absolute_delta=delta,
                    previous_prediction_step=entry.get('created_step'),
                    previous_request_id=entry.get('request_id'),
                    prediction_number=entry.get('prediction_count', 0)+1)
            if is_refresh:
                # A refresh must really regenerate, even if JPEG/action inputs match.
                # Original first-observation video reuse remains unchanged.
                payload['episode']['generation_epoch'] += ':refresh:' + payload['request_id']
            print('[FRONTIER_ONCE_GENERATE] '+json.dumps(dict(scene=context['scene_id'],
                episode=context['episode_id'], step=context['step_index'], track_id=entry['id'],
                request_id=payload['request_id'], completed_in_query=generated,
                current_frontiers=len(frontiers), refresh=payload.get('frontier_refresh'))), flush=True)
            if alignment_audit.enabled():
                alignment_audit.bind_request(payload,frontiers)
                c.alignment_last_request=dict(request_id=payload['request_id'],**payload['alignment_audit'])
            response = c.client.decide(payload)
            rows = response.get('candidate_scores', [])
            if response.get('decision') != 'call_wm' or len(rows) != 1 or rows[0]['plan_id'] != plan_id:
                raise RuntimeError('frontier_once: generation failed; abort rather than silently use baseline')
            value = float(rows[0]['imagined_itm_score'])
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise RuntimeError('frontier_once: invalid predicted score')
            entry.update(score=value, created_step=context['step_index'], request_id=payload['request_id'],
                         prediction_real_score=real, prediction_count=entry.get('prediction_count', 0)+1)
            generated += 1
            refreshed += int(is_refresh)
        scores.append(dict(plan_id='track_%d' % entry['id'], rank=frontier['rank'],
            query_id=query, x=frontier['x'], y=frontier['y'], real_semantic_score=real,
            imagined_itm_score=entry['score'], decision_semantic_score=(real+entry['score'])/2,
            score_mode='frontier_once', cache_created_step=entry['created_step'],
            source_request_id=entry['request_id'], prediction_real_score=entry['prediction_real_score'],
            prediction_count=entry['prediction_count']))
    print('[FRONTIER_ONCE] '+json.dumps(dict(episode=context['episode_id'], scene=context['scene_id'],
        step=context['step_index'], query=query, current_frontiers=len(frontiers),
        lifetime_tracks=len(c.once_registry.entries), generated=generated,
        cache_hits=len(scores)-generated, refreshed=refreshed, initial_generated=generated-refreshed,
        refresh_threshold=threshold, scores=scores)), flush=True)
    return AvicDecision(baseline_action=baseline, executed_action=baseline,
        service_decision='call_wm' if generated else 'cache', selected_plan_id=None,
        reason=REFRESH_CONTRACT if threshold > 0 else CONTRACT, latency_ms=(time.monotonic()-started)*1000,
        used_world_model=bool(generated), overridden=False,
        frontier_scores=scores, cache_hits=len(scores)-generated, cache_misses=generated)
