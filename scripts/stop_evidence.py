"""Opt-in read-only instrumentation around frozen upstream evaluation.

No model requests, action changes, threshold changes or ground-truth policy input.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def install(env, upstream, cfg, directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    original_reset, original_step = env.reset, env.step
    state = {}

    def write(path, data):
        path.write_text(json.dumps(data, indent=2) + '\n')

    def reset(*args, **kwargs):
        observation = original_reset(*args, **kwargs)
        state.clear()
        state.update(rgb=observation['rgb'].copy(), depth=observation['depth'].copy(),
                     observation=observation, step=0, started=time.monotonic())
        return observation

    def step(action, *args, **kwargs):
        # The original evaluator has already chosen action. Read only its locals.
        local = sys._getframe(1).f_locals
        index = state['step']
        agent = env.sim.get_agent_state()
        q = agent.rotation
        obs = state['observation']
        detections = []
        masks = local.get('object_masks_list', []) if index else []
        labels = local.get('label_list', []) if index else []
        scores = local.get('score_list', []) if index else []
        clouds = local.get('obj_point_cloud_list', []) if index else []
        for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            y, x = np.where(np.asarray(mask).squeeze() > 0)
            item = dict(index=i, label=int(label), confidence=float(score),
                        box=[int(x.min()), int(y.min()), int(x.max()), int(y.max())] if len(x) else None)
            if i < len(clouds) and clouds[i].data:
                cloud = np.frombuffer(clouds[i].data, dtype=np.float32).reshape(-1, 4)
                xyz = cloud[:, :3]
                item['estimated_cloud_ros'] = dict(count=len(xyz), minimum=xyz.min(0).tolist(),
                    maximum=xyz.max(0).tolist(), median=np.median(xyz, axis=0).tolist())
            detections.append(item)
        row = dict(step_before_action=index, action=int(action),
            final_state=int(upstream.final_state), exploration_result=int(upstream.expl_result),
            target=env.current_episode.object_category, detector_label=local.get('label'),
            fusion_threshold=float(upstream.fusion_threshold),
            world_position=agent.position.tolist(), quaternion_xyzw=[*map(float, q.imag), float(q.real)],
            ros_position=[float(-obs['gps'][2]), float(-obs['gps'][0]), float(obs['gps'][1])],
            yaw=float(obs['compass'][0]), detections=detections)
        if int(action) == 0 or any(d['label'] == 0 for d in detections):
            folder = root / f'step_{index:03d}'
            folder.mkdir(exist_ok=True)
            image = Image.fromarray(state['rgb'][:, :, :3])
            image.save(folder / 'rgb.png')
            marked = image.copy()
            draw = ImageDraw.Draw(marked)
            for d in detections:
                if d['box']:
                    draw.rectangle(d['box'], outline='red' if d['label'] == 0 else 'blue', width=3)
                    draw.text((d['box'][0], max(0, d['box'][1])),
                              f"detection {d['index']} label={d['label']} p={d['confidence']:.3f}", fill='yellow')
            marked.save(folder / 'marked.png')
            np.savez_compressed(folder / 'sensors.npz', depth=state['depth'],
                                masks=np.asarray(masks, dtype=np.uint8))
            write(folder / 'observation.json', row)
        with (root / 'trace.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        observation = original_step(action, *args, **kwargs)
        state.update(rgb=observation['rgb'].copy(), depth=observation['depth'].copy(),
                     observation=observation, step=index + 1)
        if env.episode_over:
            # Privileged labels/metrics are saved only AFTER navigation terminates.
            goals = [dict(object_id=g.object_id, position=g.position,
                          view_points=[v.agent_state.position for v in g.view_points])
                     for g in env.current_episode.goals]
            metrics = {k:float(v) for k,v in env.get_metrics().items() if np.isscalar(v)}
            write(root / 'terminal.json', dict(pre_stop=row, metrics=metrics,
                action_count=index+1, elapsed_seconds=time.monotonic()-state['started'],
                seed=int(cfg.habitat.seed), max_steps=int(cfg.habitat.environment.max_episode_steps),
                distance_to=str(cfg.habitat.task.measurements.distance_to_goal.distance_to),
                success_distance=float(cfg.habitat.task.measurements.success.success_distance),
                offline_only_goals=goals))
        return observation

    env.reset, env.step = reset, step
