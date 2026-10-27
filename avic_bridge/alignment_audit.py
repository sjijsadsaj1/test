"""Opt-in evidence capture only. Never changes navigation decisions."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path

SCHEMA = 'alignmentaudit_v1'


def enabled():
    return os.environ.get('APEXNAV_ALIGNMENT_AUDIT', '0') == '1'


def error(exc):
    print('[ALIGNMENT_AUDIT_ERROR] ' + str(exc), flush=True)


def emit(event, data):
    if enabled():
        try:
            print('[ALIGNMENT_AUDIT] ' + json.dumps(dict(schema_version=SCHEMA, event=event, **data), allow_nan=False), flush=True)
        except Exception as exc:
            error(exc)


def save_json(path, data):
    if not enabled(): return None
    try:
        path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
        return str(path)
    except Exception as exc:
        error(exc);return None


def save_jpeg(path, encoded):
    if not enabled(): return None
    try:
        raw=base64.b64decode(encoded, validate=True)
        path=Path(path);path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(raw)
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    except Exception as exc:
        error(exc);return None


def pose(position, rotation):
    if hasattr(rotation, 'imag') and hasattr(rotation, 'real'):
        q=[float(v) for v in rotation.imag]+[float(rotation.real)]
    elif all(hasattr(rotation,k) for k in ('x','y','z','w')):
        q=[float(getattr(rotation,k)) for k in ('x','y','z','w')]
    else:
        raise ValueError('Unsupported quaternion representation; no guessed component order')
    n=math.sqrt(sum(v*v for v in q))
    if n<=0:raise ValueError('Zero quaternion')
    x,y,z,w=[v/n for v in q];p=[float(v) for v in position]
    if len(p)!=3 or not all(math.isfinite(v) for v in p+q):raise ValueError('Invalid pose')
    matrix=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),p[0]],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),p[1]],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y),p[2]],
            [0.,0.,0.,1.]]
    return dict(position=p, quaternion_xyzw=q, c2w=matrix)


def capture_pose(env, cfg, observations, episode_start=None):
    if not enabled():return None
    try:
        state=env.sim.get_agent_state()
        candidates=[]
        for key,runtime_sensor in env.sim.sensor_suite.sensors.items():
            if any(cls.__name__=='HabitatSimRGBSensor' for cls in type(runtime_sensor).__mro__):
                uuid=str(runtime_sensor.uuid)
                if str(key)!=uuid:raise ValueError('Runtime RGB sensor dictionary/UUID mismatch')
                candidates.append(uuid)
        if len(candidates)!=1:
            raise ValueError('Runtime RGB UUID missing/ambiguous: '+repr(candidates))
        uuid=candidates[0]
        if uuid not in state.sensor_states:raise ValueError('Actual RGB sensor state missing: '+uuid)
        sensor=state.sensor_states[uuid]
        gps=[float(v) for v in observations['gps']]
        compass=float(observations['compass'][0])
        return dict(schema_version=SCHEMA, sensor_uuid=uuid,
                    body=pose(state.position,state.rotation),rgb_sensor=pose(sensor.position,sensor.rotation),
                    episode_start_body=episode_start,
                    gps=gps,compass=compass,
                    simulator_action_parameters=dict(forward_step_m=float(cfg.habitat.simulator.forward_step_size),
                                                     turn_angle_deg=float(cfg.habitat.simulator.turn_angle)),
                    ros_planar=dict(x=-gps[2],y=-gps[0],yaw_rad=compass),
                    frames=dict(sensor_c2w='Habitat world: +Y up, camera optical forward -Z',
                                ros_planar='GPS episode frame: x=-gps[2], y=-gps[0], yaw=compass; not absolute Habitat world xy',
                                svc_c2w='SVC relative camera: +Z forward, Ry @ Rx; world conversion not asserted'))
    except Exception as exc:
        error(exc);return dict(schema_version=SCHEMA,missing_reason=str(exc))


def bind_request(payload, frontiers):
    if not enabled():return
    payload['alignment_audit']=dict(schema_version=SCHEMA, episode=payload.get('episode'),
        query_id=frontiers[0].get('query_id') if frontiers else None,
        all_frontiers=frontiers, camera=payload.get('camera'),
        scope='Current experimental working copy; no image-content quality claim')


def capture_generation(plan_dir, encoded, input_path, **contract):
    """All diagnostic I/O is fail-open: a logging failure never fails generation."""
    if not enabled():return None
    try:
        plan_dir=Path(plan_dir);input_path=Path(input_path)
        contract.update(schema_version=SCHEMA,
            original_anchor=save_jpeg(plan_dir/'anchor_original.jpg',encoded),
            recompressed_input=dict(path=str(input_path),sha256=hashlib.sha256(input_path.read_bytes()).hexdigest()))
        path=save_json(plan_dir/'camera_contract.json',contract)
        return dict(schema_version=SCHEMA,camera_contract_path=path,
                    original_anchor=contract['original_anchor'],anchor_pose=contract.get('anchor_pose'))
    except Exception as exc:
        error(exc);return dict(schema_version=SCHEMA,missing_reason=str(exc))
