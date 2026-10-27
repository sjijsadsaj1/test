"""Project original ApexNav frontier cells into a registered real RGB/depth view.

This is our visualization adapter, not native ApexNav image segmentation. A 2D
cluster is extruded over an explicitly recorded height prior. No detector chooses
an entrance, no VLM removes frames, and no generation-quality size gate is used.
"""
import hashlib
import math
import numpy as np

CONTRACT='original_cluster_crop_v2'


def camera_config(cfg):
    sensors=list(cfg.habitat.simulator.agents.main_agent.sim_sensors.values())
    rgb=[s for s in sensors if s.type=='HabitatSimRGBSensor']
    depth=[s for s in sensors if s.type=='HabitatSimDepthSensor']
    if len(rgb)!=1 or len(depth)!=1:raise ValueError('one registered RGB/depth camera required')
    r,d=rgb[0],depth[0]
    for key in ('width','height','hfov','position','orientation'):
        if not np.allclose(np.asarray(r[key],float),np.asarray(d[key],float)):
            raise ValueError('RGB/depth camera registration mismatch: '+key)
    if not np.allclose(r.orientation,[0,0,0]) or not np.allclose([r.position[0],r.position[2]],[0,0]):
        raise ValueError('unsupported camera extrinsics')
    return dict(width=int(r.width),height=int(r.height),hfov=float(r.hfov),camera_height=float(r.position[1]),
                min_depth=float(d.min_depth),max_depth=float(d.max_depth),normalize_depth=bool(d.normalize_depth))


def project(frontier,view,cfg):
    cells=np.asarray(frontier.get('cluster_cells_xy',[]),dtype=float)
    if frontier.get('cluster_contract')!='original_cluster_projection_v1' or cells.ndim!=2 or cells.shape[1]!=2 or not len(cells):
        raise ValueError('original frontier cluster export missing; rebuild planner')
    if not np.isfinite(cells).all():raise ValueError('nonfinite frontier cells')
    w,h=int(cfg['width']),int(cfg['height'])
    diag=dict(contract=CONTRACT,box=None,source_step=view['step_index'],
        image_size=[w,h],frontier_xy=[frontier['x'],frontier['y']],
        cluster_sha256=hashlib.sha256(cells.astype('<f8').tobytes()).hexdigest(),
        height_prior_m=[.15,1.8],depth_tolerance_m=.15)
    if view['width']!=w or view['height']!=h:raise ValueError('RGB changed size after depth registration')
    # History normally includes level views from the initialization scan. Do not
    # invent a projection for unsupported pitch, or move to a different doorway.
    if abs(view['camera_pitch_rad'])>1e-3:return dict(diag,reason='nonlevel_view')
    depth=np.asarray(view['projection_depth'],dtype=float)
    if depth.shape!=(h,w):raise ValueError('depth registration mismatch')
    valid=np.isfinite(depth)&(depth>0)
    if cfg['normalize_depth']:
        valid&=depth<.99
        depth=depth*(cfg['max_depth']-cfg['min_depth'])+cfg['min_depth']
    valid&=(depth>cfg['min_depth'])&(depth<.99*cfg['max_depth'])
    yaw=view['robot_yaw_rad'];delta=cells-[view['robot_x'],view['robot_y']]
    forward=delta[:,0]*math.cos(yaw)+delta[:,1]*math.sin(yaw)
    left=-delta[:,0]*math.sin(yaw)+delta[:,1]*math.cos(yaw)
    fx=w/(2*math.tan(math.radians(cfg['hfov'])/2));mask=np.zeros((h,w),bool)
    for z,l in zip(forward,left):
        if z<=.1:continue
        u=w/2-fx*l/z;radius=max(.5,fx*.025/z)
        a=max(0,int(math.floor(u-radius)));b=min(w,int(math.ceil(u+radius))+1)
        top=max(0,int(math.floor(h/2-fx*(1.8-cfg['camera_height'])/z)))
        bottom=min(h,int(math.ceil(h/2-fx*(.15-cfg['camera_height'])/z)))
        if a>=b or top>=bottom:continue
        mask[top:bottom,a:b]|=valid[top:bottom,a:b]&(depth[top:bottom,a:b]>=z-.15)
    yy,xx=np.nonzero(mask)
    if len(xx)==0:return dict(diag,reason='outside_fov_or_occluded',support_pixels=0)
    box=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
    return dict(diag,box=box,reason='original_cluster_depth_supported',support_pixels=len(xx),
        support_fraction=len(xx)/((box[2]-box[0])*(box[3]-box[1])))


def select_view(controller,frontier,step):
    candidates=[]
    for v in controller.observation_history:
        if 'projection_depth' not in v:continue
        roi=project(frontier,v,controller.frontier_camera_config)
        if roi['box'] is not None:
            # Prefer supported pixels, with a small age discount; not semantic
            # score, predicted content, a detector proposal or a guessed door.
            candidates.append((roi['support_pixels']/(1+.02*(step-v['step_index'])),v,roi))
    if candidates:_,v,roi=max(candidates,key=lambda x:x[0])
    else:
        v=controller.observation_history[-1]
        roi=project(frontier,v,controller.frontier_camera_config)
    dx=frontier['x']-v['robot_x'];dy=frontier['y']-v['robot_y']
    a=math.atan2(dy,dx)-v['robot_yaw_rad']
    view={k:v[k] for k in ('rgb_jpeg_b64','observation_id','width','height','robot_x','robot_y','robot_yaw_rad','camera_pitch_rad','step_index')}
    return dict(view,age_steps=step-v['step_index'],bearing_rad=math.atan2(math.sin(a),math.cos(a)),roi=roi)
