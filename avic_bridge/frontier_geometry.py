"""Original 2-D frontier clusters -> bounded RGB reference, no object detector.

The map has no vertical segmentation: a declared upright height interval is
projected at each original cell. Depth rejects foreground occlusion. This is
NOT a measured 3-D portal segmentation. Unsupported cameras fail closed.
"""
import base64
import hashlib
import io
import math
import os
import numpy as np
from PIL import Image

CONTRACT='original_cluster_projection_v1'


def decode_cells(row, label):
    if 'cluster_cells_v1' not in label:return {}
    if len(row)<20 or row[18]!=1:raise ValueError('invalid_cluster_wire_version')
    n=int(row[19])
    if n!=row[19] or n<0 or 20+2*n>len(row):raise ValueError('invalid_cluster_wire_count')
    cells=np.asarray(row[20:20+2*n],dtype=float).reshape(-1,2)
    if not np.isfinite(cells).all():raise ValueError('nonfinite_cluster_cells')
    return dict(cluster_contract=CONTRACT,cluster_cells_xy=cells.tolist())


def camera_config(cfg):
    sensors=list(cfg.habitat.simulator.agents.main_agent.sim_sensors.values())
    rgb=[s for s in sensors if s.type=='HabitatSimRGBSensor']
    depth=[s for s in sensors if s.type=='HabitatSimDepthSensor']
    if len(rgb)!=1 or len(depth)!=1:raise ValueError('geometry_requires_one_rgb_and_depth')
    r,d=rgb[0],depth[0]
    for key in ['width','height','hfov','position','orientation']:
        if not np.allclose(np.asarray(r[key],dtype=float),np.asarray(d[key],dtype=float)):
            raise ValueError('rgb_depth_not_registered_'+key)
    if not np.allclose(r.orientation,[0,0,0]) or not np.allclose([r.position[0],r.position[2]],[0,0]):
        raise ValueError('unsupported_sensor_extrinsics')
    return dict(width=int(r.width),height=int(r.height),hfov=float(r.hfov),
                camera_height=float(r.position[1]),min_depth=float(d.min_depth),
                max_depth=float(d.max_depth),normalize_depth=bool(d.normalize_depth))


def attach_regions(frontiers, observations, config, pitch, step, rgb):
    """Same observation transaction as RGB and planner response; no stale cache."""
    from copy import deepcopy
    result=deepcopy(frontiers)
    for f in result:
        f['geometry_roi']=project(f,observations,config,pitch,step,rgb)
    return result


def project(frontier, obs, cfg, pitch, step, rgb):
    diag=dict(contract=CONTRACT,visible=False,step_index=int(step),query_id=frontier.get('query_id'),
              image_size=[int(cfg['width']),int(cfg['height'])],
              frontier_xy=[frontier['x'],frontier['y']],height_interval_m=[.15,1.8],
              depth_tolerance_m=.15,scope='original 2D cells extruded over declared height prior; not portal segmentation')
    def reject(reason):return dict(diag,reason=reason)
    if frontier.get('cluster_contract')!=CONTRACT:raise ValueError('planner_missing_original_cluster_export_rebuild_required')
    cells=np.asarray(frontier.get('cluster_cells_xy',[]),dtype=float)
    if cells.ndim!=2 or cells.shape[1]!=2 or len(cells)==0:return reject('empty_original_cluster')
    if not np.isfinite(cells).all():return reject('invalid_original_cluster')
    diag['cluster_cell_count']=len(cells)
    diag['cluster_sha256']=hashlib.sha256(cells.astype('<f8').tobytes()).hexdigest()
    if abs(pitch)>1e-3:return reject('nonlevel_camera_not_supported_v1')
    w,h=cfg['width'],cfg['height']
    depth=np.asarray(obs['depth'],dtype=float).squeeze()
    if depth.shape!=(h,w) or np.asarray(rgb).shape[:2]!=(h,w):raise ValueError('rgb_depth_shape_mismatch')
    normalized=cfg['normalize_depth']
    valid=np.isfinite(depth)&(depth>0)
    if normalized:valid &= depth<.99
    depth=depth*(cfg['max_depth']-cfg['min_depth'])+cfg['min_depth'] if normalized else depth
    valid &= (depth>cfg['min_depth'])&(depth<.99*cfg['max_depth'])
    # Same episode frame as ApexNav: x=-gps[2], y=-gps[0], yaw=compass.
    x,y=-float(obs['gps'][2]),-float(obs['gps'][0]);yaw=float(np.asarray(obs['compass']).reshape(-1)[0])
    delta=cells-[x,y];co,si=math.cos(yaw),math.sin(yaw)
    forward=delta[:,0]*co+delta[:,1]*si;left=-delta[:,0]*si+delta[:,1]*co
    fx=w/(2*math.tan(math.radians(cfg['hfov'])/2));fy=fx
    mask=np.zeros((h,w),dtype=bool)
    # Each original grid centre gets a small fixed metric footprint; no new
    # cells or detector-selected boxes are introduced. Footprint is explicit.
    diag['cell_footprint_radius_m']=.025
    for z,l in zip(forward,left):
        if z<=.1:continue
        u=w/2-fx*l/z;radius=max(.5,fx*.025/z)
        a=max(0,int(math.floor(u-radius)));b=min(w,int(math.ceil(u+radius))+1)
        top=max(0,int(math.floor(h/2-fy*(1.8-cfg['camera_height'])/z)))
        bottom=min(h,int(math.ceil(h/2-fy*(.15-cfg['camera_height'])/z)))
        if a>=b or top>=bottom:continue
        mask[top:bottom,a:b] |= valid[top:bottom,a:b] & (depth[top:bottom,a:b]>=z-.15)
    diag['depth_supported_pixels']=int(mask.sum())
    if not mask.any():return reject('outside_fov_or_depth_occluded')
    # Connected support only: never bridge two disjoint views through a wall.
    import cv2
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
    if n<2:return reject('no_connected_depth_support')
    idx=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));a,t,bw,bh,area=map(int,stats[idx])
    box=[a,t,a+bw,t+bh];diag.update(projected_box=box,support_fraction=area/(bw*bh))
    if bw<48 or bh<96 or area<.015*w*h:return reject('insufficient_projected_context')
    if area/(bw*bh)<.65:return reject('fragmented_depth_support')
    if bw*bh>.65*w*h:return reject('projection_not_local')
    # Save supporting mask in candidate for checking VLM's crop, not only box.
    support=Image.fromarray(((labels==idx)*255).astype('uint8')).crop(box)
    buf=io.BytesIO();support.save(buf,format='PNG')
    return dict(diag,visible=True,reason='original_cluster_depth_supported',box=box,
                support_png_b64=base64.b64encode(buf.getvalue()).decode())


class GeometryLocator:
    localization_source=CONTRACT
    def __init__(self,make_client):
        self.make_client=make_client;self.client=None
        self.last_attempts=0;self.last_usage={};self.last_diagnostic={};self.total_requests=0

    def locate_interior(self,real,hint):
        roi=hint['geometry_roi'];self.last_diagnostic=dict(roi)
        self.last_diagnostic.pop('support_png_b64',None)
        self.last_attempts=0;self.last_usage={}
        if not roi.get('visible'):return dict(visible=False)
        image=Image.open(io.BytesIO(base64.b64decode(real,validate=True))).convert('RGB')
        if list(image.size)!=roi['image_size']:raise ValueError('geometry_rgb_resized_after_projection')
        outer=roi['box'];patch=image.crop(outer);buf=io.BytesIO();patch.save(buf,format='JPEG',quality=95)
        if self.client is None:self.client=self.make_client()
        prompt=('First classify this supplied image: interior_space means visible room layout or contents; '
            'solid_surface means a wall, door panel, near object surface or bare texture; insufficient means uncertain. '
            'A plain surface with no frame is still solid_surface, NOT usable interior context. '
            'This image is a FIXED region projected from ONE original navigation frontier. '
            'Do not locate another entrance or choose another object. Only remove visible foreground door frames '
            'or door leaves from the edges of THIS region, retaining the largest useful visible interior context. '
            'ONLY IF interior_space and no frame is present, keep the whole region. If the image is only a solid wall/panel/close object '
            'or too little room context remains, return usable=false. Do not invent hidden contents. '
            'Return JSON {"content_type":"interior_space" or "solid_surface" or "insufficient",'
            '"usable":true or false,"box_1000":[left,top,right,bottom],"reason":"visible evidence"}. '
            'Coordinates are within THIS supplied image, independently normalized 0..1000. No search target.')
        dual=os.environ.get('AVIC_IMAGE_DUAL_CONTEXT')=='1'
        images=[dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()))]
        if dual:
            images.insert(0,dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+real)))
            prompt=('Image 1 is the full observation for spatial context ONLY. Image 2 is the FIXED '
                'frontier region cropped from image 1 at pixel box '+str(outer)+' in size '+str(image.size)+'. '
                'All classification and crop coordinates below refer to IMAGE 2. Use image 1 to understand '
                'the role of image 2, but never substitute a different entrance or borrow contents not visible '
                'in image 2 as evidence that image 2 has room context. '+prompt)
        self.last_diagnostic['conditioning_images']=2 if dual else 1
        reply=self.client._request(dict(max_tokens=192,messages=[dict(role='user',content=[dict(type='text',text=prompt),*images])]))
        self.last_attempts=self.client.last_attempts;self.last_usage=dict(self.client.last_usage);self.total_requests+=1
        self.last_diagnostic['deframe_reply']=reply
        if (not isinstance(reply,dict) or type(reply.get('usable')) is not bool or
            reply.get('content_type') not in ('interior_space','solid_surface','insufficient')):raise ValueError('invalid_deframe_schema')
        if reply['content_type']!='interior_space' or not reply['usable']:return dict(visible=False)
        box=reply.get('box_1000')
        if (not isinstance(box,list) or len(box)!=4 or any(type(v)!=int or not 0<=v<=1000 for v in box)
            or not box[0]<box[2] or not box[1]<box[3]):raise ValueError('invalid_deframe_box')
        local=[round(v*patch.size[i%2]/1000) for i,v in enumerate(box)]
        a,t,b,d=local
        if b-a<48 or d-t<96 or (b-a)*(d-t)<.015*image.width*image.height:return dict(visible=False)
        support=Image.open(io.BytesIO(base64.b64decode(roi['support_png_b64']))).convert('L')
        if (np.asarray(support.crop(local))>0).mean()<.65:return dict(visible=False)
        final=[a+outer[0],t+outer[1],b+outer[0],d+outer[1]]
        self.last_diagnostic.update(locked_outer=outer,final_box=final)
        return dict(visible=True,box=final,crop_hint=dict(hint,column_px=(final[0]+final[2])/2))
