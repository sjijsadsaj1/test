"""Offline-only: associate saved real target masks with MP3D labelled mesh.

This script never imports or calls a navigation policy or model service.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def quaternion_matrix(q):
    x, y, z, w = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--scene-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    import yaml
    cfg = yaml.safe_load(a.config.read_text())
    sensor = cfg['habitat']['simulator']['agents']['main_agent']['sim_sensors']['depth_sensor']
    height = float(sensor['position'][1])
    focal = sensor['width'] / (2*np.tan(np.deg2rad(sensor['hfov'])/2))
    prefix = a.scene_dir.name
    lines = (a.scene_dir / (prefix+'.house')).read_text().splitlines()
    categories = {int(t[1]):dict(raw=t[3], mpcat40=t[5])
                  for t in map(str.split, lines) if t and t[0]=='C'}
    objects = {int(t[1]):dict(category=categories.get(int(t[3]), dict(raw='unlabelled', mpcat40='void')), center_native=list(map(float,t[4:7])))
               for t in map(str.split, lines) if t and t[0]=='O'}
    terminal = json.loads((a.evidence/'terminal.json').read_text())
    goals = terminal['offline_only_goals']
    errors=[]
    for g in goals:
        x,y,z=objects[int(g['object_id'])]['center_native']
        errors.append(float(np.linalg.norm(np.array([x,z,-y])-g['position'])))
    assert max(errors)<1e-4, 'house object ID / goal coordinate validation failed'
    with (a.scene_dir/(prefix+'_semantic.ply')).open('rb') as f:
        nv=nf=None
        while True:
            line=f.readline().decode().strip()
            if line.startswith('element vertex '):nv=int(line.split()[-1])
            if line.startswith('element face '):nf=int(line.split()[-1])
            if line=='end_header':break
        vertices=np.fromfile(f,dtype=[('xyz','<f4',3),('rgb','u1',3)],count=nv)['xyz'].copy()
        faces=np.fromfile(f,dtype=[('n','u1'),('vi','<i4',3),('oid','<i4')],count=nf)
    assert len(faces)==nf and np.all(faces['n']==3)
    # Accept a vertex label only if all incident faces agree. Boundaries abstain.
    lo=np.full(nv,np.iinfo(np.int32).max,dtype=np.int32)
    hi=np.full(nv,-1,dtype=np.int32)
    for j in range(3):
        np.minimum.at(lo,faces['vi'][:,j],faces['oid'])
        np.maximum.at(hi,faces['vi'][:,j],faces['oid'])
    tree=cKDTree(vertices)
    trace=[json.loads(l) for l in (a.evidence/'trace.jsonl').read_text().splitlines()]
    pitch=0.; report=[]
    goal_ids={int(g['object_id']) for g in goals}
    for row in trace:
        folder=a.evidence/f"step_{row['step_before_action']:03d}"
        if (folder/'sensors.npz').exists():
            data=np.load(folder/'sensors.npz');depth=data['depth'].squeeze()
            depth=depth*(sensor['max_depth']-sensor['min_depth'])+sensor['min_depth']
            c,s=np.cos(pitch),np.sin(pitch)
            rotation=quaternion_matrix(row['quaternion_xyzw']) @ np.array([[1,0,0],[0,c,-s],[0,s,c]])
            camera=np.array(row['world_position'])+np.array([0,height,0])
            for d in row['detections']:
                if d['label']!=0:continue
                mask=data['masks'][d['index']].squeeze()>0
                yy,xx=np.where(mask & (depth>0) & (depth<sensor['max_depth']*.95))
                stride=max(1,int(np.ceil(len(xx)/1500)));xx,yy=xx[::stride],yy[::stride]
                z=depth[yy,xx]
                local=np.column_stack(((xx-sensor['width']/2)*z/focal,-(yy-sensor['height']/2)*z/focal,-z))
                world=local@rotation.T+camera
                native=world[:,[0,2,1]]*np.array([1,-1,1])
                distance,ix=tree.query(native)
                ok=(distance<.03)&(lo[ix]==hi[ix])&(hi[ix]>=0)
                ids,counts=np.unique(hi[ix[ok]],return_counts=True)
                hits=[dict(object_id=int(i),sample_count=int(n),is_valid_goal=int(i) in goal_ids,**objects[int(i)])
                      for i,n in sorted(zip(ids,counts),key=lambda z:-z[1]) if int(i) in objects]
                report.append(dict(step=row['step_before_action'],detection=d,samples=len(xx),
                    reliable_samples=int(ok.sum()),mesh_distance_median=float(np.median(distance)) if len(distance) else None,
                    surface_labels=hits))
        if row['action']==4:pitch+=np.pi/6
        if row['action']==5:pitch-=np.pi/6
    result=dict(method='real mask depth -> world -> native MP3D xyz -> nearest mesh vertex <3cm; all incident faces must agree',
                goal_house_center_max_error_m=max(errors),valid_goal_ids=sorted(goal_ids),frames=report,
                warning='Offline diagnosis only. Raster semantic rendering was not trusted; no GT enters navigation.')
    (a.evidence/'offline_mesh_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in report:
        print(r['step'],r['reliable_samples'],'/',r['samples'],
              [(x['object_id'],x['category']['raw'],x['sample_count']) for x in r['surface_labels'][:3]])


if __name__=='__main__':main()
