#!/usr/bin/env python3
"""Re-render specified logged observations; require pose/depth hashes to match."""
import argparse, gzip, hashlib, json, re, struct
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--scene',required=True);p.add_argument('--episode',type=int,required=True)
    p.add_argument('--steps',type=int,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--semantic',action='store_true',help='Offline GT diagnostic only; never enters navigation policy')
    args=p.parse_args()
    actions={};plans={};audits={};active=False;step=0
    for line in args.log.read_text().splitlines():
        if line.startswith('[SYNC_PLAN] '):
            row=json.loads(line.split(' ',1)[1]);m=re.search(r'/([^/]+)\.glb:(\d+)$',row['episode'])
            active=(m[1],int(m[2]))==(args.scene,args.episode)
            if active:plans[row['step']]=row
        elif active:
            m=re.match(r'-+Step: (\d+)-+',line)
            if m:step=int(m[1])
            m=re.match(r'Finding \[.*?\]; Action: (\d+);',line)
            if m:actions[step]=int(m[1])
            if line.startswith('[TARGET_ALIAS_AUDIT] '):audits[step]=json.loads(line.split(' ',1)[1])
    assert set(args.steps)<=plans.keys()
    import habitat_sim,magnum as mn,quaternion
    from habitat_sim.utils.common import quat_from_coeffs
    from PIL import Image
    root=Path(__file__).resolve().parents[1]
    data=json.load(gzip.open(root/f'data/datasets/objectnav/mp3d/v1/val/content/{args.scene}.json.gz'))
    ep=data['episodes'][args.episode]
    cfg=habitat_sim.SimulatorConfiguration();cfg.scene_id=str(root/f'data/scene_datasets/mp3d/{args.scene}/{args.scene}.glb')
    cfg.create_renderer=True;cfg.enable_physics=False;cfg.allow_sliding=False;cfg.gpu_device_id=-1
    nav=habitat_sim.NavMeshSettings();nav.set_defaults();nav.agent_radius=.18;nav.agent_height=.88;nav.agent_max_climb=.2;cfg.navmesh_settings=nav
    ac=habitat_sim.agent.AgentConfiguration();ac.radius=.18;ac.height=.88
    sensors=[]
    sensor_types=[('rgb',habitat_sim.SensorType.COLOR),('depth',habitat_sim.SensorType.DEPTH)]
    if args.semantic:sensor_types.append(('semantic',habitat_sim.SensorType.SEMANTIC))
    for name,kind in sensor_types:
        s=habitat_sim.CameraSensorSpec();s.uuid=name;s.sensor_type=kind;s.resolution=[480,640];s.position=mn.Vector3(0,.88,0);s.hfov=mn.Deg(79);sensors.append(s)
    ac.sensor_specifications=sensors
    names={1:'move_forward',2:'turn_left',3:'turn_right'}
    ac.action_space={name:habitat_sim.agent.ActionSpec(name,habitat_sim.agent.ActuationSpec(amount=.25 if i==1 else 30)) for i,name in names.items()}
    sim=habitat_sim.Simulator(habitat_sim.Configuration(cfg,[ac]))
    try:
        start=np.asarray(ep['start_position'],dtype=np.float32);q0=quat_from_coeffs(ep['start_rotation'])
        st=habitat_sim.AgentState();st.position=start;st.rotation=q0;agent=sim.initialize_agent(0,st)
        def rotate(q,v):return (q*quaternion.quaternion(0,*v)*q.inverse()).imag
        args.output.mkdir(parents=True,exist_ok=True);checks=[]
        for step in range(max(args.steps)+1):
            state=agent.get_state();gps=rotate(q0.inverse(),state.position-start).astype(np.float32)
            heading=rotate(state.rotation.inverse()*q0,[0,0,-1]);yaw=float(np.float32(np.arctan2(heading[0],-heading[2])))
            if step in plans:
                sy,cy=np.sin(yaw/2),np.cos(yaw/2);ori=[0.*cy-0.*sy,0.*cy+0.*sy,sy-0.*cy,cy+0.*sy]
                raw=struct.pack('<III',0,step+1,0)+struct.pack('<I',5)+b'world'+struct.pack('<I',9)+b'base_link'
                raw+=struct.pack('<85d',*[-gps[2],-gps[0],gps[1]],*ori,*([0.]*78))
                assert hashlib.sha256(raw).hexdigest()==plans[step]['robot_pose_sha256'], ('pose mismatch',step)
            if step in args.steps:
                obs=sim.get_sensor_observations();depth=(np.clip(obs['depth'],0,5)/5).astype(np.float32)
                assert hashlib.sha256(depth.tobytes()).hexdigest()==plans[step]['depth_sha256'],('depth mismatch',step)
                Image.fromarray(obs['rgb'][:,:,:3]).save(args.output/f'step_{step:03d}.png')
                frame=dict(step=step,depth_matches=True,pose_matches=True,audit=audits.get(step),
                           world_position=state.position.tolist())
                goals=data['goals_by_category'][args.scene+'.glb_'+ep['object_category']]
                views=np.array([v['agent_state']['position'] for g in goals for v in g['view_points']],dtype=np.float32)
                query=habitat_sim.MultiGoalShortestPath();query.requested_start=state.position;query.requested_ends=views
                sim.pathfinder.find_path(query);frame['gt_geodesic_distance_to_views']=float(query.geodesic_distance)
                frame['closest_goal_centers']=sorted([dict(object_id=g['object_id'],position=g['position'],
                    euclidean_distance=float(np.linalg.norm(np.array(g['position'])-state.position))) for g in goals],
                    key=lambda x:x['euclidean_distance'])[:3]
                frame['detection_center_geometry']=[]
                camera_state=state.sensor_states['rgb']
                focal=640/(2*np.tan(np.deg2rad(79)/2))
                for detection in (audits.get(step) or {}).get('raw_dino',[]):
                    box=detection['box'];u=int(np.clip((box[0]+box[2])*320,0,639));v=int(np.clip((box[1]+box[3])*240,0,479))
                    depth_m=float(obs['depth'][v,u]);local=np.array([(u-320)*depth_m/focal,-(v-240)*depth_m/focal,-depth_m])
                    world=rotate(camera_state.rotation,local)+camera_state.position
                    nearest=sorted([dict(object_id=g['object_id'],position=g['position'],
                        distance=float(np.linalg.norm(np.array(g['position'])-world))) for g in goals],key=lambda x:x['distance'])[:2]
                    frame['detection_center_geometry'].append(dict(phrase=detection['phrase'],pixel=[u,v],depth_m=depth_m,
                        world_point=world.tolist(),nearest_goal_centers=nearest))
                if args.semantic:
                    semantic=obs['semantic'];np.save(args.output/f'step_{step:03d}_semantic.npy',semantic)
                    wanted={str(g['object_id']) for g in goals};stats=[]
                    for d in (audits.get(step) or {}).get('raw_dino',[]):
                        b=d['box'];x1,y1,x2,y2=np.round(np.array(b)*[640,480,640,480]).astype(int)
                        crop=semantic[max(0,y1):min(480,y2),max(0,x1):min(640,x2)]
                        ids,counts=np.unique(crop,return_counts=True);items=[]
                        for i,n in sorted(zip(ids,counts),key=lambda x:-x[1])[:5]:
                            obj=sim.semantic_scene.objects[int(i)] if 0<=i<len(sim.semantic_scene.objects) else None
                            ident=obj.id if obj else None
                            items.append(dict(semantic_id=int(i),pixels=int(n),object_id=ident,
                                category=obj.category.name() if obj and obj.category else None,
                                is_goal=bool(ident and ident.split('_')[-1] in wanted)))
                        stats.append(dict(detection=d,box_semantic_counts=items))
                    frame['semantic_box_audit']=stats
                    frame['semantic_warning']='Semantic mesh/instance-ID alignment has NOT been independently validated; do not infer detector correctness from these labels alone.'
                checks.append(frame)
            if step==max(args.steps):break
            a=actions[step+1]
            if a in names:agent.act(names[a])
            elif a in (4,5):
                for sensor in agent._sensors.values():sensor.node.rotation=sensor.node.rotation*mn.Quaternion.rotation(mn.Deg(30 if a==4 else -30),mn.Vector3.x_axis())
            elif a!=0:raise ValueError(a)
        (args.output/'validation.json').write_text(json.dumps(dict(source_log=str(args.log),
            note='offline re-render; pose/depth verified, original RGB hash unavailable',frames=checks),indent=2)+'\n')
        print('Verified frames:',args.steps,flush=True)
    finally:sim.close()

if __name__=='__main__':main()
