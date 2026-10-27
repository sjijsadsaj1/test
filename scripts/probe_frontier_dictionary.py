#!/usr/bin/env python3
"""Live VLM on original-cluster REAL crops. Never reads generated images."""
import argparse
import base64
import json
import math
from pathlib import Path
from avic_bridge.frontier_dictionary import CONTRACT,DictionaryFrontierModel,dictionary
from avic_bridge.frontier_crop import CONTRACT as CROP_CONTRACT

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--provider',choices=['local','glm'],default='local');p.add_argument('--limit',type=int,default=3)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=ROOT/'experiments/runs/image_frontier_dual_strong3/20261022_134729_229677/avic_imagination'
    grouped={}
    for file in sorted(root.glob('*/verification.json')):
        d=json.loads(file.read_text());ep=(d['episode']['scene_id'],d['episode']['episode_id'])
        for c in d['candidates']:
            f=c['source_frontier'];roi=f.get('geometry_roi',{})
            if not (roi.get('box') or roi.get('projected_box')):continue
            grouped.setdefault(ep,[]).append((file,d,c))
    selected=[]
    for ep,items in sorted(grouped.items()):
        # Prefer the previously supported crop, else keep the first physical
        # projection rejected ONLY by the old generation size/context gates.
        selected.append(next((x for x in items if x[2]['source_frontier']['geometry_roi'].get('visible')),items[0]))
        if len(selected)==a.limit:break
    if len(selected)!=a.limit:raise ValueError('not enough independent archived cluster crops')
    (a.output/'selection.json').write_text(json.dumps([str(x[0]) for x in selected],indent=2)+'\n')
    model=DictionaryFrontierModel(a.provider);summaries=[]
    for i,(file,d,c) in enumerate(selected):
        f=c['source_frontier'];v=d['anchor_pose'];roi=f['geometry_roi']
        raw=Path(d['anchor_path']).read_bytes()
        dx=f['x']-v['robot_x'];dy=f['y']-v['robot_y'];bearing=math.atan2(dy,dx)-v['robot_yaw_rad']
        # Use archived ORIGINAL projected box, not the later VLM de-framed box.
        # No generated image, synthetic view or measured target distance is read.
        view=dict(v,rgb_jpeg_b64=base64.b64encode(raw).decode(),age_steps=0,
            bearing_rad=math.atan2(math.sin(bearing),math.cos(bearing)),
            roi=dict(contract=CROP_CONTRACT,box=roi.get('box',roi.get('projected_box')),source_step=v['step_index'],
                     image_size=roi['image_size'],frontier_xy=[f['x'],f['y']],cluster_sha256=roi['cluster_sha256'],
                     reason='archived_original_cluster_projection_before_any_VLM_deframe'))
        payload=dict(contract=CONTRACT,request_id='dictionaryprobe%d'%i,episode=d['episode'],dictionary=dictionary(),
            target=d['target']['category'].split('|')[0].strip(),candidates=[dict(id='f%d'%f['rank'],frontier=f,view=view)])
        try:
            out=model.predict(payload,a.output/('%02d'%i));row=out['candidates'][0]
            summary=dict(index=i,episode=d['episode'],probabilities=row['probabilities'],
                probability_sum=sum(row['probabilities'].values()),predicted_space=row['predicted_space'],
                seconds=out['inference_seconds'],calls=out['vlm_calls'],generated_images=0)
        except Exception as exc:summary=dict(index=i,error=getattr(exc,'diagnostic',str(exc)))
        summaries.append(summary);(a.output/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
