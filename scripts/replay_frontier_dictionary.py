#!/usr/bin/env python3
"""Replay archived real images and geometry, verifying exact model-image bytes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from avic_bridge.frontier_dictionary import DictionaryFrontierModel, CONTRACT, dictionary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-input-padding',action='store_true')
    a=p.parse_args()
    saved=json.loads((a.source/'input.json').read_text())
    candidates=[]
    for c in saved['candidates']:
        view=dict(c['view'],rgb_jpeg_b64=base64.b64encode((a.source/c['real_file']).read_bytes()).decode())
        candidates.append(dict(id=c['id'],frontier=c['frontier'],view=view))
    payload=dict(contract=CONTRACT,request_id='replay'+a.source.name,episode=saved['episode'],
                 dictionary=dictionary(),target=saved['target_for_navigation_only'],candidates=candidates)
    out=DictionaryFrontierModel('local').predict(payload,a.output)
    from PIL import Image
    from avic_bridge.vlm_transport import image_part
    layouts=json.loads((a.output/'input.json').read_text())['candidates']
    padded=[]
    for c,new in zip(saved['candidates'],layouts):
        for i,name in enumerate(c['actual_model_input_files']):
            if (a.source/name).read_bytes()==(a.output/name).read_bytes():continue
            assert a.allow_input_padding and new['image_input_layout'][i]['padding'], 'Unexpected input change: '+name
            raw_image=a.source/(c['context_file'] if i==0 else c['crop_file'])
            expected=base64.b64decode(image_part(Image.open(raw_image))['image_url']['url'].split(',',1)[1])
            assert expected==(a.output/name).read_bytes(), 'Not solely input padding: '+name
            padded.append(dict(file=name,layout=new['image_input_layout'][i]))
    report=dict(source=str(a.source),all_model_image_bytes_identical=not padded,verified_padding_only_changes=padded,
                generated_images=out['generated_images'],
                calls=out['vlm_calls'],seconds=out['inference_seconds'],
                candidates=[dict(id=c['id'],objects=len(c['probabilities']),probability_sum=sum(c['probabilities'].values()))
                            for c in out['candidates']],
                source_sha256=hashlib.sha256(Path('avic_bridge/frontier_dictionary.py').read_bytes()).hexdigest())
    (a.output/'replay_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
