#!/usr/bin/env python3
"""Fresh MP3D random 50, disjoint from fixed100 and its historical exclusions."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random

from make_fixed100_manifest import make_manifest, ROOT

NAME = 'mp3d_random50_seed20260919_runtime'


def make():
    fixed = make_manifest()
    excluded = {(e['scene_key'], str(e['episode_id'])) for e in fixed['episodes']}
    for old in fixed['criteria']['excluded_manifests']:
        d = json.loads((ROOT / old['path']).read_text())
        excluded.update((e['scene_key'], str(e['episode_id'])) for e in d['episodes'])
    excluded.update(tuple(k) for k in fixed['criteria']['excluded_first50_scene_episode_keys'])
    pool, hashes = [], []
    for p in sorted((ROOT / 'data/datasets/objectnav/mp3d/v1/val/content').glob('*.json.gz')):
        raw = p.read_bytes()
        hashes.append({'file': p.name, 'sha256': hashlib.sha256(raw).hexdigest()})
        for index, e in enumerate(json.loads(gzip.decompress(raw))['episodes']):
            key = (p.name.split('.')[0], str(index))
            if key not in excluded:
                pool.append(dict(scene_key=key[0], episode_id=key[1], object_category=e['object_category']))
    if len(pool) != 1900 or len(excluded) != 295:
        raise ValueError('Dataset/exclusions changed; do not silently redraw')
    selected = random.Random(20260919).sample(pool, 50)
    return dict(name=NAME, dataset='mp3d', split='val',
        criteria=dict(random_seed=20260919, sample_size=50, eligible_episode_count=len(pool),
            excluded_unique_episode_count=len(excluded), method='uniform without replacement; no outcome selection',
            episode_id_semantics='Habitat runtime per-scene array index, NOT raw JSON episode_id'),
        excluded_keys=sorted(excluded), dataset_content_sha256=hashes,
        scene_counts=dict(sorted(Counter(e['scene_key'] for e in selected).items())),
        category_counts=dict(sorted(Counter(e['object_category'] for e in selected).items())), episodes=selected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    data = json.loads(json.dumps(make()))
    if args.check:
        if data != json.loads((ROOT / 'experiments/manifests' / (NAME + '.json')).read_text()):
            raise SystemExit('Frozen random50 manifest differs')
        print('Verified random50 seed20260919; disjoint from 295 historical/fixed100 tasks')
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))
