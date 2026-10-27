#!/usr/bin/env python3
"""Reproduce the frozen MP3D SR development set without consulting outcomes."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
NAME = "mp3d_sr_dev100_seed20260911_runtime"


def make_manifest(root=ROOT):
    excluded = set()
    exclusions = []
    for name in (
        "mp3d_random50_seed20260906_habitat.json",
        "mp3d_random50_seed20260907_disjoint_runtime.json",
        "mp3d_random50_seed20260909_disjoint_runtime.json",
    ):
        path = root / "experiments/manifests" / name
        data = path.read_bytes()
        exclusions.append({"path": str(path.relative_to(root)), "sha256": hashlib.sha256(data).hexdigest()})
        excluded.update((e["scene_key"], str(e["episode_id"])) for e in json.loads(data)["episodes"])
    # Historical first-50 run: only identifiers are read, never success labels.
    path = root / "experiments/runs/paired50_mp3d/apexnav/20260906_005305/record.txt"
    first50 = sorted(set(re.findall(r"Scene ID: .*?/([^/]+)\.glb\s+Episode ID: (\d+)", path.read_text())))
    if len(first50) != 50:
        raise ValueError("Expected exactly 50 historical task identifiers")
    excluded.update(first50)
    pool, fingerprints, source_count = [], [], 0
    for path in sorted((root / "data/datasets/objectnav/mp3d/v1/val/content").glob("*.json.gz")):
        raw = path.read_bytes()
        fingerprints.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
        episodes = json.loads(gzip.decompress(raw))["episodes"]
        scene = path.name.split(".")[0]
        source_count += len(episodes)
        for index, episode in enumerate(episodes):
            # Habitat 0.3.1 remaps source IDs to their per-scene array index.
            if (scene, str(index)) not in excluded:
                pool.append({"scene_key": scene, "episode_id": str(index), "object_category": episode["object_category"]})
    if source_count != 2195 or len(pool) != 2000:
        raise ValueError("Dataset/exclusion pool changed; do not silently redraw the fixed set")
    selected = random.Random(20260911).sample(pool, 100)
    return {
        "name": NAME, "dataset": "mp3d", "split": "val", "selection_version": 1,
        "purpose": "fixed SR development set; not an untouched final test set after tuning",
        "criteria": {
            "random_seed": 20260911, "sample_size": 100,
            "method": "uniform random sample without replacement; no outcome-based selection",
            "pool_order": "content filename sorted, then per-scene runtime episode index",
            "selection_order": "random.Random(seed).sample output order; keep this order in every run",
            "source_episode_count": source_count, "excluded_unique_episode_count": len(excluded),
            "eligible_episode_count": len(pool), "excluded_manifests": exclusions,
            "excluded_first50_scene_episode_keys": first50,
            "episode_id_semantics": "Habitat runtime per-scene array index, NOT raw JSON episode_id",
        },
        "dataset_content_sha256": fingerprints,
        "scene_counts": dict(sorted(collections.Counter(e["scene_key"] for e in selected).items())),
        "category_counts": dict(sorted(collections.Counter(e["object_category"] for e in selected).items())),
        "episodes": selected,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare against frozen manifest; never overwrite")
    args = parser.parse_args()
    data = make_manifest()
    if args.check:
        frozen = ROOT / "experiments/manifests" / (NAME + ".json")
        if json.loads(frozen.read_text()) != json.loads(json.dumps(data)):
            raise SystemExit("Frozen manifest differs from reproduced selection")
        print("OK: 100 unique tasks, 11 scenes, 16 raw target categories, no overlap with excluded tasks")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))
