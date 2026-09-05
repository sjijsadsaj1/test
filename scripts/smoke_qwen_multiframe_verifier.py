#!/usr/bin/env python3
"""Replay saved SVC artifacts through AVIC's real Qwen multi-image calls."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from avic_bridge.service import QwenOpenAIClient


def _encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _load_candidate(audit_path: Path, index: int):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if len(audit.get("candidates", [])) != 1:
        raise ValueError("smoke audit must contain exactly one candidate: %s" % audit_path)
    source = audit["candidates"][0]
    score = source.get("score", {})
    candidate_dir = Path(source["source_video"]).parent
    plan_id = "smoke_frontier_%d" % index
    anchor = {
        "rgb_jpeg_b64": _encoded(candidate_dir / "input.jpg"),
        "anchor_quality": float(source.get("anchor_quality", 0.0)),
        "detected_objects": [],
    }
    actions = [{"id": int(action)} for action in source["actions"]]
    frames = []
    for frame in source.get("frames", []):
        frames.append(
            {
                "order": int(frame["order"]),
                "jpeg_b64": _encoded(Path(frame["artifact"])),
            }
        )
    source_frontier = {
        "semantic_score": float(score.get("real_semantic_score", 0.0)),
        "observation_evidence": float(score.get("observation_evidence", 0.0)),
        "path_length": 1.0 + index,
    }
    plan = {
        "plan_id": plan_id,
        "actions": actions,
        "frontier_anchor": anchor,
        "source_frontier": source_frontier,
    }
    candidate = {
        **plan,
        "future_frames": frames,
        "frame_itm_scores": [
            float(value) for value in score.get("frame_itm_scores", [])
        ],
    }
    return audit["target"], plan, candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verification", nargs=3, type=Path)
    args = parser.parse_args()

    target = None
    plans = []
    candidates = []
    for index, audit_path in enumerate(args.verification):
        item_target, plan, candidate = _load_candidate(audit_path, index)
        if target is None:
            target = item_target
        elif item_target != target:
            raise ValueError("all smoke candidates must have identical targets")
        plans.append(plan)
        candidates.append(candidate)

    client = QwenOpenAIClient()
    payload = {"target": target}
    prefixes = client.select_prefix_lengths(payload, plans)
    evaluations = client.evaluate_frontier_sequences(payload, candidates)
    print(json.dumps({"prefixes": prefixes, "evaluations": evaluations}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
