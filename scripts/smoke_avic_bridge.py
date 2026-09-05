#!/usr/bin/env python3
"""Send one real image and a reachable-frontier snapshot through AVIC."""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2

# Allow direct execution via ``python scripts/smoke_avic_bridge.py`` without
# requiring the caller to export PYTHONPATH first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from avic_bridge import AvicController, AvicControllerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--image", required=True)
    parser.add_argument("--target", default="chair")
    parser.add_argument("--room", default="living room")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--mode", choices=("apexnav", "avic"), default="avic")
    args = parser.parse_args()

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit("Could not read image: %s" % args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    controller = AvicController(
        AvicControllerConfig(
            mode=args.mode,
            service_url=args.url,
            timeout_s=args.timeout,
            warmup_steps=0,
            cooldown_steps=0,
            max_calls_per_episode=1,
        )
    )
    print(json.dumps(controller.check_health(), indent=2, ensure_ascii=False))
    # The smoke test has one supplied image, so register it at two synthetic
    # headings to exercise the per-frontier anchor protocol without Habitat.
    controller.record_observation(
        rgb=rgb,
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw_rad=math.atan2(0.8, 1.0),
        camera_pitch_rad=0.0,
        step_index=0,
        itm_cosine=0.20,
        detected_objects=[],
    )
    controller.record_observation(
        rgb=rgb,
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw_rad=math.atan2(0.9, -0.8),
        camera_pitch_rad=0.0,
        step_index=1,
        itm_cosine=0.20,
        detected_objects=[],
    )
    decision = controller.decide_action(
        rgb=rgb,
        baseline_action=1,
        target_category=args.target,
        target_room=args.room,
        target_related_objects=[],
        scene_id="bridge_smoke",
        episode_id="0",
        step_index=0,
        camera_pitch_rad=0.0,
        itm_cosine=0.1,
        final_state=0,
        exploration_result=0,
        frontier_candidates=[
            {
                "rank": 0,
                "x": 1.0,
                "y": 0.8,
                "semantic_score": 0.20,
                "path_length": 1.4,
                "relative_yaw_rad": -0.75,
                "first_action": 3,
                "path_action_prefix": [3, 3, 1, 1],
                "selected_by_apexnav": False,
                "query_id": 1,
                "query_active": True,
                "observation_evidence": 0.20,
                "evidence_observed": False,
            },
            {
                "rank": 1,
                "x": -0.8,
                "y": 0.9,
                "semantic_score": 0.205,
                "path_length": 1.3,
                "relative_yaw_rad": 0.75,
                "first_action": 2,
                "path_action_prefix": [2, 2, 1, 1],
                "selected_by_apexnav": True,
                "query_id": 1,
                "query_active": True,
                "observation_evidence": 0.25,
                "evidence_observed": False,
            },
        ],
    )
    print(json.dumps(decision.__dict__, indent=2, ensure_ascii=False))
    if decision.service_decision == "skip" and decision.reason.startswith("AVIC failure"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
