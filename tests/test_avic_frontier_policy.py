import base64
import unittest

from avic_bridge.controller import assess_navigation_gate
from avic_bridge.service import (
    SVCQwenBackend,
    _last_json_object,
    align_frontier_scores,
)


def payload(frontiers, active=True, imagination_ranks=None):
    plans = [
        {
            "plan_id": "frontier_%d" % item["rank"],
            "actions": [{"id": item["first_action"], "name": "probe"}],
            "source_frontier": item,
            "frontier_anchor": {
                "observation_id": "anchor_%d" % item["rank"],
                "rgb_jpeg_b64": base64.b64encode(
                    ("real-anchor-%d" % item["rank"]).encode("ascii")
                ).decode("ascii"),
                "robot_x": 0.0,
                "robot_y": 0.0,
                "robot_yaw_rad": 0.0,
                "camera_pitch_rad": 0.0,
                "frontier_distance_m": 1.0,
                "relative_yaw_to_frontier_rad": 0.0,
                "anchor_quality": 0.8,
                "detected_objects": [],
            },
        }
        for item in frontiers
    ]
    if imagination_ranks is None:
        imagination_ranks = [item["rank"] for item in frontiers]
    return {
        "request_id": "test-request",
        "target": {"category": "toilet", "related_room": "bathroom"},
        "observation": {"rgb_jpeg_b64": "fake-image"},
        "camera": {"forward_step_m": 0.25, "turn_angle_deg": 30.0, "pitch_rad": 0.0},
        "navigation": {
            "baseline_action": {"id": 1, "name": "move_forward"},
            "frontier_candidates": frontiers,
            "candidate_plans": plans,
            "imagination_plan_ids": [
                "frontier_%d" % rank for rank in imagination_ranks
            ],
        },
        "score_policy": {
            "real_weight": 0.5,
            "imagination_weight": 0.5,
            "max_imagination_bonus": 0.02,
            "min_imagination_reliability": 0.55,
            "min_imagination_evidence_gain": 0.20,
        },
        "signals": {
            "final_state": 0,
            "exploration_result": 0,
            "frontier_query_active": active,
        },
    }


def frontier(
    rank,
    semantic,
    path,
    yaw,
    action,
    selected=False,
    evidence_observed=False,
):
    return {
        "rank": rank,
        "x": float(rank),
        "y": float(rank + 1),
        "semantic_score": semantic,
        "path_length": path,
        "relative_yaw_rad": yaw,
        "first_action": action,
        "path_action_prefix": [action, 1],
        "observation_evidence": 0.8 if evidence_observed else 0.2,
        "evidence_observed": evidence_observed,
        "selected_by_apexnav": selected,
    }


def score_band_gate(frontiers):
    return assess_navigation_gate(
        frontiers,
        score_band_low=0.191149,
        score_band_high=0.227590,
        max_score_gap=1.0,
    )


class NavigationGateTests(unittest.TestCase):
    def test_only_middle_band_frontiers_need_imagination(self):
        annotated, result = score_band_gate(
            [
                frontier(0, 0.18, 1.0, -0.8, 3),
                frontier(1, 0.20, 1.1, 0.0, 1),
                frontier(2, 0.24, 1.2, 0.8, 2),
            ]
        )
        self.assertTrue(result["use_imagination"])
        self.assertEqual(result["imagination_frontier_ranks"], [1])
        self.assertEqual(
            [item["score_band"] for item in annotated],
            ["low", "middle", "high"],
        )
        self.assertEqual(
            [item["needs_imagination"] for item in annotated],
            [False, True, False],
        )

    def test_band_boundaries_are_inclusive(self):
        annotated, result = score_band_gate(
            [
                frontier(0, 0.191149, 1.0, -0.8, 3),
                frontier(1, 0.227590, 1.1, 0.8, 2),
            ]
        )
        self.assertTrue(result["use_imagination"])
        self.assertEqual(
            [item["score_band"] for item in annotated], ["middle", "middle"]
        )

    def test_no_middle_band_frontier_uses_real_scores_only(self):
        annotated, result = score_band_gate(
            [
                frontier(0, 0.18, 1.0, -0.8, 3),
                frontier(1, 0.24, 1.1, 0.8, 2),
            ]
        )
        self.assertFalse(result["use_imagination"])
        self.assertEqual(result["imagination_frontier_ranks"], [])
        self.assertFalse(any(item["needs_imagination"] for item in annotated))

    def test_clear_top_two_score_margin_skips_imagination(self):
        annotated, result = assess_navigation_gate(
            [
                frontier(0, 0.22, 1.0, -0.8, 3),
                frontier(1, 0.20, 1.1, 0.8, 2),
            ],
            score_band_low=0.191149,
            score_band_high=0.227590,
            max_score_gap=0.01,
        )
        self.assertFalse(result["use_imagination"])
        self.assertFalse(result["ambiguous_real_scores"])
        self.assertAlmostEqual(result["top_score_gap"], 0.02)
        self.assertFalse(any(item["needs_imagination"] for item in annotated))

    def test_observed_middle_frontier_returns_to_real_score(self):
        annotated, result = score_band_gate(
            [
                frontier(0, 0.20, 1.0, -0.8, 3, evidence_observed=True),
                frontier(1, 0.205, 1.1, 0.8, 2),
            ]
        )
        self.assertTrue(result["use_imagination"])
        self.assertEqual(result["observed_middle_count"], 1)
        self.assertEqual(result["eligible_middle_count"], 1)
        self.assertEqual(result["imagination_frontier_ranks"], [1])
        self.assertEqual(
            [item["needs_imagination"] for item in annotated], [False, True]
        )

    def test_all_observed_middle_frontiers_skip_imagination(self):
        annotated, result = score_band_gate(
            [
                frontier(0, 0.20, 1.0, -0.8, 3, evidence_observed=True),
                frontier(1, 0.205, 1.1, 0.8, 2, evidence_observed=True),
            ]
        )
        self.assertFalse(result["use_imagination"])
        self.assertEqual(result["imagination_frontier_ranks"], [])
        self.assertIn("sufficient real observation", result["reason"])
        self.assertFalse(any(item["needs_imagination"] for item in annotated))


class QwenResponseParsingTests(unittest.TestCase):
    def test_nested_prefix_response_keeps_top_level_container(self):
        response = (
            '{"prefixes":[{"plan_id":"frontier_0","prefix_length":2,'
            '"reason":"use future view"}]}'
        )
        parsed = _last_json_object(response)
        self.assertIn("prefixes", parsed)
        self.assertEqual(parsed["prefixes"][0]["plan_id"], "frontier_0")

    def test_markdown_wrapped_nested_response_keeps_top_level_container(self):
        response = (
            '```json\n{"prefixes":[{"plan_id":"frontier_1",'
            '"prefix_length":1,"reason":"short"}]}\n```'
        )
        parsed = _last_json_object(response)
        self.assertIn("prefixes", parsed)
        self.assertEqual(parsed["prefixes"][0]["prefix_length"], 1)


class FrontierAlignmentTests(unittest.TestCase):
    def test_equal_weight_fusion_can_change_frontier_order(self):
        plans = [
            {
                "plan_id": "frontier_0",
                "source_frontier": {
                    "rank": 0,
                    "semantic_score": 0.30,
                    "path_length": 2.0,
                },
            },
            {
                "plan_id": "frontier_1",
                "source_frontier": {
                    "rank": 1,
                    "semantic_score": 0.29,
                    "path_length": 1.0,
                },
            },
        ]
        selected, records, confidence = align_frontier_scores(
            plans,
            {"frontier_0": 0.20, "frontier_1": 0.90},
            {"frontier_0": 1.0, "frontier_1": 1.0},
        )
        self.assertEqual(selected, "frontier_1")
        self.assertEqual(records[0]["plan_id"], "frontier_1")
        self.assertAlmostEqual(records[0]["decision_semantic_score"], 0.595)
        self.assertGreater(confidence, 0.0)

    def test_imagined_score_is_equal_weight_not_max(self):
        plans = [
            {"plan_id": "frontier_0", "source_frontier": {"semantic_score": 0.30}}
        ]
        _, records, _ = align_frontier_scores(
            plans, {"frontier_0": 0.10}, {"frontier_0": 1.0}
        )
        self.assertAlmostEqual(records[0]["decision_semantic_score"], 0.20)
        self.assertEqual(records[0]["imagination_weight"], 0.5)

    def test_unimagined_candidate_keeps_real_score(self):
        plans = [
            {"plan_id": "frontier_0", "source_frontier": {"semantic_score": 0.30}}
        ]
        _, records, _ = align_frontier_scores(plans, {}, {})
        self.assertEqual(records[0]["decision_semantic_score"], 0.30)


class FakeHorizonClient:
    def __init__(self, prefixes, evaluations=None):
        self.prefixes = prefixes
        self.evaluations = evaluations
        self.calls = []
        self.verifier_calls = []

    def select_prefix_lengths(self, _payload, plans):
        self.calls.append([str(plan["plan_id"]) for plan in plans])
        return {"prefixes": list(self.prefixes)}

    def evaluate_frontier_sequences(self, _payload, candidates):
        self.verifier_calls.append([str(item["plan_id"]) for item in candidates])
        if self.evaluations is not None:
            return {"evaluations": list(self.evaluations)}
        return {
            "evaluations": [
                {
                    "plan_id": str(item["plan_id"]),
                    "reliability": 0.8,
                    "evidence_gain": 0.5,
                    "semantic_relation": "plausible related-room continuation",
                    "reason": "consistent and informative",
                }
                for item in candidates
            ]
        }


class FakeSVC:
    def __init__(self, duplicate_frames=False):
        self.generated = []
        self.duplicate_frames = duplicate_frames

    def generate_plan(self, **kwargs):
        plan = kwargs["plan"]
        self.generated.append(plan)
        frame_value = "same-frame" if self.duplicate_frames else plan["plan_id"]
        encoded_frame = base64.b64encode(frame_value.encode("ascii")).decode("ascii")
        return {
            "plan_id": plan["plan_id"],
            "actions": plan["actions"],
            "source_frontier": plan.get("source_frontier"),
            "frontier_anchor": plan.get("frontier_anchor"),
            "future_jpeg_b64": encoded_frame,
            "future_frames": [
                {
                    "order": 1,
                    "video_frame_index": 8,
                    "jpeg_b64": encoded_frame,
                    "artifact": "/tmp/%s-step.jpg" % plan["plan_id"],
                }
            ],
            "artifact": "/tmp/%s.jpg" % plan["plan_id"],
            "source_video": "/tmp/%s-pred.mp4" % plan["plan_id"],
        }


class FakeITM:
    def cosine(self, *_args):
        return 0.90


class HorizonSelectorTests(unittest.TestCase):
    def test_one_batched_request_selects_bound_route_prefixes(self):
        backend = SVCQwenBackend()
        backend._qwen = FakeHorizonClient(
            [
                {"plan_id": "frontier_0", "prefix_length": 2, "reason": "turn"},
                {"plan_id": "frontier_1", "prefix_length": 1, "reason": "short"},
            ]
        )
        test_payload = payload(
            [
                frontier(0, 0.20, 1.0, -1.0, 3),
                frontier(1, 0.21, 1.1, 1.0, 2),
            ]
        )
        test_payload["navigation"]["candidate_plans"][0]["actions"] = [
            {"id": 3, "name": "turn_right"},
            {"id": 3, "name": "turn_right"},
            {"id": 1, "name": "move_forward"},
        ]
        selected, reason, errors = backend._select_prefixes(
            test_payload, test_payload["navigation"]["candidate_plans"]
        )
        self.assertEqual(len(backend._qwen.calls), 1)
        self.assertEqual(selected["frontier_0"], [3, 3])
        self.assertEqual(selected["frontier_1"], [2])
        self.assertIn("frontier_0 prefix=2/3", reason)
        self.assertEqual(errors, [])

    def test_invalid_prefix_cannot_change_or_invent_route(self):
        backend = SVCQwenBackend()
        backend._qwen = FakeHorizonClient(
            [{"plan_id": "frontier_0", "prefix_length": 2, "reason": "invalid"}]
        )
        test_payload = payload(
            [
                frontier(0, 0.20, 1.0, -0.8, 2),
                frontier(1, 0.21, 1.1, 0.8, 3),
            ]
        )
        selected, _, errors = backend._select_prefixes(
            test_payload, test_payload["navigation"]["candidate_plans"]
        )
        self.assertEqual(selected, {})
        self.assertTrue(any("invalid prefix_length" in item for item in errors))
        self.assertTrue(
            any("missing valid prefix for frontier_1" in item for item in errors)
        )

    def test_svc_results_keep_direct_frontier_binding(self):
        backend = SVCQwenBackend()
        backend._qwen = FakeHorizonClient(
            [{"plan_id": "frontier_1", "prefix_length": 1, "reason": "one step"}]
        )
        backend._svc = FakeSVC()
        backend._itm = FakeITM()
        test_payload = payload(
            [
                frontier(0, 0.20, 1.0, -0.8, 2),
                frontier(1, 0.21, 1.1, 0.8, 3),
            ],
            imagination_ranks=[1],
        )
        result = backend.decide(test_payload)
        self.assertEqual(result["decision"], "call_wm")
        self.assertEqual(len(backend._qwen.calls), 1)
        self.assertEqual(len(backend._qwen.verifier_calls), 1)
        self.assertEqual(backend._svc.generated[0]["plan_id"], "frontier_1")
        self.assertEqual(
            backend._svc.generated[0]["frontier_anchor"]["observation_id"],
            "anchor_1",
        )
        records = {item["plan_id"]: item for item in result["candidate_scores"]}
        self.assertAlmostEqual(
            records["frontier_1"]["decision_semantic_score"], 0.218
        )
        self.assertNotIn("frontier_0", records)
        self.assertAlmostEqual(records["frontier_1"]["imagination_bonus"], 0.008)

    def test_identical_imagined_trajectories_receive_no_bonus(self):
        backend = SVCQwenBackend()
        backend._qwen = FakeHorizonClient(
            [
                {"plan_id": "frontier_0", "prefix_length": 1, "reason": "short"},
                {"plan_id": "frontier_1", "prefix_length": 1, "reason": "short"},
            ]
        )
        backend._svc = FakeSVC(duplicate_frames=True)
        backend._itm = FakeITM()
        result = backend.decide(
            payload(
                [
                    frontier(0, 0.20, 1.0, -0.8, 2),
                    frontier(1, 0.205, 1.1, 0.8, 3),
                ]
            )
        )
        self.assertEqual(result["decision"], "call_wm")
        self.assertEqual(len(result["candidate_scores"]), 2)
        self.assertTrue(
            all(item["imagination_bonus"] == 0.0 for item in result["candidate_scores"])
        )
        self.assertTrue(
            all(item["imagination_reliability"] == 0.0 for item in result["candidate_scores"])
        )
        self.assertTrue(
            all("duplicate" in item["verifier_reason"] for item in result["candidate_scores"])
        )

    def test_verifier_rejection_keeps_apexnav_real_score(self):
        backend = SVCQwenBackend()
        backend._qwen = FakeHorizonClient(
            [{"plan_id": "frontier_0", "prefix_length": 1, "reason": "short"}],
            evaluations=[
                {
                    "plan_id": "frontier_0",
                    "reliability": 0.9,
                    "evidence_gain": 0.1,
                    "semantic_relation": "no new evidence",
                    "reason": "continuation is redundant",
                }
            ],
        )
        backend._svc = FakeSVC()
        backend._itm = FakeITM()
        result = backend.decide(
            payload([frontier(0, 0.20, 1.0, -0.8, 2)])
        )
        record = result["candidate_scores"][0]
        self.assertEqual(record["imagination_bonus"], 0.0)
        self.assertEqual(record["decision_semantic_score"], 0.20)
        self.assertTrue(record["imagination_evaluated"])


if __name__ == "__main__":
    unittest.main()
