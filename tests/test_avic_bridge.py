import base64
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from avic_bridge.controller import (
    AvicController,
    AvicControllerConfig,
    filter_applicable_frontier_scores,
)
from avic_bridge.protocol import SCHEMA_VERSION, action_dict, validate_request
from avic_bridge.svc_adapter import SVCAdapter, build_relative_camera_path


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    def health(self):
        return {"ok": True, "backend": "fake"}

    def decide(self, payload):
        self.requests.append(payload)
        if self.error is not None:
            raise self.error
        response = dict(self.response)
        response["schema_version"] = SCHEMA_VERSION
        response["request_id"] = payload["request_id"]
        return response


def make_controller(mode, client):
    controller = AvicController(
        AvicControllerConfig(
            mode=mode,
            warmup_steps=0,
            cooldown_steps=0,
            max_calls_per_episode=2,
        ),
        client=client,
    )
    if mode == "avic":
        # Two different real directional observations anchor the two physical
        # frontiers.  The RGB content differs deliberately so a regression to
        # one shared starting frame is visible in the request.
        for step, (x, y, value) in enumerate(((1.0, 2.0, 32), (-1.0, 1.5, 192)), 1):
            controller.record_observation(
                rgb=np.full((16, 24, 3), value, dtype=np.uint8),
                robot_x=0.0,
                robot_y=0.0,
                robot_yaw_rad=math.atan2(y, x),
                camera_pitch_rad=0.0,
                step_index=step,
                itm_cosine=0.20,
                detected_objects=[],
            )
    return controller


def frontier_candidates(query_id=1, query_active=True):
    return [
        {
            "rank": 0,
            "x": 1.0,
            "y": 2.0,
            "semantic_score": 0.20,
            "path_length": 1.2,
            "relative_yaw_rad": -0.6,
            "first_action": 3,
            "path_action_prefix": [3, 3, 1, 1],
            "selected_by_apexnav": False,
            "query_id": query_id,
            "query_active": query_active,
            "observation_evidence": 0.20,
            "evidence_observed": False,
        },
        {
            "rank": 1,
            "x": -1.0,
            "y": 1.5,
            "semantic_score": 0.21,
            "path_length": 0.8,
            "relative_yaw_rad": 0.5,
            "first_action": 2,
            "path_action_prefix": [2, 1, 1],
            "selected_by_apexnav": True,
            "query_id": query_id,
            "query_active": query_active,
            "observation_evidence": 0.25,
            "evidence_observed": False,
        },
    ]


def decide(controller, baseline=1):
    return controller.decide_action(
        rgb=np.zeros((16, 24, 3), dtype=np.uint8),
        baseline_action=baseline,
        target_category="chair",
        target_room="living room",
        scene_id="scene.glb",
        episode_id="7",
        step_index=30,
        camera_pitch_rad=0.0,
        itm_cosine=0.1,
        final_state=0,
        exploration_result=0,
        frontier_candidates=frontier_candidates(),
    )


class ControllerTests(unittest.TestCase):
    def test_generation_epoch_changes_only_on_episode_reset(self):
        client = FakeClient({"decision": "fallback", "reason": "test"})
        controller = make_controller("avic", client)
        epoch = controller.generation_epoch
        decide(controller)
        self.assertEqual(client.requests[0]["episode"]["generation_epoch"], epoch)
        self.assertEqual(controller.generation_epoch, epoch)
        controller.reset_episode()
        self.assertNotEqual(controller.generation_epoch, epoch)
    def test_cache_context_ttl_and_reset(self):
        import copy
        controller = make_controller("avic", FakeClient())
        controller.current_step = 10
        plan = {"source_frontier": frontier_candidates()[0],
                "frontier_anchor": {"observation_id": "real-1"}}
        score = dict(imagined_itm_score=0.4, imagination_reliability=0.8, imagination_bonus=0.004)
        controller._store_cached_imagination(plan, score)
        self.assertIsNotNone(controller._find_cached_imagination(plan))
        for field, value in (("x", 1.2), ("relative_yaw_rad", 2.0), ("observation_evidence", 0.4)):
            changed = copy.deepcopy(plan)
            changed["source_frontier"][field] = value
            self.assertIsNone(controller._find_cached_imagination(changed), field)
        changed = copy.deepcopy(plan)
        changed["frontier_anchor"]["observation_id"] = "real-2"
        self.assertIsNone(controller._find_cached_imagination(changed))
        controller.current_step = 41
        self.assertIsNone(controller._find_cached_imagination(plan))
        controller.reset_episode()
        self.assertEqual(controller.frontier_imagination_cache, [])

    def test_negative_cache_expires_and_equal_bonuses_reach_hybrid(self):
        client = FakeClient({"decision": "call_wm", "reason": "equal evidence",
            "candidate_scores": [dict(plan_id="frontier_%d" % rank,
                imagined_itm_score=0.5, imagination_reliability=0.8,
                imagination_bonus=0.004, imagination_evaluated=True) for rank in range(2)]})
        controller = make_controller("avic", client)
        result = decide(controller)
        self.assertEqual([s['rank'] for s in result.frontier_scores], [0, 1])
        self.assertAlmostEqual(result.frontier_scores[0]['decision_semantic_score'], 0.204)
        self.assertAlmostEqual(result.frontier_scores[1]['decision_semantic_score'], 0.214)
        self.assertFalse(result.overridden)  # Only ROS Hybrid may choose an action.
        controller.reset_episode()
        controller.current_step = 10
        plan = {"source_frontier": frontier_candidates()[0], "frontier_anchor": {"observation_id": "x"}}
        controller._store_cached_imagination(plan, dict(imagined_itm_score=0.5,
            imagination_reliability=0.8, imagination_bonus=0.0))
        controller.current_step = 90
        self.assertIsNotNone(controller._find_cached_imagination(plan))
        controller.current_step = 91
        self.assertIsNone(controller._find_cached_imagination(plan))

    def test_hybrid_receives_small_and_weaker_bonuses_with_incumbent(self):
        for bonuses in [(0.0032, 0.0032), (0.0029, 0.0032), (0.0005, 0.0032)]:
            with self.subTest(bonuses=bonuses):
                client = FakeClient({'decision': 'call_wm', 'reason': 'Hybrid inputs',
                    'candidate_scores': [dict(plan_id='frontier_%d' % rank,
                        imagined_itm_score=0.5, imagination_reliability=0.8,
                        imagination_bonus=bonus, imagination_evaluated=True)
                        for rank, bonus in enumerate(bonuses)]})
                result = decide(make_controller('avic', client))
                self.assertEqual([s['rank'] for s in result.frontier_scores], [0, 1])
                for score, bonus in zip(result.frontier_scores, bonuses):
                    self.assertAlmostEqual(score['imagination_bonus'], bonus)
                self.assertFalse(result.overridden)

    def test_apexnav_mode_never_calls_service(self):
        client = FakeClient(error=AssertionError("service must not be called"))
        result = decide(make_controller("apexnav", client), baseline=1)
        self.assertEqual(result.executed_action, 1)
        self.assertEqual(result.service_decision, "skip")
        self.assertEqual(client.requests, [])

    def test_avic_returns_temporary_scores_but_never_overrides_action(self):
        client = FakeClient(
            {
                "decision": "call_wm",
                "selected_action": action_dict(2),
                "selected_plan_id": "frontier_0",
                "reason": "test",
                "latency_ms": 1.0,
                "candidate_scores": [
                    {
                        "plan_id": "frontier_0",
                        "real_semantic_score": 0.20,
                        "observation_evidence": 0.20,
                        "evidence_state": "uncertain_positive",
                        "imagined_itm_score": 0.36,
                        "imagination_reliability": 0.9,
                        "decision_semantic_score": 0.28,
                    },
                    {
                        "plan_id": "frontier_1",
                        "real_semantic_score": 0.21,
                        "observation_evidence": 0.25,
                        "evidence_state": "uncertain_positive",
                        "imagined_itm_score": 0.12,
                        "imagination_reliability": 0.9,
                        "decision_semantic_score": 0.165,
                    },
                ],
            }
        )
        result = decide(make_controller("avic", client), baseline=1)
        self.assertEqual(result.executed_action, 1)
        self.assertFalse(result.overridden)
        self.assertEqual(len(result.frontier_scores), 1)
        self.assertEqual(result.frontier_scores[0]["rank"], 0)
        self.assertAlmostEqual(
            result.frontier_scores[0]["decision_semantic_score"], 0.22
        )
        self.assertAlmostEqual(result.frontier_scores[0]["imagined_itm_score"], 0.36)
        validate_request(client.requests[0])
        base64.b64decode(client.requests[0]["observation"]["rgb_jpeg_b64"])

    def test_world_model_never_directly_overrides_turn_with_forward(self):
        client = FakeClient(
            {
                "decision": "call_wm",
                "selected_action": action_dict(1),
                "selected_plan_id": "frontier_0",
                "reason": "test",
                "latency_ms": 1.0,
                "candidate_scores": [],
            }
        )
        result = decide(make_controller("avic", client), baseline=2)
        self.assertEqual(result.executed_action, 2)

    def test_service_error_falls_back(self):
        result = decide(make_controller("avic", FakeClient(error=TimeoutError("late"))))
        self.assertEqual(result.executed_action, 1)
        self.assertEqual(result.service_decision, "fallback")
        self.assertTrue(result.used_world_model)
        self.assertGreater(result.latency_ms, 0.0)

    def test_middle_band_frontiers_become_imagination_plans(self):
        client = FakeClient(
            {
                "decision": "skip",
                "selected_action": action_dict(1),
                "selected_plan_id": "apexnav",
                "reason": "test",
                "latency_ms": 1.0,
            }
        )
        controller = make_controller("avic", client)
        controller.decide_action(
            rgb=np.zeros((16, 24, 3), dtype=np.uint8),
            baseline_action=1,
            target_category="chair",
            target_room="living room",
            scene_id="scene.glb",
            episode_id="7",
            step_index=30,
            camera_pitch_rad=0.0,
            itm_cosine=0.1,
            final_state=0,
            exploration_result=0,
            frontier_candidates=[
                {
                    "rank": 0,
                    "x": 1.0,
                    "y": 2.0,
                    "semantic_score": 0.20,
                    "path_length": 1.2,
                    "relative_yaw_rad": -0.6,
                    "first_action": 3,
                    "path_action_prefix": [3, 3, 1, 1],
                    "selected_by_apexnav": False,
                    "query_id": 1,
                    "query_active": True,
                    "observation_evidence": 0.20,
                },
                {
                    "rank": 1,
                    "x": -1.0,
                    "y": 1.5,
                    "semantic_score": 0.21,
                    "path_length": 0.8,
                    "relative_yaw_rad": 0.5,
                    "first_action": 2,
                    "path_action_prefix": [2, 1, 1],
                    "selected_by_apexnav": True,
                    "query_id": 1,
                    "query_active": True,
                    "observation_evidence": 0.25,
                },
            ],
        )
        request = client.requests[0]
        self.assertEqual(
            request["navigation"]["imagination_plan_ids"],
            ["frontier_0", "frontier_1"],
        )
        self.assertEqual(
            request["score_policy"],
            {
                "real_weight": 0.5,
                "mode": "strict_evidence",
                "imagination_weight": 0.5,
                "min_imagination_score_spread": 0.0001,
                "max_imagination_bonus": 0.02,
                "min_imagination_reliability": 0.55,
                "min_imagination_evidence_gain": 0.20,
            },
        )
        self.assertEqual(request["navigation"]["candidate_plans"][0]["plan_id"], "frontier_0")
        self.assertEqual(
            request["navigation"]["candidate_plans"][0]["apexnav_route_actions"][0]["id"],
            3,
        )
        anchors = [
            plan["frontier_anchor"]
            for plan in request["navigation"]["candidate_plans"]
        ]
        self.assertNotEqual(anchors[0]["observation_id"], anchors[1]["observation_id"])
        self.assertNotEqual(anchors[0]["rgb_jpeg_b64"], anchors[1]["rgb_jpeg_b64"])

    def test_only_unobserved_middle_frontier_is_sent_for_imagination(self):
        client = FakeClient(
            {
                "decision": "skip",
                "selected_action": action_dict(1),
                "selected_plan_id": "apexnav",
                "reason": "test",
                "latency_ms": 1.0,
            }
        )
        candidates = frontier_candidates()
        candidates[0]["observation_evidence"] = 0.8
        candidates[0]["evidence_observed"] = True
        controller = make_controller("avic", client)
        controller.decide_action(
            rgb=np.zeros((16, 24, 3), dtype=np.uint8),
            baseline_action=1,
            target_category="chair",
            target_room="living room",
            scene_id="scene.glb",
            episode_id="7",
            step_index=30,
            camera_pitch_rad=0.0,
            itm_cosine=0.1,
            final_state=0,
            exploration_result=0,
            frontier_candidates=candidates,
        )
        request = client.requests[0]
        self.assertEqual(
            request["navigation"]["imagination_plan_ids"], ["frontier_1"]
        )
        self.assertFalse(
            request["navigation"]["frontier_candidates"][0]["needs_imagination"]
        )
        self.assertTrue(
            request["navigation"]["frontier_candidates"][1]["needs_imagination"]
        )

    def test_same_frontier_query_is_evaluated_once(self):
        client = FakeClient(
            {
                "decision": "skip",
                "selected_action": action_dict(1),
                "selected_plan_id": None,
                "reason": "clear",
                "latency_ms": 1.0,
            }
        )
        controller = make_controller("avic", client)
        decide(controller)
        result = decide(controller)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(result.reason, "frontier decision already evaluated")

    def test_frontiers_outside_middle_band_never_call_service(self):
        client = FakeClient(error=AssertionError("service must not be called"))
        controller = make_controller("avic", client)
        candidates = frontier_candidates()
        candidates[0]["semantic_score"] = 0.18
        candidates[1]["semantic_score"] = 0.24
        result = controller.decide_action(
            rgb=np.zeros((16, 24, 3), dtype=np.uint8),
            baseline_action=1,
            target_category="chair",
            target_room="living room",
            scene_id="scene.glb",
            episode_id="7",
            step_index=30,
            camera_pitch_rad=0.0,
            itm_cosine=0.1,
            final_state=0,
            exploration_result=0,
            frontier_candidates=candidates,
        )
        self.assertEqual(client.requests, [])
        self.assertEqual(result.service_decision, "skip")
        self.assertEqual(result.frontier_scores, [])

    def test_direction_only_view_without_semantic_support_is_not_an_anchor(self):
        client = FakeClient(error=AssertionError("service must not be called"))
        controller = AvicController(
            AvicControllerConfig(
                mode="avic",
                warmup_steps=0,
                cooldown_steps=0,
                max_calls_per_episode=2,
            ),
            client=client,
        )
        for step, (x, y) in enumerate(((1.0, 2.0), (-1.0, 1.5)), 1):
            controller.record_observation(
                rgb=np.zeros((16, 24, 3), dtype=np.uint8),
                robot_x=0.0,
                robot_y=0.0,
                robot_yaw_rad=math.atan2(y, x),
                camera_pitch_rad=0.0,
                step_index=step,
                itm_cosine=0.18,
                detected_objects=[],
            )
        result = decide(controller)
        self.assertEqual(client.requests, [])
        self.assertIn("no eligible frontier has a real directional anchor", result.reason)

    def test_detected_related_object_can_anchor_a_low_itm_view(self):
        controller = AvicController(
            AvicControllerConfig(mode="avic", warmup_steps=0),
            client=FakeClient(),
        )
        controller.record_observation(
            rgb=np.zeros((16, 24, 3), dtype=np.uint8),
            robot_x=0.0,
            robot_y=0.0,
            robot_yaw_rad=math.atan2(2.0, 1.0),
            camera_pitch_rad=0.0,
            step_index=10,
            itm_cosine=0.18,
            detected_objects=[{"label": "bathtub", "score": 0.8}],
        )
        anchor = controller._select_frontier_anchor(frontier_candidates()[0], 20)
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["semantic_support_source"], "detected_object")

    def test_positive_cache_reaches_hybrid_without_accumulation_or_extra_call(self):
        client = FakeClient(
            {
                "decision": "call_wm",
                "selected_action": action_dict(1),
                "selected_plan_id": "frontier_0",
                "reason": "generated",
                "latency_ms": 1.0,
                "candidate_scores": [
                    {
                        "plan_id": "frontier_0",
                        "imagined_itm_score": 0.40,
                        "imagination_reliability": 1.0,
                    },
                    {
                        "plan_id": "frontier_1",
                        "imagined_itm_score": 0.10,
                        "imagination_reliability": 1.0,
                    },
                ],
            }
        )
        controller = make_controller("avic", client)
        first = decide(controller)
        reused = controller.decide_action(
            rgb=np.zeros((16, 24, 3), dtype=np.uint8),
            baseline_action=1,
            target_category="chair",
            target_room="living room",
            scene_id="scene.glb",
            episode_id="7",
            step_index=40,
            camera_pitch_rad=0.0,
            itm_cosine=0.1,
            final_state=0,
            exploration_result=0,
            frontier_candidates=frontier_candidates(query_id=2),
        )
        self.assertTrue(first.used_world_model)
        self.assertEqual(len(first.frontier_scores), 1)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(reused.service_decision, "cache_hit")
        self.assertFalse(reused.used_world_model)
        self.assertEqual(reused.cache_hits, 2)
        self.assertEqual(len(reused.frontier_scores), 1)
        self.assertEqual(reused.frontier_scores[0]['imagination_bonus'], first.frontier_scores[0]['imagination_bonus'])
        self.assertEqual(reused.frontier_scores[0]['decision_semantic_score'], first.frontier_scores[0]['decision_semantic_score'])

        outside = frontier_candidates(query_id=3)
        outside[0]["semantic_score"] = 0.24
        outside[1]["semantic_score"] = 0.18
        ignored = controller.decide_action(
            rgb=np.zeros((16, 24, 3), dtype=np.uint8),
            baseline_action=1,
            target_category="chair",
            target_room="living room",
            scene_id="scene.glb",
            episode_id="7",
            step_index=50,
            camera_pitch_rad=0.0,
            itm_cosine=0.1,
            final_state=0,
            exploration_result=0,
            frontier_candidates=outside,
        )
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(ignored.service_decision, "skip")
        self.assertEqual(ignored.frontier_scores, [])

    def test_observed_frontier_does_not_reuse_cached_imagination(self):
        client = FakeClient(
            {
                "decision": "call_wm",
                "selected_action": action_dict(1),
                "selected_plan_id": "frontier_0",
                "reason": "generated",
                "latency_ms": 1.0,
                "candidate_scores": [
                    {
                        "plan_id": "frontier_0",
                        "imagined_itm_score": 0.40,
                        "imagination_reliability": 1.0,
                    },
                    {
                        "plan_id": "frontier_1",
                        "imagined_itm_score": 0.30,
                        "imagination_reliability": 1.0,
                    },
                ],
            }
        )
        controller = make_controller("avic", client)
        decide(controller)

        observed = frontier_candidates(query_id=2)
        for item in observed:
            item["observation_evidence"] = 0.8
            item["evidence_observed"] = True
        result = controller.decide_action(
            rgb=np.zeros((16, 24, 3), dtype=np.uint8),
            baseline_action=1,
            target_category="chair",
            target_room="living room",
            scene_id="scene.glb",
            episode_id="7",
            step_index=40,
            camera_pitch_rad=0.0,
            itm_cosine=0.1,
            final_state=0,
            exploration_result=0,
            frontier_candidates=observed,
        )
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(result.service_decision, "skip")
        self.assertEqual(result.frontier_scores, [])

    def test_cache_ignores_route_churn_but_invalidates_yaw_change(self):
        client = FakeClient(
            {
                "decision": "call_wm",
                "selected_action": action_dict(1),
                "selected_plan_id": "frontier_0",
                "reason": "generated",
                "latency_ms": 1.0,
                "candidate_scores": [
                    {
                        "plan_id": "frontier_0",
                        "imagined_itm_score": 0.40,
                        "imagination_reliability": 1.0,
                    },
                    {
                        "plan_id": "frontier_1",
                        "imagined_itm_score": 0.30,
                        "imagination_reliability": 1.0,
                    },
                ],
            }
        )
        controller = make_controller("avic", client)
        decide(controller)

        changed = frontier_candidates(query_id=2)
        changed[0]["path_action_prefix"] = [3, 1, 1]
        changed[1]["relative_yaw_rad"] += 0.6
        controller.decide_action(
            rgb=np.zeros((16, 24, 3), dtype=np.uint8),
            baseline_action=1,
            target_category="chair",
            target_room="living room",
            scene_id="scene.glb",
            episode_id="7",
            step_index=40,
            camera_pitch_rad=0.0,
            itm_cosine=0.1,
            final_state=0,
            exploration_result=0,
            frontier_candidates=changed,
        )
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.requests[-1]["navigation"]["imagination_plan_ids"], ["frontier_1"])

    def test_rejected_imagination_is_cached_but_never_published(self):
        client = FakeClient(
            {
                "decision": "call_wm",
                "selected_action": action_dict(1),
                "selected_plan_id": "frontier_0",
                "reason": "verifier rejected both hypotheses",
                "latency_ms": 1.0,
                "candidate_scores": [
                    {
                        "plan_id": "frontier_0",
                        "imagined_itm_score": 0.40,
                        "imagination_reliability": 0.0,
                        "evidence_gain": 0.0,
                        "imagination_bonus": 0.0,
                        "imagination_evaluated": True,
                    },
                    {
                        "plan_id": "frontier_1",
                        "imagined_itm_score": 0.30,
                        "imagination_reliability": 0.0,
                        "evidence_gain": 0.0,
                        "imagination_bonus": 0.0,
                        "imagination_evaluated": True,
                    },
                ],
            }
        )
        controller = make_controller("avic", client)
        first = decide(controller)
        second = controller.decide_action(
            rgb=np.zeros((16, 24, 3), dtype=np.uint8),
            baseline_action=1,
            target_category="chair",
            target_room="living room",
            scene_id="scene.glb",
            episode_id="7",
            step_index=40,
            camera_pitch_rad=0.0,
            itm_cosine=0.1,
            final_state=0,
            exploration_result=0,
            frontier_candidates=frontier_candidates(query_id=2),
        )
        self.assertTrue(first.used_world_model)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(first.frontier_scores, [])
        self.assertEqual(second.service_decision, "cache_hit")
        self.assertEqual(second.frontier_scores, [])

    def test_filter_applicable_frontier_scores_requires_strict_gain(self):
        rows = [
            {
                "frontier_id": 0,
                "real_semantic_score": 0.20,
                "imagination_bonus": 0.0,
                "decision_semantic_score": 0.20,
            },
            {
                "frontier_id": 1,
                "real_semantic_score": 0.21,
                "imagination_bonus": 0.02,
                "decision_semantic_score": 0.23,
            },
            {
                "frontier_id": 2,
                "real_semantic_score": 0.22,
                "imagination_bonus": 0.02,
                "decision_semantic_score": 0.22,
            },
        ]

        filtered = filter_applicable_frontier_scores(rows)

        self.assertEqual([row["frontier_id"] for row in filtered], [1])

    def test_transient_unverified_result_is_not_applied_or_cached(self):
        client = FakeClient(
            {
                "decision": "call_wm",
                "selected_action": action_dict(1),
                "selected_plan_id": "frontier_0",
                "reason": "verifier timeout",
                "latency_ms": 1.0,
                "candidate_scores": [
                    {
                        "plan_id": "frontier_0",
                        "imagined_itm_score": 0.40,
                        "imagination_reliability": 0.0,
                        "evidence_gain": 0.0,
                        "imagination_bonus": 0.0,
                        "imagination_evaluated": False,
                    },
                    {
                        "plan_id": "frontier_1",
                        "imagined_itm_score": 0.30,
                        "imagination_reliability": 0.0,
                        "evidence_gain": 0.0,
                        "imagination_bonus": 0.0,
                        "imagination_evaluated": False,
                    },
                ],
            }
        )
        controller = make_controller("avic", client)
        first = decide(controller)
        second = controller.decide_action(
            rgb=np.zeros((16, 24, 3), dtype=np.uint8),
            baseline_action=1,
            target_category="chair",
            target_room="living room",
            scene_id="scene.glb",
            episode_id="7",
            step_index=40,
            camera_pitch_rad=0.0,
            itm_cosine=0.1,
            final_state=0,
            exploration_result=0,
            frontier_candidates=frontier_candidates(query_id=2),
        )
        self.assertEqual(first.service_decision, "fallback")
        self.assertEqual(second.service_decision, "fallback")
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(controller.frontier_imagination_cache, [])


class VideoFrameSamplingTests(unittest.TestCase):
    def test_single_sample_is_terminal_frame_and_multisample_includes_terminal(self):
        frames = [np.full((8, 8, 3), index, dtype=np.uint8) for index in range(10)]

        class FakeCapture:
            def __init__(self, _path):
                self.index = 0

            def isOpened(self):
                return True

            def get(self, _property):
                return len(frames)

            def set(self, _property, value):
                self.index = int(value)

            def read(self):
                return True, frames[self.index].copy()

            def release(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single_dir = root / "single"
            multi_dir = root / "multi"
            single_dir.mkdir()
            multi_dir.mkdir()
            with patch("avic_bridge.svc_adapter.cv2.VideoCapture", FakeCapture):
                single = SVCAdapter._sample_video_frames(
                    root / "fake.mp4", 1, single_dir
                )
                multi = SVCAdapter._sample_video_frames(
                    root / "fake.mp4", 3, multi_dir
                )
        self.assertEqual([item["video_frame_index"] for item in single], [9])
        self.assertEqual([item["video_frame_index"] for item in multi], [3, 6, 9])
        self.assertEqual(multi[-1]["video_frame_index"], 9)


class CameraPathTests(unittest.TestCase):
    def test_apexnav_action_units(self):
        poses = build_relative_camera_path(
            [2, 1], forward_step_m=0.25, turn_angle_deg=30.0, pitch_rad=0.0
        )
        self.assertEqual(poses.shape, (3, 4, 4))
        self.assertAlmostEqual(float(np.linalg.norm(poses[2, :3, 3])), 0.25, places=6)

if __name__ == "__main__":
    unittest.main()
