"""Fail-open AVIC frontier-score augmenter for ApexNav's exploration policy.

The controller never writes imagined observations into ApexNav maps and never
uses a world-model response as a low-level robot command.  It only returns a
temporary semantic score for each candidate; ApexNav's Hybrid policy still
chooses the final frontier and action.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .client import AvicHTTPClient
from .protocol import IMAGINATION_ACTION_IDS, SCHEMA_VERSION, action_dict, make_candidate_plan


def assess_navigation_gate(
    frontier_candidates: List[Dict[str, Any]],
    *,
    score_band_low: float,
    score_band_high: float,
    max_score_gap: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select ambiguous decisions containing an unobserved middle-band frontier.

    This is the only call/no-call gate.  Scores below or above the frozen band
    remain untouched.  A middle-band frontier also returns immediately to its
    real score once ApexNav reports sufficient real-observation evidence.  Only
    the remaining unobserved middle-band frontiers in a decision whose two best
    real semantic scores are close may use cached or fresh SVC predictions.
    The gate never looks at imagined frames or changes map state.
    """

    frontiers = [dict(item) for item in frontier_candidates]
    band_counts = {"low": 0, "middle": 0, "high": 0}
    observed_middle_count = 0
    imagination_frontier_ranks: List[int] = []
    for frontier in frontiers:
        semantic = float(frontier.get("semantic_score", 0.0))
        evidence_observed = bool(frontier.get("evidence_observed", False))
        if semantic < score_band_low:
            score_band = "low"
        elif semantic <= score_band_high:
            score_band = "middle"
            if evidence_observed:
                observed_middle_count += 1
            else:
                imagination_frontier_ranks.append(int(frontier.get("rank", 0)))
        else:
            score_band = "high"
        band_counts[score_band] += 1
        frontier["score_band"] = score_band
        frontier["needs_imagination"] = (
            score_band == "middle" and not evidence_observed
        )

    sorted_scores = sorted(
        (float(frontier.get("semantic_score", 0.0)) for frontier in frontiers),
        reverse=True,
    )
    top_score_gap = (
        sorted_scores[0] - sorted_scores[1]
        if len(sorted_scores) >= 2
        else float("inf")
    )
    ambiguous = len(frontiers) >= 2 and top_score_gap <= max_score_gap
    has_middle_band_frontier = band_counts["middle"] > 0
    has_eligible_middle_band_frontier = bool(imagination_frontier_ranks)
    use_imagination = ambiguous and has_eligible_middle_band_frontier
    if not use_imagination:
        imagination_frontier_ranks = []
        for frontier in frontiers:
            frontier["needs_imagination"] = False
    assessment: Dict[str, Any] = {
        "use_imagination": use_imagination,
        "score_band_low": float(score_band_low),
        "score_band_high": float(score_band_high),
        "max_score_gap": float(max_score_gap),
        "top_score_gap": float(top_score_gap),
        "ambiguous_real_scores": bool(ambiguous),
        "band_counts": band_counts,
        "observed_middle_count": int(observed_middle_count),
        "eligible_middle_count": len(imagination_frontier_ranks),
        "imagination_frontier_ranks": imagination_frontier_ranks,
        "reason": (
            "G1: middle-band ranks=%s band=[%.6f, %.6f] counts=%s"
            % (
                imagination_frontier_ranks,
                score_band_low,
                score_band_high,
                band_counts,
            )
            if use_imagination
            else (
                "G1: clear real-score margin %.6f > %.6f"
                % (top_score_gap, max_score_gap)
                if has_eligible_middle_band_frontier and len(frontiers) >= 2
                else (
                    "G1: all middle-band frontiers have sufficient real observation"
                    if has_middle_band_frontier
                    else "G1: no middle-band frontier"
                )
            )
        ),
    }
    return frontiers, assessment


@dataclass
class AvicControllerConfig:
    mode: str = "apexnav"  # apexnav | avic
    service_url: str = "http://127.0.0.1:8091"
    timeout_s: float = 180.0
    warmup_steps: int = 26
    max_calls_per_episode: int = 20
    cooldown_steps: int = 5
    jpeg_quality: int = 85
    max_image_side: int = 768
    score_band_low: float = 0.191149
    score_band_high: float = 0.227590
    max_score_gap: float = 0.01
    real_score_weight: float = 0.5
    imagination_score_weight: float = 0.5
    min_imagination_score_spread: float = 0.0001
    cache_match_radius_m: float = 0.5
    cache_yaw_tolerance_rad: float = math.radians(15.0)
    observation_history_size: int = 96
    anchor_horizontal_fov_rad: float = math.radians(90.0)
    anchor_max_distance_m: float = 8.0
    anchor_min_quality: float = 0.35
    max_imagination_bonus: float = 0.02
    min_imagination_reliability: float = 0.55
    min_imagination_evidence_gain: float = 0.20

    def __post_init__(self):
        if self.mode not in ("apexnav", "avic"):
            raise ValueError("navigation mode must be apexnav or avic")
        if self.timeout_s <= 0:
            raise ValueError("AVIC timeout must be positive")
        if self.warmup_steps < 0 or self.max_calls_per_episode < 0 or self.cooldown_steps < 0:
            raise ValueError("AVIC budgets must be non-negative")
        if self.max_image_side < 224:
            raise ValueError("AVIC max_image_side must be at least 224")
        if self.score_band_low < 0 or self.score_band_high < self.score_band_low:
            raise ValueError("AVIC score band must satisfy 0 <= low <= high")
        if self.max_score_gap < 0:
            raise ValueError("AVIC maximum score gap must be non-negative")
        if self.real_score_weight < 0 or self.imagination_score_weight < 0:
            raise ValueError("AVIC score weights must be non-negative")
        if self.real_score_weight + self.imagination_score_weight <= 0:
            raise ValueError("at least one AVIC score weight must be positive")
        if self.min_imagination_score_spread < 0:
            raise ValueError("AVIC imagination score spread must be non-negative")
        if self.cache_match_radius_m <= 0:
            raise ValueError("AVIC cache match radius must be positive")
        if not 0 < self.cache_yaw_tolerance_rad <= math.pi:
            raise ValueError("AVIC cache yaw tolerance must be in (0, pi]")
        if self.observation_history_size < 1:
            raise ValueError("AVIC observation history must be positive")
        if not 0 < self.anchor_horizontal_fov_rad <= 2.0 * math.pi:
            raise ValueError("AVIC anchor horizontal FOV must be in (0, 2*pi]")
        if self.anchor_max_distance_m <= 0 or not 0 <= self.anchor_min_quality <= 1:
            raise ValueError("AVIC anchor distance/quality is invalid")
        if self.max_imagination_bonus < 0:
            raise ValueError("AVIC maximum imagination bonus must be non-negative")
        if not 0 <= self.min_imagination_reliability <= 1:
            raise ValueError("AVIC minimum reliability must be in [0, 1]")
        if not 0 <= self.min_imagination_evidence_gain <= 1:
            raise ValueError("AVIC minimum evidence gain must be in [0, 1]")


@dataclass
class AvicDecision:
    baseline_action: int
    executed_action: int
    service_decision: str
    selected_plan_id: Optional[str]
    reason: str
    latency_ms: float
    used_world_model: bool
    overridden: bool
    frontier_scores: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0


class AvicController:
    """Owns episode budgets, payload construction and safety fallbacks."""

    def __init__(self, config: AvicControllerConfig, client=None):
        self.config = config
        self.client = client or AvicHTTPClient(config.service_url, config.timeout_s)
        self.reset_episode()

    @property
    def enabled(self) -> bool:
        return self.config.mode == "avic"

    def reset_episode(self) -> None:
        self.world_model_calls = 0
        self.last_world_model_step = -10**9
        self.last_frontier_query_id = -1
        self.action_history: List[int] = []
        self.frontier_imagination_cache: List[Dict[str, Any]] = []
        self.observation_history: List[Dict[str, Any]] = []

    def check_health(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"ok": True, "backend": "disabled"}
        return self.client.health()

    def record_executed_action(self, action_id: int) -> None:
        self.action_history.append(int(action_id))
        if len(self.action_history) > 12:
            self.action_history = self.action_history[-12:]

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def record_observation(
        self,
        *,
        rgb: np.ndarray,
        robot_x: float,
        robot_y: float,
        robot_yaw_rad: float,
        camera_pitch_rad: float,
        step_index: int,
        itm_cosine: Optional[float] = None,
        detected_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Store a compact real-view history used to anchor each frontier.

        Images are JPEG-compressed immediately so the history remains bounded.
        This history is decision-only: it is never published to ValueMap and is
        cleared at every episode reset.
        """

        if not self.enabled:
            return
        encoded, width, height = self._encode_rgb(rgb)
        normalized_objects = []
        for item in detected_objects or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            try:
                score = max(0.0, min(1.0, float(item.get("score", 0.0))))
            except (TypeError, ValueError):
                continue
            normalized_objects.append({"label": label, "score": score})
        digest = hashlib.sha256(base64.b64decode(encoded)).hexdigest()[:16]
        observation_id = "step_%06d_%s" % (int(step_index), digest)
        record = {
            "observation_id": observation_id,
            "rgb_jpeg_b64": encoded,
            "width": int(width),
            "height": int(height),
            "robot_x": float(robot_x),
            "robot_y": float(robot_y),
            "robot_yaw_rad": float(robot_yaw_rad),
            "camera_pitch_rad": float(camera_pitch_rad),
            "step_index": int(step_index),
            "itm_cosine": None if itm_cosine is None else float(itm_cosine),
            "detected_objects": normalized_objects,
        }
        if self.observation_history and (
            self.observation_history[-1]["observation_id"] == observation_id
        ):
            self.observation_history[-1] = record
        else:
            self.observation_history.append(record)
            self.observation_history = self.observation_history[
                -self.config.observation_history_size :
            ]

    def _select_frontier_anchor(
        self, frontier: Dict[str, Any], current_step: int
    ) -> Optional[Dict[str, Any]]:
        """Choose the real historical view that most directly observed a frontier."""

        frontier_x = float(frontier["x"])
        frontier_y = float(frontier["y"])
        best: Optional[Dict[str, Any]] = None
        best_quality = -1.0
        half_fov = self.config.anchor_horizontal_fov_rad * 0.5
        score_span = max(self.config.score_band_high - self.config.score_band_low, 1e-6)
        for observation in self.observation_history:
            dx = frontier_x - float(observation["robot_x"])
            dy = frontier_y - float(observation["robot_y"])
            distance = math.hypot(dx, dy)
            if distance <= 1e-6 or distance > self.config.anchor_max_distance_m:
                continue
            bearing = math.atan2(dy, dx)
            relative_yaw = self._wrap_angle(
                bearing - float(observation["robot_yaw_rad"])
            )
            angular_error = abs(relative_yaw)
            if angular_error > half_fov:
                continue

            alignment = max(0.0, math.cos(angular_error))
            proximity = max(0.0, 1.0 - distance / self.config.anchor_max_distance_m)
            pitch_alignment = max(
                0.0,
                math.cos(min(abs(float(observation["camera_pitch_rad"])), math.pi / 2.0)),
            )
            itm = observation.get("itm_cosine")
            itm_support = 0.0
            if isinstance(itm, (int, float)):
                itm_support = max(
                    0.0,
                    min(1.0, (float(itm) - self.config.score_band_low) / score_span),
                )
            object_support = max(
                (float(item.get("score", 0.0)) for item in observation["detected_objects"]),
                default=0.0,
            )
            # A merely well-aligned camera view is not enough.  SVC is useful
            # here only when the real view already contains partial semantic
            # evidence that it can extend (for example, a bathtub while
            # searching for a sink).  This prevents unsupported imagination
            # from being launched on visually empty frontiers.
            itm_is_partial_evidence = (
                isinstance(itm, (int, float))
                and float(itm) >= self.config.score_band_low
            )
            if not itm_is_partial_evidence and object_support <= 0.0:
                continue
            age = max(0, int(current_step) - int(observation["step_index"]))
            recency = math.exp(-float(age) / 80.0)
            quality = (
                0.45 * alignment
                + 0.15 * proximity
                + 0.20 * max(itm_support, object_support)
                + 0.10 * pitch_alignment
                + 0.10 * recency
            )
            if quality <= best_quality:
                continue
            best_quality = quality
            best = {
                **observation,
                "frontier_distance_m": float(distance),
                "relative_yaw_to_frontier_rad": float(relative_yaw),
                "anchor_quality": float(quality),
                "age_steps": int(age),
                "semantic_support": float(max(itm_support, object_support)),
                "semantic_support_source": (
                    "detected_object"
                    if object_support >= itm_support and object_support > 0.0
                    else "itm"
                ),
            }
        if best is None or best_quality < self.config.anchor_min_quality:
            return None
        return best

    def _probe_actions_from_anchor(self, anchor: Dict[str, Any]) -> List[int]:
        """Build a short virtual probe towards the bound frontier.

        These actions condition SVC only and are never sent to Habitat.  They
        begin in the anchor camera frame, unlike the current-pose A* prefix.
        """

        yaw = float(anchor["relative_yaw_to_frontier_rad"])
        turn_steps = min(2, int(round(abs(math.degrees(yaw)) / 30.0)))
        actions: List[int] = []
        if turn_steps:
            actions.extend(([2] if yaw > 0.0 else [3]) * turn_steps)
        distance = float(anchor["frontier_distance_m"])
        forward_steps = max(1, min(3, int(round(min(distance, 0.75) / 0.25))))
        actions.extend([1] * forward_steps)
        return actions[:6]

    def _skip(self, baseline_action: int, reason: str) -> AvicDecision:
        return AvicDecision(
            baseline_action=baseline_action,
            executed_action=baseline_action,
            service_decision="skip",
            selected_plan_id=None,
            reason=reason,
            latency_ms=0.0,
            used_world_model=False,
            overridden=False,
            frontier_scores=[],
        )

    def _find_cached_imagination(
        self, plan: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        frontier = plan["source_frontier"]
        x = float(frontier["x"])
        y = float(frontier["y"])
        nearest = None
        nearest_distance = float("inf")
        for entry in self.frontier_imagination_cache:
            distance = math.hypot(entry["x"] - x, entry["y"] - y)
            if distance < nearest_distance:
                nearest = entry
                nearest_distance = distance
        if nearest is None or nearest_distance > self.config.cache_match_radius_m:
            return None
        return dict(nearest, match_distance_m=nearest_distance)

    def _store_cached_imagination(
        self, plan: Dict[str, Any], score: Dict[str, Any]
    ) -> None:
        if self._find_cached_imagination(plan) is not None:
            return
        frontier = plan["source_frontier"]
        self.frontier_imagination_cache.append(
            {
                "x": float(frontier["x"]),
                "y": float(frontier["y"]),
                "imagined_itm_score": float(score["imagined_itm_score"]),
                "imagination_reliability": float(score["imagination_reliability"]),
                "imagination_bonus": float(score["imagination_bonus"]),
                "evidence_gain": float(score.get("evidence_gain", 0.0)),
                "verifier_reason": str(score.get("verifier_reason", "")),
                "anchor_id": str(score.get("anchor_id", "")),
            }
        )

    def _blend_score(self, real_score: float, imagined_score: float) -> float:
        real_weight = self.config.real_score_weight
        imagination_weight = self.config.imagination_score_weight
        return float(
            (real_weight * real_score + imagination_weight * imagined_score)
            / (real_weight + imagination_weight)
        )

    def _temporary_score_record(
        self,
        plan: Dict[str, Any],
        score: Dict[str, Any],
        *,
        cache_hit: bool,
    ) -> Dict[str, Any]:
        frontier = plan["source_frontier"]
        real_score = float(frontier["semantic_score"])
        return {
            "plan_id": str(plan["plan_id"]),
            "query_id": int(frontier.get("query_id", -1)),
            "rank": int(frontier.get("rank", -1)),
            "x": float(frontier["x"]),
            "y": float(frontier["y"]),
            "real_semantic_score": real_score,
            "observation_evidence": float(frontier.get("observation_evidence", 0.0)),
            "score_band": str(frontier.get("score_band", "middle")),
            "imagined_itm_score": float(score["imagined_itm_score"]),
            "imagination_reliability": float(score["imagination_reliability"]),
            "imagination_bonus": float(score["imagination_bonus"]),
            "evidence_gain": float(score.get("evidence_gain", 0.0)),
            "verifier_reason": str(score.get("verifier_reason", "")),
            "anchor_id": str(score.get("anchor_id", "")),
            "decision_semantic_score": float(real_score + score["imagination_bonus"]),
            "cache_hit": bool(cache_hit),
        }

    def _encode_rgb(self, rgb: np.ndarray):
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("RGB observation must have shape HxWx3")
        image = np.ascontiguousarray(image[:, :, :3].astype(np.uint8, copy=False))
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest > self.config.max_image_side:
            scale = self.config.max_image_side / float(longest)
            image = cv2.resize(
                image,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        )
        if not ok:
            raise ValueError("failed to encode RGB observation")
        encoded_height, encoded_width = image.shape[:2]
        return (
            base64.b64encode(encoded.tobytes()).decode("ascii"),
            encoded_width,
            encoded_height,
        )

    @staticmethod
    def _candidate_plans(
        baseline_action: int, frontier_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        del baseline_action  # Frontier paths, not the current primitive action, define candidates.
        plans: List[Dict[str, Any]] = []

        for frontier in frontier_candidates[:3]:
            first_action = int(frontier.get("first_action", -1))
            if first_action not in IMAGINATION_ACTION_IDS:
                continue
            raw_actions = frontier.get("path_action_prefix")
            if not isinstance(raw_actions, list) or not raw_actions:
                continue
            try:
                actions = [int(action) for action in raw_actions]
            except (TypeError, ValueError):
                continue
            if (
                len(actions) > 6
                or actions[0] != first_action
                or any(action not in IMAGINATION_ACTION_IDS for action in actions)
            ):
                continue
            plan = make_candidate_plan(
                "frontier_%d" % int(frontier.get("rank", len(plans))), actions
            )
            plan["source_frontier"] = dict(frontier)
            plans.append(plan)
        return plans

    def _repeat_count(self) -> int:
        if not self.action_history:
            return 0
        last = self.action_history[-1]
        count = 0
        for action in reversed(self.action_history):
            if action != last:
                break
            count += 1
        return count

    def decide_action(
        self,
        *,
        rgb: np.ndarray,
        baseline_action: int,
        target_category: str,
        target_room: str,
        target_related_objects: Optional[List[str]] = None,
        scene_id: str,
        episode_id: str,
        step_index: int,
        camera_pitch_rad: float,
        itm_cosine: Optional[float],
        final_state: int,
        exploration_result: int,
        frontier_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> AvicDecision:
        if not self.enabled:
            return self._skip(baseline_action, "AVIC disabled")
        if baseline_action not in IMAGINATION_ACTION_IDS:
            return self._skip(baseline_action, "protected stop/look action")
        if step_index < self.config.warmup_steps:
            return self._skip(baseline_action, "ApexNav initialization scan")
        frontiers = list(frontier_candidates or [])
        if len(frontiers) < 2:
            return self._skip(baseline_action, "fewer than two reachable frontiers")
        if int(final_state) != 0 or int(exploration_result) != 0:
            return self._skip(baseline_action, "ApexNav is not selecting an exploration frontier")
        query_id = int(frontiers[0].get("query_id", -1))
        query_active = bool(frontiers[0].get("query_active", False))
        if query_id < 0 or not query_active:
            return self._skip(baseline_action, "no new adaptive frontier decision")
        if query_id == self.last_frontier_query_id:
            return self._skip(baseline_action, "frontier decision already evaluated")
        # A high-level frontier query is evaluated at most once, including
        # score-band skips and service failures.
        self.last_frontier_query_id = query_id

        frontiers, outer_gate = assess_navigation_gate(
            frontiers,
            score_band_low=self.config.score_band_low,
            score_band_high=self.config.score_band_high,
            max_score_gap=self.config.max_score_gap,
        )
        if not outer_gate["use_imagination"]:
            print("[AVIC_G1] " + json.dumps(outer_gate, ensure_ascii=False), flush=True)
            return self._skip(baseline_action, str(outer_gate["reason"]))

        plans = self._candidate_plans(baseline_action, frontiers)
        imagination_ranks = set(outer_gate["imagination_frontier_ranks"])
        eligible_plans = [
            plan
            for plan in plans
            if int(plan["source_frontier"].get("rank", -1)) in imagination_ranks
        ]
        if not eligible_plans:
            return self._skip(
                baseline_action, "G1 selected no frontier with a valid imagination path"
            )

        anchored_plans = []
        missing_anchor_ranks = []
        anchor_log = []
        for plan in eligible_plans:
            anchor = self._select_frontier_anchor(plan["source_frontier"], step_index)
            if anchor is None:
                missing_anchor_ranks.append(
                    int(plan["source_frontier"].get("rank", -1))
                )
                continue
            probe_actions = self._probe_actions_from_anchor(anchor)
            anchored = make_candidate_plan(str(plan["plan_id"]), probe_actions)
            anchored["source_frontier"] = dict(plan["source_frontier"])
            anchored["frontier_anchor"] = anchor
            anchored["apexnav_route_actions"] = list(plan["actions"])
            anchored_plans.append(anchored)
            anchor_log.append(
                {
                    "plan_id": str(plan["plan_id"]),
                    "anchor_id": str(anchor["observation_id"]),
                    "quality": round(float(anchor["anchor_quality"]), 4),
                    "angle_deg": round(
                        math.degrees(float(anchor["relative_yaw_to_frontier_rad"])), 2
                    ),
                    "distance_m": round(float(anchor["frontier_distance_m"]), 3),
                    "age_steps": int(anchor["age_steps"]),
                    "probe_actions": probe_actions,
                }
            )
        if not anchored_plans:
            print(
                "[AVIC_ANCHOR] no eligible frontier has a real directional anchor; "
                "ranks=%s" % missing_anchor_ranks,
                flush=True,
            )
            return self._skip(
                baseline_action, "no eligible frontier has a real directional anchor"
            )
        eligible_plans = anchored_plans

        available_scores: Dict[str, Dict[str, Any]] = {}
        fresh_plans = []
        for plan in eligible_plans:
            cached = self._find_cached_imagination(plan)
            if cached is None:
                fresh_plans.append(plan)
                continue
            available_scores[str(plan["plan_id"])] = {
                "imagined_itm_score": cached["imagined_itm_score"],
                "imagination_reliability": cached["imagination_reliability"],
                "imagination_bonus": cached["imagination_bonus"],
                "evidence_gain": cached.get("evidence_gain", 0.0),
                "verifier_reason": cached.get("verifier_reason", ""),
                "anchor_id": cached.get("anchor_id", ""),
                "cache_hit": True,
            }

        cache_hits = len(available_scores)
        cache_misses = len(fresh_plans)
        service_block_reason = ""
        if fresh_plans and self.world_model_calls >= self.config.max_calls_per_episode:
            service_block_reason = "episode world-model budget exhausted"
            fresh_plans = []
        elif fresh_plans and step_index - self.last_world_model_step < self.config.cooldown_steps:
            service_block_reason = "world-model cooldown"
            fresh_plans = []

        outer_gate.update(
            {
                "cache_hit_ranks": [
                    int(plan["source_frontier"].get("rank", -1))
                    for plan in eligible_plans
                    if str(plan["plan_id"]) in available_scores
                ],
                "fresh_imagination_ranks": [
                    int(plan["source_frontier"].get("rank", -1))
                    for plan in fresh_plans
                ],
                "call_service": bool(fresh_plans),
                "service_block_reason": service_block_reason,
                "anchor_bindings": anchor_log,
                "missing_anchor_ranks": missing_anchor_ranks,
            }
        )
        print("[AVIC_G1] " + json.dumps(outer_gate, ensure_ascii=False), flush=True)

        response: Dict[str, Any] = {}
        latency_ms = 0.0
        service_called = False
        service_error = ""
        selected_plan_id = None
        if fresh_plans:
            request_id = "%s-%s-%06d-%s" % (
                str(scene_id).split("/")[-1],
                episode_id,
                step_index,
                uuid.uuid4().hex[:8],
            )
            try:
                encoded_rgb, width, height = self._encode_rgb(rgb)
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "request_id": request_id,
                    "episode": {
                        "scene_id": str(scene_id),
                        "episode_id": str(episode_id),
                        "step_index": int(step_index),
                    },
                    "target": {
                        "category": str(target_category),
                        "related_room": str(target_room),
                        "related_objects": [
                            str(item) for item in (target_related_objects or [])
                        ],
                    },
                    "observation": {
                        "rgb_jpeg_b64": encoded_rgb,
                        "width": int(width),
                        "height": int(height),
                    },
                    "camera": {
                        "forward_step_m": 0.25,
                        "turn_angle_deg": 30.0,
                        "pitch_rad": float(camera_pitch_rad),
                    },
                    "navigation": {
                        "baseline_action": action_dict(baseline_action),
                        "candidate_plans": fresh_plans,
                        "imagination_plan_ids": [
                            str(plan["plan_id"]) for plan in fresh_plans
                        ],
                        "frontier_candidates": frontiers,
                    },
                    "score_policy": {
                        "real_weight": float(self.config.real_score_weight),
                        "imagination_weight": float(
                            self.config.imagination_score_weight
                        ),
                        "min_imagination_score_spread": float(
                            self.config.min_imagination_score_spread
                        ),
                        "max_imagination_bonus": float(
                            self.config.max_imagination_bonus
                        ),
                        "min_imagination_reliability": float(
                            self.config.min_imagination_reliability
                        ),
                        "min_imagination_evidence_gain": float(
                            self.config.min_imagination_evidence_gain
                        ),
                    },
                    "signals": {
                        "itm_cosine": None if itm_cosine is None else float(itm_cosine),
                        "final_state": int(final_state),
                        "exploration_result": int(exploration_result),
                        "same_action_repeat_count": self._repeat_count(),
                        "recent_actions": [
                            action_dict(action) for action in self.action_history
                        ],
                        "frontier_count": len(frontiers),
                        "frontier_query_id": query_id,
                        "frontier_query_active": query_active,
                        "outer_gate": outer_gate,
                    },
                }
                service_called = True
                self.world_model_calls += 1
                self.last_world_model_step = step_index
                started = time.monotonic()
                response = self.client.decide(payload)
                latency_ms = float(response.get("latency_ms") or 0.0)
                if latency_ms <= 0:
                    latency_ms = (time.monotonic() - started) * 1000.0
                selected_plan_id = response.get("selected_plan_id")
            except Exception as exc:
                service_error = "AVIC failure: %s" % (exc,)

            plans_by_id = {str(plan["plan_id"]): plan for plan in fresh_plans}
            if response.get("decision") == "call_wm":
                for item in response.get("candidate_scores", []):
                    plan_id = str(item.get("plan_id", ""))
                    plan = plans_by_id.get(plan_id)
                    reliability = item.get("imagination_reliability", 0.0)
                    imagined_score = item.get("imagined_itm_score")
                    evaluated = bool(item.get("imagination_evaluated", False))
                    if (
                        plan is None
                        or not isinstance(imagined_score, (int, float))
                        or not isinstance(reliability, (int, float))
                        or (float(reliability) <= 0.0 and not evaluated)
                    ):
                        continue
                    real_score = float(plan["source_frontier"]["semantic_score"])
                    bonus = item.get("imagination_bonus")
                    if not isinstance(bonus, (int, float)):
                        decision_score = item.get("decision_semantic_score")
                        if isinstance(decision_score, (int, float)):
                            bonus = max(0.0, float(decision_score) - real_score)
                        else:
                            bonus = max(
                                0.0,
                                self._blend_score(real_score, float(imagined_score))
                                - real_score,
                            )
                    bonus = max(0.0, min(float(bonus), self.config.max_imagination_bonus))
                    available_scores[plan_id] = {
                        "imagined_itm_score": float(imagined_score),
                        "imagination_reliability": float(reliability),
                        "imagination_bonus": bonus,
                        "evidence_gain": float(item.get("evidence_gain", 0.0)),
                        "verifier_reason": str(item.get("verifier_reason", "")),
                        "anchor_id": str(item.get("anchor_id", "")),
                        "cache_hit": False,
                    }
                    self._store_cached_imagination(plan, available_scores[plan_id])

        frontier_scores = []
        for plan in eligible_plans:
            score = available_scores.get(str(plan["plan_id"]))
            if score is None:
                continue
            frontier_scores.append(
                self._temporary_score_record(
                    plan,
                    score,
                    cache_hit=bool(score["cache_hit"]),
                )
            )

        ranked = []
        augmented_by_id = {
            item["plan_id"]: item["decision_semantic_score"] for item in frontier_scores
        }
        for plan in plans:
            plan_id = str(plan["plan_id"])
            frontier = plan["source_frontier"]
            ranked.append(
                (
                    float(
                        augmented_by_id.get(plan_id, frontier["semantic_score"])
                    ),
                    -float(frontier.get("path_length", 0.0)),
                    plan_id,
                )
            )
        ranked.sort(reverse=True)
        confidence = None
        if ranked:
            selected_plan_id = ranked[0][2]
            confidence = 1.0
            if len(ranked) > 1:
                confidence = max(
                    0.0,
                    min(
                        1.0,
                        (ranked[0][0] - ranked[1][0])
                        / max(abs(ranked[0][0]), 1e-6),
                    ),
                )

        new_scores = sum(not item["cache_hit"] for item in frontier_scores)
        if new_scores:
            service_decision = "call_wm"
        elif frontier_scores:
            service_decision = "cache_hit"
        else:
            service_decision = "skip"
        reason_parts = [str(outer_gate["reason"])]
        if service_block_reason:
            reason_parts.append(service_block_reason)
        if response.get("reason"):
            reason_parts.append(str(response["reason"]))
        if service_error:
            reason_parts.append(service_error)
        reason_parts.append(
            "cache_hits=%d cache_misses=%d fresh_scores=%d"
            % (cache_hits, cache_misses, new_scores)
        )
        decision = AvicDecision(
            baseline_action=baseline_action,
            # The world model never directly controls Habitat or selects a
            # final frontier.  It only returns temporary semantic evidence;
            # ApexNav's unchanged Hybrid policy makes the final decision.
            executed_action=baseline_action,
            service_decision=service_decision,
            selected_plan_id=selected_plan_id,
            reason="; ".join(reason_parts),
            latency_ms=latency_ms,
            used_world_model=bool(service_called and new_scores),
            overridden=False,
            frontier_scores=frontier_scores,
            confidence=confidence,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )
        print("[AVIC] " + json.dumps(decision.__dict__, ensure_ascii=False), flush=True)
        return decision
