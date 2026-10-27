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
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .client import AvicHTTPClient
from . import alignment_audit
from .relaxed_imagination import CONTRACT as RELAXED_CONTRACT,blend as relaxed_blend,q25 as relaxed_q25,finite as relaxed_finite
from .protocol import (CURRENT_ROUTE_CONTRACT, IMAGINATION_ACTION_IDS,
                       IMAGE_GLM_CONTRACT, SCHEMA_VERSION, action_dict, make_candidate_plan)


MIN_APPLICABLE_IMAGINATION_BONUS = 1e-6


def aligned_base_experiment():
    return os.environ.get('APEXNAV_ALIGNED_BASE_EXPERIMENT') == '1'

def relaxed_imagination():
    return os.environ.get('APEXNAV_RELAXED_IMAGINATION')=='1'

def image_glm_enabled():
    return os.environ.get('APEXNAV_IMAGE_GLM') == '1'


def aligned_origin(plan, item, request_id, epoch):
    anchor=plan.get('frontier_anchor') or {}
    value=item.get('anchor_itm_score')
    anchor_id=str(anchor.get('observation_id',''))
    if (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
            or not -1 <= value <= 1 or not anchor_id or str(item.get('anchor_id',''))!=anchor_id):
        raise ValueError('ALIGNED_BASE_EXPERIMENT: missing/invalid same-origin anchor score or identity')
    encoded=anchor.get('rgb_jpeg_b64','')
    raw=base64.b64decode(encoded,validate=True)
    if not raw:raise ValueError('ALIGNED_BASE_EXPERIMENT: missing original anchor JPEG')
    return dict(anchor_itm_score=float(value),anchor_id=anchor_id,
                anchor_rgb_sha256=hashlib.sha256(raw).hexdigest(),
                request_origin=dict(request_id=request_id,plan_id=plan['plan_id'],epoch=epoch))


def filter_applicable_frontier_scores(
    frontier_scores: Optional[List[Dict[str, Any]]],
    *,
    min_bonus: float = MIN_APPLICABLE_IMAGINATION_BONUS,
) -> List[Dict[str, Any]]:
    """Return only temporary scores that strictly improve their real score.

    Zero-gain results still belong in the controller cache so they suppress
    repeated model calls.  They must not be published, however, because even
    a no-op score packet causes the exploration manager to replan.
    """

    applicable: List[Dict[str, Any]] = []
    for score in frontier_scores or []:
        try:
            real_score = float(score.get("real_semantic_score", 0.0))
            bonus = float(score.get("imagination_bonus", 0.0))
            decision_score = float(
                score.get("decision_semantic_score", real_score + bonus)
            )
        except (AttributeError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (real_score, bonus, decision_score)):
            continue
        if os.environ.get('APEXNAV_TEXT_FRONTIER')=='1' and score.get('score_mode')=='text_frontier':
            from .text_frontier import CONTRACT, probability
            if os.environ.get('APEXNAV_TEXT_METHOD')=='dictionary':
                from .frontier_dictionary import CONTRACT
            try:
                p=probability(score.get('target_belief'))
                if score.get('text_contract')==CONTRACT and score.get('binding_sha256') and abs(p-decision_score)<1e-9:
                    applicable.append(score)
            except (ValueError,TypeError): pass
            continue
        if image_glm_enabled() and score.get('score_mode') == 'image_glm':
            try:
                expected=relaxed_blend(real_score,score['image_score'],score['image_real_weight'],score['image_prediction_weight'])
                if score.get('image_glm_contract')==IMAGE_GLM_CONTRACT and abs(expected-decision_score)<1e-9 and abs(decision_score-real_score)>min_bonus:
                    applicable.append(score)
            except (KeyError,ValueError,TypeError): pass
            continue
        if relaxed_imagination() and score.get('relaxed_imagination') is True:
            try:
                expected=relaxed_blend(real_score,score['future_itm_q25'],score['relaxed_real_weight'],score['relaxed_imagination_weight'])
                if score.get('score_mode')=='grounded_delta' and score.get('relaxed_contract')==RELAXED_CONTRACT and abs(expected-decision_score)<1e-9 and abs(decision_score-real_score)>min_bonus:
                    applicable.append(score)
            except (KeyError,ValueError,TypeError):pass
            continue
        if score.get("score_mode") == "frontier_once":
            applicable.append(score)
            continue
        if aligned_base_experiment() and score.get('aligned_base_experiment') is True:
            origin=score.get('aligned_origin') or {}
            anchor=origin.get('anchor_itm_score')
            if (bonus > min_bonus and isinstance(anchor,(int,float)) and not isinstance(anchor,bool)
                    and math.isfinite(anchor) and -1<=anchor<=1
                    and origin.get('request_origin') and origin.get('anchor_id') and origin.get('anchor_rgb_sha256')
                    and abs(decision_score-(anchor+bonus))<1e-9):
                applicable.append(score)
            continue
        if bonus <= min_bonus or decision_score <= real_score + min_bonus:
            continue
        applicable.append(score)
    return applicable


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
    score_mode: str = "strict_evidence"
    imagination_score_weight: float = 0.5
    min_imagination_score_spread: float = 0.0001
    cache_match_radius_m: float = 0.10
    cache_yaw_tolerance_rad: float = math.radians(15.0)
    observation_history_size: int = 96
    anchor_horizontal_fov_rad: float = math.radians(90.0)
    anchor_max_distance_m: float = 8.0
    anchor_min_quality: float = 0.35
    max_imagination_bonus: float = 0.02
    min_imagination_reliability: float = 0.55
    min_imagination_evidence_gain: float = 0.20
    positive_cache_ttl_steps: int = 30
    negative_cache_ttl_steps: int = 80
    cache_evidence_tolerance: float = 0.10

    def __post_init__(self):
        if self.score_mode not in ("strict_evidence", "grounded_delta", "frontier_once", "image_glm", "text_frontier"):
            raise ValueError("invalid AVIC score mode")
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
        if min(self.positive_cache_ttl_steps, self.negative_cache_ttl_steps) < 1:
            raise ValueError("cache lifetimes must be positive")
        if not 0 <= self.cache_evidence_tolerance <= 1:
            raise ValueError("invalid evidence tolerance")
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
    stage_latencies_ms: Optional[Dict[str, float]] = None
    workload: Optional[Dict[str, int]] = None


class AvicController:
    """Owns episode budgets, payload construction and safety fallbacks."""

    def __init__(self, config: AvicControllerConfig, client=None):
        text_enabled=os.environ.get('APEXNAV_TEXT_FRONTIER')=='1'
        if (config.score_mode=='text_frontier') != text_enabled:
            raise ValueError('text_frontier requires its dedicated launcher')
        if text_enabled and (image_glm_enabled() or relaxed_imagination() or aligned_base_experiment() or os.environ.get('APEXNAV_FRONTIER_ONCE')=='1'):
            raise ValueError('text frontier excludes other scoring experiments')
        if (config.score_mode == 'image_glm') != image_glm_enabled():
            raise ValueError('image_glm requires the dedicated APEXNAV_IMAGE_GLM=1 launcher')
        if image_glm_enabled() and (relaxed_imagination() or aligned_base_experiment() or os.environ.get('APEXNAV_FRONTIER_ONCE')=='1'):
            raise ValueError('image_glm excludes other score experiments')
        if relaxed_imagination() and (config.score_mode!='grounded_delta' or aligned_base_experiment() or os.environ.get('APEXNAV_FRONTIER_ONCE')=='1'):
            raise ValueError('RELAXED_IMAGINATION requires grounded_delta and excludes aligned_base/frontier_once')
        if aligned_base_experiment() and config.score_mode != 'grounded_delta':
            raise ValueError('ALIGNED_BASE_EXPERIMENT requires grounded_delta')
        self.config = config
        self.client = client or AvicHTTPClient(config.service_url, config.timeout_s)
        print('[ALIGNED_BASE_CONFIG] enabled=%d score_mode=%s' % (aligned_base_experiment(),config.score_mode),flush=True)
        print('[RELAXED_IMAGINATION_CONFIG] enabled=%d contract=%s' % (relaxed_imagination(),RELAXED_CONTRACT),flush=True)
        print('[IMAGE_GLM_CONFIG] enabled=%d contract=%s' % (image_glm_enabled(),IMAGE_GLM_CONTRACT),flush=True)
        self.reset_episode()

    @property
    def enabled(self) -> bool:
        return self.config.mode == "avic"

    def reset_episode(self) -> None:
        if alignment_audit.enabled():
            self.alignment_last_request = None
        self.generation_epoch = uuid.uuid4().hex
        self.world_model_calls = 0
        self.last_world_model_step = -10**9
        self.last_frontier_query_id = -1
        self.action_history: List[int] = []
        self.frontier_imagination_cache: List[Dict[str, Any]] = []
        self.observation_history: List[Dict[str, Any]] = []
        self.current_step = 0
        if hasattr(self, "once_registry"):
            del self.once_registry

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
        alignment_pose: Optional[Dict[str, Any]] = None,
        projection_depth=None,
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
        if projection_depth is not None:
            record['projection_depth']=np.asarray(projection_depth,dtype=np.float32).squeeze().copy()
        if alignment_audit.enabled():
            record['alignment_pose'] = alignment_pose
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

    def _bind_current_routes(self, eligible_plans, all_plans, rgb, pitch, step):
        """Bind genuine current-pose routes; never attach them to an old image.

        A route indistinguishable within the available prefix is deferred, not
        randomly perturbed. Include non-G1 routes when locating the divergence,
        so a lone eligible candidate cannot use the incumbent's shared prefix.
        """
        encoded, _, _ = self._encode_rgb(rgb)
        observation = next((o for o in reversed(self.observation_history)
                            if o["step_index"] == step), None)
        if (observation is None or observation["rgb_jpeg_b64"] != encoded or
                not math.isclose(observation["camera_pitch_rad"], pitch, abs_tol=1e-6)):
            return [], [{"reason": "missing or mismatched current observation"}]
        routes = {p["plan_id"]: [a["id"] for a in p["actions"]] for p in all_plans}
        bound, audit = [], []
        for plan in eligible_plans:
            route = routes[plan["plan_id"]]
            minimum = 1 if self.config.score_mode == 'image_glm' else 2
            shared = []
            for other_id, other in routes.items():
                if other_id == plan["plan_id"]:
                    continue
                divergence = next((i + 1 for i, (a, b) in enumerate(zip(route, other))
                                   if a != b), None)
                if divergence is None:
                    shared.append(other_id)
                else:
                    minimum = max(minimum, divergence)
            if shared or len(route) < minimum:
                audit.append({"plan_id": plan["plan_id"], "route_actions": route,
                              "shared_with": shared, "reason": "no distinct two-view route prefix"})
                continue
            frontier = plan["source_frontier"]
            dx = float(frontier["x"]) - observation["robot_x"]
            dy = float(frontier["y"]) - observation["robot_y"]
            anchor = dict(observation, binding_source="current_observation",
                          frontier_distance_m=math.hypot(dx, dy),
                          relative_yaw_to_frontier_rad=self._wrap_angle(
                              math.atan2(dy, dx) - observation["robot_yaw_rad"]),
                          # Not the old heuristic image-quality estimate.
                          anchor_quality=0.0, anchor_quality_kind="not_assessed", age_steps=0)
            bound.append(dict(plan, frontier_anchor=anchor,
                              apexnav_route_actions=list(plan["actions"]),
                              route_contract=CURRENT_ROUTE_CONTRACT,
                              min_prefix_length=minimum))
            audit.append({"plan_id": plan["plan_id"], "anchor_id": anchor["observation_id"],
                          "route_actions": route, "min_prefix_length": minimum,
                          "reason": "current-pose A* route bound"})
        return bound, audit

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
        anchor = plan.get("frontier_anchor") or {}
        for entry in self.frontier_imagination_cache:
            ttl = (30 if relaxed_imagination() and entry.get('relaxed_imagination') else
                   self.config.positive_cache_ttl_steps if entry["imagination_bonus"] > 0
                   else self.config.negative_cache_ttl_steps)
            if self.current_step - entry["created_step"] > ttl:
                continue
            if entry["anchor_id"] != str(anchor.get("observation_id", "")):
                continue
            if plan.get("route_contract") == CURRENT_ROUTE_CONTRACT and (
                entry.get("route_actions") != [a["id"] for a in plan["actions"]]
                or entry.get("min_prefix_length") != plan.get("min_prefix_length")
            ):
                continue
            yaw = float(frontier.get("relative_yaw_rad", 0.0))
            if abs(self._wrap_angle(yaw - entry["query_yaw"])) > self.config.cache_yaw_tolerance_rad:
                continue
            if abs(float(frontier.get("observation_evidence", 0.0)) - entry["coverage"]) > self.config.cache_evidence_tolerance:
                continue
            distance = math.hypot(entry["x"] - x, entry["y"] - y)
            if distance < nearest_distance:
                nearest = entry
                nearest_distance = distance
        if nearest is None or nearest_distance > self.config.cache_match_radius_m:
            return None
        return nearest

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
                "anchor_id": str((plan.get("frontier_anchor") or {}).get("observation_id", "")),
                "created_step": self.current_step,
                "query_yaw": float(frontier.get("relative_yaw_rad", 0.0)),
                "coverage": float(frontier.get("observation_evidence", 0.0)),
                "route_actions": [a["id"] for a in plan.get("actions", [])],
                "min_prefix_length": plan.get("min_prefix_length"),
                **({'relaxed_imagination':True,'future_itm_q25':score['future_itm_q25']} if relaxed_imagination() and score.get('relaxed_imagination') else {}),
                **({'aligned_origin':score.get('aligned_origin')} if aligned_base_experiment() else {}),
                **({'alignment_source_request':score.get('alignment_source_request')} if alignment_audit.enabled() else {}),
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
        bonus=float(score['imagination_bonus'])
        aligned_fields={}
        decision_score=real_score+bonus
        if self.config.score_mode == 'image_glm':
            decision_score=relaxed_blend(real_score,score['image_score'],self.config.real_score_weight,self.config.imagination_score_weight)
            bonus=decision_score-real_score
            aligned_fields=dict(score_mode='image_glm',image_glm_contract=IMAGE_GLM_CONTRACT,
                image_score=score['image_score'],image_real_weight=self.config.real_score_weight,
                image_prediction_weight=self.config.imagination_score_weight)
        if relaxed_imagination():
            if not score.get('relaxed_imagination') or not relaxed_finite(score.get('future_itm_q25')):
                raise ValueError('RELAXED_IMAGINATION: missing valid generated future Q25')
            decision_score=relaxed_blend(real_score,score['future_itm_q25'],self.config.real_score_weight,self.config.imagination_score_weight)
            aligned_fields=dict(score_mode='grounded_delta',relaxed_imagination=True,relaxed_contract=RELAXED_CONTRACT,
                future_itm_q25=score['future_itm_q25'],relaxed_real_weight=self.config.real_score_weight,
                relaxed_imagination_weight=self.config.imagination_score_weight)
            bonus=decision_score-real_score
        if aligned_base_experiment() and bonus > MIN_APPLICABLE_IMAGINATION_BONUS:
            origin=score.get('aligned_origin')
            if not origin or not origin.get('request_origin') or not origin.get('anchor_rgb_sha256'):
                raise ValueError('ALIGNED_BASE_EXPERIMENT: positive bonus lacks original anchor association')
            decision_score=float(origin['anchor_itm_score'])+bonus
            if not math.isfinite(decision_score):raise ValueError('ALIGNED_BASE_EXPERIMENT: nonfinite final score')
            aligned_fields=dict(aligned_base_experiment=True,aligned_origin=origin)
            print('[ALIGNED_BASE_SCORE] '+json.dumps(dict(map_base=real_score,anchor_base=origin['anchor_itm_score'],
                original_bonus=bonus,test_final=decision_score,plan_id=plan['plan_id'],query_id=frontier.get('query_id'),
                source_anchor_id=origin['anchor_id'],source_anchor_hash=origin['anchor_rgb_sha256'],
                request_origin=origin['request_origin'],cache_hit=cache_hit)),flush=True)
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
            "imagination_bonus": bonus,
            "evidence_gain": float(score.get("evidence_gain", 0.0)),
            "verifier_reason": str(score.get("verifier_reason", "")),
            "anchor_id": str(score.get("anchor_id", "")),
            "decision_semantic_score": decision_score,
            **aligned_fields,
            "cache_hit": bool(cache_hit),
            **({'alignment_source_request':score.get('alignment_source_request')} if alignment_audit.enabled() else {}),
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
        if self.config.score_mode == 'text_frontier':
            if os.environ.get('APEXNAV_TEXT_METHOD')=='dictionary':
                from .frontier_dictionary import decide_dictionary
                return decide_dictionary(self,locals())
            from .text_frontier import decide_text
            return decide_text(self, locals())
        if self.config.score_mode == "frontier_once" and self.enabled:
            from .frontier_once import decide_once
            return decide_once(self, locals())
        self.current_step = int(step_index)
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
        if relaxed_imagination() or self.config.score_mode == 'image_glm':
            chosen=sorted(frontiers,key=lambda item:int(item.get('rank',0)))[:3]
            outer_gate.update(use_imagination=True,imagination_frontier_ranks=[int(f.get('rank',0)) for f in chosen],
                              reason=('image_glm_single_view_v1' if self.config.score_mode=='image_glm' else 'relaxed_blend_v1')+': top3 real exposed routes; G1 band/gap/evidence veto disabled')
            for frontier in frontiers:frontier['needs_imagination']=frontier in chosen
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
        if self.config.score_mode in ("grounded_delta", "image_glm"):
            anchored_plans, anchor_log = self._bind_current_routes(
                eligible_plans, plans, rgb, camera_pitch_rad, step_index)
        for plan in ([] if self.config.score_mode in ("grounded_delta", "image_glm") else eligible_plans):
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
            if self.config.score_mode in ("grounded_delta", "image_glm"):
                print("[AVIC_ROUTE] " + json.dumps(anchor_log), flush=True)
                return self._skip(baseline_action, "no distinct current-pose route for imagination")
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
            cached = None if self.config.score_mode == 'image_glm' else self._find_cached_imagination(plan)
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
                **({'relaxed_imagination':True,'future_itm_q25':cached.get('future_itm_q25')} if relaxed_imagination() and cached.get('relaxed_imagination') else {}),
                **({'aligned_origin':cached.get('aligned_origin')} if aligned_base_experiment() else {}),
                **({'alignment_source_request':cached.get('alignment_source_request')} if alignment_audit.enabled() else {}),
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
                        "generation_epoch": self.generation_epoch,
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
                        **({'hfov_deg': self.rgb_hfov_deg} if hasattr(self, 'rgb_hfov_deg') else {}),
                        "forward_step_m": 0.25,
                        "turn_angle_deg": 30.0,
                        "pitch_rad": float(camera_pitch_rad),
                    },
                    "navigation": {
                        "route_contract": (CURRENT_ROUTE_CONTRACT
                            if self.config.score_mode in ("grounded_delta", "image_glm") else "historical_virtual_probe_v1"),
                        "baseline_action": action_dict(baseline_action),
                        "candidate_plans": fresh_plans,
                        "imagination_plan_ids": [
                            str(plan["plan_id"]) for plan in fresh_plans
                        ],
                        "frontier_candidates": frontiers,
                    },
                    "score_policy": {
                        **({'contract':IMAGE_GLM_CONTRACT} if self.config.score_mode=='image_glm' else {}),
                        **({'relaxed_imagination':True,'contract':RELAXED_CONTRACT} if relaxed_imagination() else {}),
                        "mode": self.config.score_mode,
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
                if alignment_audit.enabled():
                    alignment_audit.bind_request(payload, frontiers)
                    self.alignment_last_request = dict(request_id=request_id, **payload['alignment_audit'])
                self.world_model_calls += 1
                self.last_world_model_step = step_index
                started = time.monotonic()
                response = self.client.decide(payload)
                latency_ms = float(response.get("latency_ms") or 0.0)
                if latency_ms <= 0:
                    latency_ms = (time.monotonic() - started) * 1000.0
                selected_plan_id = response.get("selected_plan_id")
            except Exception as exc:
                if self.config.score_mode == 'image_glm':
                    raise RuntimeError('Image/GLM transport failed; navigation aborted') from exc
                if service_called:
                    latency_ms = (time.monotonic() - started) * 1000.0
                service_error = "AVIC failure: %s" % (exc,)

            plans_by_id = {str(plan["plan_id"]): plan for plan in fresh_plans}
            if self.config.score_mode == 'image_glm' and (response.get('decision') != 'call_wm' or not response.get('candidate_scores')):
                raise RuntimeError('Image/GLM inference returned no valid scores; navigation aborted')
            if self.config.score_mode == 'image_glm':
                response_ids=[s.get('plan_id') for s in response['candidate_scores']]
                if len(response_ids)!=len(set(response_ids)) or set(response_ids)!=set(plans_by_id):
                    raise RuntimeError('Image/GLM response missing, duplicated or unknown candidate identity')
            if response.get("decision") == "call_wm":
                for item in response.get("candidate_scores", []):
                    plan_id = str(item.get("plan_id", ""))
                    plan = plans_by_id.get(plan_id)
                    reliability = item.get("imagination_reliability", 0.0)
                    imagined_score = item.get("imagined_itm_score")
                    evaluated = item.get("imagination_evaluated")
                    if self.config.score_mode == 'image_glm':
                        if item.get('geometry_skipped') is True:
                            from .frontier_geometry import CONTRACT as GEOMETRY_CONTRACT
                            if (os.environ.get('AVIC_IMAGE_FRONTIER_GEOMETRY')!='1' or plan is None or
                                item.get('geometry_contract')!=GEOMETRY_CONTRACT or
                                item.get('anchor_id')!=plan['frontier_anchor']['observation_id'] or
                                item.get('imagination_evaluated') is not False or
                                item.get('decision_semantic_score')!=plan['source_frontier']['semantic_score']):
                                raise RuntimeError('Invalid geometry skip association')
                            # No generated score, no cache entry and no temporary
                            # score publication. Original map score stays intact.
                            continue
                        prediction=item.get('image_score')
                        if (plan is None or item.get('image_glm_contract')!=IMAGE_GLM_CONTRACT or
                                not relaxed_finite(prediction) or not -1<=prediction<=1 or evaluated is not True or
                                not relaxed_finite(imagined_score) or abs(imagined_score-prediction)>1e-9 or
                                item.get('anchor_id')!=plan['frontier_anchor']['observation_id']):
                            raise RuntimeError('Image/GLM response has invalid candidate association or score')
                        available_scores[plan_id]=dict(imagined_itm_score=prediction,image_score=prediction,
                            imagination_reliability=float(reliability),imagination_bonus=0.,evidence_gain=0.,
                            verifier_reason=str(item.get('verifier_reason','')),anchor_id=str(item.get('anchor_id','')),
                            cache_hit=False,alignment_source_request=dict(request_id=request_id,plan_id=plan_id,
                                epoch=self.generation_epoch,query_id=query_id,step=step_index))
                        continue
                    relaxed_record=relaxed_imagination() and item.get('score_mode')=='grounded_delta' and item.get('relaxed_imagination') is True and item.get('relaxed_contract')==RELAXED_CONTRACT
                    if relaxed_imagination():
                        try:
                            future=relaxed_q25(item.get('frame_itm_scores',[]))
                            if not relaxed_record or not relaxed_finite(imagined_score) or not relaxed_finite(item.get('future_itm_q25')) or abs(future-item['future_itm_q25'])>1e-9:continue
                        except (ValueError,KeyError,TypeError):continue
                    if (
                        plan is None
                        or not isinstance(imagined_score, (int, float))
                        or not isinstance(reliability, (int, float))
                        or (not relaxed_record and evaluated is False)
                        or (not relaxed_record and float(reliability) <= 0.0 and not bool(evaluated))
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
                    if relaxed_record:bonus=relaxed_blend(real_score,future,self.config.real_score_weight,self.config.imagination_score_weight)-real_score
                    available_scores[plan_id] = {
                        "imagined_itm_score": float(imagined_score),
                        "imagination_reliability": float(reliability),
                        "imagination_bonus": bonus,
                        "evidence_gain": float(item.get("evidence_gain", 0.0)),
                        "verifier_reason": str(item.get("verifier_reason", "")),
                        "anchor_id": str(item.get("anchor_id", "")),
                        "cache_hit": False,
                        **({'relaxed_imagination':True,'future_itm_q25':future} if relaxed_record else {}),
                        **({'aligned_origin':aligned_origin(plan,item,request_id,self.generation_epoch)}
                           if aligned_base_experiment() and bonus > MIN_APPLICABLE_IMAGINATION_BONUS else {}),
                        **({'alignment_source_request':dict(request_id=request_id,plan_id=plan_id,
                            epoch=self.generation_epoch,query_id=query_id,step=step_index)} if alignment_audit.enabled() else {}),
                    }
                    self._store_cached_imagination(plan, available_scores[plan_id])

        candidate_score_records = []
        for plan in eligible_plans:
            score = available_scores.get(str(plan["plan_id"]))
            if score is None:
                continue
            candidate_score_records.append(
                self._temporary_score_record(
                    plan,
                    score,
                    cache_hit=bool(score["cache_hit"]),
                )
            )

        frontier_scores = filter_applicable_frontier_scores(candidate_score_records)
        # Valid positive scores (or opt-in signed fusion) enter Hybrid, including
        # incumbents and tied bonuses. Do not preselect a winner by bonus alone.
        # Context-valid cache reuse replaces the temporary score; it never adds
        # to a previous bonus or writes imagined evidence into the real map.

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
        elif service_called:
            service_decision = "fallback"
        elif cache_hits:
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
            # Record attempted work as cost even when the service times out or
            # fails before producing an applicable score.
            used_world_model=bool(service_called),
            overridden=False,
            frontier_scores=frontier_scores,
            confidence=confidence,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            stage_latencies_ms=(
                dict(response["stage_latencies_ms"])
                if isinstance(response.get("stage_latencies_ms"), dict)
                else {}
            ),
            workload=(
                dict(response["workload"])
                if isinstance(response.get("workload"), dict)
                else {}
            ),
        )
        print("[AVIC] " + json.dumps(decision.__dict__, ensure_ascii=False), flush=True)
        return decision
