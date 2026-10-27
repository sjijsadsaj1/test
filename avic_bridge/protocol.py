"""Versioned JSON protocol shared by the ApexNav and AVIC processes."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List


SCHEMA_VERSION = "1.4"
CURRENT_ROUTE_CONTRACT = "current_pose_astar_prefix_v1_20260915"
GROUNDED_SCORE_CONTRACT = "grounded_view_delta_v2_20260915"
IMAGE_GLM_CONTRACT = "image_glm_single_view_v1"

ACTION_ID_TO_NAME = {
    0: "stop",
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
    4: "look_down",
    5: "look_up",
}
ACTION_NAME_TO_ID = {name: action_id for action_id, name in ACTION_ID_TO_NAME.items()}
IMAGINATION_ACTION_IDS = frozenset((1, 2, 3))


class ProtocolError(ValueError):
    """Raised when one side of the bridge violates the JSON contract."""


def action_dict(action_id: int) -> Dict[str, Any]:
    if action_id not in ACTION_ID_TO_NAME:
        raise ProtocolError("unknown action id: %r" % (action_id,))
    return {"id": int(action_id), "name": ACTION_ID_TO_NAME[action_id]}


def make_candidate_plan(plan_id: str, action_ids: Iterable[int]) -> Dict[str, Any]:
    actions = list(action_ids)
    if not actions or len(actions) > 6:
        raise ProtocolError("a candidate plan must contain 1-6 actions")
    if any(action_id not in IMAGINATION_ACTION_IDS for action_id in actions):
        raise ProtocolError("candidate plans may only contain forward/left/right")
    return {
        "plan_id": str(plan_id),
        "actions": [action_dict(action_id) for action_id in actions],
    }


def validate_current_route_bindings(payload, plans):
    """Reject mixed observation/route epochs before spending GPU work."""
    if payload.get("navigation", {}).get("route_contract") != CURRENT_ROUTE_CONTRACT:
        raise ProtocolError("grounded_delta requires the current-pose route contract")
    step = payload.get("episode", {}).get("step_index")
    image = payload.get("observation", {}).get("rgb_jpeg_b64")
    pitch = payload.get("camera", {}).get("pitch_rad")
    if type(step) is not int or not image or not isinstance(pitch, (int, float)) or not math.isfinite(pitch):
        raise ProtocolError("invalid current observation epoch/image/pitch")
    for plan in plans:
        anchor = plan.get("frontier_anchor", {})
        actions = [a["id"] for a in plan["actions"]]
        minimum = plan.get("min_prefix_length")
        if (plan.get("route_contract") != CURRENT_ROUTE_CONTRACT or
                anchor.get("binding_source") != "current_observation" or
                anchor.get("step_index") != step or anchor.get("rgb_jpeg_b64") != image or
                anchor.get("camera_pitch_rad") != pitch or
                actions != plan.get("source_frontier", {}).get("path_action_prefix") or
                type(minimum) is not int or not (1 if payload.get('score_policy',{}).get('mode')=='image_glm' else 2) <= minimum <= len(actions)):
            raise ProtocolError("mismatched current observation / A* route: " + str(plan.get("plan_id")))
        for key in ("robot_x", "robot_y", "robot_yaw_rad", "camera_pitch_rad"):
            value = anchor.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ProtocolError("invalid current route pose: " + key)


def validate_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProtocolError("request must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("unsupported schema_version")
    if not isinstance(payload.get("request_id"), str) or not payload["request_id"]:
        raise ProtocolError("request_id must be a non-empty string")

    observation = payload.get("observation")
    if not isinstance(observation, dict):
        raise ProtocolError("observation must be an object")
    encoded_rgb = observation.get("rgb_jpeg_b64")
    if not isinstance(encoded_rgb, str) or not encoded_rgb:
        raise ProtocolError("observation.rgb_jpeg_b64 is required")

    navigation = payload.get("navigation")
    if not isinstance(navigation, dict):
        raise ProtocolError("navigation must be an object")
    baseline = navigation.get("baseline_action")
    if not isinstance(baseline, dict) or baseline.get("id") not in ACTION_ID_TO_NAME:
        raise ProtocolError("navigation.baseline_action is invalid")

    plans = navigation.get("candidate_plans")
    if not isinstance(plans, list) or not plans or len(plans) > 8:
        raise ProtocolError("navigation.candidate_plans must contain 1-8 plans")
    seen = set()
    plans_by_id = {}
    for plan in plans:
        if not isinstance(plan, dict):
            raise ProtocolError("candidate plan must be an object")
        plan_id = plan.get("plan_id")
        if not isinstance(plan_id, str) or not plan_id or plan_id in seen:
            raise ProtocolError("candidate plan_id must be unique and non-empty")
        seen.add(plan_id)
        plans_by_id[plan_id] = plan
        actions = plan.get("actions")
        if not isinstance(actions, list) or not actions or len(actions) > 6:
            raise ProtocolError("candidate plan actions must contain 1-6 actions")
        for action in actions:
            if not isinstance(action, dict) or action.get("id") not in IMAGINATION_ACTION_IDS:
                raise ProtocolError("candidate plan contains an unsupported action")

    imagination_plan_ids = navigation.get("imagination_plan_ids")
    if imagination_plan_ids is not None:
        if (
            not isinstance(imagination_plan_ids, list)
            or not imagination_plan_ids
            or any(not isinstance(plan_id, str) for plan_id in imagination_plan_ids)
            or len(set(imagination_plan_ids)) != len(imagination_plan_ids)
            or any(plan_id not in seen for plan_id in imagination_plan_ids)
        ):
            raise ProtocolError(
                "navigation.imagination_plan_ids must be a non-empty unique subset "
                "of candidate plan IDs"
            )
        for plan_id in imagination_plan_ids:
            anchor = plans_by_id[plan_id].get("frontier_anchor")
            if not isinstance(anchor, dict):
                raise ProtocolError(
                    "every imagination plan requires a frontier_anchor"
                )
            if not isinstance(anchor.get("observation_id"), str) or not anchor[
                "observation_id"
            ]:
                raise ProtocolError("frontier_anchor.observation_id is required")
            if not isinstance(anchor.get("rgb_jpeg_b64"), str) or not anchor[
                "rgb_jpeg_b64"
            ]:
                raise ProtocolError("frontier_anchor.rgb_jpeg_b64 is required")
            for key in (
                "robot_x",
                "robot_y",
                "robot_yaw_rad",
                "camera_pitch_rad",
                "frontier_distance_m",
                "relative_yaw_to_frontier_rad",
                "anchor_quality",
            ):
                if not isinstance(anchor.get(key), (int, float)):
                    raise ProtocolError("frontier_anchor.%s must be numeric" % key)
    if payload.get("score_policy", {}).get("mode") in ("grounded_delta", "image_glm"):
        validate_current_route_bindings(payload, plans)
    return payload


def validate_response(payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProtocolError("response must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("response schema_version mismatch")
    if payload.get("request_id") != request_id:
        raise ProtocolError("response request_id mismatch")
    if payload.get("decision") not in ("skip", "call_wm", "fallback"):
        raise ProtocolError("response decision is invalid")
    selected = payload.get("selected_action")
    if selected is not None:
        if not isinstance(selected, dict) or selected.get("id") not in ACTION_ID_TO_NAME:
            raise ProtocolError("response selected_action is invalid")
    selected_plan_id = payload.get("selected_plan_id")
    if selected_plan_id is not None and not isinstance(selected_plan_id, str):
        raise ProtocolError("response selected_plan_id is invalid")
    candidate_scores = payload.get("candidate_scores", [])
    if not isinstance(candidate_scores, list):
        raise ProtocolError("response candidate_scores must be a list")
    for item in candidate_scores:
        if not isinstance(item, dict) or not isinstance(item.get("plan_id"), str):
            raise ProtocolError("response candidate score is invalid")
        for key in (
            "real_semantic_score",
            "observation_evidence",
            "imagined_itm_score",
            "imagination_reliability",
            "imagination_bonus",
            "evidence_gain",
            "decision_semantic_score",
        ):
            if key in item and not isinstance(item[key], (int, float)):
                raise ProtocolError("response candidate score %s is invalid" % key)
    return payload


def fallback_response(request_id: str, reason: str, backend: str) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "decision": "fallback",
        "selected_action": None,
        "selected_plan_id": None,
        "reason": str(reason),
        "backend": backend,
        "latency_ms": 0.0,
        "artifacts": [],
        "candidate_scores": [],
    }
