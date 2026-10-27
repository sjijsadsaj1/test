"""Independent AVIC service for ApexNav.

Run this module with the ``avic`` Conda Python.  ``mock`` exercises the full
transport/control path without loading a model; ``svc_qwen`` performs real SVC
imagination from a frontier-specific real anchor, samples the imagined
trajectory, and uses Qwen-VL as AVIC's task-aware multi-frame verifier.  The
existing ApexNav BLIP-2 ITM score is retained as a diagnostic, not treated as
proof that a generated object is real.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from . import alignment_audit
from .relaxed_imagination import CONTRACT as RELAXED_CONTRACT,score_prediction as relaxed_score_prediction
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request as URLRequest, urlopen

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

from .protocol import (
    CURRENT_ROUTE_CONTRACT,
    GROUNDED_SCORE_CONTRACT,
    SCHEMA_VERSION,
    make_candidate_plan,
    fallback_response,
    validate_request,
    validate_current_route_bindings,
)


def validate_evidence(evaluation, frames):
    """Return (well_formed, supports_bonus, reason); imagery is never ground truth."""
    try:
        for key in ("reliability", "evidence_gain"):
            value = evaluation[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid " + key)
        for key in ("anchor_cue", "new_cue", "reason", "semantic_relation"):
            if not isinstance(evaluation[key], str):
                raise ValueError("invalid " + key)
        for key in ("supported_by_real_anchor", "adds_new_information", "target_only_in_imagination"):
            if type(evaluation[key]) is not bool:
                raise ValueError("invalid " + key)
        orders = evaluation["support_frame_orders"]
        valid_orders = {frame["order"] for frame in frames}
        if not isinstance(orders, list) or any(type(order) is not int or order not in valid_orders for order in orders):
            raise ValueError("invalid support_frame_orders")
    except (KeyError, TypeError, ValueError) as exc:
        return False, False, "malformed evidence: " + str(exc)
    supported = (
        evaluation["supported_by_real_anchor"] and evaluation["adds_new_information"]
        and not evaluation["target_only_in_imagination"]
        and bool(evaluation["anchor_cue"].strip()) and bool(evaluation["new_cue"].strip())
        and len(set(orders)) >= min(2, len(valid_orders)) and bool(valid_orders)
    )
    # A conservative extra guard, not a replacement for the structured contract.
    explanation = evaluation["reason"] + " " + evaluation["semantic_relation"]
    contradicts = bool(re.search(
        r"no (?:new|additional|meaningful)(?: [\w-]+){0,3} (?:evidence|information|cue)|"
        r"unsupported by (?:the )?real anchor|没有新增|无新增|缺乏真实支持",
        explanation, re.IGNORECASE))
    return True, supported and not contradicts, (
        "contradictory evidence explanation" if contradicts else
        "grounded evidence accepted" if supported else "insufficient grounded new evidence")


# By default the controller exposes at most three frontiers and verification
# sends one real anchor plus at most three sampled SVC frames for each one.
# The shared configurable limit is enforced before SVC inference, again in the
# Qwen client, and by run_apexnav.sh's live preflight.
DEFAULT_MAX_VLM_IMAGES_PER_PROMPT = 12


def validate_view_evidence(evaluation, frames):
    """Validate consistency independently of novelty (strict mode is unchanged)."""
    formed, _, reason = validate_evidence(evaluation, frames)
    if not formed:
        return False, False, reason
    if not isinstance(evaluation.get("consistency_reason"), str) or not evaluation["consistency_reason"].strip():
        return False, False, "malformed view evidence: missing consistency_reason"
    supported = (evaluation["supported_by_real_anchor"] and
                 not evaluation["target_only_in_imagination"] and
                 bool(evaluation["anchor_cue"].strip()) and
                 len(set(evaluation["support_frame_orders"])) >= 2)
    return True, supported, ("grounded view consistency accepted" if supported
                             else "insufficient real context or supporting view frames")


def grounded_view_delta(evaluation, frames, anchor_itm, frame_scores, policy):
    """A bounded forecast-utility bonus, NOT evidence that an imagined object exists."""
    formed, supported, validation = validate_view_evidence(evaluation, frames)
    diagnostic = {"score_mode": "grounded_delta", "anchor_itm_score": anchor_itm,
                  "view_delta": 0.0, "score_reason": validation}
    if not formed:
        return 0.0, diagnostic
    if (not evaluation["supported_by_real_anchor"] or
            evaluation["target_only_in_imagination"] or not evaluation["anchor_cue"].strip()):
        diagnostic["score_reason"] = "no real target-context support or imagined-only target"
        return 0.0, diagnostic
    reliability = evaluation["reliability"]
    if reliability < float(policy.get("min_imagination_reliability", 0.55)):
        diagnostic["score_reason"] = "low generation reliability"
        return 0.0, diagnostic
    if not supported:
        diagnostic["score_reason"] = "requires two consistency-supporting frames"
        return 0.0, diagnostic
    explanation = evaluation["consistency_reason"]
    if re.search(r"different room|distorted|unsupported by (?:the )?real anchor|"
                 r"unrelated (?:room|scene)|场景跳变|缺乏真实支持", explanation, re.I):
        diagnostic["score_reason"] = "contradictory generation consistency"
        return 0.0, diagnostic
    values = [anchor_itm, *frame_scores]
    if (len(frame_scores) != len(frames) or len(frame_scores) < 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) or not -1 <= v <= 1 for v in values)):
        diagnostic["score_reason"] = "requires valid anchor and at least two scored future frames"
        return 0.0, diagnostic
    # A peak in a single generated frame is insufficient. Compare against the
    # SAME real anchor/text/model, not the maximum semantic value on the map.
    future = float(np.quantile(frame_scores, 0.25))
    delta = future - anchor_itm
    diagnostic.update(future_itm_q25=future, view_delta=delta)
    if delta < 0.01 or sum(v - anchor_itm >= 0.01 for v in frame_scores) < 2:
        diagnostic["score_reason"] = "no sustained predicted visibility gain"
        return 0.0, diagnostic
    real_weight = float(policy.get("real_weight", 0.5))
    predicted_weight = float(policy.get("imagination_weight", 0.5))
    cap = float(policy.get("max_imagination_bonus", 0.02))
    if (not all(math.isfinite(v) and v >= 0 for v in (real_weight, predicted_weight, cap))
            or real_weight + predicted_weight <= 0):
        raise ValueError("invalid grounded_delta weights/cap")
    alpha = predicted_weight / (real_weight + predicted_weight)
    bonus = min(cap, alpha * reliability * delta)
    diagnostic.update(effective_imagination_weight=alpha, score_reason="grounded predicted view gain")
    return bonus, diagnostic


def verifier_frame_budget(candidate_count: int, max_images: int) -> int:
    """Return a safe per-candidate frame budget for one joint Qwen prompt.

    Each candidate consumes one real anchor in addition to its imagined
    frames.  Refuse impossible configurations before running expensive SVC
    inference instead of discovering the mismatch at verifier time.
    """

    candidate_count = int(candidate_count)
    max_images = int(max_images)
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    remaining = max_images - candidate_count
    if remaining < candidate_count:
        raise RuntimeError(
            "Qwen image budget %d cannot hold one anchor and one imagined "
            "frame for each of %d candidates" % (max_images, candidate_count)
        )
    return remaining // candidate_count


def align_frontier_scores(
    plans: List[Dict[str, Any]],
    imagined_itm_scores: Dict[str, float],
    imagination_reliability: Dict[str, float],
    *,
    real_weight: float = 0.5,
    imagination_weight: float = 0.5,
) -> tuple[str, List[Dict[str, Any]], float]:
    """Align real and imagined evidence in ApexNav's BLIP-2 score space.

    Imagined frontiers use the configured weighted mean; frontiers that were
    not imagined retain their real score.  The result is decision-only and
    never enters ValueMap.
    """

    real_weight = max(0.0, float(real_weight))
    imagination_weight = max(0.0, float(imagination_weight))
    total_weight = real_weight + imagination_weight
    if total_weight <= 0.0:
        raise ValueError("frontier score weights must have a positive sum")
    normalized_real_weight = real_weight / total_weight
    normalized_imagination_weight = imagination_weight / total_weight

    records: List[Dict[str, Any]] = []
    for plan in plans:
        plan_id = str(plan["plan_id"])
        frontier = plan.get("source_frontier") or {}
        real_score = float(frontier.get("semantic_score", 0.0))
        evidence = max(
            0.0, min(1.0, float(frontier.get("observation_evidence", 0.0)))
        )
        reliability = max(
            0.0, min(1.0, float(imagination_reliability.get(plan_id, 0.0)))
        )
        imagined_score = imagined_itm_scores.get(plan_id)
        if imagined_score is None:
            decision_score = real_score
            imagined_value = 0.0
            imagined_weight = 0.0
        else:
            imagined_value = float(imagined_score)
            decision_score = (
                normalized_real_weight * real_score
                + normalized_imagination_weight * imagined_value
            )
            imagined_weight = normalized_imagination_weight

        records.append(
            {
                "plan_id": plan_id,
                "frontier_rank": frontier.get("rank"),
                "real_semantic_score": real_score,
                "observation_evidence": evidence,
                "evidence_state": str(frontier.get("evidence_state", "unknown")),
                "imagined_itm_score": imagined_value,
                "imagination_reliability": reliability,
                "real_score_weight": (
                    normalized_real_weight if imagined_score is not None else 1.0
                ),
                "imagination_weight": imagined_weight,
                "path_length": max(0.0, float(frontier.get("path_length", 0.0))),
                "decision_semantic_score": float(decision_score),
            }
        )

    records.sort(key=lambda item: (-item["decision_semantic_score"], item["path_length"]))
    selected_plan_id = str(records[0]["plan_id"])
    confidence = 1.0
    if len(records) > 1:
        first = records[0]["decision_semantic_score"]
        second = records[1]["decision_semantic_score"]
        confidence = max(0.0, min(1.0, (first - second) / max(abs(first), 1e-6)))
    return selected_plan_id, records, float(confidence)


class ApexNavITMClient:
    """Minimal client for ApexNav's already-running BLIP-2 ITM service."""

    def __init__(self):
        self.url = os.environ.get(
            "AVIC_APEXNAV_ITM_URL", "http://127.0.0.1:12182/blip2itm"
        )
        self.timeout_s = float(os.environ.get("AVIC_APEXNAV_ITM_TIMEOUT", "30"))

    @staticmethod
    def prompt(category: str, related_room: str) -> str:
        if related_room != "everywhere":
            return "Seems like there is a %s or a %s ahead?" % (
                related_room,
                category,
            )
        return "Seems like there is a %s ahead?" % category

    @staticmethod
    def _apexnav_encoded_image(encoded_jpeg: str) -> str:
        # Match ApexNav's existing client convention: decode to RGB, then pass
        # that array through cv2.imencode before the server's cv2.imdecode.
        image = Image.open(io.BytesIO(base64.b64decode(encoded_jpeg))).convert("RGB")
        rgb = np.asarray(image)
        ok, encoded = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("failed to encode imagined RGB for BLIP-2")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def cosine(self, encoded_jpeg: str, category: str, related_room: str) -> float:
        body = json.dumps(
            {
                "image": self._apexnav_encoded_image(encoded_jpeg),
                "txt": self.prompt(category, related_room),
            }
        ).encode("utf-8")
        request = URLRequest(
            self.url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return float(payload["response"])


def _last_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # Schema-constrained Qwen responses should normally be a complete JSON
    # object.  Parsing that form first is important now that the horizon
    # contract contains nested per-frontier objects.
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value

    # Keep compatibility with responses wrapped in prose or Markdown fences.
    # Select the complete object that ends latest; when an outer and nested
    # object share an end position, prefer the wider (outer) object.  The old
    # reverse-'{' scan selected the innermost prefix record and consequently
    # discarded the top-level ``prefixes`` array.
    decoder = json.JSONDecoder()
    best: Optional[tuple[int, int, Dict[str, Any]]] = None
    for match in re.finditer(r"\{", text):
        try:
            decoded, consumed = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(decoded, dict):
            continue
        absolute_end = match.start() + consumed
        candidate = (absolute_end, consumed, decoded)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return None if best is None else best[2]


class QwenOpenAIClient:
    def __init__(self):
        from openai import OpenAI

        self.base_url = os.environ.get("AVIC_VLM_BASE_URL", "http://127.0.0.1:8001/v1")
        self.api_key = os.environ.get("AVIC_VLM_API_KEY", "EMPTY")
        self.policy_model = os.environ.get(
            "AVIC_POLICY_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        self.max_images_per_prompt = int(
            os.environ.get(
                "AVIC_QWEN_MAX_IMAGES_PER_PROMPT",
                str(DEFAULT_MAX_VLM_IMAGES_PER_PROMPT),
            )
        )
        if self.max_images_per_prompt < 2:
            raise ValueError("AVIC_QWEN_MAX_IMAGES_PER_PROMPT must be at least 2")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @staticmethod
    def _image_block(encoded: str) -> Dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + encoded},
        }

    def _complete(
        self,
        model: str,
        content: List[Dict[str, Any]],
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        image_count = sum(item.get("type") == "image_url" for item in content)
        if image_count > self.max_images_per_prompt:
            raise RuntimeError(
                "Qwen request contains %d images, exceeding configured limit %d"
                % (image_count, self.max_images_per_prompt)
            )
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            response_format=response_format or {"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        if getattr(response.choices[0], "finish_reason", None) == "length":
            raise RuntimeError("Qwen response truncated at output token limit")
        parsed = _last_json_object(text)
        if parsed is None:
            raise RuntimeError("Qwen returned no valid JSON: %s" % text[:200])
        return parsed

    def select_prefix_lengths(
        self, payload: Dict[str, Any], plans: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select one virtual-probe prefix length for every anchored frontier.

        G1 has already made the only call/no-call decision.  Qwen cannot skip,
        select a frontier, or alter any route action; it only chooses how much
        of each fixed, anchor-relative probe should condition SVC.  All candidates are
        handled in one deterministic, schema-constrained model request.
        """

        target = payload["target"]
        action_names = {
            1: "move-forward",
            2: "turn-left",
            3: "turn-right",
        }
        routes = []
        max_prefix = 1
        for plan in plans:
            action_ids = [int(action["id"]) for action in plan["actions"]]
            max_prefix = max(max_prefix, len(action_ids))
            routes.append(
                {
                    "plan_id": str(plan["plan_id"]),
                    "fixed_route": [action_names[action] for action in action_ids],
                    "max_prefix_length": len(action_ids),
                    **({"min_prefix_length": plan["min_prefix_length"]}
                       if "min_prefix_length" in plan else {}),
                }
            )
        plan_ids = [route["plan_id"] for route in routes]
        prompt = (
            "You are the HOW-MUCH horizon selector for SVC visual imagination in "
            "ObjectNav. ApexNav's fixed score-band gate and real-anchor matcher already "
            "selected every virtual probe "
            "listed below for imagination. You MUST return one prefix length for every "
            "plan. Do not skip a plan, select a frontier, change direction, reorder or "
            "invent actions. For each plan choose an integer from 1 through that plan's "
            "max_prefix_length. A longer prefix should only be used when the additional "
            "fixed actions are useful for exposing a future view relevant to the target.\n"
            "Routes: %s\nTarget: %s. Related room: %s. Related objects: %s. "
            "Return only the required JSON object."
            % (
                json.dumps(routes, ensure_ascii=False),
                target.get("category", ""),
                target.get("related_room", ""),
                json.dumps(target.get("related_objects", []), ensure_ascii=False),
            )
        )
        if payload.get("score_policy", {}).get("mode") == "frontier_once":
            prompt = ("You choose ONLY a prefix length for each fixed current-pose A* route. "
                      "This is an ungated diagnostic: do not skip candidates, choose a frontier, "
                      "change direction, reorder or invent actions. Choose from each route's "
                      "min_prefix_length through max_prefix_length. Prefer a length that exposes "
                      "a useful target-related view. Return only the required JSON object. "
                      "Routes: " + json.dumps(routes) + " Target: " + json.dumps(target))
        if payload.get("score_policy", {}).get("mode") == "grounded_delta":
            prompt += (" These are current-pose A* routes, NOT historical-anchor virtual probes. "
                       "Each min_prefix_length is mandatory: it preserves route divergence and "
                       "at least two predicted action views. Never truncate before that bound.")
        schema = {
            "type": "object",
            "properties": {
                "prefixes": {
                    "type": "array",
                    "minItems": len(routes),
                    "maxItems": len(routes),
                    "items": {
                        "type": "object",
                        "properties": {
                            "plan_id": {"type": "string", "enum": plan_ids},
                            "prefix_length": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": max_prefix,
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["plan_id", "prefix_length", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["prefixes"],
            "additionalProperties": False,
        }
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for plan in plans:
            anchor = plan.get("frontier_anchor") or {}
            content.append(
                {
                    "type": "text",
                    "text": "Real directional anchor for %s:" % plan["plan_id"],
                }
            )
            content.append(self._image_block(str(anchor.get("rgb_jpeg_b64", ""))))
        return self._complete(
            self.policy_model,
            content,
            temperature=float(os.environ.get("AVIC_GATE_TEMPERATURE", "0.0")),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "avic_prefix_lengths",
                    "strict": True,
                    "schema": schema,
                },
            },
        )

    def evaluate_frontier_sequences(
        self, payload: Dict[str, Any], candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Jointly verify real anchors and SVC trajectories for ObjectNav.

        The model estimates only evidence added by imagination.  It cannot
        reduce ApexNav's real score, select a low-level action, or write a map.
        """

        target = payload["target"]
        plan_ids = [str(item["plan_id"]) for item in candidates]
        view_mode = payload.get("score_policy", {}).get("mode") == "grounded_delta"
        prompt = (
            "You are AVIC's independent multi-frame verifier for ObjectNav. "
            "The robot seeks target '%s'; its likely room is '%s'; related real-world "
            "objects are %s. For every frontier you will see (1) a REAL directional "
            "anchor and (2) an ordered SVC-IMAGINED continuation. Treat imagined "
            "pixels as hypotheses, never as observations. Judge whether the sequence "
            "preserves the anchor's room/object structure and adds useful evidence for "
            "the target. A related cue can be useful (for example bathtub -> bathroom "
            "when seeking a sink) even if the target is not rendered. Distorted scene "
            "changes, invented objects, redundant frames, or evidence unsupported by "
            "the real anchor must receive low reliability/evidence_gain. Do not choose "
            "an action and do not copy the ApexNav score. Return exactly one evaluation "
            "for every plan_id. reliability is generation consistency in [0,1]; "
            "evidence_gain is NEW target-directed information beyond the real anchor in "
            "[0,1]. Also report anchor_cue (a cue actually visible in the REAL image), "
            "new_cue, supported_by_real_anchor, adds_new_information, "
            "target_only_in_imagination (true if support relies solely on an invented "
            "target), and support_frame_orders (the shown imagined frame orders). "
            "For multi-frame continuations cite at least two supporting frames. "
            "A negative explanation must not accompany a positive evidence_gain.\nTarget metadata: %s"
            % (
                target.get("category", ""),
                target.get("related_room", ""),
                json.dumps(target.get("related_objects", []), ensure_ascii=False),
                json.dumps(target, ensure_ascii=False),
            )
        )
        if view_mode:
            prompt = (
                "You are AVIC's multi-frame predicted-view verifier for ObjectNav. "
                "Each plan starts at the SAME REAL current observation and follows a fixed "
                "current-pose A* route. Imagined pixels are hypotheses, never observations. "
                "Return one evaluation per plan_id. Judge THREE SEPARATE properties:\n"
                "1. reliability: ONLY spatial/object-structure consistency of the generated "
                "continuation, in [0,1]. Explain it in consistency_reason. A view of the same "
                "objects, unchanged content, or no NEW information must NOT lower reliability. "
                "Actual geometric distortion, scene jumps or unsupported invented objects "
                "must lower reliability. Do not conflate 'unhelpful' with 'inconsistent'.\n"
                "2. supported_by_real_anchor: whether a target or target-related context cue "
                "is actually visible in the REAL image, regardless of novelty. Describe it "
                "in anchor_cue. Related room/objects are soft priors, not requirements: a real "
                "target outside its typical room is valid. target_only_in_imagination is true "
                "only when target support relies solely on invention. Cite at least TWO shown "
                "frame orders preserving the real cue/structure in support_frame_orders. "
                "Use [] if there is no such support, not merely because nothing is new.\n"
                "3. evidence_gain, adds_new_information and new_cue describe NOVEL information "
                "only. They may be 0, false and empty while reliability is high. Existing "
                "targets becoming closer/clearer can have navigation value without new objects. "
                "Explain usefulness separately in reason and semantic_relation. Do not invent "
                "a benefit: unchanged or deteriorating visibility need not receive a reward. "
                "The caller independently measures visibility change with BLIP. Never select "
                "an action or copy a navigation score. Target metadata: "
                + json.dumps(target, ensure_ascii=False)
            )
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for item in candidates:
            frontier = item.get("source_frontier") or {}
            anchor = item.get("frontier_anchor") or {}
            frames = item.get("future_frames", [])
            if not frames or len(frames) != len(item.get("frame_itm_scores", [])):
                raise ValueError("verifier frame/score count mismatch for " + str(item["plan_id"]))
            frame_instruction = ("Use [] when no frame supports REAL context/structure." if view_mode
                                 else "Use [] when no frame supports NEW evidence.")
            content.append({"type": "text", "text": (
                "Plan %s allowed support_frame_orders: %s. "
                "These are frame labels, NOT action numbers or zero-based indices. "
                % (item["plan_id"], json.dumps([f["order"] for f in frames]))) + frame_instruction})
            content.append(
                {
                    "type": "text",
                    "text": (("Plan %s. The next image is the REAL current observation. "
                              "No heuristic image-quality score is supplied; judge its visible structure."
                              % item["plan_id"]) if view_mode else (
                        "Plan %s. Coverage=%.3f, path "
                        "cost=%.3f, anchor_quality=%.3f, anchor detected objects=%s. "
                        "The next image is REAL."
                        % (
                            item["plan_id"],
                            float(frontier.get("observation_evidence", 0.0)),
                            float(frontier.get("path_length", 0.0)),
                            float(anchor.get("anchor_quality", 0.0)),
                            json.dumps(anchor.get("detected_objects", []), ensure_ascii=False),
                        )
                    )),
                }
            )
            content.append(self._image_block(str(anchor.get("rgb_jpeg_b64", ""))))
            for frame, itm_score in zip(
                item.get("future_frames", []), item.get("frame_itm_scores", [])
            ):
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "Plan %s imagined step %s:"
                            % (
                                item["plan_id"],
                                frame.get("order", "?"),
                            )
                        ),
                    }
                )
                content.append(self._image_block(str(frame["jpeg_b64"])))

        schema = {
            "type": "object",
            "properties": {
                "evaluations": {
                    "type": "array",
                    "minItems": len(candidates),
                    "maxItems": len(candidates),
                    "items": {
                        "type": "object",
                        "properties": {
                            "plan_id": {"type": "string", "enum": plan_ids},
                            "reliability": {"type": "number", "minimum": 0, "maximum": 1},
                            "evidence_gain": {"type": "number", "minimum": 0, "maximum": 1},
                            "semantic_relation": {"type": "string"},
                            "reason": {"type": "string"},
                            "anchor_cue": {"type": "string"},
                            "new_cue": {"type": "string"},
                            "supported_by_real_anchor": {"type": "boolean"},
                            "adds_new_information": {"type": "boolean"},
                            "target_only_in_imagination": {"type": "boolean"},
                            "support_frame_orders": {"type": "array", "items": {"type": "integer",
                                "enum": sorted({f["order"] for c in candidates for f in c["future_frames"]})}},
                        },
                        "required": [
                            "plan_id",
                            "reliability",
                            "evidence_gain",
                            "semantic_relation",
                            "reason",
                            "anchor_cue", "new_cue", "supported_by_real_anchor",
                            "adds_new_information", "target_only_in_imagination",
                            "support_frame_orders",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["evaluations"],
            "additionalProperties": False,
        }
        if view_mode:
            evaluation_schema = schema["properties"]["evaluations"]["items"]
            evaluation_schema["properties"]["consistency_reason"] = {"type": "string"}
            evaluation_schema["required"].append("consistency_reason")
        return self._complete(
            self.policy_model,
            content,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "avic_frontier_evidence",
                    "strict": True,
                    "schema": schema,
                },
            },
            max_tokens=3072,
        )


class MockBackend:
    name = "mock"

    def health(self) -> Dict[str, Any]:
        return {"ok": True, "backend": self.name, "model_loaded": False}

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        plans = payload["navigation"]["candidate_plans"]
        imagination_plan_ids = set(
            payload["navigation"].get(
                "imagination_plan_ids", [str(plan["plan_id"]) for plan in plans]
            )
        )
        imaginable_plans = [
            plan for plan in plans if str(plan["plan_id"]) in imagination_plan_ids
        ]
        requested_plan_id = os.environ.get("AVIC_MOCK_SELECTED_PLAN", "frontier_0")
        selected_plan = next(
            (plan for plan in imaginable_plans if plan["plan_id"] == requested_plan_id),
            imaginable_plans[0],
        )
        imagined_itm_scores = {
            str(plan["plan_id"]): (
                0.36 if plan["plan_id"] == selected_plan["plan_id"] else 0.12
            )
            for plan in imaginable_plans
        }
        reliability = {str(plan["plan_id"]): 1.0 for plan in imaginable_plans}
        score_policy = payload.get("score_policy", {})
        selected_plan_id, candidate_scores, confidence = align_frontier_scores(
            plans,
            imagined_itm_scores,
            reliability,
            real_weight=float(score_policy.get("real_weight", 0.5)),
            imagination_weight=float(score_policy.get("imagination_weight", 0.5)),
        )
        return {
            "decision": "call_wm",
            # Kept for protocol compatibility only.  ApexNav never executes
            # this field; C++ reruns the original Hybrid policy using only the
            # temporary candidate_scores below.
            "selected_action": selected_plan["actions"][0],
            "selected_plan_id": selected_plan_id,
            "reason": "mock score-band->horizon->SVC->ApexNav-BLIP aligned scores",
            "artifacts": [],
            "candidate_scores": candidate_scores,
            "confidence": confidence,
        }


class SVCQwenBackend:
    name = "svc_qwen"

    def __init__(self, max_images_per_prompt: Optional[int] = None):
        self._svc = None
        self._qwen = None
        self._itm = None
        if max_images_per_prompt is None:
            max_images_per_prompt = int(
                os.environ.get(
                    "AVIC_QWEN_MAX_IMAGES_PER_PROMPT",
                    str(DEFAULT_MAX_VLM_IMAGES_PER_PROMPT),
                )
            )
        self.max_images_per_prompt = int(max_images_per_prompt)
        if self.max_images_per_prompt < 2:
            raise ValueError("AVIC Qwen image budget must be at least 2")

    def _get_qwen(self) -> QwenOpenAIClient:
        if self._qwen is None:
            self._qwen = QwenOpenAIClient()
        return self._qwen

    def _get_svc(self):
        if self._svc is None:
            from .svc_adapter import SVCAdapter

            avic_root = os.environ.get("AVIC_ROOT")
            if not avic_root:
                raise RuntimeError(
                    "AVIC_ROOT is required for the svc_qwen backend; "
                    "copy config/avic.env.example to config/avic.env and configure it"
                )
            project_root = Path(__file__).resolve().parents[1]
            self._svc = SVCAdapter(
                avic_root=avic_root,
                model_dir=os.environ.get(
                    "SVC_MODEL_DIR", avic_root + "/.runtime/models/stable-virtual-camera"
                ),
                vae_dir=os.environ.get(
                    "SVC_VAE_DIR", avic_root + "/.runtime/models/stable-diffusion-2-1-base"
                ),
                hf_home=os.environ.get("HF_HOME", avic_root + "/.runtime/huggingface"),
                output_root=os.environ.get(
                    "AVIC_OUTPUT_ROOT",
                    str(project_root / "experiments" / "runs" / "avic"),
                ),
                device=os.environ.get("AVIC_SVC_DEVICE", "cuda:0"),
            )
        return self._svc

    def _get_itm(self) -> ApexNavITMClient:
        if self._itm is None:
            self._itm = ApexNavITMClient()
        return self._itm

    def health(self) -> Dict[str, Any]:
        svc = self._get_svc()
        return {
            "ok": bool(svc.resource_status()["ready"]),
            "backend": self.name,
            "gate_mode": "apexnav_middle_score_band_then_real_anchor",
            "gate_contract": "frontier_anchor_grounded_verifier_v2",
            "alignment_audit_contract": dict(schema_version=alignment_audit.SCHEMA, enabled=alignment_audit.enabled()),
            "relaxed_imagination_contract": RELAXED_CONTRACT,
            **({'alignment_audit_schema':alignment_audit.SCHEMA} if alignment_audit.enabled() else {}),
            "generation_cache_contract": "svc_input_video_cache_v1",
            "score_modes": ["strict_evidence", "grounded_delta", "frontier_once"],
            "frontier_once_contract": "frontier_once_spatial_q25_half_v1_20260923",
            "frontier_refresh_contract": "frontier_score_delta_refresh_v1_20260925",
            "score_contract": GROUNDED_SCORE_CONTRACT,
            "route_contract": CURRENT_ROUTE_CONTRACT,
            "horizon_requests_per_frontier_decision": 1,
            "frontier_binding": "per_frontier_real_directional_anchor",
            "semantic_scorer": "qwen_multiframe_verifier_with_apexnav_blip_diagnostics",
            "score_fusion": "positive_bounded_imagination_bonus",
            "required_vlm_images_per_prompt": self.max_images_per_prompt,
            "max_vlm_images_per_prompt": self.max_images_per_prompt,
            "model_loaded": svc._model is not None,
            "resources": svc.resource_status(),
            "note": "SVC config.yaml is not consumed by the reproduced loader",
        }

    def _select_prefixes(
        self, payload: Dict[str, Any], plans: List[Dict[str, Any]]
    ) -> tuple[Dict[str, List[int]], str, List[str]]:
        """Make one Qwen request and validate each bound-route prefix."""

        result = self._get_qwen().select_prefix_lengths(payload, plans)
        raw_prefixes = result.get("prefixes", [])
        if not isinstance(raw_prefixes, list):
            raise RuntimeError("horizon selector returned no prefixes array")
        routes = {
            str(plan["plan_id"]): [int(action["id"]) for action in plan["actions"]]
            for plan in plans
        }
        minimums = {str(plan["plan_id"]): int(plan.get("min_prefix_length", 1)) for plan in plans}
        selected: Dict[str, List[int]] = {}
        reasons = []
        errors = []
        for item in raw_prefixes:
            if not isinstance(item, dict):
                errors.append("non-object prefix item")
                continue
            plan_id = str(item.get("plan_id", ""))
            fixed_actions = routes.get(plan_id)
            if fixed_actions is None or plan_id in selected:
                errors.append("unknown or duplicate plan_id=%r" % plan_id)
                continue
            try:
                prefix_length = int(item.get("prefix_length", 0))
            except (TypeError, ValueError):
                prefix_length = 0
            if not 1 <= prefix_length <= len(fixed_actions):
                errors.append(
                    "invalid prefix_length=%r for %s route=%s"
                    % (item.get("prefix_length"), plan_id, fixed_actions)
                )
                continue
            minimum = minimums[plan_id]
            if not 1 <= minimum <= len(fixed_actions):
                errors.append("invalid minimum route prefix for " + plan_id)
                continue
            if prefix_length < minimum:
                reasons.append("%s raised prefix %d->%d to preserve route divergence/two views"
                               % (plan_id, prefix_length, minimum))
                prefix_length = minimum
            selected[plan_id] = fixed_actions[:prefix_length]
            reasons.append(
                "%s prefix=%d/%d: %s"
                % (
                    plan_id,
                    prefix_length,
                    len(fixed_actions),
                    str(item.get("reason", "")),
                )
            )
        for plan_id in routes:
            if plan_id not in selected:
                errors.append("missing valid prefix for %s" % plan_id)
        return selected, " | ".join(reasons), errors

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        signals = payload.get("signals", {})
        score_mode = payload.get("score_policy", {}).get("mode", "strict_evidence")
        if payload.get('score_policy',{}).get('relaxed_imagination') is True and (
                score_mode!='grounded_delta' or payload['score_policy'].get('contract')!=RELAXED_CONTRACT):
            raise ValueError('relaxed imagination requires explicit grounded_delta relaxed_blend_v1 contract')
        if score_mode == "frontier_once":
            from .frontier_once_backend import evaluate_once
            return evaluate_once(self, payload)
        if score_mode not in ("strict_evidence", "grounded_delta"):
            raise ValueError("unknown SVC score mode: " + str(score_mode))
        navigation = payload.get("navigation", {})
        plans = navigation.get("candidate_plans", [])
        if not bool(signals.get("frontier_query_active", False)) or not plans:
            return {
                "decision": "skip",
                "selected_action": navigation["baseline_action"],
                "selected_plan_id": "apexnav",
                "reason": "bridge: no active frontier-scoring context",
                "artifacts": [],
            }

        requested_ids = navigation.get("imagination_plan_ids", [])
        requested_id_set = set(str(plan_id) for plan_id in requested_ids)
        imagination_plans = [
            plan for plan in plans if str(plan["plan_id"]) in requested_id_set
        ]
        if not imagination_plans:
            return {
                "decision": "skip",
                "selected_action": navigation["baseline_action"],
                "selected_plan_id": "apexnav",
                "reason": "G1 selected no valid bound frontier plan",
                "artifacts": [],
            }

        if score_mode == "grounded_delta":
            validate_current_route_bindings(payload, imagination_plans)

        max_frames_per_plan = verifier_frame_budget(
            len(imagination_plans), self.max_images_per_prompt
        )
        if score_mode == "grounded_delta" and min(
                max_frames_per_plan, int(os.environ.get("AVIC_SVC_MAX_RERANK_FRAMES", "6"))) < 2:
            raise ValueError("grounded_delta requires a budget of at least two future frames per plan")

        stage_latencies_ms: Dict[str, float] = {}
        workload = {
            "qwen_horizon_requests": 0,
            "qwen_verifier_requests": 0,
            "svc_videos": 0,
            "svc_reused_videos": 0,
            "svc_candidate_plans": 0,
            "blip_frame_scores": 0,
        }
        imagined = []
        errors = []
        svc = self._get_svc()
        stage_started = time.monotonic()
        selected_prefixes, horizon_reason, horizon_errors = self._select_prefixes(
            payload, imagination_plans
        )
        stage_latencies_ms["qwen_horizon"] = (
            time.monotonic() - stage_started
        ) * 1000.0
        workload["qwen_horizon_requests"] = 1
        errors.extend(horizon_errors)
        episode = payload.get("episode", {})
        generation_scope = ""
        if episode.get("generation_epoch"):
            generation_scope = json.dumps([episode["generation_epoch"],
                episode.get("scene_id"), episode.get("episode_id")])
        stage_started = time.monotonic()
        for frontier_plan in imagination_plans:
            plan_id = str(frontier_plan["plan_id"])
            actions = selected_prefixes.get(plan_id)
            if not actions:
                continue
            # Preserve the frontier plan_id from G1 through SVC and verification.
            # No nearest-direction matching or post-hoc reassignment occurs.
            bound_plan = make_candidate_plan(plan_id, actions)
            bound_plan["source_frontier"] = frontier_plan.get("source_frontier")
            bound_plan["frontier_anchor"] = frontier_plan.get("frontier_anchor")
            bound_plan["max_rerank_frames"] = max_frames_per_plan
            try:
                anchor = bound_plan.get("frontier_anchor") or {}
                anchor_camera = dict(payload["camera"])
                anchor_camera["pitch_rad"] = float(
                    anchor.get("camera_pitch_rad", anchor_camera.get("pitch_rad", 0.0))
                )
                generated = svc.generate_plan(
                        request_id=payload["request_id"],
                        plan=bound_plan,
                        rgb_jpeg_b64=str(anchor["rgb_jpeg_b64"]),
                        camera=anchor_camera,
                        generation_scope=generation_scope,
                    )
                imagined.append(generated)
                workload["svc_candidate_plans"] += 1
                if generated.get("generation_cache_hit", False):
                    workload["svc_reused_videos"] += 1
                else:
                    workload["svc_videos"] += 1
            except Exception as exc:
                errors.append("%s: %s" % (frontier_plan.get("plan_id"), exc))
        stage_latencies_ms["svc_generation"] = (
            time.monotonic() - stage_started
        ) * 1000.0
        if not imagined:
            raise RuntimeError("no valid SVC result: " + "; ".join(errors))

        target = payload["target"]
        imagined_itm_scores: Dict[str, float] = {}
        verified_candidates = []
        stage_started = time.monotonic()
        for item in imagined:
            try:
                frontier_id = str(item["plan_id"])
                if frontier_id not in requested_id_set:
                    raise RuntimeError("SVC changed the bound frontier plan_id")
                frame_scores = []
                for frame in item.get("future_frames", []):
                    frame_scores.append(
                        self._get_itm().cosine(
                            frame["jpeg_b64"],
                            str(target.get("category", "")),
                            str(target.get("related_room", "everywhere")),
                        )
                    )
                if not frame_scores:
                    raise RuntimeError("SVC returned no sampled trajectory frames")
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                       not math.isfinite(v) or not -1 <= v <= 1 for v in frame_scores):
                    raise ValueError("BLIP returned invalid future cosine score")
                workload["blip_frame_scores"] += len(frame_scores)
                imagined_itm_scores[frontier_id] = max(frame_scores)
                item["frame_itm_scores"] = frame_scores
                if score_mode == "grounded_delta":
                    item["anchor_itm_score"] = self._get_itm().cosine(
                        str(item["frontier_anchor"]["rgb_jpeg_b64"]),
                        str(target.get("category", "")), str(target.get("related_room", "everywhere")))
                    anchor_score = item["anchor_itm_score"]
                    if (isinstance(anchor_score, bool) or not isinstance(anchor_score, (int, float)) or
                            not math.isfinite(anchor_score) or not -1 <= anchor_score <= 1):
                        raise ValueError("BLIP returned invalid real-anchor cosine score")
                    workload["blip_anchor_scores"] = workload.get("blip_anchor_scores", 0) + 1
                verified_candidates.append(item)
            except Exception as exc:
                errors.append("%s BLIP-2: %s" % (item.get("plan_id"), exc))
        stage_latencies_ms["blip_scoring"] = (
            time.monotonic() - stage_started
        ) * 1000.0
        if not verified_candidates:
            raise RuntimeError("ApexNav BLIP-2 failed for every imagined view")

        score_policy = payload.get("score_policy", {})
        max_bonus = max(0.0, float(score_policy.get("max_imagination_bonus", 0.02)))
        min_reliability = max(
            0.0, min(1.0, float(score_policy.get("min_imagination_reliability", 0.55)))
        )
        min_evidence_gain = max(
            0.0, min(1.0, float(score_policy.get("min_imagination_evidence_gain", 0.20)))
        )
        imagined_values = list(imagined_itm_scores.values())
        imagined_score_spread = (
            max(imagined_values) - min(imagined_values)
            if len(imagined_values) >= 2
            else None
        )

        # Exact duplicate trajectories cannot distinguish two frontiers.  Mark
        # every member of such a group unreliable instead of accepting an
        # arbitrary BLIP difference or running it again on the next query.
        signature_to_ids: Dict[str, List[str]] = {}
        for item in verified_candidates:
            digest = hashlib.sha256()
            for frame in item.get("future_frames", []):
                digest.update(base64.b64decode(frame["jpeg_b64"]))
            signature_to_ids.setdefault(digest.hexdigest(), []).append(
                str(item["plan_id"])
            )
        duplicate_ids = {
            plan_id
            for ids in signature_to_ids.values()
            if len(ids) > 1
            for plan_id in ids
        }

        verifier_error = ""
        stage_started = time.monotonic()
        try:
            workload["qwen_verifier_requests"] = 1
            verifier_result = self._get_qwen().evaluate_frontier_sequences(
                payload, verified_candidates
            )
            raw_evaluations = verifier_result.get("evaluations", [])
        except Exception as exc:
            verifier_error = "%s: %s" % (type(exc).__name__, exc)
            errors.append("multi-frame verifier: " + verifier_error)
            raw_evaluations = []
        stage_latencies_ms["qwen_verifier"] = (
            time.monotonic() - stage_started
        ) * 1000.0
        evaluation_by_id = {}
        duplicate_evaluation_ids = set()
        for evaluation in raw_evaluations:
            if not isinstance(evaluation, dict):
                continue
            plan_id = str(evaluation.get("plan_id", ""))
            if plan_id not in requested_id_set:
                continue
            if plan_id in evaluation_by_id:
                duplicate_evaluation_ids.add(plan_id)
                continue
            evaluation_by_id[plan_id] = evaluation
        for plan_id in duplicate_evaluation_ids:
            evaluation_by_id.pop(plan_id, None)
            errors.append("duplicate verifier plan_id: " + plan_id)

        candidate_scores = []
        for item in verified_candidates:
            plan_id = str(item["plan_id"])
            frontier = item.get("source_frontier") or {}
            anchor = item.get("frontier_anchor") or {}
            evaluation = evaluation_by_id.get(plan_id, {})
            evidence_validator = validate_view_evidence if score_mode == "grounded_delta" else validate_evidence
            imagination_evaluated, evidence_supported, evidence_validation = evidence_validator(
                evaluation, item.get("future_frames", []))
            reliability = max(
                0.0, min(1.0, float(evaluation["reliability"]))
            ) if imagination_evaluated else 0.0
            evidence_gain = max(
                0.0, min(1.0, float(evaluation["evidence_gain"]))
            ) if imagination_evaluated else 0.0
            verifier_reason = str(
                evaluation.get("reason", verifier_error or "missing verifier result")
            )
            if plan_id in duplicate_ids:
                reliability = 0.0
                evidence_gain = 0.0
                verifier_reason = "exact duplicate imagined trajectory across frontiers"
            accepted = (
                imagination_evaluated and evidence_supported and reliability >= min_reliability
                and evidence_gain >= min_evidence_gain
            )
            bonus = max_bonus * reliability * evidence_gain if accepted else 0.0
            score_diagnostic = {"score_mode": score_mode, "score_reason": evidence_validation}
            if score_mode == "grounded_delta":
                bonus, score_diagnostic = grounded_view_delta(evaluation,
                    item.get("future_frames", []), item.get("anchor_itm_score"),
                    item.get("frame_itm_scores", []), score_policy)
            if plan_id in duplicate_ids:
                bonus = 0.0
                score_diagnostic["score_reason"] = "exact duplicate imagined trajectory across frontiers"
            real_score = float(frontier.get("semantic_score", 0.0))
            if score_policy.get('relaxed_imagination') is True:
                try:
                    score_diagnostic.update(relaxed_score_prediction(score_policy,real_score,
                        item.get('frame_itm_scores',[]),item.get('future_frames',[])))
                    bonus=score_diagnostic['imagination_bonus']
                except (ValueError,TypeError) as exc:
                    errors.append('relaxed prediction rejected: '+str(exc));continue
            candidate_scores.append(
                {
                    "plan_id": plan_id,
                    "frontier_rank": frontier.get("rank"),
                    **score_diagnostic,
                    "real_semantic_score": real_score,
                    "observation_evidence": float(
                        frontier.get("observation_evidence", 0.0)
                    ),
                    "imagined_itm_score": float(imagined_itm_scores[plan_id]),
                    "imagination_reliability": reliability,
                    "evidence_gain": evidence_gain,
                    "imagination_bonus": float(bonus),
                    "decision_semantic_score": float(real_score + bonus),
                    # A valid zero-valued verifier judgement is cacheable.  A
                    # missing judgement (timeout, malformed/partial response)
                    # is transient and must be retried rather than fossilized
                    # as stable negative evidence for the entire episode.
                    "imagination_evaluated": imagination_evaluated,
                    "evidence_validation": evidence_validation,
                    "structured_evidence": evaluation,
                    "semantic_relation": str(
                        evaluation.get("semantic_relation", "")
                    ),
                    "verifier_reason": verifier_reason,
                    "anchor_id": str(anchor.get("observation_id", "")),
                    "anchor_quality": float(anchor.get("anchor_quality", 0.0)),
                    "frame_itm_scores": [
                        float(value) for value in item.get("frame_itm_scores", [])
                    ],
                }
            )
        candidate_scores.sort(
            key=lambda record: (
                -record["decision_semantic_score"],
                float(
                    next(
                        (
                            candidate.get("source_frontier", {}).get("path_length", 0.0)
                            for candidate in verified_candidates
                            if str(candidate["plan_id"]) == record["plan_id"]
                        ),
                        0.0,
                    )
                ),
            )
        )
        selected_plan_id = (
            str(candidate_scores[0]["plan_id"]) if candidate_scores else "apexnav"
        )
        confidence = 0.0
        if candidate_scores:
            confidence = 1.0
            if len(candidate_scores) > 1:
                first = candidate_scores[0]["decision_semantic_score"]
                second = candidate_scores[1]["decision_semantic_score"]
                confidence = max(
                    0.0, min(1.0, (first - second) / max(abs(first), 1e-6))
                )
        accepted_count = sum(
            item["imagination_bonus"] > 0.0 for item in candidate_scores
        )
        print("[AVIC_SCORE_AUDIT] " + json.dumps({
            "request_id": payload["request_id"], "score_policy": score_policy,
            "evaluated": len(candidate_scores), "positive": accepted_count,
            **({'negative': sum(s['imagination_bonus'] < -1e-6 for s in candidate_scores),
                'relaxed_imagination': True} if score_policy.get('relaxed_imagination') is True else {}),
            "candidates": [{key: item.get(key) for key in (
                "plan_id", "anchor_id", "evidence_validation", "score_reason", "anchor_itm_score",
                "future_itm_q25", "view_delta", "imagination_reliability", "imagination_bonus",
                "real_semantic_score", "decision_semantic_score")} for item in candidate_scores]
        }, ensure_ascii=False), flush=True)
        artifacts = []
        for item in imagined:
            artifacts.append(item["artifact"])
            artifacts.extend(
                frame["artifact"] for frame in item.get("future_frames", [])
            )
        # Keep a compact, image-free audit record beside the generated videos.
        # This makes it possible to inspect whether a run really used distinct
        # anchors, which frames were scored, and why a bonus was accepted,
        # without scraping interleaved ROS/service logs.
        try:
            request_dir = Path(str(imagined[0]["artifact"])).parent.parent
            audit_path = request_dir / "verification.json"
            audit_candidates = []
            scores_by_id = {
                str(record["plan_id"]): record for record in candidate_scores
            }
            for item in imagined:
                plan_id = str(item["plan_id"])
                anchor = item.get("frontier_anchor") or {}
                audit_candidates.append(
                    {
                        "plan_id": plan_id,
                        "anchor_id": str(anchor.get("observation_id", "")),
                        "source_frontier": item.get("source_frontier", {}),
                        "route_contract": payload.get("navigation", {}).get("route_contract", "historical_virtual_probe_v1"),
                        "anchor_binding_source": anchor.get("binding_source", "historical_directional"),
                        "anchor_step_index": anchor.get("step_index"),
                        "requested_route_actions": [a["id"] for a in next(
                            p for p in imagination_plans if p["plan_id"] == plan_id)["actions"]],
                        "min_prefix_length": next(p for p in imagination_plans if p["plan_id"] == plan_id).get("min_prefix_length", 1),
                        "anchor_rgb_sha256": hashlib.sha256(base64.b64decode(anchor.get("rgb_jpeg_b64", ""))).hexdigest(),
                        "anchor_pitch_rad": anchor.get("camera_pitch_rad"),
                        "anchor_quality": float(anchor.get("anchor_quality", 0.0)),
                        "semantic_support": float(anchor.get("semantic_support", 0.0)),
                        "semantic_support_source": str(
                            anchor.get("semantic_support_source", "")
                        ),
                        "actions": [
                            int(action["id"]) for action in item.get("actions", [])
                        ],
                        "source_video": str(item.get("source_video", "")),
                        "generation_key": item.get("generation_key", ""),
                        "generation_cache_hit": item.get("generation_cache_hit", False),
                        "generation_origin_request": item.get("generation_origin_request", payload["request_id"]),
                        "generation_origin_plan": item.get("generation_origin_plan", plan_id),
                        "frames": [
                            {
                                "order": int(frame.get("order", 0)),
                                "video_frame_index": int(
                                    frame.get("video_frame_index", -1)
                                ),
                                "artifact": str(frame.get("artifact", "")),
                            }
                            for frame in item.get("future_frames", [])
                        ],
                        "score": scores_by_id.get(plan_id, {}),
                    }
                )
                if alignment_audit.enabled():
                    audit_candidates[-1]['alignment_audit'] = item.get('alignment_audit')
                    audit_candidates[-1]['anchor_pose'] = anchor.get('alignment_pose')
            audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "request_id": str(payload["request_id"]),
                        "score_policy": score_policy,
                        "target": payload.get("target", {}),
                        "horizon_reason": horizon_reason,
                        "duplicate_plan_ids": sorted(duplicate_ids),
                        "errors": errors,
                        "stage_latencies_ms": stage_latencies_ms,
                        "workload": workload,
                        "candidates": audit_candidates,
                        **({'alignment_audit':dict(payload.get('alignment_audit') or {},
                            schema_version=alignment_audit.SCHEMA,episode=payload.get('episode'),
                            camera=payload.get('camera'),backend=self.name)} if alignment_audit.enabled() else {}),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            artifacts.append(str(audit_path))
        except Exception as exc:
            errors.append("write verification audit: %s" % exc)
        return {
            "decision": "call_wm",
            "selected_action": payload["navigation"]["baseline_action"],
            "selected_plan_id": selected_plan_id,
            "reason": (('%s; relaxed_blend_v1: %d valid signed scores; Qwen diagnostic only'
                % (horizon_reason, len(candidate_scores)))
                if score_policy.get('relaxed_imagination') is True else (
                "%s; AVIC real-anchor multi-frame verifier accepted=%d/%d; "
                "positive bonus only"
                % (horizon_reason, accepted_count, len(candidate_scores))
            )),
            "artifacts": artifacts,
            "candidate_errors": errors,
            "candidate_scores": candidate_scores,
            "confidence": confidence,
            "imagination_score_spread": imagined_score_spread,
            "stage_latencies_ms": stage_latencies_ms,
            "workload": workload,
        }


def _make_backend():
    name = os.environ.get("AVIC_BACKEND", "mock")
    if name == "mock":
        return MockBackend()
    if name == "svc_qwen":
        return SVCQwenBackend()
    if name == "qwen_image_glm":
        from .image_glm_backend import QwenImageGLMBackend
        return QwenImageGLMBackend()
    raise ValueError("AVIC_BACKEND must be mock, svc_qwen or qwen_image_glm")


BACKEND = _make_backend()
app = FastAPI(title="ApexNav AVIC bridge", version=SCHEMA_VERSION)


@app.get("/health")
def health():
    return {"schema_version": SCHEMA_VERSION, **BACKEND.health()}


@app.post("/v1/decide")
async def decide(request: Request):
    started = time.monotonic()
    request_id = "unknown"
    try:
        payload = validate_request(await request.json())
        request_id = payload["request_id"]
        result = BACKEND.decide(payload)
        response = {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "backend": BACKEND.name,
            **result,
        }
    except Exception as exc:
        response = fallback_response(request_id, "%s: %s" % (type(exc).__name__, exc), BACKEND.name)
    response["latency_ms"] = (time.monotonic() - started) * 1000.0
    return JSONResponse(response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("AVIC_SERVICE_HOST", "127.0.0.1"),
        port=int(os.environ.get("AVIC_SERVICE_PORT", "8091")),
        workers=1,
    )
