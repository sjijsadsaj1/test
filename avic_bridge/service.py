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
    SCHEMA_VERSION,
    make_candidate_plan,
    fallback_response,
    validate_request,
)


# The controller exposes at most three frontiers.  Verification sends one real
# anchor and three sampled SVC frames for each one, so the Qwen deployment must
# accept twelve images in a single request.  run_apexnav.sh enforces this with
# a live preflight instead of treating a successful /v1/models call as enough.
REQUIRED_VLM_IMAGES_PER_PROMPT = 12


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
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            response_format=response_format or {"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
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
            "[0,1].\nTarget metadata: %s"
            % (
                target.get("category", ""),
                target.get("related_room", ""),
                json.dumps(target.get("related_objects", []), ensure_ascii=False),
                json.dumps(target, ensure_ascii=False),
            )
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for item in candidates:
            frontier = item.get("source_frontier") or {}
            anchor = item.get("frontier_anchor") or {}
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Plan %s. ApexNav real score=%.6f, coverage=%.3f, path "
                        "cost=%.3f, anchor_quality=%.3f, anchor detected objects=%s. "
                        "The next image is REAL."
                        % (
                            item["plan_id"],
                            float(frontier.get("semantic_score", 0.0)),
                            float(frontier.get("observation_evidence", 0.0)),
                            float(frontier.get("path_length", 0.0)),
                            float(anchor.get("anchor_quality", 0.0)),
                            json.dumps(anchor.get("detected_objects", []), ensure_ascii=False),
                        )
                    ),
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
                            "Plan %s imagined step %s (BLIP diagnostic %.6f):"
                            % (
                                item["plan_id"],
                                frame.get("order", "?"),
                                float(itm_score),
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
                        },
                        "required": [
                            "plan_id",
                            "reliability",
                            "evidence_gain",
                            "semantic_relation",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["evaluations"],
            "additionalProperties": False,
        }
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
            max_tokens=768,
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

    def __init__(self):
        self._svc = None
        self._qwen = None
        self._itm = None

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
            "gate_contract": "frontier_anchor_multiframe_verifier_v1",
            "horizon_requests_per_frontier_decision": 1,
            "frontier_binding": "per_frontier_real_directional_anchor",
            "semantic_scorer": "qwen_multiframe_verifier_with_apexnav_blip_diagnostics",
            "score_fusion": "positive_bounded_imagination_bonus",
            "required_vlm_images_per_prompt": REQUIRED_VLM_IMAGES_PER_PROMPT,
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

        imagined = []
        errors = []
        svc = self._get_svc()
        selected_prefixes, horizon_reason, horizon_errors = self._select_prefixes(
            payload, imagination_plans
        )
        errors.extend(horizon_errors)
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
            try:
                anchor = bound_plan.get("frontier_anchor") or {}
                anchor_camera = dict(payload["camera"])
                anchor_camera["pitch_rad"] = float(
                    anchor.get("camera_pitch_rad", anchor_camera.get("pitch_rad", 0.0))
                )
                imagined.append(
                    svc.generate_plan(
                        request_id=payload["request_id"],
                        plan=bound_plan,
                        rgb_jpeg_b64=str(anchor["rgb_jpeg_b64"]),
                        camera=anchor_camera,
                    )
                )
            except Exception as exc:
                errors.append("%s: %s" % (frontier_plan.get("plan_id"), exc))
        if not imagined:
            raise RuntimeError("no valid SVC result: " + "; ".join(errors))

        target = payload["target"]
        imagined_itm_scores: Dict[str, float] = {}
        verified_candidates = []
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
                imagined_itm_scores[frontier_id] = max(frame_scores)
                item["frame_itm_scores"] = frame_scores
                verified_candidates.append(item)
            except Exception as exc:
                errors.append("%s BLIP-2: %s" % (item.get("plan_id"), exc))
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
        try:
            verifier_result = self._get_qwen().evaluate_frontier_sequences(
                payload, verified_candidates
            )
            raw_evaluations = verifier_result.get("evaluations", [])
        except Exception as exc:
            verifier_error = "%s: %s" % (type(exc).__name__, exc)
            errors.append("multi-frame verifier: " + verifier_error)
            raw_evaluations = []
        evaluation_by_id = {}
        for evaluation in raw_evaluations:
            if not isinstance(evaluation, dict):
                continue
            plan_id = str(evaluation.get("plan_id", ""))
            if plan_id not in requested_id_set or plan_id in evaluation_by_id:
                continue
            evaluation_by_id[plan_id] = evaluation

        candidate_scores = []
        for item in verified_candidates:
            plan_id = str(item["plan_id"])
            frontier = item.get("source_frontier") or {}
            anchor = item.get("frontier_anchor") or {}
            evaluation = evaluation_by_id.get(plan_id, {})
            reliability = max(
                0.0, min(1.0, float(evaluation.get("reliability", 0.0)))
            )
            evidence_gain = max(
                0.0, min(1.0, float(evaluation.get("evidence_gain", 0.0)))
            )
            verifier_reason = str(
                evaluation.get("reason", verifier_error or "missing verifier result")
            )
            if plan_id in duplicate_ids:
                reliability = 0.0
                evidence_gain = 0.0
                verifier_reason = "exact duplicate imagined trajectory across frontiers"
            accepted = (
                reliability >= min_reliability
                and evidence_gain >= min_evidence_gain
            )
            bonus = max_bonus * reliability * evidence_gain if accepted else 0.0
            real_score = float(frontier.get("semantic_score", 0.0))
            candidate_scores.append(
                {
                    "plan_id": plan_id,
                    "frontier_rank": frontier.get("rank"),
                    "real_semantic_score": real_score,
                    "observation_evidence": float(
                        frontier.get("observation_evidence", 0.0)
                    ),
                    "imagined_itm_score": float(imagined_itm_scores[plan_id]),
                    "imagination_reliability": reliability,
                    "evidence_gain": evidence_gain,
                    "imagination_bonus": float(bonus),
                    "decision_semantic_score": float(real_score + bonus),
                    "imagination_evaluated": True,
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
                        "anchor_quality": float(anchor.get("anchor_quality", 0.0)),
                        "semantic_support": float(anchor.get("semantic_support", 0.0)),
                        "semantic_support_source": str(
                            anchor.get("semantic_support_source", "")
                        ),
                        "actions": [
                            int(action["id"]) for action in item.get("actions", [])
                        ],
                        "source_video": str(item.get("source_video", "")),
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
            audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "request_id": str(payload["request_id"]),
                        "target": payload.get("target", {}),
                        "horizon_reason": horizon_reason,
                        "duplicate_plan_ids": sorted(duplicate_ids),
                        "errors": errors,
                        "candidates": audit_candidates,
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
            "reason": (
                "%s; AVIC real-anchor multi-frame verifier accepted=%d/%d; "
                "positive bonus only"
                % (horizon_reason, accepted_count, len(candidate_scores))
            ),
            "artifacts": artifacts,
            "candidate_errors": errors,
            "candidate_scores": candidate_scores,
            "confidence": confidence,
            "imagination_score_spread": imagined_score_spread,
        }


def _make_backend():
    name = os.environ.get("AVIC_BACKEND", "mock")
    if name == "mock":
        return MockBackend()
    if name == "svc_qwen":
        return SVCQwenBackend()
    raise ValueError("AVIC_BACKEND must be mock or svc_qwen")


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
