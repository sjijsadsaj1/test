"""Adapter around the reproduced AVIC Stable Virtual Camera implementation.

This module is imported only by the AVIC service when ``AVIC_BACKEND=svc_qwen``.
It deliberately avoids modifying the AVIC checkout: the local SVC checkpoint
and compatible VAE are injected at runtime, replacing two upstream hard-coded
download paths.
"""

from __future__ import annotations

import base64
import io
import hashlib
import json
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np
from PIL import Image
from . import alignment_audit


class SVCAdapterError(RuntimeError):
    pass


def _rotation_matrix_xyz(pitch: float, yaw: float, roll: float = 0.0) -> np.ndarray:
    """Match scipy ``Rotation.from_euler('xyz', ...)`` without importing scipy."""
    sx, cx = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    sz, cz = np.sin(roll), np.cos(roll)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def build_relative_camera_path(
    actions: Iterable[int],
    *,
    forward_step_m: float,
    turn_angle_deg: float,
    pitch_rad: float,
) -> np.ndarray:
    """Convert ApexNav primitive actions to the c2w sequence expected by SVC."""
    position = np.zeros(3, dtype=np.float64)
    yaw = 0.0
    poses = []

    def append_pose():
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = _rotation_matrix_xyz(pitch_rad, yaw)
        c2w[:3, 3] = position
        poses.append(np.round(c2w, 6))

    append_pose()
    for action_id in actions:
        if action_id == 2:
            yaw -= np.deg2rad(turn_angle_deg)
        elif action_id == 3:
            yaw += np.deg2rad(turn_angle_deg)
        elif action_id == 1:
            position += np.array(
                [forward_step_m * np.sin(yaw), 0.0, forward_step_m * np.cos(yaw)]
            )
        else:
            raise SVCAdapterError("SVC received unsupported action %r" % (action_id,))
        append_pose()
    return np.stack(poses, axis=0)


class SVCAdapter:
    """Lazy, serialized SVC inference using the local AVIC reproduction."""

    def __init__(
        self,
        *,
        avic_root: str,
        model_dir: str,
        vae_dir: str,
        hf_home: str,
        output_root: str,
        device: str = "cuda:0",
        generation_cache_size: int = 128,
    ):
        self.avic_root = Path(avic_root)
        self.model_dir = Path(model_dir)
        self.vae_dir = Path(vae_dir)
        self.hf_home = Path(hf_home)
        self.output_root = Path(output_root)
        self.device = device
        self._model = None
        self._demo = None
        self._lock = threading.Lock()
        if generation_cache_size < 1:
            raise ValueError("generation_cache_size must be positive")
        self.generation_cache_size = generation_cache_size
        # Only raw videos are reused, never target-dependent verifier scores.
        # Entries are paths, not GPU tensors or full videos held in RAM.
        self._generation_cache = OrderedDict()

    @staticmethod
    def generation_fingerprint(input_jpeg: bytes, c2ws: np.ndarray, cfg: float,
                               num_targets: int) -> str:
        """Identity of the actual, fixed-recipe SVC input (not frontier IDs)."""
        digest = hashlib.sha256(b"svc-video-input-v1\0")
        digest.update(input_jpeg)
        digest.update(np.asarray(c2ws, dtype="<f8").tobytes())
        digest.update(json.dumps({"cfg": cfg, "num_targets": num_targets},
                                 sort_keys=True, allow_nan=False).encode())
        return digest.hexdigest()

    def _clip_weight(self) -> Path:
        explicit = os.environ.get("SVC_CLIP_WEIGHT")
        if explicit:
            return Path(explicit)
        matches = list(
            self.hf_home.glob(
                "hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/"
                "snapshots/*/open_clip_model.safetensors"
            )
        )
        return matches[0] if matches else Path("/missing/open_clip_model.safetensors")

    def resource_status(self) -> Dict[str, Any]:
        weight = self.model_dir / "model.safetensors"
        vae_weight = self.vae_dir / "vae" / "diffusion_pytorch_model.safetensors"
        if not vae_weight.exists():
            vae_weight = self.vae_dir / "diffusion_pytorch_model.safetensors"
        config = self.model_dir / "config.yaml"
        clip_weight = self._clip_weight()
        output_root_writable = self.output_root.is_dir() and os.access(
            self.output_root, os.W_OK
        )
        return {
            "avic_root": str(self.avic_root),
            "output_root": str(self.output_root),
            "output_root_writable": output_root_writable,
            "svc_weight": str(weight),
            "svc_weight_bytes": weight.stat().st_size if weight.exists() else 0,
            "svc_config_bytes": config.stat().st_size if config.exists() else 0,
            "vae_weight": str(vae_weight),
            "vae_weight_bytes": vae_weight.stat().st_size if vae_weight.exists() else 0,
            "clip_weight": str(clip_weight),
            "clip_weight_bytes": clip_weight.stat().st_size if clip_weight.exists() else 0,
            "ready": (
                weight.exists()
                and weight.stat().st_size > 1_000_000_000
                and vae_weight.exists()
                and vae_weight.stat().st_size > 100_000_000
                and clip_weight.exists()
                and clip_weight.stat().st_size > 1_000_000_000
                and output_root_writable
            ),
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        status = self.resource_status()
        if not status["ready"]:
            raise SVCAdapterError("local SVC checkpoint is missing or incomplete")

        navigation_root = self.avic_root / "navigation"
        if not navigation_root.is_dir():
            raise SVCAdapterError("AVIC navigation package not found: %s" % navigation_root)
        if str(navigation_root) not in sys.path:
            sys.path.insert(0, str(navigation_root))

        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("PYTORCH_SDP_FORCE_FALLBACK", "1")

        import kornia
        import open_clip
        import torch
        from diffusers.models import AutoencoderKL
        from torch import nn

        from stable_virtual_camera import demo
        from stable_virtual_camera.seva import utils as seva_utils

        vae_dir = self.vae_dir
        if (vae_dir / "vae").is_dir():
            vae_parent = str(vae_dir)
            vae_subfolder = "vae"
        else:
            vae_parent = str(vae_dir)
            vae_subfolder = None

        class LocalAutoEncoder(nn.Module):
            scale_factor = 0.18215
            downsample = 8

            def __init__(self, chunk_size=None):
                super().__init__()
                kwargs = {
                    "force_download": False,
                    "local_files_only": True,
                    "low_cpu_mem_usage": False,
                }
                if vae_subfolder is not None:
                    kwargs["subfolder"] = vae_subfolder
                self.module = AutoencoderKL.from_pretrained(vae_parent, **kwargs)
                self.module.eval().requires_grad_(False)
                self.chunk_size = chunk_size

            def _encode(self, value):
                return self.module.encode(value).latent_dist.mean * self.scale_factor

            def encode(self, value, chunk_size=None):
                size = chunk_size or self.chunk_size
                if size is None:
                    return self._encode(value)
                return torch.cat([self._encode(chunk) for chunk in value.split(size)], dim=0)

            def _decode(self, value):
                return self.module.decode(value / self.scale_factor).sample

            def decode(self, value, chunk_size=None):
                size = chunk_size or self.chunk_size
                if size is None:
                    return self._decode(value)
                return torch.cat([self._decode(chunk) for chunk in value.split(size)], dim=0)

            def forward(self, value):
                return self.decode(self.encode(value))

        clip_weight = str(self._clip_weight())

        class LocalCLIPConditioner(nn.Module):
            def __init__(self):
                super().__init__()
                self.module = open_clip.create_model_and_transforms(
                    "ViT-H-14", pretrained=clip_weight
                )[0]
                self.module.eval().requires_grad_(False)
                self.register_buffer(
                    "mean",
                    torch.tensor([0.48145466, 0.4578275, 0.40821073]),
                    persistent=False,
                )
                self.register_buffer(
                    "std",
                    torch.tensor([0.26862954, 0.26130258, 0.27577711]),
                    persistent=False,
                )

            def preprocess(self, value):
                value = kornia.geometry.resize(
                    value,
                    (224, 224),
                    interpolation="bicubic",
                    align_corners=True,
                    antialias=True,
                )
                value = (value + 1.0) / 2.0
                return kornia.enhance.normalize(value, self.mean, self.std)

            def forward(self, value):
                return self.module.encode_image(self.preprocess(value))

        original_load_model = seva_utils.load_model
        local_model_dir = str(self.model_dir)

        def load_local_model(device="cpu", verbose=True, **_kwargs):
            return original_load_model(
                pretrained_model_name_or_path=local_model_dir,
                device=device,
                verbose=verbose,
            )

        # Model resolves these names from demo.py at construction time.
        demo.AutoEncoder = LocalAutoEncoder
        demo.CLIPConditioner = LocalCLIPConditioner
        demo.load_model = load_local_model
        demo.device = self.device

        self._demo = demo
        self._model = demo.Model()
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _decode_input(encoded: str) -> Image.Image:
        try:
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        except Exception as exc:
            raise SVCAdapterError("invalid input JPEG") from exc

    @staticmethod
    def _last_video_frame(video_path: Path) -> Image.Image:
        """Read the terminal frame from SVC's generated ``pred.mp4``."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise SVCAdapterError("SVC did not produce a readable pred.mp4")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise SVCAdapterError("could not read the final SVC-generated frame")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _sample_video_frames(
        video_path: Path, sample_count: int, output_dir: Path
    ) -> List[Dict[str, Any]]:
        """Sample an ordered SVC trajectory, approximately one view per action."""

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise SVCAdapterError("SVC did not produce a readable pred.mp4")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            capture.release()
            raise SVCAdapterError("SVC produced an empty pred.mp4")
        count = max(1, min(int(sample_count), max(1, frame_count - 1)))
        # ``np.linspace(start, stop, num=1)`` returns ``start``.  That made a
        # one-action horizon sample frame 1 instead of the terminal frame and
        # also made ``svc_final.jpg`` misleading.  A one-frame sample is the
        # terminal prediction by definition; multi-frame samples remain
        # ordered and always include that terminal frame.
        if count == 1:
            indices = [frame_count - 1]
        else:
            # Video frame 0 is the anchor. Distribute samples across future
            # action intervals (e.g. middle/end for two actions), rather than
            # using frame 1, which is almost an anchor copy in interpolated SVC.
            indices = np.linspace(0, frame_count - 1, num=count + 1, dtype=int)[1:].tolist()
        indices = list(dict.fromkeys(int(index) for index in indices))
        frames: List[Dict[str, Any]] = []
        audit_fps = float(capture.get(cv2.CAP_PROP_FPS)) if alignment_audit.enabled() else None
        for order, frame_index in enumerate(indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            artifact = output_dir / ("svc_step_%02d.jpg" % order)
            image.save(artifact, quality=92)
            frames.append(
                {
                    "order": order,
                    "video_frame_index": frame_index,
                    "jpeg_b64": SVCAdapter._jpeg_b64(image),
                    "artifact": str(artifact),
                }
            )
            if alignment_audit.enabled():
                frames[-1]['alignment_sample'] = dict(actual_media_index=frame_index,
                    actual_media_total=frame_count,actual_media_fps=audit_fps,
                    target_pose_index=None,missing_reason='interp media index to conditioning target pose mapping not archived/proven')
        capture.release()
        if not frames:
            raise SVCAdapterError("could not sample any SVC-generated frame")
        return frames

    @staticmethod
    def _jpeg_b64(image: Image.Image, quality: int = 88) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def generate_plan(
        self,
        *,
        request_id: str,
        plan: Dict[str, Any],
        rgb_jpeg_b64: str,
        camera: Dict[str, Any],
        generation_scope: str = "",
    ) -> Dict[str, Any]:
        with self._lock:
            self._load()
            assert self._demo is not None and self._model is not None

            plan_id = str(plan["plan_id"])
            action_ids = [int(action["id"]) for action in plan["actions"]]
            c2ws = build_relative_camera_path(
                action_ids,
                forward_step_m=float(camera.get("forward_step_m", 0.25)),
                turn_angle_deg=float(camera.get("turn_angle_deg", 30.0)),
                pitch_rad=float(camera.get("pitch_rad", 0.0)),
            )

            request_dir = self.output_root / request_id
            plan_dir = request_dir / plan_id
            plan_dir.mkdir(parents=True, exist_ok=True)
            input_path = plan_dir / "input.jpg"
            self._decode_input(rgb_jpeg_b64).save(input_path, quality=95)
            cfg = float(os.environ.get("AVIC_SVC_CFG", "4.0"))
            num_targets = max(1, min(len(action_ids), 16))
            fingerprint = self.generation_fingerprint(input_path.read_bytes(), c2ws, cfg, num_targets)
            # Legacy clients without an episode epoch only reuse within one request.
            scope = generation_scope or "request:" + request_id
            cache_key = (scope, str(self.avic_root), str(self.model_dir), str(self.vae_dir),
                         self.device, fingerprint)
            cached = self._generation_cache.get(cache_key)
            if cached is not None and not Path(cached["source_video"]).is_file():
                del self._generation_cache[cache_key]
                cached = None
            cache_hit = cached is not None
            if cache_hit:
                video_path = Path(cached["source_video"])
                origin_request, origin_plan = cached["request_id"], cached["plan_id"]
                self._generation_cache.move_to_end(cache_key)
            else:
                self._demo.svc_main(
                    model=self._model,
                    data_path=str(plan_dir),
                    task="img2trajvid_s-prob",
                    replace_or_include_input=True,
                    traj_prior="apexnav-relative-camera",
                    cfg=cfg,
                    guider=1,
                    L_short=576,
                    num_targets=num_targets,
                    use_traj_prior=True,
                    output_path=str(plan_dir),
                    chunk_strategy="interp",
                    c2ws=c2ws,
                    scene=[str(input_path)],
                )
                video_path = plan_dir / "pred.mp4"
                origin_request, origin_plan = request_id, plan_id
            configured_max_rerank_frames = max(
                1, int(os.environ.get("AVIC_SVC_MAX_RERANK_FRAMES", "6"))
            )
            request_max_rerank_frames = max(
                1,
                int(
                    plan.get(
                        "max_rerank_frames", configured_max_rerank_frames
                    )
                ),
            )
            max_rerank_frames = min(
                configured_max_rerank_frames, request_max_rerank_frames
            )
            try:
                future_frames = self._sample_video_frames(
                    video_path,
                    min(len(action_ids), max_rerank_frames),
                    plan_dir,
                )
            except Exception:
                self._generation_cache.pop(cache_key, None)
                raise
            final_frame = self._decode_input(future_frames[-1]["jpeg_b64"])
            final_path = plan_dir / "svc_final.jpg"
            final_frame.save(final_path, quality=92)
            self._generation_cache[cache_key] = dict(source_video=str(video_path),
                request_id=origin_request, plan_id=origin_plan)
            self._generation_cache.move_to_end(cache_key)
            while len(self._generation_cache) > self.generation_cache_size:
                self._generation_cache.popitem(last=False)
            result = {
                "plan_id": plan_id,
                "actions": plan["actions"],
                "source_frontier": plan.get("source_frontier"),
                "frontier_anchor": plan.get("frontier_anchor"),
                "future_jpeg_b64": self._jpeg_b64(final_frame),
                "future_frames": future_frames,
                "artifact": str(final_path),
                "source_video": str(video_path),
                "generation_key": fingerprint,
                "generation_cache_hit": cache_hit,
                "generation_origin_request": origin_request,
                "generation_origin_plan": origin_plan,
            }
            if alignment_audit.enabled():
                result['alignment_audit']=alignment_audit.capture_generation(plan_dir,rgb_jpeg_b64,input_path,
                    request_id=request_id,plan_id=plan_id,
                    selected_actions=action_ids,camera=dict(camera),c2ws=c2ws.tolist(),
                    c2ws_frame='SVC relative +Z-forward camera contract, Ry @ Rx; not Habitat world',
                    cfg=cfg,num_targets=num_targets,
                    svc_call_recipe=dict(task='img2trajvid_s-prob',replace_or_include_input=True,
                        traj_prior='apexnav-relative-camera',guider=1,L_short=576,
                        use_traj_prior=True,chunk_strategy='interp'),
                    source_frontier=plan.get('source_frontier'),anchor_pose=(plan.get('frontier_anchor') or {}).get('alignment_pose'),
                    scope=scope,generation_key=fingerprint,cache_hit=cache_hit,
                    cache_origin=dict(request_id=origin_request,plan_id=origin_plan,
                        contract_path=str(self.output_root/origin_request/origin_plan/'camera_contract.json')),
                    samples=[dict(order=f.get('order'),video_frame_index=f.get('video_frame_index'),
                                  sample=f.get('alignment_sample')) for f in future_frames],
                    missing_frame_pose_reason='Conditioning poses do not prove generated content obeys poses; interp target/frame mapping unknown')
            return result
