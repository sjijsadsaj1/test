"""Exercise real adapter/cache/video sampling, replacing only GPU inference."""
import copy
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from avic_bridge.protocol import make_candidate_plan
from avic_bridge.svc_adapter import SVCAdapter
from avic_bridge.service import SVCQwenBackend
from test_avic_frontier_policy import FakeHorizonClient, FakeITM, payload, frontier


class GenerationCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        env = patch.dict(os.environ, {"AVIC_SVC_CFG": "4.0", "AVIC_SVC_MAX_RERANK_FRAMES": "6"})
        env.start()
        self.addCleanup(env.stop)
        self.adapter = SVCAdapter(avic_root="unused", model_dir="unused", vae_dir="unused",
                                  hf_home="unused", output_root=str(self.root), device="cpu")
        self.calls = 0
        self.fail_next = False
        self.adapter._model = object()
        self.adapter._demo = SimpleNamespace(svc_main=self.generate_video)
        self.rgb = SVCAdapter._jpeg_b64(Image.new("RGB", (32, 24), (32, 64, 96)))

    def generate_video(self, **kwargs):
        self.calls += 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("temporary inference failure")
        output = str(Path(kwargs["output_path"]) / "pred.mp4")
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), 8, (32, 24))
        self.assertTrue(writer.isOpened())
        try:
            for value in range(16):
                writer.write(np.full((24, 32, 3), value * 8, dtype=np.uint8))
        finally:
            writer.release()

    def generate(self, request="request-1", plan_id="frontier_0", scope="episode-1",
                 actions=(2, 1), rgb=None, camera=None, max_frames=6):
        plan = make_candidate_plan(plan_id, actions)
        plan.update(source_frontier={"rank": int(plan_id.rsplit("_", 1)[-1])},
                    frontier_anchor={"observation_id": "anchor-" + plan_id},
                    max_rerank_frames=max_frames)
        return self.adapter.generate_plan(request_id=request, plan=plan,
            rgb_jpeg_b64=rgb or self.rgb,
            camera=camera or dict(pitch_rad=0.0, forward_step_m=0.25, turn_angle_deg=30.0),
            generation_scope=scope)

    def test_same_input_different_frontiers_share_video_not_bindings(self):
        first = self.generate()
        second = self.generate(plan_id="frontier_2")
        self.assertEqual(self.calls, 1)
        self.assertFalse(first["generation_cache_hit"])
        self.assertTrue(second["generation_cache_hit"])
        self.assertEqual(first["source_video"], second["source_video"])
        self.assertNotEqual(first["artifact"], second["artifact"])
        self.assertEqual(second["source_frontier"]["rank"], 2)
        self.assertEqual(second["frontier_anchor"]["observation_id"], "anchor-frontier_2")
        self.assertEqual(second["generation_origin_plan"], "frontier_0")

    def test_cross_step_reuse_and_episode_epoch_isolation(self):
        self.generate()
        reused = self.generate(request="request-2")
        self.assertTrue(reused["generation_cache_hit"])
        self.assertEqual(reused["generation_origin_request"], "request-1")
        self.assertTrue(Path(reused["artifact"]).is_file())
        self.generate(request="request-3", scope="episode-2")
        self.assertEqual(self.calls, 2)

    def test_legacy_without_epoch_only_reuses_within_request(self):
        self.generate(scope="")
        self.generate(scope="", plan_id="frontier_1")
        self.generate(scope="", request="request-2")
        self.assertEqual(self.calls, 2)

    def test_input_image_actions_camera_and_cfg_invalidate_cache(self):
        self.generate()
        self.generate(request="image", rgb=SVCAdapter._jpeg_b64(Image.new("RGB", (32, 24), "red")))
        self.generate(request="actions", actions=(3, 1))
        self.generate(request="pitch", camera=dict(pitch_rad=0.2))
        self.generate(request="step", camera=dict(forward_step_m=0.5))
        self.generate(request="yaw", camera=dict(turn_angle_deg=60))
        with patch.dict(os.environ, {"AVIC_SVC_CFG": "5.0"}):
            self.generate(request="cfg")
        self.assertEqual(self.calls, 7)

    def test_sampling_budget_resamples_same_raw_video(self):
        first = self.generate(actions=(1, 1, 1), max_frames=1)
        second = self.generate(request="request-2", actions=(1, 1, 1), max_frames=3)
        self.assertEqual(self.calls, 1)
        self.assertEqual(len(first["future_frames"]), 1)
        self.assertEqual(len(second["future_frames"]), 3)
        self.assertEqual(first["future_frames"][-1]["jpeg_b64"], second["future_frames"][-1]["jpeg_b64"])

    def test_failed_generation_is_not_cached(self):
        self.fail_next = True
        with self.assertRaisesRegex(RuntimeError, "temporary"):
            self.generate()
        self.assertEqual(len(self.adapter._generation_cache), 0)
        self.generate()
        self.assertEqual(self.calls, 2)

    def test_missing_or_corrupt_video_invalidates_cache(self):
        first = self.generate()
        path = Path(first["source_video"])
        path.rename(path.with_suffix(".backup"))
        self.generate(request="request-2")
        self.assertEqual(self.calls, 2)
        cached_path = Path(next(iter(self.adapter._generation_cache.values()))["source_video"])
        cached_path.write_bytes(b"corrupt video")
        with self.assertRaises(RuntimeError):
            self.generate(request="request-3")
        self.assertEqual(len(self.adapter._generation_cache), 0)
        self.generate(request="request-4")
        self.assertEqual(self.calls, 3)

    def test_lru_eviction_does_not_delete_artifacts(self):
        self.adapter.generation_cache_size = 2
        first = self.generate(scope="one")
        self.generate(request="r2", scope="two")
        self.generate(request="r3", scope="three")
        self.assertEqual(len(self.adapter._generation_cache), 2)
        self.assertTrue(Path(first["source_video"]).exists())
        self.generate(request="r4", scope="one")
        self.assertEqual(self.calls, 4)

    def test_concurrent_duplicates_generate_once(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda i: self.generate(request="r%d" % i), range(4)))
        self.assertEqual(self.calls, 1)
        self.assertEqual(sum(r["generation_cache_hit"] for r in results), 3)

    def test_backend_reports_reuse_preserves_duplicate_guard_and_audits(self):
        import json
        backend = SVCQwenBackend()
        backend._svc = self.adapter
        backend._qwen = FakeHorizonClient([
            dict(plan_id="frontier_%d" % i, prefix_length=1, reason="short") for i in range(3)])
        backend._itm = FakeITM()
        request = payload([frontier(i, .20, 1, 0, 2) for i in range(3)])
        request["episode"] = dict(generation_epoch="epoch-1", scene_id="scene", episode_id="1")
        for plan in request["navigation"]["candidate_plans"]:
            plan["frontier_anchor"]["rgb_jpeg_b64"] = self.rgb
        first = backend.decide(request)
        self.assertEqual(self.calls, 1)
        self.assertEqual(first["workload"]["svc_videos"], 1)
        self.assertEqual(first["workload"]["svc_reused_videos"], 2)
        self.assertEqual(first["workload"]["svc_candidate_plans"], 3)
        self.assertTrue(all(s["imagination_bonus"] == 0 for s in first["candidate_scores"]))
        second_request = copy.deepcopy(request)
        second_request["request_id"] = "request-2"
        second = backend.decide(second_request)
        self.assertEqual(second["workload"]["svc_videos"], 0)
        self.assertEqual(second["workload"]["svc_reused_videos"], 3)
        for request_id in (request["request_id"], "request-2"):
            audit = json.loads((self.root / request_id / "verification.json").read_text())
            self.assertEqual(audit["request_id"], request_id)
            self.assertEqual(len(audit["candidates"]), 3)
            self.assertEqual(len(set(c["generation_key"] for c in audit["candidates"])), 1)
        self.assertEqual(self.calls, 1)


if __name__ == "__main__":
    unittest.main()
