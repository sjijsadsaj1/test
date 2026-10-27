import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_upstream_test", ROOT / "scripts/evaluate_upstream.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


class Fixed100Tests(unittest.TestCase):
    def test_frozen_set(self):
        d = json.loads((ROOT / "experiments/manifests/mp3d_sr_dev100_seed20260911_runtime.json").read_text())
        keys = {(e["scene_key"], e["episode_id"]) for e in d["episodes"]}
        self.assertEqual(len(d["episodes"]), 100)
        self.assertEqual(len(keys), 100)
        self.assertEqual(len(d["scene_counts"]), 11)
        self.assertEqual(sum(d["scene_counts"].values()), 100)
        self.assertEqual(sum(d["category_counts"].values()), 100)
        for entry in d["criteria"]["excluded_manifests"]:
            old = json.loads((ROOT / entry["path"]).read_text())
            self.assertFalse(keys & {(e["scene_key"], e["episode_id"]) for e in old["episodes"]})
        self.assertFalse(keys & {tuple(k) for k in d["criteria"]["excluded_first50_scene_episode_keys"]})

    def test_upstream_selector_uses_scene_id_and_category(self):
        episodes = [SimpleNamespace(scene_id="data/a/a.glb", episode_id="1", object_category="chair"),
                    SimpleNamespace(scene_id="data/b/b.glb", episode_id="1", object_category="table")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text(json.dumps({"episodes": [
                {"scene_key": "b", "episode_id": "1", "object_category": "table"},
                {"scene_key": "a", "episode_id": "1", "object_category": "chair"}]}))
            self.assertEqual(wrapper.select(episodes, path), episodes[::-1])
            self.assertEqual(wrapper.select(episodes, path, 1), [episodes[1]])
            with self.assertRaises(ValueError):
                wrapper.select(episodes[:1], path)


if __name__ == "__main__":
    unittest.main()
