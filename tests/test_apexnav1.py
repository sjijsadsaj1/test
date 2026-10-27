import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('download_apexnav1', ROOT / 'scripts/download_apexnav1.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class Apexnav1Tests(unittest.TestCase):
    def fixture(self, root):
        name = 'habitat_evaluation.py'
        for tree in ('upstream', 'runtime'):
            (root / tree).mkdir()
            (root / tree / name).write_text('original\n')
        (root / 'runtime/src').mkdir()
        (root / 'runtime/src/CMakeLists.txt').symlink_to('/usr/share/catkin/cmake/toplevel.cmake')
        (root / 'SOURCE.json').write_text(json.dumps(dict(repository=download.REPO,
            commit=download.COMMIT, compat_patch_sha256=download.sha(download.PATCH.read_bytes()),
            files={name: dict(upstream_sha256=download.sha(b'original\n'),
                              runtime_sha256=download.sha(b'original\n'))})))

    def test_download_selection_excludes_assets_and_traversal(self):
        for p in ('assets/video_plant.gif', '../src/a.cpp', '/src/a.cpp', 'real_world_test_example/a.py'):
            self.assertFalse(download.wanted(p))
        self.assertTrue(download.wanted('src/planner/path_searching/src/astar2d.cpp'))
        self.assertTrue(download.wanted('habitat_evaluation.py'))

    def test_mutated_original_and_runtime_are_rejected(self):
        for tree in ('upstream', 'runtime'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root)
                download.verify(root)
                (root / tree / 'habitat_evaluation.py').write_text('modified\n')
                with self.assertRaisesRegex(ValueError, 'modified/missing'):
                    download.verify(root)

    def test_injected_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'runtime/avic_bridge.py').write_text('pass\n')
            with self.assertRaisesRegex(ValueError, 'Unexpected baseline code'):
                download.verify(root)

    def test_fresh_random50_reproduces(self):
        subprocess.run(['python3', str(ROOT / 'scripts/make_random50_20260919.py'), '--check'], check=True)
        d = json.loads((ROOT / 'experiments/manifests/mp3d_random50_seed20260919_runtime.json').read_text())
        keys = {(e['scene_key'], e['episode_id']) for e in d['episodes']}
        self.assertEqual(len(keys), 50)
        self.assertFalse(keys & {tuple(k) for k in d['excluded_keys']})

    def test_baseline_entry_cannot_use_old_snapshot(self):
        entry = (ROOT / 'scripts/run_upstream_apexnav.sh').read_text()
        self.assertIn('/Apexnav_1/runtime', entry)
        self.assertNotIn('/experiments/upstream_baseline/', entry)
        patch = download.PATCH.read_text()
        self.assertNotIn('diff --git a/src/planner/exploration_manager/', patch)
        self.assertNotIn('diff --git a/vlm/detector/grounding_dino.py', patch)


if __name__ == '__main__':
    unittest.main()
