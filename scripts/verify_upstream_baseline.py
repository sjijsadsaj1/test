#!/usr/bin/env python3
"""Fail closed if the isolated upstream/compatibility snapshot has drifted."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPAT = "1c8d6d264bce4a55b4055b5848f04a9c290ba41c"
ORIGINAL = "049971b"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def verify(snapshot):
    snapshot = Path(snapshot).resolve()
    if snapshot == ROOT.parent / 'Apexnav_1/runtime':
        from download_apexnav1 import verify as verify_download
        return verify_download(snapshot.parent)
    checked = 0
    for entry in git("ls-tree", "-rz", COMPAT).split(b"\0"):
        if not entry:
            continue
        meta, raw_path = entry.split(b"\t", 1)
        mode, kind, obj = meta.split()
        path = raw_path.decode()
        if kind != b"blob" or mode == b"120000":
            continue
        expected = git("cat-file", "blob", obj.decode())
        actual = snapshot / path
        if not actual.is_file() or actual.read_bytes() != expected:
            raise ValueError("Upstream snapshot modified/missing: " + path)
        checked += 1
    # Inspect every tracked planner C++ source/header and the evaluator.
    changed = git("diff", "--name-only", ORIGINAL, COMPAT, "--", "src", "habitat_evaluation.py").decode().splitlines()
    allowed = {
        "src/planner/path_searching/CMakeLists.txt", "src/planner/plan_env/CMakeLists.txt",
        "src/planner/trajectory_manager/CMakeLists.txt", "src/planner/utils/vis_utils/CMakeLists.txt",
        "src/planner/trajectory_manager/include/gcopter/geo_utils.hpp",
    }
    if set(changed) - allowed:
        raise ValueError("Unreviewed algorithm changes in compatibility commit: " + repr(changed))
    print(f"Verified {checked} upstream/compatibility files; evaluator and exploration/A* logic remain at {ORIGINAL}")


if __name__ == "__main__":
    verify(sys.argv[1])
