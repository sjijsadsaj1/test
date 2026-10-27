#!/usr/bin/env python3
"""Capture the code/config identity, or verify a fixed-set run completed."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--verify-results", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.capture:
        # Include untracked source files: git diff alone misses new modules.
        paths = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT).split(b"\0")
        hashes = {}
        for raw in paths:
            if not raw:
                continue
            path = ROOT / raw.decode()
            if path.is_file() and path.suffix in {".py", ".sh", ".cpp", ".h", ".hpp", ".xml", ".launch", ".yaml", ".json", ".txt", ".cmake"}:
                hashes[raw.decode()] = hashlib.sha256(path.read_bytes()).hexdigest()
        args.capture.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
        with tarfile.open(args.capture.parent / "source_snapshot.tar.gz", "w:gz") as archive:
            for name in sorted(hashes):
                archive.add(ROOT / name, arcname=name, recursive=False)
    if args.verify_results:
        records = list(args.verify_results.glob("*/record.txt"))
        if len(records) != 1:
            raise ValueError("Expected exactly one timestamped run in " + str(args.verify_results))
        expected = {(e["scene_key"], str(e["episode_id"])) for e in json.loads(args.manifest.read_text())["episodes"]}
        text = records[0].read_text()
        keys = re.findall(r"Scene ID: .*?/([^/]+)\.glb\s+Episode ID: (\d+)", text)
        numbers = re.findall(r"No\.(\d+) task is finished", text)
        if len(keys) != len(expected) or set(keys) != expected or sorted(map(int, numbers)) != list(range(1, len(expected) + 1)):
            raise ValueError("Incomplete or mismatched fixed-set results: " + str(records[0]))
        print("Verified complete paired-task membership:", records[0])


if __name__ == "__main__":
    main()
