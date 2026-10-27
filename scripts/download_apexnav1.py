#!/usr/bin/env python3
"""Download only pinned official runtime sources; keep pristine and adapted copies separate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parent / 'Apexnav_1'
COMMIT = '049971b613fcd2554b034bd4a853d79cc1e61613'
REPO = 'Robotics-STAR-Lab/ApexNav'
PATCH = ROOT / 'scripts/apexnav1_compat.patch'
PREFIXES = {'src', 'basic_utils', 'habitat2ros', 'config', 'llm', 'vlm'}
FILES = {'LICENSE', 'params.py', 'habitat_evaluation.py', 'apexnav_environment.yaml', 'pyproject.toml'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def wanted(path):
    p = PurePosixPath(path)
    return bool(p.parts) and not p.is_absolute() and '..' not in p.parts and (p.parts[0] in PREFIXES or path in FILES)


def fetch(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ApexNav-pinned-baseline'})
            with urllib.request.urlopen(req, timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def verify(dest=DEST):
    dest = Path(dest).resolve()
    meta = json.loads((dest / 'SOURCE.json').read_text())
    if meta['commit'] != COMMIT or meta['repository'] != REPO or meta['compat_patch_sha256'] != sha(PATCH.read_bytes()):
        raise ValueError('Baseline provenance or compatibility patch changed')
    for name, record in meta['files'].items():
        if not wanted(name):
            raise ValueError('Unexpected source path: ' + name)
        for tree, key in [('upstream', 'upstream_sha256'), ('runtime', 'runtime_sha256')]:
            p = dest / tree / name
            if p.is_symlink() or not p.is_file() or sha(p.read_bytes()) != record[key]:
                raise ValueError('Baseline modified/missing: ' + str(p))
    for tree in ('upstream', 'runtime'):
        for p in (dest / tree).rglob('*'):
            name = p.relative_to(dest / tree).as_posix()
            if name.split('/')[0] in {'build', 'devel', '.ros'} or '__pycache__' in p.parts:
                continue
            if p.is_file() and p.suffix in {'.py', '.cpp', '.h', '.hpp', '.xml', '.launch', '.yaml'} and name not in meta['files']:
                raise ValueError('Unexpected baseline code: ' + str(p))
    entry = dest / 'runtime/src/CMakeLists.txt'
    if not entry.is_symlink() or entry.resolve() != Path('/usr/share/catkin/cmake/toplevel.cmake'):
        raise ValueError('Unexpected catkin entrypoint')
    print(f'Verified GitHub baseline {COMMIT}: {len(meta["files"])} source files; {dest / "runtime"}')
    return meta


def download():
    if DEST.exists():
        return verify()
    tree_url = f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1'
    tree = json.loads(fetch(tree_url))
    if tree.get('truncated') or 'tree' not in tree:
        raise ValueError('Incomplete official source tree')
    entries = [e for e in tree['tree'] if e['type'] == 'blob' and e['mode'] != '120000' and wanted(e['path'])]
    if not entries or not any(e['path'] == 'habitat_evaluation.py' for e in entries):
        raise ValueError('Missing official evaluator')
    print(f'Downloading {len(entries)} files, {sum(e["size"] for e in entries):,} source bytes; no assets/weights/data/history', flush=True)
    staging = Path(tempfile.mkdtemp(prefix='.Apexnav_1-download-', dir=ROOT.parent))
    def get(entry):
        name = entry['path']
        raw = fetch(f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}')
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if blob != entry['sha']:
            raise ValueError('GitHub blob checksum mismatch: ' + name)
        p = staging / 'upstream' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        p.chmod(0o755 if entry['mode'] == '100755' else 0o644)
        return name, {'git_blob': blob, 'upstream_sha256': sha(raw)}
    records = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (name, record) in enumerate(pool.map(get, entries), 1):
            records[name] = record
            if index % 50 == 0:
                print(f'Downloaded {index}/{len(entries)}', flush=True)
    shutil.copytree(staging / 'upstream', staging / 'runtime')
    subprocess.run(['git', 'apply', '--check', str(PATCH)], cwd=staging / 'runtime', check=True)
    subprocess.run(['git', 'apply', str(PATCH)], cwd=staging / 'runtime', check=True)
    for name, record in records.items():
        record['runtime_sha256'] = sha((staging / 'runtime' / name).read_bytes())
    (staging / 'runtime/src/CMakeLists.txt').symlink_to('/usr/share/catkin/cmake/toplevel.cmake')
    shutil.copy2(PATCH, staging / 'compatibility.patch')
    (staging / 'SOURCE.json').write_text(json.dumps(dict(repository=REPO, commit=COMMIT,
        tree_url=tree_url, downloaded_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        compat_patch_sha256=sha(PATCH.read_bytes()), files=records), indent=2) + '\n')
    staging.chmod(0o755)
    staging.rename(DEST)  # no existing workspace is overwritten
    verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    download() if args.download else verify()
