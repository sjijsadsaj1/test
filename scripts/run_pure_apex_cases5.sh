#!/usr/bin/env bash
# Five recorded failure representatives, frozen upstream rules, real video.
set -Eeuo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
check_only=false
accept_existing=false
while [[ "${1:-}" == --check-only || "${1:-}" == --reuse-existing-perception ]]; do
    case "$1" in
        --check-only) check_only=true;;
        --reuse-existing-perception) accept_existing=true;;
    esac
    shift
done
service_mode="$(python3 - <<'PY'
import socket
ports = (12181, 12182, 12183, 12184)
occupied = []
for port in ports:
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=2):
            occupied.append(port)
    except ConnectionRefusedError:
        pass
    except OSError as exc:
        raise SystemExit(f'Cannot safely inspect perception port {port}: {exc}')
if not occupied:
    print('start')
elif len(occupied) == len(ports):
    print('reuse')
else:
    raise SystemExit(f'Partial/unknown perception stack {occupied}; refusing to load duplicate models or stop processes.')
PY
)"
service_args=()
if [[ "$accept_existing" == true && "$service_mode" != reuse ]]; then
    echo 'Explicit reuse requested, but the four existing perception endpoints are not all available. No new models will be loaded.' >&2
    exit 1
fi
if [[ "$service_mode" == reuse ]]; then
    # No model inference/loading here. Refuse early if an evaluation is busy,
    # the owner cannot be inspected, or the expected HTTP interface is absent.
    if [[ "$accept_existing" == true ]]; then
        echo 'Explicit deployment reuse: checks real inference schema, not process ownership or loaded-weight hashes.'
        python3 scripts/check_reusable_vlm.py --project "$project_dir" --accept-existing-owner
        service_args+=(--reuse-existing-perception)
    else
        python3 scripts/check_reusable_vlm.py --project "$project_dir"
        service_args+=(--reuse-vlm-services)
    fi
fi
python3 - <<'PY'
import json, socket
from pathlib import Path
manifest = json.loads(Path('experiments/manifests/mp3d_pure_apex_failure_cases5_20261006.json').read_text())
expected = [('Z6MFQCViBuw', '97', 'chair'), ('TbHJrupSAjP', '181', 'toilet'),
            ('oLBMNvg9in8', '79', 'picture'), ('Z6MFQCViBuw', '98', 'chair'),
            ('oLBMNvg9in8', '183', 'sink')]
actual = [(e['scene_key'], str(e['episode_id']), e['object_category']) for e in manifest['episodes']]
if actual != expected:
    raise SystemExit('Five-case manifest mismatch; refusing to start.')
try:
    with socket.create_connection(('127.0.0.1', 11321), timeout=2):
        raise SystemExit('Dedicated ROS port 11321 is occupied; no existing process has been stopped.')
except ConnectionRefusedError:
    pass
except OSError as exc:
    raise SystemExit(f'Cannot safely inspect dedicated ROS port 11321: {exc}')
print('Manifest: five expected cases; dedicated ROS port 11321: free.')
PY
if [[ "$check_only" == true ]]; then
    echo "Preflight passed: perception mode=$service_mode. No model, ROS or evaluation started."
    echo 'This checks startup prerequisites only, not numerical perception quality or navigation success.'
    exit 0
fi
exec bash scripts/run_upstream_apexnav.sh \
    --dataset mp3d \
    --episode-manifest experiments/manifests/mp3d_pure_apex_failure_cases5_20261006.json \
    --max-episodes 5 --video \
    --result-dir experiments/runs/pure_apex_failure_cases5 \
    --ros-port 11321 "${service_args[@]}" --detach --follow "$@"
