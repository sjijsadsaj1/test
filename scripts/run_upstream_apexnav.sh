#!/usr/bin/env bash
set -Eeuo pipefail
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ " ${*} " == *" --avic-mode "* || " ${*} " == *" --avic-shadow "* ]]; then
    echo "This entry always runs the frozen upstream baseline, never AVIC/shadow." >&2
    exit 2
fi
bash "$project_dir/scripts/prepare_upstream_baseline.sh"
export APEXNAV_UPSTREAM_COMPAT_ROOT="$(dirname -- "$project_dir")/Apexnav_1/runtime"
exec bash "$project_dir/scripts/run_apexnav.sh" --avic-mode apexnav "$@"
