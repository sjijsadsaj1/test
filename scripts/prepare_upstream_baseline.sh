#!/usr/bin/env bash
# Baselines now exclusively use the fresh GitHub download in Apexnav_1.
set -Eeuo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
snapshot="$(dirname -- "$project_dir")/Apexnav_1/runtime"
python3 "$project_dir/scripts/download_apexnav1.py"
printf 'Upstream baseline workspace: %s\n' "$snapshot"
