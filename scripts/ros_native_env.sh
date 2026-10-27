#!/usr/bin/env bash
# Scope native ABI selection to ROS; never change the model/Habitat environment.
set -Eeuo pipefail
if [[ $# == 0 ]]; then
    echo 'Usage: bash scripts/ros_native_env.sh COMMAND [ARGS...]' >&2
    exit 2
fi
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
planner_workspace="${APEXNAV_ROS_WORKSPACE:-$project_dir}"
export LD_LIBRARY_PATH="$planner_workspace/devel/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/lib:/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$@"
