#!/usr/bin/env bash
# Frozen upstream and dictionary probability model, 30 identical episodes each.
set -Eeuo pipefail
main() {
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python=/data/wzy/miniconda3/envs/apexnav/bin/python
if [[ "${1:-}" != --worker ]]; then
    reference_args=()
    if [[ $# == 2 && "$1" == --reuse-baseline ]]; then
        reference_args=(--reference "$2")
    elif [[ $# != 0 ]]; then
        echo 'Usage: bash scripts/run_frontier_dictionary30.sh [--reuse-baseline previous_batch]'; return 2
    fi
    batch="$project_dir/experiments/runs/frontier_dictionary30/$(date +%Y%m%d_%H%M%S)_$$"
    "$python" scripts/frontier_dictionary_batch.py prepare --batch "$batch" "${reference_args[@]}"
    touch "$batch/progress.log"
    nohup setsid bash "$project_dir/scripts/run_frontier_dictionary30.sh" --worker "$batch" >"$batch/progress.log" 2>&1 </dev/null &
    pid=$!; printf '%s\n' "$pid" >"$batch/worker.pid"
    echo "Results: $batch; Worker PID: $pid"
    echo "tail -n 40 -F $batch/progress.log"
    echo 'Ctrl+C stops viewing only; the experiment continues.'
    tail --pid="$pid" -n 40 -F "$batch/progress.log"
    [[ -f "$batch/worker.exit_code" ]] || return 1
    return "$(<"$batch/worker.exit_code")"
fi
[[ $# == 2 ]] || return 2
batch="$2"; bridge_pid=''
cleanup() {
    status=$?; trap - EXIT INT TERM
    if [[ -n "$bridge_pid" ]]; then
        kill -TERM -- "-$bridge_pid" 2>/dev/null || true
        wait "$bridge_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$status" >"$batch/worker.exit_code"; exit "$status"
}
trap cleanup EXIT; trap 'exit 130' INT TERM
if [[ -f config/avic.env ]]; then set -a; source config/avic.env; set +a; fi
if [[ -f config/image_glm.env ]]; then set -a; source config/image_glm.env; set +a; fi
# Prefer the dedicated VLM config; legacy local image_glm.env remains a compatibility fallback.
if [[ -f config/text_frontier.env ]]; then set -a; source config/text_frontier.env; set +a; fi
export AVIC_ENV_FILE=/dev/null APEXNAV_ROOT="$project_dir" APEXNAV_STAGNATION_RECOVERY=false
export APEXNAV_IMAGE_GLM=0 APEXNAV_IMAGE_GLM_STRONG=0 APEXNAV_FRONTIER_CYCLE_MEMORY=0
export APEXNAV_ALIGNED_BASE_EXPERIMENT=0 APEXNAV_RELAXED_IMAGINATION=0 APEXNAV_EXTREME_SAFETY=0
export APEXNAV_HYPOTHESIS_PROGRESS=0 APEXNAV_ROOM_PRIOR=0 APEXNAV_TARGET_ALIASES=off
export APEXNAV_LITERATURE_METHOD=off APEXNAV_ALIGNMENT_AUDIT=1 APEXNAV_PERCEPTION_AUDIT=1
export AVIC_IMAGE_INTERIOR=0 AVIC_IMAGE_FRONTIER_GEOMETRY=0 AVIC_IMAGE_DUAL_CONTEXT=0
unset APEXNAV_UPSTREAM_COMPAT_ROOT APEXNAV_FRONTIER_ONCE APEXNAV_FRONTIER_REFRESH_DELTA APEXNAV_V1_FRONTIER_POLICY
unset APEXNAV_SCORE_RESPONSE_ARCHIVE APEXNAV_SCORE_RESPONSE_REPLAY APEXNAV_OPEN_ESCAPE_TRIGGER
export AVIC_TEXT_PROVIDER=local APEXNAV_TEXT_METHOD=dictionary
export APEXNAV_TARGET_REVIEW=0
export AVIC_OBJECT_DICTIONARY="$project_dir/config/frontier_object_dictionary.json"
export AVIC_TEXT_OUTPUT_ROOT="$batch/predictions"
export PYTHONPATH="$project_dir" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
"$python" scripts/frontier_dictionary_batch.py verify --batch "$batch"
"$python" scripts/experiment_utils.py ports --batch "$batch" >"$batch/ports.txt"
read -r bridge_port ros_port <"$batch/ports.txt"
export APEXNAV_TEXT_FRONTIER=0 APEXNAV_LOG_ROOT="$batch/baseline/logs"
echo 'Arm 1/2: frozen original ApexNav, 30 tasks'
if [[ -f "$batch/baseline_reference.json" ]]; then
    "$python" scripts/verify_upstream_baseline.py "$project_dir/../Apexnav_1/runtime"
    echo 'Reusing completed original baseline: task list, provenance, runner/config and result hashes verified.'
else
bash scripts/run_upstream_apexnav.sh --dataset mp3d --episode-manifest "$batch/manifest.json" \
    --max-episodes 30 --reuse-existing-perception --ros-port "$ros_port" --result-dir "$batch/baseline/results"
fi
"$python" scripts/frontier_dictionary_batch.py verify --batch "$batch"
# Re-select ports after the baseline; never stop an unrelated service.
"$python" scripts/experiment_utils.py ports --batch "$batch" >"$batch/ports.txt"
read -r bridge_port ros_port <"$batch/ports.txt"
export APEXNAV_TEXT_FRONTIER=1 APEXNAV_LOG_ROOT="$batch/dictionary/logs" AVIC_SERVICE_PORT="$bridge_port"
echo 'Arm 2/2: dictionary, local VLM, no generated images; 500 queries max, cooldown 0'
setsid "$project_dir/.runtime/image_glm_env/bin/python" -u -m avic_bridge.text_service >"$batch/bridge.log" 2>&1 &
bridge_pid=$!; printf '%s\n' "$bridge_pid" >"$batch/bridge.pid"
deadline=$((SECONDS+60))
while ! curl -fsS --max-time 2 "http://127.0.0.1:$bridge_port/health" >"$batch/health.json" 2>/dev/null; do
    kill -0 "$bridge_pid" || { echo "Bridge exited; see $batch/bridge.log"; exit 1; }
    ((SECONDS<deadline)) || { echo 'Bridge health timeout'; exit 1; }
    sleep 1
done
"$python" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["contract"]=="frontier_object_marginals_v1" and d["generated_images"]==0 and d["provider"]=="local"' "$batch/health.json"
bash scripts/run_apexnav.sh --dataset mp3d --episode-manifest "$batch/manifest.json" \
    --max-episodes 30 --avic-mode avic --avic-service-url "http://127.0.0.1:$bridge_port" \
    --avic-score-mode text_frontier --avic-timeout 600 --avic-max-calls 500 --avic-cooldown-steps 0 \
    --reuse-existing-perception --ros-port "$ros_port" --result-dir "$batch/dictionary/results"
"$python" scripts/frontier_dictionary_batch.py compare --batch "$batch"
}
main "$@"
