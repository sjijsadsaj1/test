#!/usr/bin/env bash
set -Eeuo pipefail
main() {
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python=/data/wzy/miniconda3/envs/apexnav/bin/python
batch="$project_dir/experiments/runs/target_review_pilot/$(date +%Y%m%d_%H%M%S)_$$"
"$python" scripts/target_review_pilot.py prepare --batch "$batch"
echo "Pilot results: $batch"
bridge_pid=''
cleanup() {
 status=$?;trap - EXIT INT TERM
 if [[ -n "$bridge_pid" ]];then kill -TERM -- "-$bridge_pid" 2>/dev/null || true; wait "$bridge_pid" 2>/dev/null || true;fi
 printf '%s\n' "$status" >"$batch/exit_code";exit "$status"
}
trap cleanup EXIT;trap 'exit 130' INT TERM
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

export PYTHONPATH="$project_dir" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1

"$python" scripts/verify_upstream_baseline.py "$project_dir/../Apexnav_1/runtime"
for arm in review dictionary combined; do
 "$python" scripts/target_review_pilot.py verify --batch "$batch"
 mkdir -p "$batch/$arm"
 "$python" scripts/experiment_utils.py ports --batch "$batch" >"$batch/ports.txt"
 read -r bridge_port ros_port <"$batch/ports.txt"
 export AVIC_SERVICE_PORT="$bridge_port" AVIC_TEXT_OUTPUT_ROOT="$batch/$arm/predictions"
 export APEXNAV_REVIEW_URL="http://127.0.0.1:$bridge_port" APEXNAV_LOG_ROOT="$batch/$arm/logs"
 export APEXNAV_TARGET_REVIEW=1 APEXNAV_TEXT_FRONTIER=1
 mode=avic;score_mode=text_frontier
 if [[ "$arm" == review ]];then export APEXNAV_TEXT_FRONTIER=0;mode=apexnav;score_mode=grounded_delta;fi
 if [[ "$arm" == dictionary ]];then export APEXNAV_TARGET_REVIEW=0;fi
 setsid "$project_dir/.runtime/image_glm_env/bin/python" -u -m avic_bridge.text_service >"$batch/$arm/bridge.log" 2>&1 &
 bridge_pid=$!
 deadline=$((SECONDS+60))
 until curl -fsS --max-time 2 "$APEXNAV_REVIEW_URL/health" >"$batch/$arm/health.json" 2>/dev/null; do
   kill -0 "$bridge_pid" || exit 1
   ((SECONDS<deadline)) || exit 1
   sleep 1
 done
 echo "Running $arm: 3 cases, review=$APEXNAV_TARGET_REVIEW dictionary=$APEXNAV_TEXT_FRONTIER"
 bash scripts/run_apexnav.sh --dataset mp3d --episode-manifest "$batch/manifest.json" --max-episodes 3 \
   --avic-mode "$mode" --avic-service-url "$APEXNAV_REVIEW_URL" --avic-score-mode "$score_mode" \
   --avic-max-calls 4 --avic-cooldown-steps 0 --avic-timeout 600 --reuse-existing-perception \
   --ros-port "$ros_port" --result-dir "$batch/$arm/results" >"$batch/$arm/progress.log" 2>&1
 kill -TERM -- "-$bridge_pid" 2>/dev/null || true
 wait "$bridge_pid" 2>/dev/null || true
 bridge_pid=''
done
"$python" scripts/target_review_pilot.py compare --batch "$batch"
}
main "$@"
