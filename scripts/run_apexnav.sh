#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="${APEXNAV_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
readonly AVIC_ENV_FILE="${AVIC_ENV_FILE:-$PROJECT_ROOT/config/avic.env}"

if [[ -f "$AVIC_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$AVIC_ENV_FILE"
fi

readonly CONDA_SETUP="${APEXNAV_CONDA_SETUP:-}"
readonly CONDA_ENV="${APEXNAV_CONDA_ENV:-apexnav}"
readonly AVIC_VLLM="${AVIC_VLLM:-vllm}"
readonly AVIC_QWEN_HF_HOME="${AVIC_QWEN_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
readonly AVIC_QWEN_MODEL="${AVIC_QWEN_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
readonly AVIC_QWEN_PORT="${AVIC_QWEN_PORT:-8001}"
readonly AVIC_BRIDGE_PORT="${AVIC_BRIDGE_PORT:-8091}"
readonly AVIC_QWEN_MAX_IMAGES_PER_PROMPT="${AVIC_QWEN_MAX_IMAGES_PER_PROMPT:-12}"
readonly AVIC_QWEN_REQUIRED_IMAGES="${AVIC_QWEN_REQUIRED_IMAGES:-12}"
readonly AVIC_QWEN_MAX_MODEL_LEN="${AVIC_QWEN_MAX_MODEL_LEN:-16384}"
readonly APEXNAV_LOG_ROOT="${APEXNAV_LOG_ROOT:-$PROJECT_ROOT/logs}"
readonly APEXNAV_OUTPUT_ROOT="${APEXNAV_OUTPUT_ROOT:-$PROJECT_ROOT/experiments/runs}"

dataset="hm3dv2"
episode="0"
run_full=false
need_video=false
startup_timeout=300
avic_mode="apexnav"
avic_service_url="${AVIC_SERVICE_URL:-http://127.0.0.1:${AVIC_BRIDGE_PORT}}"
avic_timeout="180"
avic_warmup_steps="26"
avic_max_calls="20"
avic_cooldown_steps="5"
avic_score_low="0.191149"
avic_score_high="0.227590"
avic_max_score_gap="0.01"
avic_real_weight="0.5"
avic_imagination_weight="0.5"
avic_min_score_spread="0.0001"
avic_cache_radius="0.5"
avic_cache_yaw_deg="15.0"
avic_anchor_fov_deg="90.0"
avic_anchor_max_distance="8.0"
avic_anchor_min_quality="0.35"
avic_max_bonus="0.02"
avic_min_reliability="0.55"
avic_min_evidence_gain="0.20"
max_episodes="0"
episode_manifest=""
result_dir=""
detach=false

usage() {
    cat <<'EOF'
Usage:
  ./scripts/run_apexnav.sh [options]

Options:
  --dataset hm3dv1|hm3dv2|mp3d  Dataset to evaluate (default: hm3dv2)
  --episode N                    Run one episode (default: 0)
  --full                         Run the complete validation split
  --video                        Save evaluation video
  --startup-timeout SECONDS      Model/ROS startup timeout (default: 300)
  --avic-mode apexnav|avic       Navigation mode; avic auto-starts Qwen + SVC bridge
  --avic-service-url URL         AVIC bridge URL (default: http://127.0.0.1:8091)
  --avic-timeout SECONDS         Per-request timeout (default: 180)
  --avic-warmup-steps N          Skip ApexNav initialization actions (default: 26)
  --avic-max-calls N             Episode WM-batch budget (default: 20)
  --avic-cooldown-steps N        Minimum steps between WM batches (default: 5)
  --avic-score-low X             Frozen lower score percentile (default: 0.191149)
  --avic-score-high X            Frozen upper score percentile (default: 0.227590)
  --avic-max-score-gap X         Max top-two real-score gap for G1 (default: 0.01)
  --avic-real-weight X           Legacy/mock compatibility option (default: 0.5)
  --avic-imagination-weight X    Legacy/mock compatibility option (default: 0.5)
  --avic-min-score-spread X      Legacy/mock compatibility option (default: 0.0001)
  --avic-cache-radius METERS     Episode cache match radius (default: 0.5)
  --avic-cache-yaw-deg DEGREES   Legacy option; cache now matches physical position (default: 15.0)
  --avic-anchor-fov-deg DEGREES  Real anchor horizontal FOV (default: 90.0)
  --avic-anchor-max-distance M   Maximum anchor-to-frontier distance (default: 8.0)
  --avic-anchor-min-quality X    Minimum real-anchor quality (default: 0.35)
  --avic-max-bonus X             Maximum positive score bonus (default: 0.02)
  --avic-min-reliability X       Minimum SVC verifier reliability (default: 0.55)
  --avic-min-evidence-gain X     Minimum new-evidence score (default: 0.20)
  --max-episodes N               Stop after the first N split episodes (0: all)
  --episode-manifest PATH        Run the exact episodes listed in a JSON manifest
  --result-dir PATH              Result parent; each run creates a timestamp child
  --detach                       Run independently of the current VSCode/SSH terminal
  -h, --help                     Show this help

Examples:
  ./scripts/run_apexnav.sh
  ./scripts/run_apexnav.sh --dataset mp3d --episode 10
  ./scripts/run_apexnav.sh --dataset hm3dv2 --full
  ./scripts/run_apexnav.sh --dataset hm3dv2 --episode-manifest experiments/manifests/hm3dv2_clear_targets_50_habitat.json
  ./scripts/run_apexnav.sh --dataset hm3dv2 --episode 0 --avic-mode avic
EOF
}

while (($#)); do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || { echo "--dataset requires a value" >&2; exit 2; }
            dataset="$2"
            shift 2
            ;;
        --episode)
            [[ $# -ge 2 ]] || { echo "--episode requires a value" >&2; exit 2; }
            episode="$2"
            run_full=false
            shift 2
            ;;
        --full)
            run_full=true
            shift
            ;;
        --video)
            need_video=true
            shift
            ;;
        --startup-timeout)
            [[ $# -ge 2 ]] || { echo "--startup-timeout requires a value" >&2; exit 2; }
            startup_timeout="$2"
            shift 2
            ;;
        --avic-mode)
            [[ $# -ge 2 ]] || { echo "--avic-mode requires a value" >&2; exit 2; }
            avic_mode="$2"
            shift 2
            ;;
        --avic-service-url)
            [[ $# -ge 2 ]] || { echo "--avic-service-url requires a value" >&2; exit 2; }
            avic_service_url="$2"
            shift 2
            ;;
        --avic-timeout)
            [[ $# -ge 2 ]] || { echo "--avic-timeout requires a value" >&2; exit 2; }
            avic_timeout="$2"
            shift 2
            ;;
        --avic-warmup-steps)
            [[ $# -ge 2 ]] || { echo "--avic-warmup-steps requires a value" >&2; exit 2; }
            avic_warmup_steps="$2"
            shift 2
            ;;
        --avic-max-calls)
            [[ $# -ge 2 ]] || { echo "--avic-max-calls requires a value" >&2; exit 2; }
            avic_max_calls="$2"
            shift 2
            ;;
        --avic-cooldown-steps)
            [[ $# -ge 2 ]] || { echo "--avic-cooldown-steps requires a value" >&2; exit 2; }
            avic_cooldown_steps="$2"
            shift 2
            ;;
        --avic-score-low)
            [[ $# -ge 2 ]] || { echo "--avic-score-low requires a value" >&2; exit 2; }
            avic_score_low="$2"
            shift 2
            ;;
        --avic-score-high)
            [[ $# -ge 2 ]] || { echo "--avic-score-high requires a value" >&2; exit 2; }
            avic_score_high="$2"
            shift 2
            ;;
        --avic-max-score-gap)
            [[ $# -ge 2 ]] || { echo "--avic-max-score-gap requires a value" >&2; exit 2; }
            avic_max_score_gap="$2"
            shift 2
            ;;
        --avic-real-weight)
            [[ $# -ge 2 ]] || { echo "--avic-real-weight requires a value" >&2; exit 2; }
            avic_real_weight="$2"
            shift 2
            ;;
        --avic-imagination-weight)
            [[ $# -ge 2 ]] || { echo "--avic-imagination-weight requires a value" >&2; exit 2; }
            avic_imagination_weight="$2"
            shift 2
            ;;
        --avic-min-score-spread)
            [[ $# -ge 2 ]] || { echo "--avic-min-score-spread requires a value" >&2; exit 2; }
            avic_min_score_spread="$2"
            shift 2
            ;;
        --avic-cache-radius)
            [[ $# -ge 2 ]] || { echo "--avic-cache-radius requires a value" >&2; exit 2; }
            avic_cache_radius="$2"
            shift 2
            ;;
        --avic-cache-yaw-deg)
            [[ $# -ge 2 ]] || { echo "--avic-cache-yaw-deg requires a value" >&2; exit 2; }
            avic_cache_yaw_deg="$2"
            shift 2
            ;;
        --avic-anchor-fov-deg)
            [[ $# -ge 2 ]] || { echo "--avic-anchor-fov-deg requires a value" >&2; exit 2; }
            avic_anchor_fov_deg="$2"
            shift 2
            ;;
        --avic-anchor-max-distance)
            [[ $# -ge 2 ]] || { echo "--avic-anchor-max-distance requires a value" >&2; exit 2; }
            avic_anchor_max_distance="$2"
            shift 2
            ;;
        --avic-anchor-min-quality)
            [[ $# -ge 2 ]] || { echo "--avic-anchor-min-quality requires a value" >&2; exit 2; }
            avic_anchor_min_quality="$2"
            shift 2
            ;;
        --avic-max-bonus)
            [[ $# -ge 2 ]] || { echo "--avic-max-bonus requires a value" >&2; exit 2; }
            avic_max_bonus="$2"
            shift 2
            ;;
        --avic-min-reliability)
            [[ $# -ge 2 ]] || { echo "--avic-min-reliability requires a value" >&2; exit 2; }
            avic_min_reliability="$2"
            shift 2
            ;;
        --avic-min-evidence-gain)
            [[ $# -ge 2 ]] || { echo "--avic-min-evidence-gain requires a value" >&2; exit 2; }
            avic_min_evidence_gain="$2"
            shift 2
            ;;
        --max-episodes)
            [[ $# -ge 2 ]] || { echo "--max-episodes requires a value" >&2; exit 2; }
            max_episodes="$2"
            shift 2
            ;;
        --episode-manifest)
            [[ $# -ge 2 ]] || { echo "--episode-manifest requires a value" >&2; exit 2; }
            episode_manifest="$2"
            run_full=true
            shift 2
            ;;
        --result-dir)
            [[ $# -ge 2 ]] || { echo "--result-dir requires a value" >&2; exit 2; }
            result_dir="$2"
            shift 2
            ;;
        --detach)
            detach=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$dataset" in
    hm3dv1|hm3dv2|mp3d) ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

case "$avic_mode" in
    apexnav|avic) ;;
    *) echo "Unsupported AVIC mode: $avic_mode" >&2; exit 2 ;;
esac

if [[ "$run_full" == false && ! "$episode" =~ ^[0-9]+$ ]]; then
    echo "--episode must be a non-negative integer" >&2
    exit 2
fi

if [[ ! "$startup_timeout" =~ ^[1-9][0-9]*$ ]]; then
    echo "--startup-timeout must be a positive integer" >&2
    exit 2
fi

if [[ ! "$avic_timeout" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_warmup_steps" =~ ^[0-9]+$ ]] ||
   [[ ! "$avic_max_calls" =~ ^[0-9]+$ ]] ||
   [[ ! "$avic_cooldown_steps" =~ ^[0-9]+$ ]] ||
   [[ ! "$avic_score_low" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_score_high" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_max_score_gap" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_real_weight" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_imagination_weight" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_min_score_spread" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_cache_radius" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_cache_yaw_deg" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_anchor_fov_deg" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_anchor_max_distance" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_anchor_min_quality" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_max_bonus" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_min_reliability" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$avic_min_evidence_gain" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$max_episodes" =~ ^[0-9]+$ ]]; then
    echo "Invalid AVIC timeout/budget option." >&2
    exit 2
fi

# A VSCode/code-server terminal is a pseudo-terminal managed by ptyHost. When
# its UI disconnects, verbose writers attached to that PTY can be flow-controlled
# or receive SIGHUP if the terminal is disposed. Detach before loading models so
# every byte goes to a regular file and the evaluation has no terminal dependency.
if [[ "$detach" == true ]]; then
    detached_dir="$APEXNAV_LOG_ROOT/detached"
    mkdir -p "$detached_dir"
    detached_stamp="$(date +%Y%m%d_%H%M%S)"
    detached_log="$detached_dir/${detached_stamp}_${dataset}_${avic_mode}.log"
    detached_pid="$detached_log.pid"

    detached_args=(--dataset "$dataset")
    if [[ "$run_full" == true ]]; then
        detached_args+=(--full)
    else
        detached_args+=(--episode "$episode")
    fi
    if [[ "$need_video" == true ]]; then
        detached_args+=(--video)
    fi
    detached_args+=(--startup-timeout "$startup_timeout")
    detached_args+=(--avic-mode "$avic_mode")
    detached_args+=(--avic-service-url "$avic_service_url")
    detached_args+=(--avic-timeout "$avic_timeout")
    detached_args+=(--avic-warmup-steps "$avic_warmup_steps")
    detached_args+=(--avic-max-calls "$avic_max_calls")
    detached_args+=(--avic-cooldown-steps "$avic_cooldown_steps")
    detached_args+=(--avic-score-low "$avic_score_low")
    detached_args+=(--avic-score-high "$avic_score_high")
    detached_args+=(--avic-max-score-gap "$avic_max_score_gap")
    detached_args+=(--avic-real-weight "$avic_real_weight")
    detached_args+=(--avic-imagination-weight "$avic_imagination_weight")
    detached_args+=(--avic-min-score-spread "$avic_min_score_spread")
    detached_args+=(--avic-cache-radius "$avic_cache_radius")
    detached_args+=(--avic-cache-yaw-deg "$avic_cache_yaw_deg")
    detached_args+=(--avic-anchor-fov-deg "$avic_anchor_fov_deg")
    detached_args+=(--avic-anchor-max-distance "$avic_anchor_max_distance")
    detached_args+=(--avic-anchor-min-quality "$avic_anchor_min_quality")
    detached_args+=(--avic-max-bonus "$avic_max_bonus")
    detached_args+=(--avic-min-reliability "$avic_min_reliability")
    detached_args+=(--avic-min-evidence-gain "$avic_min_evidence_gain")
    detached_args+=(--max-episodes "$max_episodes")
    if [[ -n "$episode_manifest" ]]; then
        detached_args+=(--episode-manifest "$episode_manifest")
    fi
    if [[ -n "$result_dir" ]]; then
        detached_args+=(--result-dir "$result_dir")
    fi

    nohup "$0" "${detached_args[@]}" >"$detached_log" 2>&1 </dev/null &
    detached_process=$!
    printf '%s\n' "$detached_process" >"$detached_pid"

    # Catch immediate argument/environment failures while still returning a
    # useful log path to the caller.
    sleep 1
    if ! kill -0 "$detached_process" 2>/dev/null; then
        echo "Detached ApexNav process exited during startup." >&2
        echo "Log: $detached_log" >&2
        tail -n 80 "$detached_log" >&2 || true
        exit 1
    fi

    echo "ApexNav detached from the terminal."
    echo "PID: $detached_process"
    echo "PID file: $detached_pid"
    echo "Log: $detached_log"
    echo "Observe: tail -f $detached_log"
    exit 0
fi

conda_setup="$CONDA_SETUP"
if [[ -z "$conda_setup" ]] && command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$conda_base" ]]; then
        conda_setup="$conda_base/etc/profile.d/conda.sh"
    fi
fi
if [[ -z "$conda_setup" || ! -r "$conda_setup" ]]; then
    echo "Conda setup not found. Set APEXNAV_CONDA_SETUP in config/avic.env." >&2
    exit 1
fi

# Do not inherit a parent/root shell's active Conda environment. Sourcing a
# different Conda installation does not clear those variables by itself.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
unset CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M

set +u
# shellcheck disable=SC1090
source "$conda_setup"
conda activate "$CONDA_ENV"
set -u
cd "$PROJECT_ROOT"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="$APEXNAV_LOG_ROOT/one_click/${timestamp}_$$"
mkdir -p "$log_dir"

# Never resume or overwrite an earlier experiment implicitly.  A caller may
# group runs under --result-dir, but every invocation receives its own
# timestamp-named child directory.  Without --result-dir, keep the same
# separation by dataset and mode under experiments/runs.
if [[ -n "$result_dir" ]]; then
    result_parent="$result_dir"
else
    result_parent="$APEXNAV_OUTPUT_ROOT/$dataset/$avic_mode"
fi
result_dir="${result_parent%/}/${timestamp}"
mkdir -p "$result_dir"
echo "Result directory: $result_dir"

avic_output_root="$result_dir/avic_imagination"
if [[ "$avic_mode" == "avic" ]]; then
    mkdir -p "$avic_output_root"
    if [[ ! -d "$avic_output_root" || ! -w "$avic_output_root" ]]; then
        echo "AVIC output directory is not writable by $(id -un): $avic_output_root" >&2
        exit 1
    fi
    echo "AVIC imagination directory: $avic_output_root"
fi

declare -a managed_pids=()
declare -a managed_names=()
declare -A service_pids=()
declare -A service_logs=()

port_open() {
    local port="$1"
    (exec 3<>"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM

    if ((${#managed_pids[@]})); then
        echo
        echo "Stopping services started by this script..."
        local index pid
        for ((index=${#managed_pids[@]} - 1; index >= 0; index--)); do
            pid="${managed_pids[index]}"
            kill -TERM -- "-${pid}" 2>/dev/null || true
        done

        local deadline=$((SECONDS + 10))
        while ((SECONDS < deadline)); do
            local alive=false
            for pid in "${managed_pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    alive=true
                    break
                fi
            done
            [[ "$alive" == false ]] && break
            sleep 0.2
        done

        for pid in "${managed_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL -- "-${pid}" 2>/dev/null || true
            fi
            wait "$pid" 2>/dev/null || true
        done
    fi

    echo "Logs: $log_dir"
    exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

start_background() {
    local name="$1"
    shift
    local logfile="$log_dir/${name}.log"

    setsid "$@" >"$logfile" 2>&1 &
    local pid=$!
    managed_pids+=("$pid")
    managed_names+=("$name")
    service_pids["$name"]="$pid"
    service_logs["$name"]="$logfile"
    echo "  $name: PID $pid, log $logfile"
}

wait_for_port() {
    local name="$1"
    local port="$2"
    local pid="${service_pids[$name]}"
    local logfile="${service_logs[$name]}"
    local deadline=$((SECONDS + startup_timeout))

    while ! port_open "$port"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$name exited before opening port $port." >&2
            tail -n 80 "$logfile" >&2 || true
            return 1
        fi
        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for $name on port $port." >&2
            tail -n 80 "$logfile" >&2 || true
            return 1
        fi
        sleep 1
    done
    echo "  $name ready on port $port"
}

ensure_local_avic_stack() {
    local qwen_models avic_health

    echo "[AVIC] Ensuring local Qwen policy service..."
    if port_open "$AVIC_QWEN_PORT"; then
        echo "  Reusing service on port $AVIC_QWEN_PORT"
    else
        command -v "$AVIC_VLLM" >/dev/null 2>&1 || {
            echo "AVIC vLLM executable not found: $AVIC_VLLM" >&2
            return 1
        }
        start_background avic_qwen env \
            HF_HOME="$AVIC_QWEN_HF_HOME" \
            "$AVIC_VLLM" serve "$AVIC_QWEN_MODEL" \
            --host 127.0.0.1 \
            --port "$AVIC_QWEN_PORT" \
            --served-model-name "$AVIC_QWEN_MODEL" \
            --limit-mm-per-prompt "image=${AVIC_QWEN_MAX_IMAGES_PER_PROMPT}" \
            --max-model-len "$AVIC_QWEN_MAX_MODEL_LEN" \
            --gpu-memory-utilization 0.50
        wait_for_port avic_qwen "$AVIC_QWEN_PORT"
    fi

    qwen_models="$(curl -fsS --max-time 30 \
        "http://127.0.0.1:${AVIC_QWEN_PORT}/v1/models")" || {
        echo "Qwen port is open but /v1/models is not ready." >&2
        return 1
    }
    if [[ "$qwen_models" != *"$AVIC_QWEN_MODEL"* ]]; then
        echo "Port $AVIC_QWEN_PORT does not serve $AVIC_QWEN_MODEL." >&2
        return 1
    fi
    echo "  Qwen ready: $AVIC_QWEN_MODEL"

    echo "[AVIC] Verifying Qwen multi-image capacity..."
    if ! python -m avic_bridge.qwen_preflight \
        --base-url "http://127.0.0.1:${AVIC_QWEN_PORT}/v1" \
        --model "$AVIC_QWEN_MODEL" \
        --images "$AVIC_QWEN_REQUIRED_IMAGES" \
        --timeout "$startup_timeout"; then
        echo "Qwen cannot accept the ${AVIC_QWEN_REQUIRED_IMAGES}-image AVIC verifier request." >&2
        echo "If port ${AVIC_QWEN_PORT} was reused, stop that stale Qwen service and retry." >&2
        echo "ApexNav evaluation has not started." >&2
        return 1
    fi

    echo "[AVIC] Ensuring local SVC/Qwen bridge..."
    if port_open "$AVIC_BRIDGE_PORT"; then
        echo "  Reusing service on port $AVIC_BRIDGE_PORT"
    else
        start_background avic_bridge env \
            AVIC_OUTPUT_ROOT="$avic_output_root" \
            "$PROJECT_ROOT/scripts/run_avic_service.sh" \
            --backend svc_qwen \
            --host 127.0.0.1 \
            --port "$AVIC_BRIDGE_PORT" \
            --device cuda:0
        wait_for_port avic_bridge "$AVIC_BRIDGE_PORT"
    fi

    avic_health="$(curl -fsS --max-time 30 \
        "http://127.0.0.1:${AVIC_BRIDGE_PORT}/health")" || {
        echo "AVIC bridge port is open but its health endpoint failed." >&2
        return 1
    }
    if [[ "$avic_health" != *'"ok": true'* && "$avic_health" != *'"ok":true'* ]]; then
        echo "AVIC bridge is not healthy: $avic_health" >&2
        return 1
    fi
    if [[ "$avic_health" != *'"backend": "svc_qwen"'* && \
          "$avic_health" != *'"backend":"svc_qwen"'* ]]; then
        echo "Port $AVIC_BRIDGE_PORT is not running the svc_qwen backend." >&2
        return 1
    fi
    if [[ "$avic_health" != *'"frontier_binding": "per_frontier_real_directional_anchor"'* && \
          "$avic_health" != *'"frontier_binding":"per_frontier_real_directional_anchor"'* ]]; then
        echo "Port $AVIC_BRIDGE_PORT is running a stale AVIC bridge. Stop it and retry." >&2
        return 1
    fi
    if [[ "$avic_health" != *'"gate_contract": "frontier_anchor_multiframe_verifier_v1"'* && \
          "$avic_health" != *'"gate_contract":"frontier_anchor_multiframe_verifier_v1"'* ]]; then
        echo "Port $AVIC_BRIDGE_PORT is running an obsolete G2 call/skip contract. Stop it and retry." >&2
        return 1
    fi
    if [[ "$avic_health" != *"\"output_root\": \"$avic_output_root\""* && \
          "$avic_health" != *"\"output_root\":\"$avic_output_root\""* ]]; then
        echo "Port $AVIC_BRIDGE_PORT is using a different AVIC output directory." >&2
        echo "Expected: $avic_output_root" >&2
        echo "Health: $avic_health" >&2
        return 1
    fi
    echo "  AVIC bridge ready: score-band G1 + batched horizon selector + SVC"
}

if [[ "$avic_mode" == "avic" ]]; then
    case "$avic_service_url" in
        http://127.0.0.1:8091|http://localhost:8091)
            ensure_local_avic_stack
            ;;
        *)
            echo "[AVIC] Using external bridge: $avic_service_url"
            curl -fsS --max-time 30 "${avic_service_url%/}/health" >/dev/null || {
                echo "External AVIC bridge is not healthy: $avic_service_url" >&2
                exit 1
            }
            ;;
    esac
fi

for port in 12181 12182 12183 12184 11311; do
    if port_open "$port"; then
        echo "Port $port is already in use. Stop the previous ApexNav/ROS process first." >&2
        exit 1
    fi
done

echo "[ApexNav] Starting four VLM services..."
start_background grounding_dino python -u -m vlm.detector.grounding_dino --port 12181
start_background blip2itm python -u -m vlm.itm.blip2itm --port 12182
start_background mobile_sam python -u -m vlm.segmentor.sam --port 12183
start_background yolov7 python -u -m vlm.detector.yolov7 --port 12184

wait_for_port grounding_dino 12181
wait_for_port blip2itm 12182
wait_for_port mobile_sam 12183
wait_for_port yolov7 12184

echo "[ApexNav] Starting ROS exploration nodes..."
start_background ros roslaunch exploration_manager exploration.launch
wait_for_port ros 11311

ros_deadline=$((SECONDS + startup_timeout))
while true; do
    ros_nodes="$(timeout 5 rosnode list 2>/dev/null || true)"
    if grep -qx '/exploration_node' <<<"$ros_nodes" && grep -qx '/tsp_solver' <<<"$ros_nodes"; then
        break
    fi
    if ! kill -0 "${service_pids[ros]}" 2>/dev/null; then
        echo "ROS exited before its nodes became ready." >&2
        tail -n 120 "${service_logs[ros]}" >&2 || true
        exit 1
    fi
    if ((SECONDS >= ros_deadline)); then
        echo "Timed out waiting for /exploration_node and /tsp_solver." >&2
        tail -n 120 "${service_logs[ros]}" >&2 || true
        exit 1
    fi
    sleep 1
done
echo "  ROS ready: /exploration_node, /tsp_solver"

eval_command=(
    python -u habitat_evaluation.py
    --dataset "$dataset"
    --avic-mode "$avic_mode"
    --avic-service-url "$avic_service_url"
    --avic-timeout "$avic_timeout"
    --avic-warmup-steps "$avic_warmup_steps"
    --avic-max-calls "$avic_max_calls"
    --avic-cooldown-steps "$avic_cooldown_steps"
    --avic-score-low "$avic_score_low"
    --avic-score-high "$avic_score_high"
    --avic-max-score-gap "$avic_max_score_gap"
    --avic-real-weight "$avic_real_weight"
    --avic-imagination-weight "$avic_imagination_weight"
    --avic-min-score-spread "$avic_min_score_spread"
    --avic-cache-radius "$avic_cache_radius"
    --avic-cache-yaw-deg "$avic_cache_yaw_deg"
    --avic-anchor-fov-deg "$avic_anchor_fov_deg"
    --avic-anchor-max-distance "$avic_anchor_max_distance"
    --avic-anchor-min-quality "$avic_anchor_min_quality"
    --avic-max-bonus "$avic_max_bonus"
    --avic-min-reliability "$avic_min_reliability"
    --avic-min-evidence-gain "$avic_min_evidence_gain"
    --max-episodes "$max_episodes"
)
if [[ -n "$episode_manifest" ]]; then
    eval_command+=(--episode-manifest "$episode_manifest")
fi
eval_command+=(--result-dir "$result_dir")
if [[ "$run_full" == false ]]; then
    eval_command+=("test_epi_num=${episode}")
fi
if [[ "$need_video" == true ]]; then
    eval_command+=("need_video=true")
fi

echo "[ApexNav] Running Habitat evaluation..."
printf '  Command:'
printf ' %q' "${eval_command[@]}"
printf '\n'
echo "  Evaluation log: $log_dir/evaluation.log"
echo "  Observe: tail -f $log_dir/evaluation.log"
printf '\n'

set +e
if [[ -t 1 ]]; then
    # Do not stream verbose per-step output into a VSCode/code-server PTY.
    # Its 100k unacknowledged-character flow-control threshold pauses the PTY
    # while the browser is disconnected, which back-pressures Habitat itself.
    "${eval_command[@]}" >"$log_dir/evaluation.log" 2>&1
    evaluation_status=$?
else
    # Detached/non-interactive launches may safely mirror output because their
    # stdout is a regular log file rather than a terminal managed by ptyHost.
    "${eval_command[@]}" 2>&1 | tee "$log_dir/evaluation.log"
    evaluation_status=${PIPESTATUS[0]}
fi
set -e

if ((evaluation_status == 0)); then
    echo "Evaluation completed successfully."
else
    echo "Evaluation failed with exit code $evaluation_status." >&2
fi
exit "$evaluation_status"
