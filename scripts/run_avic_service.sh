#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="${APEXNAV_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
readonly AVIC_ENV_FILE="${AVIC_ENV_FILE:-$PROJECT_ROOT/config/avic.env}"

if [[ -f "$AVIC_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$AVIC_ENV_FILE"
fi

readonly AVIC_PYTHON="${AVIC_PYTHON:-python}"

backend="${AVIC_BACKEND:-mock}"
host="${AVIC_SERVICE_HOST:-127.0.0.1}"
port="${AVIC_SERVICE_PORT:-8091}"
device="${AVIC_SVC_DEVICE:-cuda:0}"

usage() {
    printf '%s\n' \
        "Usage: ./scripts/run_avic_service.sh [options]" \
        "" \
        "Options:" \
        "  --backend mock|svc_qwen       Backend (default: mock)" \
        "  --host ADDRESS                Listen address (default: 127.0.0.1)" \
        "  --port PORT                   Listen port (default: 8091)" \
        "  --device DEVICE               SVC torch device (default: cuda:0)" \
        "  -h, --help                    Show this help"
}

while (($#)); do
    case "$1" in
        --backend) backend="$2"; shift 2 ;;
        --host) host="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$backend" in mock|svc_qwen) ;; *) printf 'Invalid backend: %s\n' "$backend" >&2; exit 2 ;; esac
[[ "$port" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid port: %s\n' "$port" >&2; exit 2; }

command -v "$AVIC_PYTHON" >/dev/null 2>&1 || {
    printf 'AVIC Python not found: %s\n' "$AVIC_PYTHON" >&2
    exit 1
}

if [[ "$backend" == "svc_qwen" && -z "${AVIC_ROOT:-}" ]]; then
    printf 'AVIC_ROOT is required for backend svc_qwen. Configure %s first.\n' \
        "$AVIC_ENV_FILE" >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT"
export AVIC_BACKEND="$backend"
export AVIC_SERVICE_HOST="$host"
export AVIC_SERVICE_PORT="$port"
export AVIC_SVC_DEVICE="$device"
export HF_HOME="${HF_HOME:-${AVIC_ROOT:-$PROJECT_ROOT}/.runtime/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export SVC_MODEL_DIR="${SVC_MODEL_DIR:-${AVIC_ROOT:-$PROJECT_ROOT}/.runtime/models/stable-virtual-camera}"
export SVC_VAE_DIR="${SVC_VAE_DIR:-${AVIC_ROOT:-$PROJECT_ROOT}/.runtime/models/stable-diffusion-2-1-base}"
export AVIC_OUTPUT_ROOT="${AVIC_OUTPUT_ROOT:-$PROJECT_ROOT/experiments/runs/avic}"

mkdir -p "$AVIC_OUTPUT_ROOT" || {
    printf 'Cannot create AVIC output directory: %s\n' "$AVIC_OUTPUT_ROOT" >&2
    exit 1
}
[[ -d "$AVIC_OUTPUT_ROOT" && -w "$AVIC_OUTPUT_ROOT" ]] || {
    printf 'AVIC output directory is not writable by %s: %s\n' \
        "$(id -un)" "$AVIC_OUTPUT_ROOT" >&2
    exit 1
}

if [[ "$backend" == "svc_qwen" ]]; then
    [[ -s "$SVC_MODEL_DIR/model.safetensors" ]] || {
        printf 'SVC checkpoint missing: %s\n' "$SVC_MODEL_DIR/model.safetensors" >&2
        exit 1
    }
    [[ -s "$SVC_VAE_DIR/vae/diffusion_pytorch_model.safetensors" ]] || {
        printf 'SVC VAE missing: %s\n' "$SVC_VAE_DIR/vae/diffusion_pytorch_model.safetensors" >&2
        exit 1
    }
fi

cd "$PROJECT_ROOT"
printf 'Starting AVIC bridge: backend=%s binding=per_frontier_real_directional_anchor gate_contract=frontier_anchor_multiframe_verifier_v1 internal_gate=apexnav_middle_score_band scorer=qwen_multiframe_verifier address=%s:%s\n' \
    "$backend" "$host" "$port"
exec "$AVIC_PYTHON" -u -m avic_bridge.service
