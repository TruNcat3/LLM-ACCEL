#!/usr/bin/env bash
set -euo pipefail

script_path="$(realpath "$0")"
cd "$(dirname "${script_path}")/.."

work_root="${VITIS_8X64_QWEN3B_WORK_ROOT:-/tmp/llm_accel_qwen3b_q214_resident_fix}"
device="${DEVICE:-xilinx_u50_gen3x16_xdma_5_202210_1}"
build_dir="${work_root}/build.hw_emu.${device}"
host_exe="${VITIS_8X64_CHECKPOINT_HOST_EXE:-/tmp/host_qwen_8x64_checkpoint.exe}"
xclbin="${build_dir}/qwen_8x64_dual.xclbin"
emconfig="${build_dir}/emconfig.json"
env_script="${VITIS_ENV_SCRIPT:-}"
prompt_tokens="${VITIS_8X64_CHECKPOINT_TOKENS:-8}"
layers="${VITIS_8X64_CHECKPOINT_LAYERS:-3}"
tolerance="${VITIS_8X64_CHECKPOINT_TOLERANCE:-0}"
continue_on_failure="${VITIS_8X64_CHECKPOINT_CONTINUE_ON_FAILURE:-0}"
seed="${VITIS_8X64_RESIDENT_SEED:-20260718}"
timeout_seconds="${VITIS_8X64_HW_EMU_TIMEOUT:-172800}"
min_available_gib="${VITIS_MIN_AVAILABLE_GIB:-80}"

# Resolve the reproducible public-repository environment before the worker is
# placed in tmux.  VITIS_ENV_SCRIPT may still be supplied explicitly for a
# site-specific Vitis installation.
source scripts/setup_environment.sh >/dev/null
env_script="${VITIS_ENV_SCRIPT}"

if [ "${1:-}" = "--worker" ]; then
    for input in "${env_script}" "${host_exe}" "${xclbin}" "${emconfig}"; do
        if [ ! -s "${input}" ]; then
            echo "Missing or empty checkpoint HW-Emu input: ${input}" >&2
            exit 66
        fi
    done
    source "${env_script}" >/dev/null 2>&1
    if ! rg -a -q -- 'verify-composed-prefill-checkpoints' "${host_exe}" ||
       ! rg -a -q -- 'checkpoint-tolerance' "${host_exe}" ||
       ! rg -a -q -- 'checkpoint-continue-on-failure' "${host_exe}"; then
        echo "Host executable does not contain the checkpoint diagnostic mode" >&2
        exit 65
    fi

    token_csv=""
    for ((token = 0; token < prompt_tokens; token++)); do
        if [ -n "${token_csv}" ]; then
            token_csv+=","
        fi
        token_csv+="${token}"
    done

    echo "started_at=$(date -Is)"
    echo "gate=qwen3b_prefill_checkpoint_localization"
    echo "target=hw_emu"
    echo "profile=qwen2.5-3b"
    echo "prompt_tokens=${prompt_tokens}"
    echo "layers=${layers}"
    echo "expected_checkpoints=$((2 * layers + 1))"
    echo "seed=${seed}"
    echo "strict_tolerance=0"
    echo "numeric_tolerance=${tolerance}"
    echo "acceptance_policy=max_abs_raw_error"
    echo "continue_on_failure=${continue_on_failure}"
    echo "checkpoint_host_copy=1"
    echo "checkpoint_copy_scope=diagnostic_only_excluded_from_production_path"
    echo "timeout_seconds=${timeout_seconds}"
    echo "host_exe=$(realpath "${host_exe}")"
    echo "host_exe_sha256=$(sha256sum "${host_exe}" | awk '{print $1}')"
    echo "xclbin_sha256=$(sha256sum "${xclbin}" | awk '{print $1}')"
    echo "emconfig_sha256=$(sha256sum "${emconfig}" | awk '{print $1}')"

    host_exe_abs="$(realpath "${host_exe}")"
    build_dir_abs="$(realpath "${build_dir}")"
    continue_args=()
    if [ "${continue_on_failure}" -eq 1 ]; then
        continue_args+=(--checkpoint-continue-on-failure)
    fi
    cd "${build_dir_abs}"
    set +e
    XCL_EMULATION_MODE=hw_emu \
    EMCONFIG_PATH="${build_dir_abs}" \
        timeout "${timeout_seconds}" \
        "${host_exe_abs}" \
        --xclbin ./qwen_8x64_dual.xclbin \
        --mode verify-composed-prefill-checkpoints \
        --profile qwen2.5-3b \
        --random-model \
        --seed "${seed}" \
        --tokens "${token_csv}" \
        --layers "${layers}" \
        --checkpoint-tolerance "${tolerance}" \
        "${continue_args[@]}"
    host_status="$?"
    set -e
    echo "finished_at=$(date -Is)"
    echo "host_exit_status=${host_status}"
    exit "${host_status}"
fi

dry_run=0
if [ "${1:-}" = "--dry-run" ]; then
    dry_run=1
    shift
fi
if [ "$#" -ne 0 ]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

for value_name in prompt_tokens layers tolerance continue_on_failure seed timeout_seconds min_available_gib; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "${value_name} must be a non-negative integer" >&2
        exit 2
    fi
done
if [ "${prompt_tokens}" -lt 1 ] || [ "${prompt_tokens}" -gt 8 ]; then
    echo "VITIS_8X64_CHECKPOINT_TOKENS must be in 1..8" >&2
    exit 2
fi
if [ "${continue_on_failure}" -gt 1 ]; then
    echo "VITIS_8X64_CHECKPOINT_CONTINUE_ON_FAILURE must be 0 or 1" >&2
    exit 2
fi
if [ "${layers}" -lt 1 ] || [ "${layers}" -gt 36 ]; then
    echo "VITIS_8X64_CHECKPOINT_LAYERS must be in 1..36" >&2
    exit 2
fi
if [ "${tolerance}" -gt 32767 ]; then
    echo "VITIS_8X64_CHECKPOINT_TOLERANCE must be in 0..32767" >&2
    exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required for the long-running HW-Emu diagnostic" >&2
    exit 66
fi

if [ "${dry_run}" -eq 0 ]; then
    available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    if [ "${available_kib}" -lt $((min_available_gib * 1024 * 1024)) ]; then
        echo "Need at least ${min_available_gib} GiB available memory" >&2
        exit 75
    fi
    for input in "${env_script}" "${host_exe}" "${xclbin}" "${emconfig}"; do
        if [ ! -s "${input}" ]; then
            echo "Missing or empty checkpoint HW-Emu input: ${input}" >&2
            exit 66
        fi
    done
    if pgrep -af '[h]ost_qwen_8x64.*verify-composed-prefill-checkpoints' >/dev/null ||
       pgrep -af '[x]simk' >/dev/null; then
        echo "Another checkpoint Host or XSim process is already running" >&2
        pgrep -af '[h]ost_qwen_8x64.*verify-composed-prefill-checkpoints|[x]simk' >&2 || true
        exit 70
    fi
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
tolerance_tag=""
if [ "${tolerance}" -ne 0 ]; then
    tolerance_tag="_tol${tolerance}"
fi
if [ "${continue_on_failure}" -eq 1 ]; then
    tolerance_tag+="_continue"
fi
session="llm_qwen3b_checkpoint_p${prompt_tokens}_l${layers}${tolerance_tag}_${timestamp}"
log_path="$PWD/logs/qwen3b_checkpoint_hwemu_p${prompt_tokens}_l${layers}${tolerance_tag}_${timestamp}.log"
pid_path="${log_path%.log}.pid"
worker_argv=(
    env
    "VITIS_8X64_QWEN3B_WORK_ROOT=${work_root}"
    "DEVICE=${device}"
    "VITIS_8X64_CHECKPOINT_HOST_EXE=${host_exe}"
    "VITIS_8X64_CHECKPOINT_TOKENS=${prompt_tokens}"
    "VITIS_8X64_CHECKPOINT_LAYERS=${layers}"
    "VITIS_8X64_CHECKPOINT_TOLERANCE=${tolerance}"
    "VITIS_8X64_CHECKPOINT_CONTINUE_ON_FAILURE=${continue_on_failure}"
    "VITIS_8X64_RESIDENT_SEED=${seed}"
    "VITIS_8X64_HW_EMU_TIMEOUT=${timeout_seconds}"
    "VITIS_MIN_AVAILABLE_GIB=${min_available_gib}"
    "VITIS_ENV_SCRIPT=${env_script}"
    "${script_path}"
    --worker
)
printf -v worker_command '%q ' "${worker_argv[@]}"
printf -v quoted_log '%q' "${log_path}"

if [ "${dry_run}" -eq 1 ]; then
    printf 'dry_run=1\nprompt_tokens=%s\nlayers=%s\ntolerance=%s\ncontinue_on_failure=%s\nexpected_checkpoints=%s\n' \
        "${prompt_tokens}" "${layers}" "${tolerance}" \
        "${continue_on_failure}" "$((2 * layers + 1))"
    printf 'worker_command=%s\nlog=%s\nbuild_dir=%s\n' \
        "${worker_command}" "${log_path}" "${build_dir}"
    exit 0
fi

mkdir -p logs
tmux new-session -d -s "${session}" \
    "exec ${worker_command} >>${quoted_log} 2>&1"
pid="$(tmux list-panes -t "${session}" -F '#{pane_pid}' | head -n 1)"
printf '%s\n' "${pid}" > "${pid_path}"
printf 'pid=%s\nsession=%s\nlog=%s\npidfile=%s\nbuild_dir=%s\n' \
    "${pid}" "${session}" "${log_path}" "${pid_path}" "${build_dir}"
