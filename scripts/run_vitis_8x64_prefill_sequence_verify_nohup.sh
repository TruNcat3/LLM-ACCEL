#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

profile="${VITIS_8X64_MODEL_PROFILE:-small}"
sequence_tokens="${VITIS_8X64_VERIFY_SEQUENCE_TOKENS:-16}"
block_size="${VITIS_8X64_VERIFY_BLOCK_SIZE:-8}"
seed="${VITIS_8X64_RESIDENT_SEED:-20260718}"
timeout_seconds="${VITIS_8X64_HW_EMU_TIMEOUT:-43200}"
device="${DEVICE:-xilinx_u50_gen3x16_xdma_5_202210_1}"
env_script="${VITIS_ENV_SCRIPT:-/home/hepc/env/vitis_env_22.sh}"
build_dir="${VITIS_8X64_VERIFY_BUILD_DIR:-vitis_8x64/build.small.resident_layer.t8.d2.ii2.r0.wr33.cw1.block_prefill_q214_exact.hw_emu.${device}}"
host_exe="${VITIS_8X64_VERIFY_HOST_EXE:-${build_dir}/host_qwen_8x64.exe}"
xclbin="${build_dir}/qwen_8x64_dual.xclbin"
emconfig="${build_dir}/emconfig.json"

if [ "${1:-}" = "--worker" ]; then
    for input in "${env_script}" "${host_exe}" "${xclbin}" "${emconfig}"; do
        if [ ! -s "${input}" ]; then
            echo "Missing or empty sequence-verification input: ${input}" >&2
            exit 66
        fi
    done
    source "${env_script}" >/dev/null 2>&1
    echo "started_at=$(date -Is)"
    echo "profile=${profile}"
    echo "sequence_tokens=${sequence_tokens}"
    echo "block_size=${block_size}"
    echo "seed=${seed}"
    echo "host_exe=${host_exe}"
    echo "xclbin=${xclbin}"
    host_exe_abs="$(realpath "${host_exe}")"
    cd "${build_dir}"
    set +e
    XCL_EMULATION_MODE=hw_emu \
    EMCONFIG_PATH="$PWD" \
    XRT_INI_PATH="$PWD/xrt.ini" \
        timeout "${timeout_seconds}" \
        "${host_exe_abs}" \
        --xclbin ./qwen_8x64_dual.xclbin \
        --mode verify-composed-prefill-sequence \
        --profile "${profile}" \
        --random-model \
        --seed "${seed}" \
        --prefill-len "${sequence_tokens}" \
        --prefill-block-size "${block_size}"
    status="$?"
    set -e
    echo "finished_at=$(date -Is)"
    echo "exit_status=${status}"
    exit "${status}"
fi

if [ "${sequence_tokens}" -lt 1 ]; then
    echo "VITIS_8X64_VERIFY_SEQUENCE_TOKENS must be positive" >&2
    exit 2
fi
if [ "${block_size}" -lt 1 ] || [ "${block_size}" -gt 8 ]; then
    echo "VITIS_8X64_VERIFY_BLOCK_SIZE must be in 1..8" >&2
    exit 2
fi

mkdir -p logs
timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="$PWD/logs/prefill_sequence_${profile}_p${sequence_tokens}_b${block_size}_${timestamp}.log"
pid_path="${log_path%.log}.pid"
session_name="llm_prefill_seq_${profile//[^a-zA-Z0-9]/_}_p${sequence_tokens}_${timestamp}"

tmux new-session -d -s "${session_name}" \
    "exec '$PWD/$0' --worker >>'${log_path}' 2>&1"
pid="$(tmux display-message -p -t "${session_name}:0.0" '#{pane_pid}')"
printf '%s\n' "${pid}" >"${pid_path}"
printf 'pid=%s\nsession=%s\nlog=%s\npidfile=%s\n' \
    "${pid}" "${session_name}" "${log_path}" "${pid_path}"
