#!/usr/bin/env bash
set -euo pipefail

script_path="$(realpath "$0")"
cd "$(dirname "${script_path}")/.."

device="${DEVICE:-xilinx_u50_gen3x16_xdma_5_202210_1}"
profile="${VITIS_8X64_MODEL_PROFILE:-qwen-layer}"
fifo_depth="${CC8_WEIGHT_TILE_FIFO_DEPTH:-2}"
load_ii="${CC8_WEIGHT_TILE_LOAD_II:-2}"
wave_repeat="${CC8_ENABLE_MM_WAVE_REPEAT:-0}"
wave_result_depth="${CC8_MM_WAVE_RESULT_FIFO_DEPTH:-33}"
cross_wave_dataflow="${CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW:-1}"
resident_token_rows="${CC8_RESIDENT_TOKEN_ROWS:-8}"
variant_tag="${VITIS_8X64_RESIDENT_VARIANT_TAG:-block_prefill_q214_resident_fix}"
profile_tag="${profile//./_}"
profile_tag="${profile_tag//-/_}"
variant_tag="${variant_tag//./_}"
variant_tag="${variant_tag//-/_}"
tag="${profile_tag}.resident_layer.t${resident_token_rows}.d${fifo_depth}.ii${load_ii}.r${wave_repeat}.wr${wave_result_depth}.cw${cross_wave_dataflow}.${variant_tag}"
build_dir="${VITIS_8X64_BUILD_DIR:-vitis_8x64/build.${tag}.hw_emu.${device}}"
host_exe="${VITIS_8X64_DIAG_HOST_EXE:-${build_dir}/host_qwen_8x64.exe}"
xclbin="${build_dir}/qwen_8x64_dual.xclbin"
emconfig="${build_dir}/emconfig.json"
seed="${VITIS_8X64_RESIDENT_SEED:-20260718}"
token_count="${VITIS_8X64_DIAG_QUERY_TOKENS:-2}"
timeout_seconds="${VITIS_8X64_HW_EMU_TIMEOUT:-21600}"
env_script="${VITIS_ENV_SCRIPT:-/home/hepc/env/vitis_env_22.sh}"

if [ "${1:-}" = "--worker" ]; then
    for input in "${env_script}" "${host_exe}" "${xclbin}" "${emconfig}"; do
        if [ ! -s "${input}" ]; then
            echo "Missing or empty P${token_count} attention diagnostic input: ${input}" >&2
            exit 66
        fi
    done
    source "${env_script}" >/dev/null 2>&1
    echo "started_at=$(date -Is)"
    echo "gate=standard_qwen_composed_attention_p${token_count}"
    echo "profile=${profile}"
    echo "active_query_rows=${token_count}"
    echo "seed=${seed}"
    echo "timeout_seconds=${timeout_seconds}"
    echo "host_exe_sha256=$(sha256sum "${host_exe}" | awk '{print $1}')"
    echo "xclbin_sha256=$(sha256sum "${xclbin}" | awk '{print $1}')"
    echo "emconfig_sha256=$(sha256sum "${emconfig}" | awk '{print $1}')"
    host_abs="$(realpath "${host_exe}")"
    build_abs="$(realpath "${build_dir}")"
    cd "${build_abs}"
    set +e
    XCL_EMULATION_MODE=hw_emu \
    EMCONFIG_PATH="${build_abs}" \
        timeout "${timeout_seconds}" \
        "${host_abs}" \
        --xclbin ./qwen_8x64_dual.xclbin \
        --mode verify-composed-prefill-attention \
        --profile "${profile}" \
        --random-model \
        --seed "${seed}" \
        --position 0 \
        --prefill-block-size "${token_count}"
    status="$?"
    set -e
    echo "finished_at=$(date -Is)"
    echo "exit_status=${status}"
    exit "${status}"
fi

if [ "${profile}" != "qwen-layer" ]; then
    echo "The P${token_count} numerical diagnostic requires VITIS_8X64_MODEL_PROFILE=qwen-layer" >&2
    exit 2
fi
if [ "${resident_token_rows}" != "8" ]; then
    echo "The P${token_count} numerical diagnostic requires CC8_RESIDENT_TOKEN_ROWS=8" >&2
    exit 2
fi
if ! [[ "${token_count}" =~ ^[0-9]+$ ]] ||
   [ "${token_count}" -lt 2 ] || [ "${token_count}" -gt 8 ]; then
    echo "VITIS_8X64_DIAG_QUERY_TOKENS must be in 2..8" >&2
    exit 2
fi
if ! [[ "${timeout_seconds}" =~ ^[0-9]+$ ]] ||
   [ "${timeout_seconds}" -lt 1 ]; then
    echo "VITIS_8X64_HW_EMU_TIMEOUT must be positive" >&2
    exit 2
fi
for input in "${env_script}" "${host_exe}" "${xclbin}" "${emconfig}"; do
    if [ ! -s "${input}" ]; then
        echo "Missing or empty P${token_count} attention diagnostic input: ${input}" >&2
        exit 66
    fi
done
if pgrep -af '[h]ost_qwen_8x64.*verify-composed-prefill-attention' >/dev/null; then
    echo "A composed-prefill attention diagnostic already appears to be running" >&2
    pgrep -af '[h]ost_qwen_8x64.*verify-composed-prefill-attention' >&2 || true
    exit 70
fi

mkdir -p logs
timestamp="$(date +%Y%m%d_%H%M%S)"
session="llm_qwen_attention_p${token_count}_diag_${timestamp}"
log_path="$PWD/logs/qwen_attention_p${token_count}_diag_${timestamp}.log"
pid_path="${log_path%.log}.pid"
tmux new-session -d -s "${session}" \
    "exec '${script_path}' --worker >>'${log_path}' 2>&1"
pid="$(tmux list-panes -t "${session}" -F '#{pane_pid}' | head -n 1)"
printf '%s\n' "${pid}" > "${pid_path}"
printf 'pid=%s\nsession=%s\nlog=%s\npidfile=%s\n' \
    "${pid}" "${session}" "${log_path}" "${pid_path}"
