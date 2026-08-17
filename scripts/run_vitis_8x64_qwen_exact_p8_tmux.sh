#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Re-run the standard-dimension, exact-compute P8 numerical gate under a real
# tmux session.  Keeping the Host/XRT process group attached to tmux avoids the
# simulator teardown seen when a long HW Emu job outlives a transient shell.
export VITIS_8X64_MODEL_PROFILE="${VITIS_8X64_MODEL_PROFILE:-qwen-layer}"
export CC8_WEIGHT_TILE_FIFO_DEPTH="${CC8_WEIGHT_TILE_FIFO_DEPTH:-2}"
export CC8_WEIGHT_TILE_LOAD_II="${CC8_WEIGHT_TILE_LOAD_II:-2}"
export CC8_ENABLE_MM_WAVE_REPEAT="${CC8_ENABLE_MM_WAVE_REPEAT:-0}"
export CC8_MM_WAVE_RESULT_FIFO_DEPTH="${CC8_MM_WAVE_RESULT_FIFO_DEPTH:-33}"
export CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW="${CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW:-1}"
export CC8_RESIDENT_TOKEN_ROWS="${CC8_RESIDENT_TOKEN_ROWS:-8}"
export FREQUENCY="${FREQUENCY:-200}"
export VITIS_8X64_HWEMU_DEBUG="${VITIS_8X64_HWEMU_DEBUG:-0}"
export VITIS_8X64_RESIDENT_VARIANT_TAG="${VITIS_8X64_RESIDENT_VARIANT_TAG:-block_prefill_q214}"
export VITIS_8X64_BUILD_EXACT_COMPUTE_XO="${VITIS_8X64_BUILD_EXACT_COMPUTE_XO:-1}"
export VITIS_8X64_RESIDENT_SEED="${VITIS_8X64_RESIDENT_SEED:-20260718}"
export VITIS_8X64_HW_EMU_TIMEOUT="${VITIS_8X64_HW_EMU_TIMEOUT:-43200}"

if [ "${1:-}" = "--worker" ]; then
    echo "started_at=$(date -Is)"
    echo "gate=standard_qwen_exact_p8"
    echo "session_owner=tmux"
    set +e
    scripts/build_vitis_8x64_resident_layer_hwemu.sh run-block
    status="$?"
    set -e
    echo "finished_at=$(date -Is)"
    echo "exit_status=${status}"
    exit "${status}"
fi

if [ "${VITIS_8X64_MODEL_PROFILE}" != "qwen-layer" ]; then
    echo "This numerical gate requires VITIS_8X64_MODEL_PROFILE=qwen-layer." >&2
    exit 2
fi
if [ "${CC8_RESIDENT_TOKEN_ROWS}" != "8" ]; then
    echo "This numerical gate requires CC8_RESIDENT_TOKEN_ROWS=8." >&2
    exit 2
fi
if [ "${VITIS_8X64_BUILD_EXACT_COMPUTE_XO}" != "1" ]; then
    echo "This numerical gate requires the exact compute XO." >&2
    exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required for the long-running HW Emu gate." >&2
    exit 66
fi

profile_tag="qwen_layer"
variant_tag="${VITIS_8X64_RESIDENT_VARIANT_TAG//./_}"
variant_tag="${variant_tag//-/_}"
tag="${profile_tag}.resident_layer.t8.d${CC8_WEIGHT_TILE_FIFO_DEPTH}.ii${CC8_WEIGHT_TILE_LOAD_II}.r${CC8_ENABLE_MM_WAVE_REPEAT}.wr${CC8_MM_WAVE_RESULT_FIFO_DEPTH}.cw${CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW}.${variant_tag}"
device="${DEVICE:-xilinx_u50_gen3x16_xdma_5_202210_1}"
build_dir="${VITIS_8X64_BUILD_DIR:-vitis_8x64/build.${tag}_exact.hw_emu.${device}}"
export VITIS_8X64_BUILD_DIR="${build_dir}"

for input in \
    "${build_dir}/qwen_8x64_dual.xclbin" \
    "${build_dir}/host_qwen_8x64.exe" \
    "${build_dir}/emconfig.json"
do
    if [ ! -s "${input}" ]; then
        echo "Missing or empty exact-P8 HW Emu input: ${input}" >&2
        exit 66
    fi
done

if pgrep -af 'host_qwen_8x64.*verify-composed-prefill-block.*profile qwen-layer' >/dev/null; then
    echo "A standard Qwen exact-P8 Host already appears to be running." >&2
    pgrep -af 'host_qwen_8x64.*verify-composed-prefill-block.*profile qwen-layer' >&2
    exit 70
fi

mkdir -p logs
timestamp="$(date +%Y%m%d_%H%M%S)"
session="llm_qwen_exact_p8_${timestamp}"
log_path="$PWD/logs/qwen_exact_p8_hwemu_tmux_${timestamp}.log"
pid_path="$PWD/logs/qwen_exact_p8_hwemu_tmux_${timestamp}.pid"
worker_env=(
    "VITIS_8X64_MODEL_PROFILE=${VITIS_8X64_MODEL_PROFILE}"
    "CC8_WEIGHT_TILE_FIFO_DEPTH=${CC8_WEIGHT_TILE_FIFO_DEPTH}"
    "CC8_WEIGHT_TILE_LOAD_II=${CC8_WEIGHT_TILE_LOAD_II}"
    "CC8_ENABLE_MM_WAVE_REPEAT=${CC8_ENABLE_MM_WAVE_REPEAT}"
    "CC8_MM_WAVE_RESULT_FIFO_DEPTH=${CC8_MM_WAVE_RESULT_FIFO_DEPTH}"
    "CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW=${CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW}"
    "CC8_RESIDENT_TOKEN_ROWS=${CC8_RESIDENT_TOKEN_ROWS}"
    "FREQUENCY=${FREQUENCY}"
    "VITIS_8X64_HWEMU_DEBUG=${VITIS_8X64_HWEMU_DEBUG}"
    "VITIS_8X64_RESIDENT_VARIANT_TAG=${VITIS_8X64_RESIDENT_VARIANT_TAG}"
    "VITIS_8X64_BUILD_EXACT_COMPUTE_XO=${VITIS_8X64_BUILD_EXACT_COMPUTE_XO}"
    "VITIS_8X64_RESIDENT_SEED=${VITIS_8X64_RESIDENT_SEED}"
    "VITIS_8X64_HW_EMU_TIMEOUT=${VITIS_8X64_HW_EMU_TIMEOUT}"
    "VITIS_8X64_BUILD_DIR=${VITIS_8X64_BUILD_DIR}"
)
worker="exec env"
for assignment in "${worker_env[@]}"; do
    printf -v quoted_assignment '%q' "${assignment}"
    worker+=" ${quoted_assignment}"
done
printf -v quoted_script '%q' "$PWD/scripts/run_vitis_8x64_qwen_exact_p8_tmux.sh"
printf -v quoted_log '%q' "${log_path}"
worker+=" ${quoted_script} --worker >>${quoted_log} 2>&1"

tmux new-session -d -s "${session}" "${worker}"
pid="$(tmux list-panes -t "${session}" -F '#{pane_pid}' | head -n 1)"
printf '%s\n' "${pid}" > "${pid_path}"

printf 'pid=%s\nsession=%s\nlog=%s\npidfile=%s\nbuild_dir=%s\n' \
    "${pid}" "${session}" "${log_path}" "${pid_path}" "${build_dir}"
