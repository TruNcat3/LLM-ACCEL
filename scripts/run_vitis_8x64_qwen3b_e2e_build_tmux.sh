#!/usr/bin/env bash
set -euo pipefail

script_path="$(realpath "$0")"
cd "$(dirname "${script_path}")/.."

phase="${1:-all}"
case "${phase}" in
    compute-xo|control-xo|status-xo|link|host|all) ;;
    *)
        echo "usage: $0 [compute-xo|control-xo|status-xo|link|host|all]" >&2
        exit 2
        ;;
esac

device="${DEVICE:-xilinx_u50_gen3x16_xdma_5_202210_1}"
root="${VITIS_8X64_QWEN3B_WORK_ROOT:-/tmp/llm_accel_qwen3b_q214_resident_fix}"
scratch_root="${VITIS_8X64_QWEN3B_TMP_ROOT:-${root}/tmp}"
threads="${THREADS:-8}"
min_available_gib="${VITIS_MIN_AVAILABLE_GIB:-80}"
min_tmp_gib="${VITIS_MIN_TMP_GIB:-100}"

if [ "${2:-}" = "--worker" ]; then
    echo "started_at=$(date -Is)"
    echo "phase=${phase}"
    echo "work_root=${root}"
    echo "tmp_root=${scratch_root}"
    echo "compute_profile=qwen2.5-3b"
    echo "threads=${threads}"
    echo "build_source_manifest_begin=1"
    while IFS=$'\t' read -r path sha bytes role; do
        [ "${path}" = "path" ] && continue
        printf 'build_source\t%s\t%s\t%s\t%s\n' \
            "${path}" "${sha}" "${bytes}" "${role}"
    done < <(scripts/report_qwen3b_source_snapshot.sh "$PWD")
    echo "build_source_manifest_end=1"
    mkdir -p "${scratch_root}"
    export TMPDIR="${scratch_root}"
    # MAX_SEQ_LEN contributes to the compute service-loop bound. A qwen-layer
    # XO is only compiled for 96 positions and can stop reading before a
    # long-context qwen2.5-3b Task 18 emits last_task. Build the compute XO
    # with the same 2048-position profile as the controller.
    set +e
    VITIS_8X64_MODEL_PROFILE=qwen2.5-3b \
    VITIS_8X64_RESIDENT_VARIANT_TAG=block_prefill_q214_resident_fix_e2e \
    CC8_RESIDENT_TOKEN_ROWS=8 \
    CC8_WEIGHT_TILE_FIFO_DEPTH=2 \
    CC8_WEIGHT_TILE_LOAD_II=2 \
    CC8_ENABLE_MM_WAVE_REPEAT=0 \
    CC8_MM_WAVE_RESULT_FIFO_DEPTH=33 \
    CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW=1 \
    VITIS_8X64_BUILD_EXACT_COMPUTE_XO=1 \
    VITIS_8X64_RESIDENT_XO_DIR="${root}/xo" \
    VITIS_8X64_BUILD_DIR="${root}/build.hw_emu.${device}" \
    VITIS_8X64_TEMP_DIR="${root}/temp.hw_emu.${device}" \
    VITIS_8X64_REPORT_DIR="${root}/reports.hw_emu.${device}" \
    VITIS_8X64_HLS_PROJECT_ROOT="${root}/hls" \
    VITIS_8X64_CONN_CFG=conn_u50_8x64_dual_full_resident.cfg \
    THREADS="${threads}" \
        scripts/build_vitis_8x64_resident_layer_hwemu.sh "${phase}"
    status="$?"
    set -e
    echo "finished_at=$(date -Is)"
    echo "exit_status=${status}"
    exit "${status}"
fi

available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
required_kib=$((min_available_gib * 1024 * 1024))
if [ "${available_kib}" -lt "${required_kib}" ]; then
    echo "Need at least ${min_available_gib} GiB available memory" >&2
    exit 75
fi
mkdir -p "${root}" "${scratch_root}" logs
tmp_available_kib="$(df -Pk "${scratch_root}" | awk 'NR == 2 {print $4}')"
tmp_required_kib=$((min_tmp_gib * 1024 * 1024))
if [ "${tmp_available_kib}" -lt "${tmp_required_kib}" ]; then
    echo "Need at least ${min_tmp_gib} GiB free under ${scratch_root}" >&2
    exit 75
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
session_name="llm_qwen3b_${phase//[^a-zA-Z0-9]/_}_${timestamp}"
log_path="$PWD/logs/qwen3b_e2e_${phase}_${timestamp}.log"
pid_path="${log_path%.log}.pid"

printf -v worker_command \
    'exec env DEVICE=%q VITIS_8X64_QWEN3B_WORK_ROOT=%q VITIS_8X64_QWEN3B_TMP_ROOT=%q THREADS=%q VITIS_MIN_AVAILABLE_GIB=%q VITIS_MIN_TMP_GIB=%q %q %q --worker >>%q 2>&1' \
    "${device}" "${root}" "${scratch_root}" "${threads}" \
    "${min_available_gib}" "${min_tmp_gib}" "${script_path}" "${phase}" \
    "${log_path}"
tmux new-session -d -s "${session_name}" "${worker_command}"
pid="$(tmux display-message -p -t "${session_name}:0.0" '#{pane_pid}')"
printf '%s\n' "${pid}" >"${pid_path}"
printf 'pid=%s\nsession=%s\nlog=%s\npidfile=%s\nwork_root=%s\n' \
    "${pid}" "${session_name}" "${log_path}" "${pid_path}" "${root}"
