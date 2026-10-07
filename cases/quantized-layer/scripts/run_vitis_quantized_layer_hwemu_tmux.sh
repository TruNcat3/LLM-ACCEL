#!/usr/bin/env bash
set -euo pipefail

script_path="$(realpath "$0")"
repo_root="$(cd "$(dirname "${script_path}")/.." && pwd -P)"
cd "${repo_root}"
tmux_bin="${TMUX_BIN:-tmux}"

precision="${QUANTIZED_LAYER_PRECISION:-${1:-w4}}"
if [[ "${precision}" == "w4" ]]; then
    build_script="$PWD/scripts/build_vitis_quantized_w4_layer.sh"
    root="${VITIS_QUANTIZED_W4_LAYER_ROOT:-${COWAVE_QUANTIZED_LAYER_OUTPUT_DIR:-${repo_root}/.build}/w4}"
    prefill="${QUANTIZED_W4_LAYER_PREFILL:-10}"
    block_size="${QUANTIZED_W4_LAYER_BLOCK_SIZE:-8}"
    weight_mode="${QUANTIZED_W4_LAYER_WEIGHTS:-zero}"
    timeout_seconds="${QUANTIZED_W4_LAYER_TIMEOUT:-86400}"
    host_name=host_control_cache_quantized_w4_layer_hwemu.exe
    xclbin_name=quantized_w4_layer.xclbin
elif [[ "${precision}" == "w8" ]]; then
    build_script="$PWD/scripts/build_vitis_quantized_w8_layer.sh"
    root="${VITIS_QUANTIZED_W8_LAYER_ROOT:-${COWAVE_QUANTIZED_LAYER_OUTPUT_DIR:-${repo_root}/.build}/w8}"
    prefill="${QUANTIZED_W8_LAYER_PREFILL:-6}"
    block_size="${QUANTIZED_W8_LAYER_BLOCK_SIZE:-4}"
    weight_mode="${QUANTIZED_W8_LAYER_WEIGHTS:-zero}"
    timeout_seconds="${QUANTIZED_W8_LAYER_TIMEOUT:-86400}"
    host_name=host_control_cache_quantized_w8_layer_hwemu.exe
    xclbin_name=quantized_w8_layer.xclbin
else
    echo "precision must be w4 or w8" >&2
    exit 2
fi

source scripts/quantized_layer_profiles.sh
quantized_layer_profile_apply "$precision"
attention_variant="$QUANTIZED_LAYER_ATTENTION_VARIANT"
worker_silu_lanes=""
[[ "${QUANTIZED_LAYER_SILU_LANES_REQUESTED:-0}" == 1 ]] && \
    worker_silu_lanes="${QUANTIZED_LAYER_SILU_LANES}"
worker_silu_requested=0
[[ -n "${worker_silu_lanes}" ]] && worker_silu_requested=1
worker_rms_lanes=""
[[ "${QUANTIZED_LAYER_RMS_LANES_REQUESTED:-0}" == 1 ]] && \
    worker_rms_lanes="${QUANTIZED_LAYER_RMS_LANES}"
worker_rms_requested=0
[[ -n "${worker_rms_lanes}" ]] && worker_rms_requested=1
worker_weight_outstanding=""
[[ "${QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING_REQUESTED:-0}" == 1 ]] && \
    worker_weight_outstanding="${QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING}"
worker_weight_requested=0
[[ -n "${worker_weight_outstanding}" ]] && worker_weight_requested=1
worker_attention_wave=""
[[ "${QUANTIZED_LAYER_ATTENTION_WAVE_REQUESTED:-0}" == 1 ]] && \
    worker_attention_wave="${QUANTIZED_LAYER_ATTENTION_WAVE}"
worker_wave_requested=0
[[ -n "${worker_attention_wave}" ]] && worker_wave_requested=1
worker_prefill_ffn_overlap=""
[[ "${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED:-0}" == 1 ]] && \
    worker_prefill_ffn_overlap="${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP}"
worker_prefill_ffn_overlap_requested=0
[[ -n "${worker_prefill_ffn_overlap}" ]] && worker_prefill_ffn_overlap_requested=1

frequency="${FREQUENCY:-200}"
device="${DEVICE:-xilinx_u50_gen3x16_xdma_5_202210_1}"
target="${TARGET:-hw_emu}"
tag="hw_emu.f${frequency}"
[[ "${target}" == hw_emu ]] || { echo "HWEmu launcher requires TARGET=hw_emu" >&2; exit 2; }
[[ "${root}" == /* ]] || root="$PWD/${root}"
build_dir="${root}/build.${tag}.${device}"
host_exe="${build_dir}/${host_name}"
xclbin="${build_dir}/${xclbin_name}"
emconfig="${build_dir}/emconfig.json"
min_available_gib="${VITIS_MIN_AVAILABLE_GIB:-40}"
allow_parallel="${QUANTIZED_LAYER_ALLOW_PARALLEL:-0}"
[[ "${allow_parallel}" == 0 || "${allow_parallel}" == 1 ]] || {
    echo "QUANTIZED_LAYER_ALLOW_PARALLEL must be 0 or 1" >&2
    exit 2
}

if [[ "${1:-}" == "--worker" ]]; then
    run_lock_path="$(realpath "${build_dir}")/.quantized_layer_hwemu.lock"
    if ! exec {run_lock_fd}>"${run_lock_path}"; then
        echo "Unable to open HW-Emu run lock: ${run_lock_path}" >&2
        echo "exit_status=73"
        exit 73
    fi
    if ! flock -n "${run_lock_fd}"; then
        echo "Another quantized-layer HW Emu worker is already using build_dir=${build_dir}" >&2
        echo "exit_status=71"
        exit 71
    fi
    echo "run_lock=${run_lock_path}"
    if [[ "${QUANTIZED_LAYER_TRACE:-0}" == 1 ]]; then
        export VITIS_LAUNCH_WAVEFORM_BATCH=1
        export USER_PRE_SIM_SCRIPT="$PWD/tcl/quantized_layer_trace_pre.tcl"
        export USER_POST_SIM_SCRIPT="$PWD/tcl/quantized_layer_trace_post.tcl"
        echo "trace_pre_sha256=$(sha256sum "${USER_PRE_SIM_SCRIPT}" | awk '{print $1}')"
        echo "trace_post_sha256=$(sha256sum "${USER_POST_SIM_SCRIPT}" | awk '{print $1}')"
        echo "trace_profile_init_sha256=$(sha256sum "$PWD/tcl/quantized_profile_init.tcl" | awk '{print $1}')"
        if [[ "${QUANTIZED_LAYER_STAGE_TRACE:-0}" == 1 ]]; then
            echo "trace_projection_helpers_sha256=$(sha256sum "$PWD/tcl/quantized_projection_trace_helpers.tcl" | awk '{print $1}')"
        fi
    fi
    echo "started_at=$(date -Is)"
    echo "precision=${precision}"
    echo "target=hw_emu"
    echo "frequency=${frequency}"
    echo "prefill=${prefill}"
    echo "block_size=${block_size}"
    echo "physical_blocks=$(((prefill + block_size - 1) / block_size))"
    echo "decode_tokens=1"
    echo "weight_mode=${weight_mode}"
    echo "attention_variant=${attention_variant}"
    echo "layer_profile=${QUANTIZED_LAYER_PROFILE}"
    echo "prefill_ffn_overlap=${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP}"
    echo "prefill_ffn_overlap_requested=${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED:-0}"
    echo "rms_lanes=${CU_RMS_LANES_CONFIG}"
    echo "profile_flags_sha256=${QUANTIZED_LAYER_PROFILE_CFLAGS_SHA256}"
    echo "allow_parallel=${allow_parallel}"
    echo "rtl_handshake_trace=${QUANTIZED_LAYER_TRACE:-0}"
    echo "timeout_seconds=${timeout_seconds}"
    echo "host_sha256=$(sha256sum "${host_exe}" | awk '{print $1}')"
    echo "xclbin_sha256=$(sha256sum "${xclbin}" | awk '{print $1}')"
    echo "emconfig_sha256=$(sha256sum "${emconfig}" | awk '{print $1}')"
    set +e
    if [[ "${precision}" == "w4" ]]; then
        QUANTIZED_W4_LAYER_PREFILL="${prefill}" \
        QUANTIZED_W4_LAYER_BLOCK_SIZE="${block_size}" \
        QUANTIZED_W4_LAYER_WEIGHTS="${weight_mode}" \
        QUANTIZED_W4_LAYER_TIMEOUT="${timeout_seconds}" \
        FREQUENCY="${frequency}" DEVICE="${device}" \
            bash scripts/run_quantized_hwemu_guard.sh "${build_script}" run
    else
        QUANTIZED_W8_LAYER_PREFILL="${prefill}" \
        QUANTIZED_W8_LAYER_BLOCK_SIZE="${block_size}" \
        QUANTIZED_W8_LAYER_WEIGHTS="${weight_mode}" \
        QUANTIZED_W8_LAYER_TIMEOUT="${timeout_seconds}" \
        FREQUENCY="${frequency}" DEVICE="${device}" \
            bash scripts/run_quantized_hwemu_guard.sh "${build_script}" run
    fi
    status="$?"
    set -e
    if [[ "$status" == 0 && -n "${QUANTIZED_LAYER_DUMP_OUTPUT:-}" ]]; then
        [[ -s "$QUANTIZED_LAYER_DUMP_OUTPUT" ]] || {
            echo "Missing requested numerical dump" >&2
            status=65
        }
        if [[ "$status" == 0 ]]; then
            echo "numerical_dump_sha256=$(sha256sum "$QUANTIZED_LAYER_DUMP_OUTPUT" | awk '{print $1}')"
        fi
    fi
    echo "finished_at=$(date -Is)"
    echo "exit_status=${status}"
    exit "${status}"
fi

if [[ "$#" -gt 1 ]] || { [[ "$#" -eq 1 ]] && [[ "$1" != "w4" ]] && [[ "$1" != "w8" ]]; }; then
    echo "usage: $0 [w4|w8]" >&2
    exit 2
fi
for value_name in frequency prefill block_size timeout_seconds min_available_gib; do
    value="${!value_name}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "${value_name} must be a positive integer" >&2
        exit 2
    }
done
[[ "${weight_mode}" == zero || "${weight_mode}" == random ]] || {
    echo "weight mode must be zero or random" >&2
    exit 2
}
for input in "${build_script}" "${host_exe}" "${xclbin}" "${emconfig}"; do
    [[ -s "${input}" ]] || { echo "Missing HW-Emu input: ${input}" >&2; exit 66; }
done
if [[ "${QUANTIZED_LAYER_TRACE:-0}" == 1 ]]; then
    # Read the current binary, not a potentially stale .info sidecar.
    xclbinutil="${XCLBINUTIL:-}"
    [[ -n "${xclbinutil}" ]] || xclbinutil="${XILINX_XRT:-}/bin/xclbinutil"
    [[ -x "${xclbinutil}" ]] || {
        echo "Trace requires xclbinutil; set XCLBINUTIL or XILINX_XRT" >&2
        exit 66
    }
    image_info="$("${xclbinutil}" --input "${xclbin}" --info)"
    if ! rg -q -- '(^|[[:space:]])--debug([[:space:]]|$)' <<<"${image_info}"; then
        echo "Trace requires a debug HW Emu image. Relink with VITIS_QUANTIZED_LAYER_DEBUG=1." >&2
        exit 65
    fi
fi
if [[ "${allow_parallel}" == 0 ]] && pgrep -af '[h]ost_control_cache_quantized_w[48]_layer_hwemu|[x]simk.*quantized_w[48]_layer' >/dev/null; then
    echo "Another quantized-layer HW Emu is already running" >&2
    pgrep -af '[h]ost_control_cache_quantized_w[48]_layer_hwemu|[x]simk.*quantized_w[48]_layer' >&2 || true
    exit 70
fi
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if [[ "${available_kib}" -lt $((min_available_gib * 1024 * 1024)) ]]; then
    echo "Need at least ${min_available_gib} GiB available memory" >&2
    exit 75
fi

log_root="${COWAVE_QUANTIZED_LAYER_OUTPUT_DIR:-${repo_root}/.build}/logs"
mkdir -p "${log_root}"
timestamp="$(date +%Y%m%d_%H%M%S)"
session="llm_quantized_${precision}_hwemu_p${prefill}_b${block_size}_${timestamp}"
log_path="${log_root}/quantized_${precision}_layer_hwemu_p${prefill}_b${block_size}_${timestamp}.log"
pid_path="${log_path%.log}.pid"
worker_argv=(
    env
    "VITIS_ENV_SCRIPT=${VITIS_ENV_SCRIPT:-}"
    "XILINX_HLS=${XILINX_HLS:-}"
    "VITIS_HLS_INCLUDE=${VITIS_HLS_INCLUDE:-}"
    "XILINX_XRT=${XILINX_XRT:-}"
    "XCLBINUTIL=${XCLBINUTIL:-}"
    "XPLATFORM=${XPLATFORM:-}"
    "TARGET=${target}"
    "THREADS=${THREADS:-16}"
    "PATH=${PATH:-}"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    "PYTHONPATH=${PYTHONPATH:-}"
    "COWAVE_QUANTIZED_LAYER_OUTPUT_DIR=${COWAVE_QUANTIZED_LAYER_OUTPUT_DIR:-}"
    "COWAVE_QUANTIZED_LAYER_CONFIG_DIR=${COWAVE_QUANTIZED_LAYER_CONFIG_DIR:-}"
    "QUANTIZED_LAYER_PRECISION=${precision}"
    "FREQUENCY=${frequency}"
    "DEVICE=${device}"
    "VITIS_MIN_AVAILABLE_GIB=${min_available_gib}"
    "QUANTIZED_LAYER_ALLOW_PARALLEL=${allow_parallel}"
    "QUANTIZED_LAYER_TRACE=${QUANTIZED_LAYER_TRACE:-0}"
    "QUANTIZED_LAYER_STAGE_TRACE=${QUANTIZED_LAYER_STAGE_TRACE:-0}"
    "QUANTIZED_LAYER_FUNCTION_TRACE=${QUANTIZED_LAYER_FUNCTION_TRACE:-0}"
    "QUANTIZED_LAYER_ATTENTION_VARIANT=${attention_variant}"
    "QUANTIZED_LAYER_PROFILE=${QUANTIZED_LAYER_PROFILE}"
    "QUANTIZED_LAYER_SILU_LANES=${worker_silu_lanes}"
    "QUANTIZED_LAYER_SILU_LANES_REQUESTED=${worker_silu_requested}"
    "QUANTIZED_LAYER_RMS_LANES=${worker_rms_lanes}"
    "QUANTIZED_LAYER_RMS_LANES_REQUESTED=${worker_rms_requested}"
    "QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING=${worker_weight_outstanding}"
    "QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING_REQUESTED=${worker_weight_requested}"
    "QUANTIZED_LAYER_ATTENTION_WAVE=${worker_attention_wave}"
    "QUANTIZED_LAYER_ATTENTION_WAVE_REQUESTED=${worker_wave_requested}"
    "QUANTIZED_LAYER_PREFILL_FFN_OVERLAP=${worker_prefill_ffn_overlap}"
    "QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED=${worker_prefill_ffn_overlap_requested}"
    "QUANTIZED_LAYER_DUMP_OUTPUT=${QUANTIZED_LAYER_DUMP_OUTPUT:-}"
)
if [[ "${precision}" == "w4" ]]; then
    worker_argv+=(
        "VITIS_QUANTIZED_W4_LAYER_ROOT=${root}"
        "QUANTIZED_W4_LAYER_PREFILL=${prefill}"
        "QUANTIZED_W4_LAYER_BLOCK_SIZE=${block_size}"
        "QUANTIZED_W4_LAYER_WEIGHTS=${weight_mode}"
        "QUANTIZED_W4_LAYER_TIMEOUT=${timeout_seconds}"
    )
else
    worker_argv+=(
        "VITIS_QUANTIZED_W8_LAYER_ROOT=${root}"
        "QUANTIZED_W8_LAYER_PREFILL=${prefill}"
        "QUANTIZED_W8_LAYER_BLOCK_SIZE=${block_size}"
        "QUANTIZED_W8_LAYER_WEIGHTS=${weight_mode}"
        "QUANTIZED_W8_LAYER_TIMEOUT=${timeout_seconds}"
    )
fi
worker_argv+=("${script_path}" --worker)
printf -v worker_command '%q ' "${worker_argv[@]}"
printf -v quoted_log '%q' "${log_path}"
"${tmux_bin}" new-session -d -s "${session}" "exec ${worker_command} >>${quoted_log} 2>&1"
pid="$("${tmux_bin}" display-message -p -t "${session}:0.0" '#{pane_pid}')"
printf '%s\n' "${pid}" >"${pid_path}"
printf 'pid=%s\nsession=%s\nlog=%s\npidfile=%s\n' \
    "${pid}" "${session}" "${log_path}" "${pid_path}"
