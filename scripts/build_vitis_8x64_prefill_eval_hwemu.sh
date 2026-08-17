#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

phase="${1:-all}"
case "${phase}" in
    compute-xo|control-xo|status-xo|link|host|all|run) ;;
    *)
        echo "usage: $0 [compute-xo|control-xo|status-xo|link|host|all|run]" >&2
        exit 2
        ;;
esac

# Dedicated hw_emu image for the staged block-prefill layer.  It retains the
# projection/vector operators used by the host plus CC8_OP_ATTN_PREFILL_BLOCK,
# while pruning decode, resident-layer, and diagnostic-attention schedulers.
env_script="${VITIS_ENV_SCRIPT:-/home/hepc/env/vitis_env_22.sh}"
profile="${VITIS_8X64_MODEL_PROFILE:-qwen-layer}"
fifo_depth="${CC8_WEIGHT_TILE_FIFO_DEPTH:-2}"
load_ii="${CC8_WEIGHT_TILE_LOAD_II:-2}"
wave_result_depth="${CC8_MM_WAVE_RESULT_FIFO_DEPTH:-33}"
cross_wave_dataflow="${CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW:-1}"
variant="${CC8_PREFILL_VARIANT:-}"
frequency="${FREQUENCY:-200}"
threads="${THREADS:-16}"
device="${DEVICE:-xilinx_u50_gen3x16_xdma_5_202210_1}"
platform="${XPLATFORM:-/opt/xilinx/platforms/${device}/${device}.xpfm}"
conn_cfg="${VITIS_8X64_CONN_CFG:-conn_u50_8x64_dual.cfg}"
prefill_len="${VITIS_8X64_PREFILL_LEN:-8}"
prefill_start="${VITIS_8X64_PREFILL_START:-0}"
seed="${VITIS_8X64_PREFILL_SEED:-20260722}"
timeout_seconds="${VITIS_8X64_HW_EMU_TIMEOUT:-604800}"

if [ "${profile}" != "qwen-layer" ] && [ "${profile}" != "qwen-layer-long" ]; then
    echo "prefill evaluation requires qwen-layer or qwen-layer-long" >&2
    exit 2
fi
if [ "${prefill_len}" -lt 1 ] || [ "${prefill_len}" -gt 2048 ]; then
    echo "VITIS_8X64_PREFILL_LEN must be in 1..2048" >&2
    exit 2
fi
if [ "${prefill_start}" -lt 0 ] || [ "${prefill_start}" -ge "${prefill_len}" ]; then
    echo "VITIS_8X64_PREFILL_START must be in 0..prefill_len-1" >&2
    exit 2
fi
if [ ! -r "${env_script}" ]; then
    echo "Missing Vitis 2022.2 environment script: ${env_script}" >&2
    exit 66
fi
if [ -n "${variant}" ] && ! [[ "${variant}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "CC8_PREFILL_VARIANT may only contain letters, digits, '.', '_' and '-'" >&2
    exit 2
fi

profile_tag="${profile//./_}"
profile_tag="${profile_tag//-/_}"
tag="${profile_tag}.prefill_layer.d${fifo_depth}.ii${load_ii}.wr${wave_result_depth}.cw${cross_wave_dataflow}.scratch"
if [ -n "${variant}" ]; then
    tag="${tag}.${variant}"
fi
xo_dir="${VITIS_8X64_PREFILL_XO_DIR:-vitis_8x64/xo.${tag}}"
build_dir="${VITIS_8X64_BUILD_DIR:-vitis_8x64/build.${tag}.hw_emu.${device}}"
temp_dir="${VITIS_8X64_TEMP_DIR:-vitis_8x64/_x.${tag}.hw_emu.${device}}"
report_dir="${VITIS_8X64_REPORT_DIR:-reports/vitis_8x64/${tag}/hw_emu.${device}}"
default_compute_xo="vitis_8x64/xo.${profile_tag}.dynamic_bounds/compute_core_8x64_unified_nk.xo"
compute_xo="${VITIS_8X64_COMPUTE_XO:-${default_compute_xo}}"
control_xo="${xo_dir}/control_cache_8x64_dual_core_nk.xo"
status_xo="${VITIS_8X64_STATUS_XO:-${xo_dir}/cc8_status_sink_nk.xo}"
xclbin="${build_dir}/qwen_8x64_dual.xclbin"

source "${env_script}" >/dev/null 2>&1

echo "phase=${phase}"
echo "profile=${profile}"
echo "resident_layer_only=0"
echo "prefill_layer_only=1"
echo "prefill_len=${prefill_len}"
echo "prefill_start=${prefill_start}"
echo "fifo_depth=${fifo_depth} load_ii=${load_ii} wave_result_depth=${wave_result_depth}"
echo "cross_wave_dataflow=${cross_wave_dataflow} frequency=${frequency}"
echo "variant=${variant:-baseline}"
echo "control_xo=${control_xo}"
echo "compute_xo=${compute_xo}"
echo "status_xo=${status_xo}"
echo "xclbin=${xclbin}"

build_compute_xo() {
    mkdir -p "$(dirname "${compute_xo}")"
    LLM_FPGA_MODEL_PROFILE="${profile}" \
    LLM_FPGA_XO_DIR="$(realpath "$(dirname "${compute_xo}")")" \
    LLM_FPGA_HLS_PROJECT_NAME="qwen_hls_compute_core_8x64_nk_prj_${profile_tag}.dynamic_bounds" \
        scripts/run_vitis_hls.sh tcl/build_compute_core_8x64_nk_xo.tcl
}

build_control_xo() {
    mkdir -p "${xo_dir}"
    LLM_FPGA_MODEL_PROFILE="${profile}" \
    LLM_FPGA_XO_DIR="$(realpath "${xo_dir}")" \
    LLM_FPGA_HLS_PROJECT_NAME="qwen_hls_control_cache_8x64_nk_prj_${tag}" \
    HLS_EXTRA_CFLAGS="${HLS_EXTRA_CFLAGS:-} -DCC8_RESIDENT_LAYER_ONLY=0 -DCC8_PREFILL_LAYER_ONLY" \
    CC8_WEIGHT_TILE_FIFO_DEPTH="${fifo_depth}" \
    CC8_WEIGHT_TILE_LOAD_II="${load_ii}" \
    CC8_ENABLE_MM_WAVE_REPEAT=0 \
    CC8_MM_WAVE_RESULT_FIFO_DEPTH="${wave_result_depth}" \
    CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW="${cross_wave_dataflow}" \
        scripts/run_vitis_hls.sh tcl/build_control_cache_8x64_nk_xo.tcl
}

build_status_xo() {
    mkdir -p "${xo_dir}"
    LLM_FPGA_MODEL_PROFILE="${profile}" \
    LLM_FPGA_XO_DIR="$(realpath "${xo_dir}")" \
    LLM_FPGA_HLS_PROJECT_NAME="qwen_hls_cc8_status_sink_nk_prj_${tag}" \
        scripts/run_vitis_hls.sh tcl/build_cc8_status_sink_nk_xo.tcl
}

link_hwemu() {
    for input in "${control_xo}" "${compute_xo}" "${status_xo}" "${conn_cfg}" "${platform}"; do
        if [ ! -s "${input}" ]; then
            echo "Missing or empty hw_emu link input: ${input}" >&2
            exit 66
        fi
    done
    mkdir -p "${build_dir}" "${temp_dir}" "${report_dir}"
    v++ -l -t hw_emu --platform "${platform}" --save-temps \
        --optimize 3 --report_level 2 -I"$PWD/include" \
        --kernel_frequency "${frequency}" --config "${conn_cfg}" \
        --vivado.synth.jobs "${threads}" --vivado.impl.jobs "${threads}" \
        --temp_dir "${temp_dir}" --report_dir "${report_dir}" \
        -o "${xclbin}" "${control_xo}" "${compute_xo}" "${status_xo}"
}

build_host() {
    make vitis_8x64_qwen_host vitis_8x64_emconfig \
        TARGET=hw_emu DEVICE="${device}" \
        VITIS_8X64_MODEL_PROFILE="${profile}" \
        VITIS_8X64_BUILD_DIR="${build_dir}"
}

run_hwemu() {
    scripts/run_vitis_8x64_qwen_profile_hwemu.sh \
        --build-dir "${build_dir}" \
        --profile "${profile}" \
        --mode profile-prefill-block \
        --prefill-len "${prefill_len}" \
        --prefill-start "${prefill_start}" \
        --seed "${seed}" \
        --random-model \
        --timeout "${timeout_seconds}"
}

case "${phase}" in
    compute-xo) build_compute_xo ;;
    control-xo) build_control_xo ;;
    status-xo) build_status_xo ;;
    link) link_hwemu ;;
    host) build_host ;;
    run) run_hwemu ;;
    all)
        build_compute_xo
        build_control_xo
        build_status_xo
        link_hwemu
        build_host
        ;;
esac

echo "completed_at=$(date -Is)"
