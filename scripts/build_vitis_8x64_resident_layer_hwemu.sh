#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

phase="${1:-all}"
case "${phase}" in
    control-xo|status-xo|link|host|all|run) ;;
    *)
        echo "usage: $0 [control-xo|status-xo|link|host|all|run]" >&2
        exit 2
        ;;
esac

# This flow is intentionally hw_emu-only.  The resident layer must pass the
# complete functional/performance loop before it is promoted to TARGET=hw.
env_script="${VITIS_ENV_SCRIPT:-/home/hepc/env/vitis_env_22.sh}"
profile="${VITIS_8X64_MODEL_PROFILE:-qwen-layer}"
fifo_depth="${CC8_WEIGHT_TILE_FIFO_DEPTH:-2}"
load_ii="${CC8_WEIGHT_TILE_LOAD_II:-2}"
wave_repeat="${CC8_ENABLE_MM_WAVE_REPEAT:-0}"
wave_result_depth="${CC8_MM_WAVE_RESULT_FIFO_DEPTH:-33}"
cross_wave_dataflow="${CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW:-1}"
frequency="${FREQUENCY:-200}"
threads="${THREADS:-16}"
device="${DEVICE:-xilinx_u50_gen3x16_xdma_5_202210_1}"
platform="${XPLATFORM:-/opt/xilinx/platforms/${device}/${device}.xpfm}"
conn_cfg="${VITIS_8X64_CONN_CFG:-conn_u50_8x64_dual.cfg}"
timeout_seconds="${VITIS_8X64_HW_EMU_TIMEOUT:-43200}"
seed="${VITIS_8X64_RESIDENT_SEED:-20260718}"
debug_build="${VITIS_8X64_HWEMU_DEBUG:-0}"

if [ "${profile}" != "qwen-layer" ]; then
    echo "resident-layer hw_emu currently requires VITIS_8X64_MODEL_PROFILE=qwen-layer" >&2
    exit 2
fi

profile_tag="${profile//./_}"
profile_tag="${profile_tag//-/_}"
tag="${profile_tag}.resident_layer.d${fifo_depth}.ii${load_ii}.r${wave_repeat}.wr${wave_result_depth}.cw${cross_wave_dataflow}.scratch"

default_compute_dir="vitis_8x64/xo.${profile_tag}.dynamic_bounds"
if [ ! -s "${default_compute_dir}/compute_core_8x64_unified_nk.xo" ]; then
    default_compute_dir="vitis_8x64/xo.${profile_tag}"
fi

compute_xo_dir="${VITIS_8X64_BASE_COMPUTE_XO_DIR:-${default_compute_dir}}"
xo_dir="${VITIS_8X64_RESIDENT_XO_DIR:-vitis_8x64/xo.${tag}}"
build_dir="${VITIS_8X64_BUILD_DIR:-vitis_8x64/build.${tag}.hw_emu.${device}}"
temp_dir="${VITIS_8X64_TEMP_DIR:-vitis_8x64/_x.${tag}.hw_emu.${device}}"
report_dir="${VITIS_8X64_REPORT_DIR:-reports/vitis_8x64/${tag}/hw_emu.${device}}"

if [ ! -r "${env_script}" ]; then
    echo "Missing Vitis 2022.2 environment script: ${env_script}" >&2
    exit 66
fi
source "${env_script}" >/dev/null 2>&1

control_xo="${xo_dir}/control_cache_8x64_dual_core_nk.xo"
status_xo="${xo_dir}/cc8_status_sink_nk.xo"
compute_xo="${compute_xo_dir}/compute_core_8x64_unified_nk.xo"
xclbin="${build_dir}/qwen_8x64_dual.xclbin"
host_exe="${build_dir}/host_qwen_8x64.exe"
emconfig="${build_dir}/emconfig.json"

echo "phase=${phase}"
echo "target=hw_emu"
echo "profile=${profile}"
echo "resident_layer_only=1"
echo "fifo_depth=${fifo_depth}"
echo "load_ii=${load_ii}"
echo "wave_repeat=${wave_repeat}"
echo "wave_result_fifo_depth=${wave_result_depth}"
echo "cross_wave_dataflow=${cross_wave_dataflow}"
echo "frequency=${frequency}"
echo "debug_build=${debug_build}"
echo "control_xo=${control_xo}"
echo "compute_xo=${compute_xo}"
echo "status_xo=${status_xo}"
echo "xclbin=${xclbin}"

build_control_xo() {
    mkdir -p "${xo_dir}"
    local extra_cflags="${HLS_EXTRA_CFLAGS:-} -DCC8_RESIDENT_LAYER_ONLY=1"
    LLM_FPGA_MODEL_PROFILE="${profile}" \
    LLM_FPGA_XO_DIR="$(realpath "${xo_dir}")" \
    LLM_FPGA_HLS_PROJECT_NAME="qwen_hls_control_cache_8x64_nk_prj_${tag}" \
    HLS_EXTRA_CFLAGS="${extra_cflags}" \
    CC8_WEIGHT_TILE_FIFO_DEPTH="${fifo_depth}" \
    CC8_WEIGHT_TILE_LOAD_II="${load_ii}" \
    CC8_ENABLE_MM_WAVE_REPEAT="${wave_repeat}" \
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
    for input in "${control_xo}" "${compute_xo}" "${status_xo}" "${conn_cfg}"; do
        if [ ! -s "${input}" ]; then
            echo "Missing or empty hw_emu link input: ${input}" >&2
            exit 66
        fi
    done
    if [ ! -s "${platform}" ]; then
        echo "Missing platform: ${platform}" >&2
        exit 66
    fi

    mkdir -p "${build_dir}" "${temp_dir}" "${report_dir}"
    local debug_flags=()
    if [ "${debug_build}" = "1" ]; then
        debug_flags=(-g)
    fi
    v++ -l -t hw_emu "${debug_flags[@]}" \
        --platform "${platform}" --save-temps \
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
    for input in "${xclbin}" "${host_exe}" "${emconfig}"; do
        if [ ! -s "${input}" ]; then
            echo "Missing or empty resident-layer hw_emu input: ${input}" >&2
            exit 66
        fi
    done
    (
        cd "${build_dir}"
        XCL_EMULATION_MODE=hw_emu \
        timeout "${timeout_seconds}" \
            ./host_qwen_8x64.exe \
            --xclbin ./qwen_8x64_dual.xclbin \
            --mode verify-resident-layer \
            --profile "${profile}" \
            --random-model \
            --seed "${seed}" \
            --position 0
    )
}

case "${phase}" in
    control-xo) build_control_xo ;;
    status-xo) build_status_xo ;;
    link) link_hwemu ;;
    host) build_host ;;
    run) run_hwemu ;;
    all)
        build_control_xo
        build_status_xo
        link_hwemu
        build_host
        ;;
esac

echo "completed_at=$(date -Is)"
