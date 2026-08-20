#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

phase="${1:-all}"
case "${phase}" in
    compute-xo|control-xo|status-xo|link|host|all|run|run-composed|run-block|run-block-stack|run-block-sequence|run-stack|run-generate) ;;
    *)
        echo "usage: $0 [compute-xo|control-xo|status-xo|link|host|all|run|run-composed|run-block|run-block-stack|run-block-sequence|run-stack|run-generate]" >&2
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
resident_token_rows="${CC8_RESIDENT_TOKEN_ROWS:-8}"
frequency="${FREQUENCY:-200}"
threads="${THREADS:-16}"
device="${DEVICE:-xilinx_u50_gen3x16_xdma_5_202210_1}"
platform="${XPLATFORM:-/opt/xilinx/platforms/${device}/${device}.xpfm}"
timeout_seconds="${VITIS_8X64_HW_EMU_TIMEOUT:-43200}"
seed="${VITIS_8X64_RESIDENT_SEED:-20260718}"
debug_build="${VITIS_8X64_HWEMU_DEBUG:-0}"
variant_tag="${VITIS_8X64_RESIDENT_VARIANT_TAG:-persistent_aux}"
build_exact_compute_xo="${VITIS_8X64_BUILD_EXACT_COMPUTE_XO:-0}"
e2e_tokens="${VITIS_8X64_E2E_TOKENS:-0,1}"
e2e_max_new_tokens="${VITIS_8X64_E2E_MAX_NEW_TOKENS:-3}"
e2e_layers="${VITIS_8X64_E2E_LAYERS:-0}"
e2e_prefill_block_size="${VITIS_8X64_E2E_PREFILL_BLOCK_SIZE:-${resident_token_rows}}"
verify_sequence_tokens="${VITIS_8X64_VERIFY_SEQUENCE_TOKENS:-16}"

case "${profile}" in
    qwen-layer|qwen2.5-3b|small) ;;
    *)
        echo "resident-layer hw_emu supports VITIS_8X64_MODEL_PROFILE=qwen-layer, qwen2.5-3b, or small" >&2
        exit 2
        ;;
esac

if [ "${resident_token_rows}" -lt 1 ] || [ "${resident_token_rows}" -gt 8 ]; then
    echo "CC8_RESIDENT_TOKEN_ROWS must be in 1..8" >&2
    exit 2
fi
if [ "${resident_token_rows}" -gt 4 ]; then
    resident_dual_vector_ports=1
else
    resident_dual_vector_ports=0
fi
if [ "${e2e_prefill_block_size}" -lt 1 ] || [ "${e2e_prefill_block_size}" -gt "${resident_token_rows}" ]; then
    echo "VITIS_8X64_E2E_PREFILL_BLOCK_SIZE must be in 1..CC8_RESIDENT_TOKEN_ROWS" >&2
    exit 2
fi
if [ "${build_exact_compute_xo}" != "0" ] && [ "${build_exact_compute_xo}" != "1" ]; then
    echo "VITIS_8X64_BUILD_EXACT_COMPUTE_XO must be 0 or 1" >&2
    exit 2
fi
if [ "${profile}" = "qwen2.5-3b" ] && [ "${build_exact_compute_xo}" != "1" ]; then
    echo "qwen2.5-3b requires a profile-matched compute XO; set VITIS_8X64_BUILD_EXACT_COMPUTE_XO=1" >&2
    exit 2
fi

if [ -n "${VITIS_8X64_CONN_CFG:-}" ]; then
    conn_cfg="${VITIS_8X64_CONN_CFG}"
elif [ "${profile}" = "qwen2.5-3b" ]; then
    conn_cfg="conn_u50_8x64_dual_full_resident.cfg"
else
    conn_cfg="conn_u50_8x64_dual.cfg"
fi

profile_tag="${profile//./_}"
profile_tag="${profile_tag//-/_}"
variant_tag="${variant_tag//./_}"
variant_tag="${variant_tag//-/_}"
tag="${profile_tag}.resident_layer.t${resident_token_rows}.d${fifo_depth}.ii${load_ii}.r${wave_repeat}.wr${wave_result_depth}.cw${cross_wave_dataflow}.${variant_tag}"

if [ "${profile}" = "small" ]; then
    default_compute_dir="vitis_8x64/xo"
else
    default_compute_dir="vitis_8x64/xo.${profile_tag}.dynamic_bounds"
    if [ ! -s "${default_compute_dir}/compute_core_8x64_unified_nk.xo" ]; then
        default_compute_dir="vitis_8x64/xo.${profile_tag}"
    fi
fi

xo_dir="${VITIS_8X64_RESIDENT_XO_DIR:-vitis_8x64/xo.${tag}}"
if [ "${build_exact_compute_xo}" = "1" ]; then
    compute_xo_dir="${xo_dir}"
else
    compute_xo_dir="${VITIS_8X64_BASE_COMPUTE_XO_DIR:-${default_compute_dir}}"
fi
build_dir="${VITIS_8X64_BUILD_DIR:-vitis_8x64/build.${tag}.hw_emu.${device}}"
temp_dir="${VITIS_8X64_TEMP_DIR:-vitis_8x64/_x.${tag}.hw_emu.${device}}"
report_dir="${VITIS_8X64_REPORT_DIR:-reports/vitis_8x64/${tag}/hw_emu.${device}}"
hls_project_root="${VITIS_8X64_HLS_PROJECT_ROOT:-$PWD}"

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
echo "resident_token_rows=${resident_token_rows}"
echo "resident_dual_vector_ports=${resident_dual_vector_ports}"
echo "frequency=${frequency}"
echo "debug_build=${debug_build}"
echo "build_exact_compute_xo=${build_exact_compute_xo}"
echo "verify_sequence_tokens=${verify_sequence_tokens}"
echo "control_xo=${control_xo}"
echo "compute_xo=${compute_xo}"
echo "status_xo=${status_xo}"
echo "xclbin=${xclbin}"
echo "hls_project_root=${hls_project_root}"

build_compute_xo() {
    mkdir -p "${compute_xo_dir}" "${hls_project_root}"
    LLM_FPGA_MODEL_PROFILE="${profile}" \
    LLM_FPGA_XO_DIR="$(realpath "${compute_xo_dir}")" \
    LLM_FPGA_HLS_PROJECT_ROOT="$(realpath "${hls_project_root}")" \
    LLM_FPGA_HLS_PROJECT_NAME="qwen_hls_compute_core_8x64_nk_prj_${tag}" \
    CC8_WEIGHT_TILE_FIFO_DEPTH="${fifo_depth}" \
    CC8_WEIGHT_TILE_LOAD_II="${load_ii}" \
    CC8_ENABLE_MM_WAVE_REPEAT="${wave_repeat}" \
    CC8_MM_WAVE_RESULT_FIFO_DEPTH="${wave_result_depth}" \
    CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW="${cross_wave_dataflow}" \
        scripts/run_vitis_hls.sh tcl/build_compute_core_8x64_nk_xo.tcl
}

build_control_xo() {
    mkdir -p "${xo_dir}" "${hls_project_root}"
    local extra_cflags="${HLS_EXTRA_CFLAGS:-} -DCC8_RESIDENT_LAYER_ONLY=1 -DCC8_RESIDENT_TOKEN_ROWS=${resident_token_rows} -DCC8_RESIDENT_DUAL_VECTOR_PORTS=${resident_dual_vector_ports}"
    LLM_FPGA_MODEL_PROFILE="${profile}" \
    LLM_FPGA_XO_DIR="$(realpath "${xo_dir}")" \
    LLM_FPGA_HLS_PROJECT_ROOT="$(realpath "${hls_project_root}")" \
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
    mkdir -p "${xo_dir}" "${hls_project_root}"
    LLM_FPGA_MODEL_PROFILE="${profile}" \
    LLM_FPGA_XO_DIR="$(realpath "${xo_dir}")" \
    LLM_FPGA_HLS_PROJECT_ROOT="$(realpath "${hls_project_root}")" \
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
        BUILD_DIR="${build_dir}"
}

run_hwemu() {
    local mode="${1:-verify-resident-layer}"
    local extra_args=()
    if [ "${mode}" = "verify-composed-prefill-block" ] ||
       [ "${mode}" = "verify-composed-prefill-stack" ] ||
       [ "${mode}" = "verify-composed-prefill-sequence" ]; then
        extra_args+=(--prefill-block-size "${resident_token_rows}")
    fi
    if [ "${mode}" = "verify-composed-prefill-sequence" ]; then
        extra_args+=(--prefill-len "${verify_sequence_tokens}")
    fi
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
            --mode "${mode}" \
            --profile "${profile}" \
            --random-model \
            --seed "${seed}" \
            --position 0 \
            "${extra_args[@]}"
    )
}

run_generate_hwemu() {
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
            --mode generate \
            --profile "${profile}" \
            --random-model \
            --seed "${seed}" \
            --tokens "${e2e_tokens}" \
            --max-new-tokens "${e2e_max_new_tokens}" \
            --prefill-block-size "${e2e_prefill_block_size}" \
            --layers "${e2e_layers}" \
            --coarse-tasks
    )
}

case "${phase}" in
    compute-xo) build_compute_xo ;;
    control-xo) build_control_xo ;;
    status-xo) build_status_xo ;;
    link) link_hwemu ;;
    host) build_host ;;
    run) run_hwemu verify-resident-layer ;;
    run-composed) run_hwemu verify-composed-layer ;;
    run-block) run_hwemu verify-composed-prefill-block ;;
    run-block-stack) run_hwemu verify-composed-prefill-stack ;;
    run-block-sequence) run_hwemu verify-composed-prefill-sequence ;;
    run-stack) run_hwemu verify-composed-stack ;;
    run-generate) run_generate_hwemu ;;
    all)
        if [ "${build_exact_compute_xo}" = "1" ]; then
            build_compute_xo
        fi
        build_control_xo
        build_status_xo
        link_hwemu
        build_host
        ;;
esac

echo "completed_at=$(date -Is)"
