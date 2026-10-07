#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"
phase="${1:-all}"
case "${phase}" in controller-xo|compute-xo|link|host|emconfig|run|all) ;;
*) echo "usage: $0 [controller-xo|compute-xo|link|host|emconfig|run|all]" >&2; exit 2 ;;
esac

source scripts/quantized_layer_profiles.sh
quantized_layer_profile_apply w8

env_script="${VITIS_ENV_SCRIPT:-}"
if [[ -n "${env_script}" ]]; then
    [[ -r "${env_script}" ]] || { echo "Missing Vitis environment: ${env_script}" >&2; exit 66; }
    source "${env_script}" >/dev/null 2>&1
fi
device="${DEVICE:-xilinx_u50_gen3x16_xdma_5_202210_1}"
platform="${XPLATFORM:-}"
target="${TARGET:-hw_emu}"
frequency="${FREQUENCY:-200}"
threads="${THREADS:-16}"
prefill="${QUANTIZED_W8_LAYER_PREFILL:-66}"
block_size="${QUANTIZED_W8_LAYER_BLOCK_SIZE:-4}"
weight_mode="${QUANTIZED_W8_LAYER_WEIGHTS:-zero}"
timeout_seconds="${QUANTIZED_W8_LAYER_TIMEOUT:-86400}"
profile="${QUANTIZED_LAYER_PROFILE}"
attention_variant="${QUANTIZED_LAYER_ATTENTION_VARIANT}"
synth_only="${QUANTIZED_LAYER_SYNTH_ONLY:-0}"
output_root="${COWAVE_QUANTIZED_LAYER_OUTPUT_DIR:-${repo_root}/.build}"
root="${VITIS_QUANTIZED_W8_LAYER_ROOT:-${output_root}/w8}"
tag="${target}.f${frequency}"
xo_dir="${VITIS_QUANTIZED_W8_LAYER_XO_DIR:-${root}/xo.${tag}}"
hls_root="${VITIS_QUANTIZED_W8_LAYER_HLS_ROOT:-${root}/hls.${tag}}"
build_dir="${VITIS_QUANTIZED_W8_LAYER_BUILD_DIR:-${root}/build.${tag}.${device}}"
temp_dir="${VITIS_QUANTIZED_W8_LAYER_TEMP_DIR:-${root}/_x.${tag}.${device}}"
run_tmp_dir="${VITIS_QUANTIZED_W8_LAYER_RUN_TMP_DIR:-${root}/run-tmp.${tag}.${device}}"
report_dir="${VITIS_QUANTIZED_W8_LAYER_REPORT_DIR:-${root}/reports.${tag}.${device}}"
xclbin="${VITIS_QUANTIZED_W8_LAYER_XCLBIN:-${build_dir}/quantized_w8_layer.xclbin}"
controller_xo="${xo_dir}/control_cache_quantized_w8_layer.xo"
compute_xo="${xo_dir}/compute_core_quantized_w8_unified_nk.xo"
host_exe="${build_dir}/host_control_cache_quantized_w8_layer_hwemu.exe"
config_dir="${COWAVE_QUANTIZED_LAYER_CONFIG_DIR:-${repo_root}/config}"
conn_cfg="${VITIS_QUANTIZED_W8_LAYER_CONN_CFG:-${config_dir}/cowave-int8-4-4-128.cfg}"

export QUANTIZED_LAYER_ARTIFACT_TARGET="${target}"
export QUANTIZED_LAYER_ARTIFACT_DEVICE="${device}"
export QUANTIZED_LAYER_ARTIFACT_FREQUENCY="${frequency}"
export QUANTIZED_LAYER_ARTIFACT_PLATFORM="${platform}"
export QUANTIZED_LAYER_ARTIFACT_CONN_CFG="${conn_cfg}"

[[ "${target}" == hw_emu || "${target}" == hw ]] || { echo "TARGET must be hw_emu or hw" >&2; exit 2; }
hls_include="${VITIS_HLS_INCLUDE:-}"
if [[ -z "${hls_include}" && -n "${XILINX_HLS:-}" ]]; then
    hls_include="${XILINX_HLS}/include"
fi
if [[ -z "${hls_include}" ]]; then
    vitis_hls_bin="$(command -v vitis_hls || true)"
    [[ -z "${vitis_hls_bin}" ]] || hls_include="$(cd "$(dirname "${vitis_hls_bin}")/../include" && pwd -P)"
fi
[[ "${prefill}" =~ ^[1-9][0-9]*$ && "${prefill}" -le 2047 ]] || { echo "QUANTIZED_W8_LAYER_PREFILL must be in 1..2047" >&2; exit 2; }
[[ "${block_size}" =~ ^[1-4]$ ]] || { echo "QUANTIZED_W8_LAYER_BLOCK_SIZE must be in 1..4" >&2; exit 2; }
[[ "${weight_mode}" == zero || "${weight_mode}" == random ]] || { echo "QUANTIZED_W8_LAYER_WEIGHTS must be zero or random" >&2; exit 2; }
[[ "${synth_only}" == 0 || "${synth_only}" == 1 ]] || {
    echo "QUANTIZED_LAYER_SYNTH_ONLY must be 0 or 1" >&2; exit 2;
}
if [[ "${phase}" == link || "${phase}" == all || "${phase}" == run ]]; then
    quantized_layer_profile_check_xo_reuse w8 "${controller_xo}"
    quantized_layer_profile_check_xo_reuse w8 "${compute_xo}"
fi
printf 'profile=%s attention_variant=%s synth_only=%s\n' \
    "${profile}" "${attention_variant}" "${synth_only}"
printf 'profile_source_sha256=%s profile_cflags_sha256=%s\n' \
    "${QUANTIZED_LAYER_PROFILE_SOURCE_SHA256}" \
    "${QUANTIZED_LAYER_PROFILE_CFLAGS_SHA256}"

build_xo() {
    local kind="$1"
    local tcl project output
    command -v vitis_hls >/dev/null 2>&1 || {
        echo "Missing vitis_hls; activate the toolchain or set VITIS_ENV_SCRIPT" >&2
        exit 66
    }
    if [[ "${kind}" == controller ]]; then
        tcl=tcl/build_control_cache_quantized_w8_layer_xo.tcl
        project=qwen_hls_control_cache_quantized_w8_layer_${tag}_prj
        output="${controller_xo}"
    else
        tcl=tcl/build_compute_core_quantized_unified_xo.tcl
        project=qwen_hls_compute_core_quantized_w8_unified_${tag}_prj
        output="${compute_xo}"
    fi
    mkdir -p "${xo_dir}" "${hls_root}"
    echo "building_${kind}_xo=${output}"
    FREQUENCY="${frequency}" LLM_FPGA_XO_DIR="$(realpath "${xo_dir}")" \
    LLM_FPGA_HLS_PROJECT_ROOT="$(realpath "${hls_root}")" \
    LLM_FPGA_HLS_PROJECT_NAME="${project}" \
    QUANTIZED_ALIGNMENT_PRECISION=w8 \
        scripts/run_vitis_hls.sh "${tcl}"
    if [[ "${synth_only}" == 1 ]]; then
        echo "synth_only_complete=${kind} output_not_exported=${output}"
    else
        [[ -s "${output}" ]] || {
            echo "Missing exported ${kind} XO: ${output}" >&2
            exit 66
        }
        quantized_layer_profile_write_xo_manifest w8 "${output}"
    fi
}

ensure_xos() {
    [[ -s "${controller_xo}" ]] || build_xo controller
    [[ -s "${compute_xo}" ]] || build_xo compute
}

link_xclbin() {
    [[ "${synth_only}" == 0 ]] || {
        echo "link is unavailable with QUANTIZED_LAYER_SYNTH_ONLY=1" >&2
        exit 2
    }
    command -v v++ >/dev/null 2>&1 || { echo "Missing v++; activate the Vitis toolchain" >&2; exit 66; }
    [[ -n "${platform}" ]] || { echo "Set XPLATFORM to the target .xpfm platform" >&2; exit 66; }
    ensure_xos
    for input in "${controller_xo}" "${compute_xo}" "${conn_cfg}" "${platform}"; do
        [[ -s "${input}" ]] || { echo "Missing link input: ${input}" >&2; exit 66; }
    done
    mkdir -p "${build_dir}" "${temp_dir}" "${report_dir}"
    local debug_args=()
    # Pin-level performance evaluation requires a debuggable XSim snapshot.
    if [[ "${VITIS_QUANTIZED_LAYER_DEBUG:-${QUANTIZED_LAYER_TRACE:-0}}" == 1 ]]; then
        debug_args=(-g)
    fi
    v++ -l "${debug_args[@]}" -t "${target}" --platform "${platform}" --save-temps \
        --optimize 3 --report_level 2 -I"${PWD}/include" \
        --kernel_frequency "${frequency}" --config "${conn_cfg}" \
        --vivado.synth.jobs "${threads}" --vivado.impl.jobs "${threads}" \
        --temp_dir "${temp_dir}" --report_dir "${report_dir}" \
        -o "${xclbin}" "${controller_xo}" "${compute_xo}"
    quantized_layer_profile_write_xclbin_manifest w8 "${xclbin}" \
        "${controller_xo}" "${compute_xo}" "${platform}" "${conn_cfg}"
}

build_host() {
    command -v g++ >/dev/null 2>&1 || { echo "Missing g++" >&2; exit 66; }
    [[ -n "${XILINX_XRT:-}" && -d "${XILINX_XRT}/include" && -d "${XILINX_XRT}/lib" ]] || {
        echo "Host build requires XILINX_XRT with include/ and lib/" >&2
        exit 66
    }
    [[ -d "${hls_include}" ]] || {
        echo "Host build requires Vitis HLS headers; set VITIS_HLS_INCLUDE or XILINX_HLS" >&2
        exit 66
    }
    mkdir -p "${build_dir}"
    g++ -std=c++14 -O3 -Wall -Wextra -Wno-unknown-pragmas \
        "${QUANTIZED_LAYER_PROFILE_CFLAGS[@]}" \
        -DMM_STREAM_QUANTIZED_SINGLE_ACCUM_BANK \
        -DMM_STREAM_QUANTIZED_NARROW_ACCUM \
        -Iinclude -Icommon/include -I"${XILINX_XRT}/include" \
        -I"${hls_include}" \
        -o "${host_exe}" common/include/xcl2.cpp \
        host/host_control_cache_quantized_w8_layer_hwemu.cpp \
        -L"${XILINX_XRT}/lib" -lOpenCL -lpthread -lrt -ldl
    quantized_layer_profile_write_host_manifest w8 "${host_exe}"
}

build_emconfig() {
    [[ "${target}" == hw_emu ]] || { echo "emconfig requires TARGET=hw_emu" >&2; exit 2; }
    command -v emconfigutil >/dev/null 2>&1 || { echo "Missing emconfigutil" >&2; exit 66; }
    [[ -n "${platform}" ]] || { echo "Set XPLATFORM to the target .xpfm platform" >&2; exit 66; }
    mkdir -p "${build_dir}"
    emconfigutil --platform "${platform}" --od "${build_dir}"
}

run_hwemu() {
    [[ "${target}" == hw_emu ]] || { echo "run supports TARGET=hw_emu only" >&2; exit 2; }
    quantized_layer_profile_check_xclbin_reuse w8 "${xclbin}" \
        "${controller_xo}" "${compute_xo}" "${platform}" "${conn_cfg}"
    quantized_layer_profile_check_host_reuse w8 "${host_exe}"
    for input in "${xclbin}" "${host_exe}" "${build_dir}/emconfig.json"; do
        [[ -s "${input}" ]] || { echo "Missing run input: ${input}" >&2; exit 66; }
    done
    mkdir -p "${run_tmp_dir}"
    local run_tmp_abs
    run_tmp_abs="$(realpath "${run_tmp_dir}")"
    local dump_args=()
    if [[ -n "${QUANTIZED_LAYER_DUMP_OUTPUT:-}" ]]; then
        dump_args=(--dump-output "$(realpath -m "${QUANTIZED_LAYER_DUMP_OUTPUT}")")
    fi
    ( cd "${build_dir}"; \
      TMPDIR="${run_tmp_abs}" \
      EMCONFIG_PATH="$PWD" \
      XCL_EMULATION_MODE=hw_emu timeout "${timeout_seconds}" \
        "./$(basename "${host_exe}")" --xclbin "$(basename "${xclbin}")" \
        --prefill "${prefill}" --block-size "${block_size}" \
        --weights "${weight_mode}" "${dump_args[@]}" )
}

echo "phase=${phase} target=${target} frequency=${frequency} prefill=${prefill} block_size=${block_size} weights=${weight_mode} profile=${profile} attention_variant=${attention_variant}"
echo "build_dir=${build_dir} temp_dir=${temp_dir} run_tmp_dir=${run_tmp_dir}"
case "${phase}" in
controller-xo) build_xo controller ;;
compute-xo) build_xo compute ;;
link) link_xclbin ;;
host) build_host ;;
emconfig) build_emconfig ;;
run) run_hwemu ;;
    all)
        if [[ "${synth_only}" == 1 ]]; then
            build_xo controller
            build_xo compute
        else
            ensure_xos
            link_xclbin
            build_host
            if [[ "${target}" == hw_emu ]]; then build_emconfig; fi
        fi
        ;;
esac
echo "completed_at=$(date -Is)"
