#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"

usage() {
    cat >&2 <<'EOF'
usage:
  scripts/cowave_quantized_layer.sh check
  scripts/cowave_quantized_layer.sh dry-run <w4|w8> [phase]
  scripts/cowave_quantized_layer.sh test-native
  scripts/cowave_quantized_layer.sh build <w4|w8> [phase]
  scripts/cowave_quantized_layer.sh host <w4|w8>
  scripts/cowave_quantized_layer.sh cosim <w4|w8> [prepare|cosim|all]
  scripts/cowave_quantized_layer.sh hwemu <w4|w8>
  scripts/cowave_quantized_layer.sh reference
  scripts/cowave_quantized_layer.sh performance [--check|--worker ...]
EOF
    exit 2
}

set_release_profile() {
    export QUANTIZED_LAYER_PROFILE="${QUANTIZED_LAYER_PROFILE:-integrated}"
    export QUANTIZED_LAYER_SILU_LANES="${QUANTIZED_LAYER_SILU_LANES:-4}"
    export QUANTIZED_LAYER_RMS_LANES="${QUANTIZED_LAYER_RMS_LANES:-2}"
    export QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING="${QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING:-32}"
    export QUANTIZED_LAYER_ATTENTION_WAVE="${QUANTIZED_LAYER_ATTENTION_WAVE:-0}"
    export QUANTIZED_LAYER_PREFILL_FFN_OVERLAP="${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP:-1}"
}

build_script() {
    case "$1" in
        w4) printf '%s\n' scripts/build_vitis_quantized_w4_layer.sh ;;
        w8) printf '%s\n' scripts/build_vitis_quantized_w8_layer.sh ;;
        *) echo "precision must be w4 or w8" >&2; exit 2 ;;
    esac
}

valid_phase() {
    case "${1:-all}" in
        controller-xo|compute-xo|link|host|emconfig|run|all) ;;
        *) echo "phase must be controller-xo, compute-xo, link, host, emconfig, run or all" >&2; exit 2 ;;
    esac
}

command_name="${1:-}"
case "${command_name}" in
    check)
        [[ $# == 1 ]] || usage
        exec scripts/verify_source.sh
        ;;
    test-native)
        [[ $# == 1 ]] || usage
        exec scripts/run_native_regression.sh
        ;;
    dry-run)
        [[ $# -ge 2 && $# -le 3 ]] || usage
        precision="$2"
        phase="${3:-all}"
        valid_phase "${phase}"
        set_release_profile
        source scripts/quantized_layer_profiles.sh
        quantized_layer_profile_apply "${precision}"
        script="$(build_script "${precision}")"
        output_root="${COWAVE_QUANTIZED_LAYER_OUTPUT_DIR:-${repo_root}/.build}"
        echo "dry_run=1"
        echo "precision=${precision} phase=${phase}"
        echo "profile=${QUANTIZED_LAYER_PROFILE} silu_lanes=${CU_NL_LANES_CONFIG} rms_lanes=${CU_RMS_LANES_CONFIG} weight_read_outstanding=${QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING} attention_wave=${QUANTIZED_ATTENTION_WAVE_PIPELINE} prefill_ffn_overlap=${QUANTIZED_PREFILL_FFN_OVERLAP}"
        echo "output_root=${output_root}"
        echo "command=VITIS_ENV_SCRIPT=... XPLATFORM=... TARGET=hw_emu bash ${script} ${phase}"
        echo "host_command=VITIS_ENV_SCRIPT=... XILINX_HLS=... XILINX_XRT=... bash ${script} host"
        echo "cosim_command=HLS_QUANTIZED_LAYER_INTEGRATION_STAGE=prepare bash scripts/run_vitis_hls.sh tcl/run_quantized_layer_integration_cosim.tcl"
        ;;
    build|host)
        [[ $# == 2 || ("${command_name}" == build && $# == 3) ]] || usage
        precision="$2"
        phase="host"
        [[ "${command_name}" == build ]] && phase="${3:-all}"
        valid_phase "${phase}"
        set_release_profile
        script="$(build_script "${precision}")"
        exec bash "${script}" "${phase}"
        ;;
    cosim)
        [[ $# -ge 2 && $# -le 3 ]] || usage
        precision="$2"
        stage="${3:-all}"
        case "${stage}" in prepare|cosim|all) ;; *) usage ;; esac
        set_release_profile
        output_root="${COWAVE_QUANTIZED_LAYER_OUTPUT_DIR:-${repo_root}/.build}"
        export QUANTIZED_LAYER_PRECISION="${precision}"
        export HLS_QUANTIZED_LAYER_INTEGRATION_STAGE="${stage}"
        export LLM_FPGA_HLS_PROJECT_ROOT="${LLM_FPGA_HLS_PROJECT_ROOT:-${output_root}/cosim/${precision}}"
        exec scripts/run_vitis_hls.sh tcl/run_quantized_layer_integration_cosim.tcl
        ;;
    hwemu)
        [[ $# == 2 ]] || usage
        set_release_profile
        exec scripts/run_vitis_quantized_layer_hwemu_tmux.sh "$2"
        ;;
    reference)
        [[ $# == 1 ]] || usage
        set_release_profile
        exec scripts/run_quantized_layer_numeric_reference.sh
        ;;
    performance)
        set_release_profile
        shift
        exec scripts/run_quantized_layer_performance_eval.sh "$@"
        ;;
    *) usage ;;
esac
