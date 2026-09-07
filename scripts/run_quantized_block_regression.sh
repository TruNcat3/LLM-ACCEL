#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-all}"
case "${mode}" in
    csim|synth|cosim|all) ;;
    *)
        echo "Usage: $0 [csim|synth|cosim|all]" >&2
        exit 2
        ;;
esac

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="logs/quantized_block_regression_${timestamp}"
mkdir -p "${log_dir}"

run_target() {
    local label="$1"
    shift
    local log_path="${log_dir}/${label}.log"
    echo
    echo "=== ${label} ==="
    echo "log=${log_path}"
    "$@" 2>&1 | tee "${log_path}"
}

run_variant() {
    local stage="$1"
    local label="$2"
    local flags="$3"
    local tcl="$4"
    local hls_env=()
    if [ -n "${flags}" ]; then
        hls_env+=(HLS_EXTRA_CFLAGS="${flags}")
    fi
    case "${stage}" in
        csim)
            run_target "${label}" env "${hls_env[@]}" HLS_CSIM_ONLY=1 \
                scripts/run_vitis_hls.sh "${tcl}"
            ;;
        synth)
            run_target "${label}" env "${hls_env[@]}" \
                HLS_COSIM_PREPARE=1 HLS_COSIM_PREPARE_ONLY=1 \
                HLS_COSIM_SKIP_CSIM=1 scripts/run_vitis_hls.sh "${tcl}"
            ;;
        cosim)
            run_target "${label}" env "${hls_env[@]}" \
                scripts/run_vitis_hls.sh "${tcl}"
            ;;
    esac
}

variants=(
    "baseline_w4||cases/quantized-block/tcl/run_cosim_mm_stream_8x64_int4x4_block.tcl"
    "baseline_w8||cases/quantized-block/tcl/run_cosim_mm_stream_4x128_int8x8_block.tcl"
    "dspacc_w4|-DMM_STREAM_QUANTIZED_USE_DSP_ACCUM|cases/quantized-block/tcl/run_cosim_mm_stream_8x64_int4x4_block_dspacc.tcl"
    "dspacc_narrow_w8|-DMM_STREAM_QUANTIZED_USE_DSP_ACCUM -DMM_STREAM_QUANTIZED_NARROW_ACCUM|cases/quantized-block/tcl/run_cosim_mm_stream_4x128_int8x8_block_dspacc_narrow.tcl"
    "single_narrow_w4|-DMM_STREAM_QUANTIZED_SINGLE_ACCUM_BANK -DMM_STREAM_QUANTIZED_NARROW_ACCUM|cases/quantized-block/tcl/run_cosim_mm_stream_8x64_int4x4_block_single_narrow.tcl"
    "single_narrow_w8|-DMM_STREAM_QUANTIZED_SINGLE_ACCUM_BANK -DMM_STREAM_QUANTIZED_NARROW_ACCUM|cases/quantized-block/tcl/run_cosim_mm_stream_4x128_int8x8_block_single_narrow.tcl"
)

for stage in csim synth cosim; do
    if [ "${mode}" != "all" ] && [ "${mode}" != "${stage}" ]; then
        continue
    fi
    for variant in "${variants[@]}"; do
        IFS='|' read -r label flags tcl <<<"${variant}"
        run_variant "${stage}" "${stage}_${label}" "${flags}" "${tcl}"
    done
done

echo
echo "QUANTIZED BLOCK REGRESSION PASS mode=${mode} logs=${log_dir}"
