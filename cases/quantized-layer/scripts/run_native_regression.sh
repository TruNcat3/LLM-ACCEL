#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"

if [[ -n "${VITIS_ENV_SCRIPT:-}" ]]; then
    [[ -r "${VITIS_ENV_SCRIPT}" ]] || {
        echo "Missing Vitis environment: ${VITIS_ENV_SCRIPT}" >&2
        exit 66
    }
    source "${VITIS_ENV_SCRIPT}" >/dev/null 2>&1
fi

hls_root="${XILINX_HLS:-}"
if [[ -z "${hls_root}" ]]; then
    vitis_hls_bin="$(command -v vitis_hls || true)"
    if [[ -n "${vitis_hls_bin}" ]]; then
        hls_root="$(cd "$(dirname "${vitis_hls_bin}")/.." && pwd -P)"
    fi
fi
hls_include="${VITIS_HLS_INCLUDE:-${hls_root}/include}"
[[ -d "${hls_include}" ]] || {
    echo "Native regression requires Vitis HLS headers; set XILINX_HLS or VITIS_HLS_INCLUDE" >&2
    exit 66
}

cxx="${CXX:-g++}"
command -v "${cxx}" >/dev/null 2>&1 || {
    echo "C++ compiler not found: ${cxx}" >&2
    exit 66
}
output_root="${COWAVE_QUANTIZED_LAYER_OUTPUT_DIR:-${repo_root}/.build}"
native_root="${COWAVE_QUANTIZED_LAYER_NATIVE_OUTPUT_DIR:-${output_root}/native}"
mkdir -p "${native_root}"

common_flags=(
    -std=c++14 -O0 -Wall -Wextra -Wno-unknown-pragmas
    -Iinclude -Icommon/include "-I${hls_include}"
)

compile_run() {
    local name="$1"
    shift
    echo "native_test=${name}"
    "${cxx}" "${common_flags[@]}" "$@" -o "${native_root}/${name}"
    "${native_root}/${name}"
}

compile_run quantized_layer_task_tb \
    tests/quantized_layer_task_tb.cpp
compile_run quantized_layer_schedule_tb \
    tests/quantized_layer_schedule_tb.cpp
compile_run quantized_w4_layer_math_tb \
    -DMM_STREAM_QUANTIZED_NARROW_ACCUM \
    tests/quantized_w4_layer_math_tb.cpp
compile_run quantized_w8_layer_math_tb \
    -DQUANTIZED_ALIGNMENT_W8 -DMM_STREAM_QUANTIZED_SINGLE_ACCUM_BANK \
    -DMM_STREAM_QUANTIZED_NARROW_ACCUM tests/quantized_w8_layer_math_tb.cpp
compile_run quantized_w4_layer_runtime_tb \
    -DQWEN_TEST_SMALL -DMM_STREAM_QUANTIZED_NARROW_ACCUM \
    tests/quantized_w4_layer_runtime_tb.cpp
compile_run quantized_w8_layer_runtime_tb \
    -DQWEN_TEST_SMALL -DQUANTIZED_ALIGNMENT_W8 \
    -DMM_STREAM_QUANTIZED_SINGLE_ACCUM_BANK -DMM_STREAM_QUANTIZED_NARROW_ACCUM \
    tests/quantized_w8_layer_runtime_tb.cpp
compile_run control_cache_quantized_w4_layer_small_tb \
    -DQWEN_TEST_SMALL -DMM_STREAM_QUANTIZED_NARROW_ACCUM \
    tests/control_cache_quantized_w4_layer_small_tb.cpp \
    kernel/control_cache_quantized_w4_layer.cpp
compile_run control_cache_quantized_w8_layer_small_tb \
    -DQWEN_TEST_SMALL -DQUANTIZED_ALIGNMENT_W8 \
    -DMM_STREAM_QUANTIZED_SINGLE_ACCUM_BANK -DMM_STREAM_QUANTIZED_NARROW_ACCUM \
    tests/control_cache_quantized_w8_layer_small_tb.cpp \
    kernel/control_cache_quantized_w8_layer.cpp

echo "quantized-layer native regression: PASS"
