#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"
tmux_bin="${TMUX_BIN:-tmux}"
prefill="${QUANTIZED_LAYER_EVAL_PREFILL:-10}"
source scripts/quantized_layer_profiles.sh
quantized_layer_profile_apply w4
profile="$QUANTIZED_LAYER_PROFILE"
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
read -r -a precisions <<<"${QUANTIZED_LAYER_EVAL_PRECISIONS:-w4 w8}"
[[ ${#precisions[@]} -ge 1 && ${#precisions[@]} -le 2 ]] || exit 2
for precision in "${precisions[@]}"; do
    [[ "$precision" == w4 || "$precision" == w8 ]] || exit 2
done
compare_reference="${QUANTIZED_LAYER_COMPARE_REFERENCE_ROOT:-}"
[[ -z "$compare_reference" ]] || compare_reference="$(realpath "$compare_reference")"
[[ "$prefill" =~ ^[1-9][0-9]*$ && "$prefill" -le 2047 ]] || exit 2
if [[ "${1:-}" != --worker ]]; then
    [[ $# == 0 ]] || { echo "usage: $0" >&2; exit 2; }
    output_root="${COWAVE_QUANTIZED_LAYER_OUTPUT_DIR:-${repo_root}/.build}"
    reference_output_root="${COWAVE_QUANTIZED_LAYER_REFERENCE_OUTPUT_DIR:-${output_root}/references}"
    run_root="${reference_output_root}/quantized_layer_numeric_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$run_root/source"
    cp -a include kernel tests host common scripts tcl config Makefile "$run_root/source/"
    (
        cd "$run_root/source"
        rg --files include kernel tests host common scripts tcl config | LC_ALL=C sort | xargs sha256sum
        sha256sum Makefile
    ) >"$run_root/source.sha256"
    printf -v command '%q ' env \
        "VITIS_ENV_SCRIPT=${VITIS_ENV_SCRIPT:-}" \
        "XILINX_HLS=${XILINX_HLS:-}" \
        "VITIS_HLS_INCLUDE=${VITIS_HLS_INCLUDE:-}" \
        "XILINX_XRT=${XILINX_XRT:-}" \
        "XPLATFORM=${XPLATFORM:-}" \
        "TARGET=${TARGET:-hw_emu}" \
        "DEVICE=${DEVICE:-xilinx_u50_gen3x16_xdma_5_202210_1}" \
        "FREQUENCY=${FREQUENCY:-200}" \
        "THREADS=${THREADS:-16}" \
        "PATH=${PATH:-}" \
        "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" \
        "PYTHONPATH=${PYTHONPATH:-}" \
        "COWAVE_QUANTIZED_LAYER_OUTPUT_DIR=${COWAVE_QUANTIZED_LAYER_OUTPUT_DIR:-}" \
        "COWAVE_QUANTIZED_LAYER_CONFIG_DIR=${COWAVE_QUANTIZED_LAYER_CONFIG_DIR:-}" \
        "QUANTIZED_LAYER_EVAL_PREFILL=$prefill" \
        "QUANTIZED_LAYER_PROFILE=$profile" \
        "QUANTIZED_LAYER_EVAL_PRECISIONS=${precisions[*]}" \
        "QUANTIZED_LAYER_ATTENTION_VARIANT=$attention_variant" \
        "QUANTIZED_LAYER_SILU_LANES=$worker_silu_lanes" \
        "QUANTIZED_LAYER_SILU_LANES_REQUESTED=${worker_silu_requested}" \
        "QUANTIZED_LAYER_RMS_LANES=$worker_rms_lanes" \
        "QUANTIZED_LAYER_RMS_LANES_REQUESTED=${worker_rms_requested}" \
        "QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING=$worker_weight_outstanding" \
        "QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING_REQUESTED=${worker_weight_requested}" \
        "QUANTIZED_LAYER_ATTENTION_WAVE=$worker_attention_wave" \
        "QUANTIZED_LAYER_ATTENTION_WAVE_REQUESTED=${worker_wave_requested}" \
        "QUANTIZED_LAYER_PREFILL_FFN_OVERLAP=$worker_prefill_ffn_overlap" \
        "QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED=${worker_prefill_ffn_overlap_requested}" \
        "QUANTIZED_LAYER_COMPARE_REFERENCE_ROOT=$compare_reference" \
        bash "$run_root/source/scripts/run_quantized_layer_numeric_reference.sh" --worker "$run_root"
    printf -v logfile '%q' "$run_root/pipeline.log"
    session="quantized_layer_numeric_$(date +%Y%m%d_%H%M%S)"
    "${tmux_bin}" new-session -d -s "$session" "exec $command >$logfile 2>&1"
    printf 'session=%s\nrun_root=%s\n' "$session" "$run_root"
    exit 0
fi
run_root="$(realpath "$2")"
trap 'rc=$?; printf "%s\n" "$rc" >"$run_root/exit.status"; printf "pipeline_exit_status=%s\nfinished_at=%s\n" "$rc" "$(date -Is)"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ -n "${VITIS_ENV_SCRIPT:-}" ]]; then
    [[ -r "${VITIS_ENV_SCRIPT}" ]] || {
        echo "Missing Vitis environment: ${VITIS_ENV_SCRIPT}" >&2
        exit 66
    }
    source "${VITIS_ENV_SCRIPT}" >/dev/null 2>&1
fi
if [[ -z "${XILINX_HLS:-}" && -z "${VITIS_HLS_INCLUDE:-}" ]]; then
    vitis_hls_bin="$(command -v vitis_hls || true)"
    if [[ -n "${vitis_hls_bin}" ]]; then
        XILINX_HLS="$(cd "$(dirname "${vitis_hls_bin}")/.." && pwd -P)"
        export XILINX_HLS
    fi
fi
hls_include="${VITIS_HLS_INCLUDE:-${XILINX_HLS:-}/include}"
[[ -d "${hls_include}" ]] || {
    echo "Numeric reference requires Vitis HLS headers; set VITIS_HLS_INCLUDE or XILINX_HLS" >&2
    exit 66
}
mkdir -p "$run_root/bin"
echo "profile=$profile attention_variant=$attention_variant rms_lanes=${CU_RMS_LANES_CONFIG} prefill_ffn_overlap=${QUANTIZED_PREFILL_FFN_OVERLAP} prefill=$prefill reference_kind=production_C_model"
for precision in "${precisions[@]}"; do
    quantized_layer_profile_apply "$precision"
    printf '%s\n' "profile=$profile" "${QUANTIZED_LAYER_PROFILE_CFLAGS[@]}" >"$run_root/${precision}_profile.txt"
    flags=(-DQUANTIZED_CSIM_FEEDBACK -DMM_STREAM_QUANTIZED_NARROW_ACCUM)
    if [[ "$precision" == w4 ]]; then
        flags+=(-DMM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE_CONFIG=128 -DMM_STREAM_8X128_INT4X4_PACKED_ACCUM_CONFIG=0)
        matrix=kernel/mm_stream_8x128_int4x4_block.cpp
        block=8
    else
        flags+=(-DQUANTIZED_ALIGNMENT_W8 -DMM_STREAM_QUANTIZED_SINGLE_ACCUM_BANK)
        matrix=kernel/mm_stream_4x128_int8x8_block.cpp
        block=4
    fi
    [[ "$profile" != integrated ]] || matrix=kernel/quantized_decode_rows.cpp
    for shape in small standard; do
        model_flags=()
        [[ "$shape" != small ]] || model_flags=(-DQWEN_TEST_SMALL)
        echo "stage=${precision}_${shape}_compile started_at=$(date -Is)"
        g++ -std=c++14 -O2 -Wno-unknown-pragmas -Iinclude -I"$hls_include" \
            "${flags[@]}" "${QUANTIZED_LAYER_PROFILE_CFLAGS[@]}" "${model_flags[@]}" tests/quantized_layer_numeric_reference.cpp \
            "kernel/control_cache_quantized_${precision}_layer.cpp" \
            "kernel/compute_core_quantized_${precision}_unified.cpp" "$matrix" \
            kernel/compute_stream.cpp -o "$run_root/bin/${precision}_${shape}" \
            >"$run_root/${precision}_${shape}.build.log" 2>&1
        modes=(random)
        [[ "$shape" != small ]] || modes=(zero random)
        for mode in "${modes[@]}"; do
            echo "stage=${precision}_${shape}_${mode}_reference started_at=$(date -Is)"
            run_prefill="$prefill"
            [[ "$shape" != small ]] || run_prefill=10
            timeout 14400 "$run_root/bin/${precision}_${shape}" --prefill "$run_prefill" --block-size "$block" \
                --weights "$mode" --output "$run_root/${precision}_${shape}_${mode}.tsv" \
                >"$run_root/${precision}_${shape}_${mode}.log" 2>&1
            rg 'C_MODEL_PHASE_COMPLETE|C_MODEL_REFERENCE_COMPLETE' "$run_root/${precision}_${shape}_${mode}.log"
        done
    done
    echo "stage=${precision}_host_build started_at=$(date -Is)"
    env "VITIS_QUANTIZED_${precision^^}_LAYER_ROOT=$run_root/$precision" \
        bash "scripts/build_vitis_quantized_${precision}_layer.sh" host \
        >"$run_root/${precision}_host_build.log" 2>&1
done
if [[ -n "$compare_reference" ]]; then
    echo "stage=compare_reference waiting_for=$compare_reference"
    deadline=$((SECONDS + 8 * 3600))
    until rg -q '^pipeline_exit_status=' "$compare_reference/pipeline.log"; do
        [[ "$SECONDS" -lt "$deadline" ]] || exit 124
        sleep 60
    done
    rg -q '^pipeline_exit_status=0$' "$compare_reference/pipeline.log"
    for precision in "${precisions[@]}"; do
        python3 scripts/compare_quantized_layer_outputs.py \
            "$compare_reference/${precision}_standard_random.tsv" \
            "$run_root/${precision}_standard_random.tsv" \
            >"$run_root/${precision}_reference_comparison.json"
    done
    echo 'ATTENTION_VARIANT_C_MODEL_EQUIVALENCE_PASS tolerance_raw=0'
fi
echo 'QUANTIZED_LAYER_C_REFERENCES_COMPLETE hwemu_comparison_pending=1'
