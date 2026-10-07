#!/usr/bin/env bash
set -euo pipefail
script_path="$(realpath "$0")"
repo_root="$(cd "$(dirname "${script_path}")/.." && pwd -P)"
cd "${repo_root}"
tmux_bin="${TMUX_BIN:-tmux}"
prefill="${QUANTIZED_LAYER_EVAL_PREFILL:-10}"
reference_root="${QUANTIZED_LAYER_REFERENCE_ROOT:-}"
source scripts/quantized_layer_profiles.sh
quantized_layer_profile_apply w4
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
[[ ${#precisions[@]} -gt 0 && ${#precisions[@]} -le 2 ]] || exit 2
for precision in "${precisions[@]}"; do
    [[ "$precision" == w4 || "$precision" == w8 ]] || exit 2
    [[ -n "${reference_root}" && -s "${reference_root}/${precision}_standard_random.tsv" ]] || {
        echo "Set QUANTIZED_LAYER_REFERENCE_ROOT to completed full-layer C reference dumps" >&2
        exit 66
    }
done
reference_root="$(realpath "${reference_root}")"
rg -q '^QUANTIZED_LAYER_C_REFERENCES_COMPLETE ' "$reference_root/pipeline.log"
rg -q '^pipeline_exit_status=0$' "$reference_root/pipeline.log"
[[ "${prefill}" =~ ^[1-9][0-9]*$ && "${prefill}" -le 2047 ]] || {
    echo "QUANTIZED_LAYER_EVAL_PREFILL must be in 1..2047" >&2
    exit 2
}
# Reject stale/wrong-shape references before an hours-long RTL run. This is a
# production C-model comparison, not an independent arithmetic oracle.
python3 - "$PWD/scripts" "$reference_root" "$prefill" "${precisions[@]}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from compare_quantized_layer_outputs import read_dump
root, prefill = Path(sys.argv[2]), int(sys.argv[3])
for precision in sys.argv[4:]:
    metadata, _ = read_dump(root / f"{precision}_standard_random.tsv")
    expected = dict(bits=int(precision[1]), hidden=2048, ffn=11008,
                    q_heads=16, kv_heads=2, head_dim=128, prefill=prefill,
                    block=8 if precision == "w4" else 4, layer=0,
                    weights="random", fixture="deterministic_v1")
    if metadata != expected:
        raise SystemExit(f"Reference workload mismatch: {precision}: {metadata}")
    for path in (root / f"{precision}_standard_random.log",
                 root / "source.sha256", root / "bin" / f"{precision}_standard"):
        if not path.is_file() or not path.stat().st_size:
            raise SystemExit(f"Missing reference provenance: {path}")
    print(f"reference_preflight=PASS precision={precision} prefill={prefill}")
PY
if [[ "${1:-}" == --check ]]; then
    [[ "$#" == 1 ]] || exit 2
    exit 0
fi

if [[ "${1:-}" != --worker ]]; then
    if [[ "$#" -gt 1 ]]; then
        echo "usage: $0 [EXISTING_WORKER_LOG_TO_WAIT_FOR]" >&2
        exit 2
    fi
    wait_log="${1:-}"
    [[ -z "${wait_log}" ]] || wait_log="$(realpath "${wait_log}")"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    output_root="${COWAVE_QUANTIZED_LAYER_OUTPUT_DIR:-${repo_root}/.build}"
    output="${COWAVE_QUANTIZED_LAYER_EVALUATION_OUTPUT_DIR:-${output_root}/evaluations}/quantized_layer_performance_${timestamp}"
    mkdir -p "${output}"
    session="llm_quantized_layer_eval_${timestamp}"
    printf -v command '%q ' env \
        "VITIS_ENV_SCRIPT=${VITIS_ENV_SCRIPT:-}" \
        "XILINX_HLS=${XILINX_HLS:-}" \
        "VITIS_HLS_INCLUDE=${VITIS_HLS_INCLUDE:-}" \
        "XILINX_XRT=${XILINX_XRT:-}" \
        "XCLBINUTIL=${XCLBINUTIL:-}" \
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
        "VITIS_MIN_AVAILABLE_GIB=${VITIS_MIN_AVAILABLE_GIB:-40}" \
        "QUANTIZED_LAYER_EVAL_PREFILL=${prefill}" \
        "QUANTIZED_LAYER_EVAL_PRECISIONS=${precisions[*]}" \
        "QUANTIZED_LAYER_ATTENTION_VARIANT=${attention_variant}" \
        "QUANTIZED_LAYER_PROFILE=${QUANTIZED_LAYER_PROFILE}" \
        "QUANTIZED_LAYER_SILU_LANES=${worker_silu_lanes}" \
        "QUANTIZED_LAYER_SILU_LANES_REQUESTED=${worker_silu_requested}" \
        "QUANTIZED_LAYER_RMS_LANES=${worker_rms_lanes}" \
        "QUANTIZED_LAYER_RMS_LANES_REQUESTED=${worker_rms_requested}" \
        "QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING=${worker_weight_outstanding}" \
        "QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING_REQUESTED=${worker_weight_requested}" \
        "QUANTIZED_LAYER_ATTENTION_WAVE=${worker_attention_wave}" \
        "QUANTIZED_LAYER_ATTENTION_WAVE_REQUESTED=${worker_wave_requested}" \
        "QUANTIZED_LAYER_PREFILL_FFN_OVERLAP=${worker_prefill_ffn_overlap}" \
        "QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED=${worker_prefill_ffn_overlap_requested}" \
        "QUANTIZED_LAYER_ALLOW_PARALLEL=${QUANTIZED_LAYER_ALLOW_PARALLEL:-0}" \
        "QUANTIZED_W4_LAYER_TIMEOUT=${QUANTIZED_W4_LAYER_TIMEOUT:-86400}" \
        "QUANTIZED_W8_LAYER_TIMEOUT=${QUANTIZED_W8_LAYER_TIMEOUT:-86400}" \
        "QUANTIZED_LAYER_REFERENCE_ROOT=${reference_root}" \
        "${script_path}" --worker "${output}" "${wait_log}"
    printf -v quoted_log '%q' "${output}/evaluation.log"
    "${tmux_bin}" new-session -d -s "${session}" "exec ${command} >${quoted_log} 2>&1"
    printf 'session=%s\noutput=%s\n' "${session}" "${output}"
    exit 0
fi

output="$2"
wait_log="${3:-}"
echo "attention_variant=${attention_variant} prefill=${prefill} reference_root=${reference_root}"
echo "layer_profile=${QUANTIZED_LAYER_PROFILE}"
echo "prefill_ffn_overlap=${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP}"
echo "prefill_ffn_overlap_requested=${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED:-0}"
echo "rms_lanes=${CU_RMS_LANES_CONFIG}"
trap 'rc=$?; printf "evaluation_exit_status=%s\nfinished_at=%s\n" "$rc" "$(date -Is)"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

stop_run_tree() {
    local pid="$1" child
    for child in $(pgrep -P "${pid}" || true); do
        stop_run_tree "${child}"
    done
    kill -TERM "${pid}" 2>/dev/null || true
}

wait_for_run() {
    local log="$1" pid_file="${1%.log}.pid" pid state sim_dir
    [[ -s "${log}" && -s "${pid_file}" ]] || { echo "Missing worker log/PID: ${log}" >&2; return 66; }
    pid="$(<"${pid_file}")"
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 65
    while ! rg -q '^exit_status=' "${log}"; do
        sim_dir="$(sed -n 's/^INFO: \[HW-EMU 05\] Path of the simulation directory : //p' "${log}" | tail -n 1)"
        if [[ -n "${sim_dir}" && -f "${sim_dir}/simulate.log" ]] &&
            rg -q 'QUANTIZED_TRACE_(RECORDING_FAILED|MISSING)' "${sim_dir}/simulate.log"; then
            echo "RTL recording failed; stopping this evaluation worker: ${pid}" >&2
            stop_run_tree "${pid}"
            return 65
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "Worker exited without status: ${log}" >&2
            return 65
        fi
        sleep 60
    done
    state="$(sed -n 's/^exit_status=//p' "${log}" | tail -n 1)"
    [[ "${state}" == 0 ]] || { echo "Worker failed: ${log} exit=${state}" >&2; return 65; }
}

if [[ -n "${wait_log}" ]]; then
    echo "waiting_for_existing_run=${wait_log}"
    wait_for_run "${wait_log}"
fi

for precision in "${precisions[@]}"; do
    echo "starting_precision=${precision} at=$(date -Is)"
    quantized_layer_profile_apply "$precision"
    # The integrated D1 uses merged decode projections, not the legacy four
    # processes. Keep full P/D and function evidence without mislabeling its
    # activity as legacy projection-stage overlap.
    stage_trace=1
    [[ "$QUANTIZED_LAYER_PROFILE" != integrated ]] || stage_trace=0
    result_dir="${output}/${precision}"
    mkdir "${result_dir}"
    if [[ "${precision}" == w4 ]]; then
        block=8
        launch="$(FREQUENCY=200 QUANTIZED_LAYER_TRACE=1 QUANTIZED_LAYER_STAGE_TRACE="$stage_trace" QUANTIZED_LAYER_FUNCTION_TRACE=1 \
            QUANTIZED_LAYER_PREFILL_FFN_OVERLAP="${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP}" \
            QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED="${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED:-0}" \
            QUANTIZED_W4_LAYER_WEIGHTS=random \
            QUANTIZED_LAYER_DUMP_OUTPUT="${result_dir}/actual_outputs.tsv" \
            QUANTIZED_W4_LAYER_PREFILL="${prefill}" QUANTIZED_W4_LAYER_BLOCK_SIZE=8 \
            scripts/run_vitis_quantized_layer_hwemu_tmux.sh w4)"
    else
        block=4
        launch="$(FREQUENCY=200 QUANTIZED_LAYER_TRACE=1 QUANTIZED_LAYER_STAGE_TRACE="$stage_trace" QUANTIZED_LAYER_FUNCTION_TRACE=1 \
            QUANTIZED_LAYER_PREFILL_FFN_OVERLAP="${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP}" \
            QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED="${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED:-0}" \
            QUANTIZED_W8_LAYER_WEIGHTS=random \
            QUANTIZED_LAYER_DUMP_OUTPUT="${result_dir}/actual_outputs.tsv" \
            QUANTIZED_W8_LAYER_PREFILL="${prefill}" QUANTIZED_W8_LAYER_BLOCK_SIZE=4 \
            scripts/run_vitis_quantized_layer_hwemu_tmux.sh w8)"
    fi
    printf '%s\n' "${launch}" | tee "${result_dir}/launch.txt"
    log="$(sed -n 's/^log=//p' <<<"${launch}")"
    # The tmux child may not have opened its output yet.
    for attempt in {1..20}; do
        [[ -s "${log}" ]] && break
        sleep 1
    done
    wait_for_run "${log}"
    sim_dir="$(sed -n 's/^INFO: \[HW-EMU 05\] Path of the simulation directory : //p' "${log}" | tail -n 1)"
    [[ -d "${sim_dir}" ]] || { echo "Missing simulation directory" >&2; exit 66; }
    cp "${log}" "${result_dir}/host.log"
    for file in quantized_layer_transitions.tsv quantized_function_transitions.tsv profile_kernels.csv timeline_kernels.csv simulate.log; do
        cp "${sim_dir}/${file}" "${result_dir}/${file}"
    done
    cp "${reference_root}/${precision}_standard_random.tsv" "${result_dir}/reference_outputs.tsv"
    cp "${reference_root}/${precision}_standard_random.log" "${result_dir}/reference.log"
    cp "${reference_root}/source.sha256" "${result_dir}/reference_source.sha256"
    sha256sum "${reference_root}/bin/${precision}_standard" >"${result_dir}/reference_binary.sha256"
    python3 scripts/compare_quantized_layer_outputs.py \
        "${result_dir}/reference_outputs.tsv" "${result_dir}/actual_outputs.tsv" \
        >"${result_dir}/numerical.json"
    scripts/report_quantized_full_layer_hwemu.sh \
        "${result_dir}/quantized_layer_transitions.tsv" "${precision}" "${prefill}" "${block}" 200 \
        "${result_dir}/host.log" \
        --numerical-reference "${result_dir}/reference_outputs.tsv" \
        --numerical-output "${result_dir}/actual_outputs.tsv" >"${result_dir}/performance.json"
    python3 scripts/report_quantized_function_trace.py \
        "${result_dir}/quantized_function_transitions.tsv" "${result_dir}/performance.json" \
        >"${result_dir}/function_activity.json"
    extra_evidence=(quantized_function_transitions.tsv function_activity.json)
    if [[ "${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP}" == 1 ]]; then
        cp "${sim_dir}/quantized_prefill_silu_nested_transitions.tsv" "${result_dir}/"
        python3 scripts/report_quantized_prefill_silu_overlap.py \
            "${result_dir}/quantized_prefill_silu_nested_transitions.tsv" \
            "${result_dir}/performance.json" \
            >"${result_dir}/prefill_silu_overlap.json"
        extra_evidence+=(quantized_prefill_silu_nested_transitions.tsv prefill_silu_overlap.json)
    fi
    if [[ "$stage_trace" == 1 ]]; then
        cp "${sim_dir}/quantized_projection_stage_transitions.tsv" "${result_dir}/"
        python3 scripts/analyze_quantized_projection_overlap.py \
        "${result_dir}/quantized_projection_stage_transitions.tsv" \
        "${result_dir}/quantized_layer_transitions.tsv" "${result_dir}/host.log" \
        "${precision}" "${prefill}" "${block}" 200 >"${result_dir}/projection_overlap.json"
        extra_evidence+=(quantized_projection_stage_transitions.tsv projection_overlap.json)
    fi
    (cd "${result_dir}"; sha256sum host.log launch.txt quantized_layer_transitions.tsv \
        profile_kernels.csv timeline_kernels.csv simulate.log performance.json \
        reference_outputs.tsv actual_outputs.tsv numerical.json reference.log \
        reference_source.sha256 reference_binary.sha256 "${extra_evidence[@]}" >checksums.sha256)
    echo "validated_precision=${precision} result=${result_dir}/performance.json"
done
echo "QUANTIZED_LAYER_PERFORMANCE_EVALUATION_PASS"
