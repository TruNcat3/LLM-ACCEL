#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

env_script="${VITIS_ENV_SCRIPT:-/home/hepc/env/vitis_env_22.sh}"
build_dir=""
profile="qwen-layer"
mode="run"
layers="1"
tokens="0"
seed="20260701"
model_kind="random"
timeout_seconds="604800"
skip_weight_preload=0
verbose_ops=0
profile_op="q"
profile_wave="0"
profile_wave_count="1"
profile_k_limit=""
profile_debug_stage=""
profile_core_mask=""
profile_token_count="1"
profile_single_launch=0
profile_zero_weight_stream=0
attention_position="0"
attention_prefill_len=""
attention_prefill_start="0"
attention_phase="pd"

usage() {
    echo "Usage: $0 --build-dir DIR [options]"
    echo
    echo "Options:"
    echo "  --profile NAME          Model/xclbin profile (default: qwen-layer)"
    echo "  --mode MODE             run, load-only, decode-smoke, profile-mm-wave, profile-attention, profile-attention-pd, profile-attention-block, profile-attention-sublayer, profile-ffn-sublayer, profile-prefill-block, or profile-prefill-vector"
    echo "  --layers N              Decoder layers for run mode"
    echo "  --tokens IDS            Comma-separated prompt token IDs"
    echo "  --seed N                Deterministic random seed"
    echo "  --random-model          Deterministic non-zero weights (default)"
    echo "  --zero-model            Deterministic zero weights"
    echo "  --skip-weight-preload   Do not migrate weight buffers"
    echo "  --verbose-ops           Print every controller invocation"
    echo "  --timeout SECONDS       Host wall-clock timeout"
    echo
    echo "profile-mm-wave options:"
    echo "  --op NAME               q,k,v,o,ffn-gate,ffn-up,ffn-down"
    echo "  --wave N                First output wave"
    echo "  --wave-count N          Number of waves; 0 means all remaining"
    echo "  --k-limit N             Optional K limit; omit for full K"
    echo "  --profile-debug-stage N Debug short-circuit stage"
    echo "  --profile-core-mask N   1=core0, 2=core1, 3=both"
    echo "  --profile-tokens N      Active MM rows; 1..8"
    echo "  --profile-single-launch Run requested waves in one controller launch"
    echo "  --profile-zero-weight-stream"
    echo
    echo "attention profile options:"
    echo "  --position N            Decode position to test"
    echo "  --prefill-len N         P/D attention prefill length"
    echo "  --prefill-start N       First P-stage position (resume support)"
    echo "  --phase p|d|pd          P/D attention phase selection"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --build-dir)
            build_dir="$2"
            shift 2
            ;;
        --profile)
            profile="$2"
            shift 2
            ;;
        --mode)
            mode="$2"
            shift 2
            ;;
        --layers)
            layers="$2"
            shift 2
            ;;
        --tokens)
            tokens="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --random-model)
            model_kind="random"
            shift
            ;;
        --zero-model)
            model_kind="zero"
            shift
            ;;
        --skip-weight-preload)
            skip_weight_preload=1
            shift
            ;;
        --verbose-ops)
            verbose_ops=1
            shift
            ;;
        --timeout)
            timeout_seconds="$2"
            shift 2
            ;;
        --op)
            profile_op="$2"
            shift 2
            ;;
        --wave)
            profile_wave="$2"
            shift 2
            ;;
        --wave-count)
            profile_wave_count="$2"
            shift 2
            ;;
        --k-limit)
            profile_k_limit="$2"
            shift 2
            ;;
        --profile-debug-stage)
            profile_debug_stage="$2"
            shift 2
            ;;
        --profile-core-mask)
            profile_core_mask="$2"
            shift 2
            ;;
        --profile-tokens)
            profile_token_count="$2"
            shift 2
            ;;
        --profile-single-launch)
            profile_single_launch=1
            shift
            ;;
        --profile-zero-weight-stream)
            profile_zero_weight_stream=1
            shift
            ;;
        --position)
            attention_position="$2"
            shift 2
            ;;
        --prefill-len)
            attention_prefill_len="$2"
            shift 2
            ;;
        --prefill-start)
            attention_prefill_start="$2"
            shift 2
            ;;
        --phase)
            attention_phase="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "${profile}" in
    small|medium|qwen-layer|qwen-layer-long|qwen2.5-3b)
        ;;
    *)
        echo "Unsupported profile: ${profile}" >&2
        exit 2
        ;;
esac

case "${mode}" in
    run|load-only|decode-smoke|profile-mm-wave|profile-attention|profile-attention-pd|profile-attention-block|profile-attention-sublayer|profile-ffn-sublayer|profile-prefill-block|profile-prefill-vector)
        ;;
    *)
        echo "Unsupported mode: ${mode}" >&2
        exit 2
        ;;
esac

if [ -z "${build_dir}" ]; then
    echo "--build-dir is required." >&2
    exit 2
fi
if [ ! -r "${env_script}" ]; then
    echo "Missing Vitis environment script: ${env_script}" >&2
    exit 66
fi

build_dir="$(realpath "${build_dir}")"
for input in \
    "${build_dir}/qwen_8x64_dual.xclbin" \
    "${build_dir}/host_qwen_8x64.exe" \
    "${build_dir}/emconfig.json"
do
    if [ ! -s "${input}" ]; then
        echo "Missing or empty hw_emu input: ${input}" >&2
        exit 66
    fi
done

source "${env_script}" >/dev/null 2>&1

if [ "${model_kind}" = "random" ]; then
    model_args=(--random-model --seed "${seed}")
else
    model_args=(--zero-model)
fi

case "${mode}" in
    run)
        host_args=(
            --xclbin ./qwen_8x64_dual.xclbin
            --mode run
            --profile "${profile}"
            --tokens "${tokens}"
            --layers "${layers}"
            "${model_args[@]}"
        )
        ;;
    load-only)
        host_args=(
            --xclbin ./qwen_8x64_dual.xclbin
            --mode run
            --profile "${profile}"
            --tokens "${tokens}"
            --layers "${layers}"
            --load-only
            "${model_args[@]}"
        )
        ;;
    decode-smoke)
        host_args=(
            --xclbin ./qwen_8x64_dual.xclbin
            --mode verify-decode-smoke
            --profile "${profile}"
            --seed "${seed}"
        )
        ;;
    profile-mm-wave)
        host_args=(
            --xclbin ./qwen_8x64_dual.xclbin
            --mode profile-mm-wave
            --profile "${profile}"
            --op "${profile_op}"
            --wave "${profile_wave}"
            --wave-count "${profile_wave_count}"
            --profile-tokens "${profile_token_count}"
            "${model_args[@]}"
        )
        if [ -n "${profile_k_limit}" ]; then
            host_args+=(--k-limit "${profile_k_limit}")
        fi
        if [ -n "${profile_debug_stage}" ]; then
            host_args+=(--profile-debug-stage "${profile_debug_stage}")
        fi
        if [ -n "${profile_core_mask}" ]; then
            host_args+=(--profile-core-mask "${profile_core_mask}")
        fi
        if [ "${profile_single_launch}" -eq 1 ]; then
            host_args+=(--profile-single-launch)
        fi
        if [ "${profile_zero_weight_stream}" -eq 1 ]; then
            host_args+=(--profile-zero-weight-stream)
        fi
        ;;
    profile-attention)
        host_args=(
            --xclbin ./qwen_8x64_dual.xclbin
            --mode profile-attention
            --profile "${profile}"
            --position "${attention_position}"
            "${model_args[@]}"
        )
        ;;
    profile-attention-pd)
        host_args=(
            --xclbin ./qwen_8x64_dual.xclbin
            --mode profile-attention-pd
            --profile "${profile}"
            --phase "${attention_phase}"
            "${model_args[@]}"
        )
        if [ -n "${attention_prefill_len}" ]; then
            host_args+=(--prefill-len "${attention_prefill_len}")
        fi
        if [ "${attention_prefill_start}" != "0" ]; then
            host_args+=(--prefill-start "${attention_prefill_start}")
        fi
        ;;
    profile-attention-block)
        host_args=(
            --xclbin ./qwen_8x64_dual.xclbin
            --mode profile-attention-block
            --profile "${profile}"
            --position "${attention_position}"
            "${model_args[@]}"
        )
        ;;
    profile-attention-sublayer)
        host_args=(
            --xclbin ./qwen_8x64_dual.xclbin
            --mode profile-attention-sublayer
            --profile "${profile}"
            --position "${attention_position}"
            "${model_args[@]}"
        )
        ;;
    profile-ffn-sublayer)
        host_args=(
            --xclbin ./qwen_8x64_dual.xclbin
            --mode profile-ffn-sublayer
            --profile "${profile}"
            --position "${attention_position}"
            "${model_args[@]}"
        )
        ;;
    profile-prefill-block)
        host_args=(
            --xclbin ./qwen_8x64_dual.xclbin
            --mode profile-prefill-block
            --profile "${profile}"
            "${model_args[@]}"
        )
        if [ -n "${attention_prefill_len}" ]; then
            host_args+=(--prefill-len "${attention_prefill_len}")
        fi
        if [ "${attention_prefill_start}" != "0" ]; then
            host_args+=(--prefill-start "${attention_prefill_start}")
        fi
        ;;
    profile-prefill-vector)
        host_args=(
            --xclbin ./qwen_8x64_dual.xclbin
            --mode profile-prefill-vector
            --profile "${profile}"
            "${model_args[@]}"
        )
        if [ -n "${attention_prefill_len}" ]; then
            host_args+=(--prefill-len "${attention_prefill_len}")
        fi
        ;;
esac

if [ "${skip_weight_preload}" -eq 1 ]; then
    host_args+=(--skip-weight-preload)
fi
if [ "${verbose_ops}" -eq 1 ]; then
    host_args+=(--verbose-ops)
fi

mkdir -p logs
profile_tag="${profile//[^A-Za-z0-9]/_}"
timestamp="$(date +%Y%m%d_%H%M%S_%N)_${BASHPID}"
log_path="$PWD/logs/vitis_8x64_hwemu_${profile_tag}_${mode}_${timestamp}.log"
status_path="$PWD/logs/vitis_8x64_hwemu_${profile_tag}_${mode}_${timestamp}.status"

echo "started_at=$(date -Is)" | tee "${log_path}"
echo "build_dir=${build_dir}" | tee -a "${log_path}"
echo "profile=${profile} mode=${mode} layers=${layers} tokens=${tokens}" | tee -a "${log_path}"
if [ "${mode}" = "profile-mm-wave" ]; then
    echo "profile_mm_wave op=${profile_op} wave=${profile_wave} wave_count=${profile_wave_count} k_limit=${profile_k_limit:-full} token_count=${profile_token_count}" | tee -a "${log_path}"
fi
if [ "${mode}" = "profile-attention" ] || [ "${mode}" = "profile-attention-block" ] || [ "${mode}" = "profile-attention-sublayer" ] || [ "${mode}" = "profile-ffn-sublayer" ]; then
    echo "profile_position=${attention_position}" | tee -a "${log_path}"
fi
if [ "${mode}" = "profile-attention-pd" ]; then
    echo "attention_pd phase=${attention_phase} prefill_len=${attention_prefill_len:-default} prefill_start=${attention_prefill_start}" | tee -a "${log_path}"
fi
if [ "${mode}" = "profile-prefill-block" ] || [ "${mode}" = "profile-prefill-vector" ]; then
    echo "prefill_stage mode=${mode} prefill_start=${attention_prefill_start} prefill_len=${attention_prefill_len:-8}" | tee -a "${log_path}"
fi
echo "model=${model_kind} timeout=${timeout_seconds}" | tee -a "${log_path}"

set +e
(
    cd "${build_dir}"
    XCL_EMULATION_MODE=hw_emu \
        timeout "${timeout_seconds}" \
        ./host_qwen_8x64.exe "${host_args[@]}"
) 2>&1 | tee -a "${log_path}"
status="${PIPESTATUS[0]}"
set -e

printf '%s\n' "${status}" > "${status_path}"
echo "finished_at=$(date -Is)" | tee -a "${log_path}"
echo "exit_status=${status}" | tee -a "${log_path}"
echo "log_file=${log_path}" | tee -a "${log_path}"
echo "status_file=${status_path}" | tee -a "${log_path}"
exit "${status}"
