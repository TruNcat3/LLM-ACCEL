#!/usr/bin/env bash
set -euo pipefail

script_path="$(realpath "$0")"
cd "$(dirname "${script_path}")/.."

work_root="${VITIS_8X64_QWEN3B_WORK_ROOT:-/tmp/llm_accel_qwen3b_q214_resident_fix}"
device="${DEVICE:-xilinx_u50_gen3x16_xdma_5_202210_1}"
build_dir="${work_root}/build.hw_emu.${device}"
prompt_tokens="${VITIS_8X64_E2E_PROMPT_TOKENS:-8}"
# G2 is the minimum end-to-end generation gate that includes a real decode
# forward.  G1 stops after the first LM-head sample and therefore measures
# prefill/TTFT only.
generated_tokens="${VITIS_8X64_E2E_MAX_NEW_TOKENS:-2}"
layers="${VITIS_8X64_E2E_LAYERS:-36}"
block_size="${VITIS_8X64_E2E_PREFILL_BLOCK_SIZE:-8}"
seed="${VITIS_8X64_RESIDENT_SEED:-20260718}"
model_source="${VITIS_8X64_E2E_MODEL_SOURCE:-random}"
checkpoint_dir="${VITIS_8X64_E2E_DATA_DIR:-$PWD/data}"
checkpoint_dir="$(realpath -m "${checkpoint_dir}")"
tie_embeddings="${VITIS_8X64_E2E_TIE_EMBEDDINGS:-1}"
verify_e2e_golden="${VITIS_8X64_E2E_VERIFY_GOLDEN:-1}"
# One standard P8 layer takes roughly three hours in RTL HW Emu on this host.
# P8 prefill plus one decode forward traverses 72 decoder layers, so retain a
# two-week guard by default rather than terminating a healthy run after 7 days.
timeout_seconds="${VITIS_8X64_HW_EMU_TIMEOUT:-1209600}"

line_has_fields() {
    local padded=" $1 "
    shift
    local required
    for required in "$@"; do
        if [[ "${padded}" != *" ${required} "* ]]; then
            return 1
        fi
    done
}

line_has_keys() {
    local line="$1"
    shift
    local key word found
    for key in "$@"; do
        found=0
        for word in ${line}; do
            if [[ "${word}" == "${key}="* ]] && [ -n "${word#*=}" ]; then
                found=1
                break
            fi
        done
        if [ "${found}" -ne 1 ]; then
            return 1
        fi
    done
}
min_available_gib="${VITIS_MIN_AVAILABLE_GIB:-80}"
env_script="${VITIS_ENV_SCRIPT:-/home/hepc/env/vitis_env_22.sh}"
host_exe="${build_dir}/host_qwen_8x64.exe"
xclbin="${build_dir}/qwen_8x64_dual.xclbin"
emconfig="${build_dir}/emconfig.json"

if [ "${1:-}" = "--worker" ]; then
    for input in "${env_script}" "${host_exe}" "${xclbin}" "${emconfig}"; do
        if [ ! -s "${input}" ]; then
            echo "Missing or empty qwen2.5-3b HW-Emu input: ${input}" >&2
            exit 66
        fi
    done
    source "${env_script}" >/dev/null 2>&1
    xclbin_kernel_clock_mhz="$(
        xclbinutil --info --input "${xclbin}" 2>/dev/null |
        awk '
            $1 == "Name:" && $2 == "KERNEL_CLK" { kernel = 1; next }
            kernel && $1 == "Frequency:" { print $2; kernel = 0 }
        '
    )"
    if ! [[ "${xclbin_kernel_clock_mhz}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Cannot read KERNEL_CLK frequency from ${xclbin}" >&2
        exit 65
    fi
    token_csv=""
    for ((token = 0; token < prompt_tokens; token++)); do
        if [ -n "${token_csv}" ]; then
            token_csv+=","
        fi
        token_csv+="${token}"
    done
    prompt_blocks=$(((prompt_tokens + block_size - 1) / block_size))
    decode_forwards=$((generated_tokens - 1))
    expected_coarse_tasks=$((
        prompt_blocks * 2 * layers + 1 +
        decode_forwards * (2 * layers + 1)
    ))
    echo "started_at=$(date -Is)"
    echo "profile=qwen2.5-3b"
    echo "prompt_tokens=${prompt_tokens}"
    echo "generated_tokens=${generated_tokens}"
    echo "layers=${layers}"
    echo "block_size=${block_size}"
    echo "prompt_blocks=${prompt_blocks}"
    echo "decode_forwards=${decode_forwards}"
    echo "expected_coarse_tasks=${expected_coarse_tasks}"
    echo "seed=${seed}"
    echo "model_source=${model_source}"
    echo "checkpoint_dir=${checkpoint_dir}"
    echo "tie_embeddings=${tie_embeddings}"
    echo "verify_e2e_golden=${verify_e2e_golden}"
    echo "timeout_seconds=${timeout_seconds}"
    echo "xclbin_kernel_clock_mhz=${xclbin_kernel_clock_mhz}"
    echo "timing_note=HW_Emu_host_wall_is_simulator_runtime_use_CU_trace_for_modeled_cycles"
    echo "host_exe_sha256=$(sha256sum "${host_exe}" | awk '{print $1}')"
    echo "xclbin_sha256=$(sha256sum "${xclbin}" | awk '{print $1}')"
    echo "emconfig_sha256=$(sha256sum "${emconfig}" | awk '{print $1}')"
    host_exe_abs="$(realpath "${host_exe}")"
    model_args=(--seed "${seed}")
    if [ "${model_source}" = "random" ]; then
        model_args+=(--random-model)
    else
        model_args+=(--data "${checkpoint_dir}")
    fi
    if [ "${tie_embeddings}" = "1" ]; then
        model_args+=(--tie-embeddings)
    fi
    if [ "${verify_e2e_golden}" = "1" ]; then
        if ! rg -a -q -- '--verify-e2e-golden' "${host_exe_abs}"; then
            echo "Host executable does not implement the E2E numerical contract" >&2
            exit 65
        fi
        model_args+=(--verify-e2e-golden)
    fi
    cd "${build_dir}"
    run_capture="$(mktemp "${TMPDIR:-/tmp}/llm-qwen3b-e2e.XXXXXX.log")"
    trap 'rm -f "${run_capture:-}"' EXIT
    set +e
    XCL_EMULATION_MODE=hw_emu \
    EMCONFIG_PATH="$PWD" \
        timeout "${timeout_seconds}" \
        "${host_exe_abs}" \
        --xclbin ./qwen_8x64_dual.xclbin \
        --mode generate \
        --profile qwen2.5-3b \
        "${model_args[@]}" \
        --tokens "${token_csv}" \
        --max-new-tokens "${generated_tokens}" \
        --prefill-block-size "${block_size}" \
        --layers "${layers}" \
        --coarse-tasks \
        2>&1 | tee "${run_capture}"
    status="${PIPESTATUS[0]}"
    set -e
    if [ "${status}" -eq 0 ] && [ "${verify_e2e_golden}" = "1" ]; then
        numeric_steps="$(
            rg -c '^QWEN_8X64_E2E_NUMERIC_STEP .* PASS$' \
                "${run_capture}" || true
        )"
        numeric_verify_line="$(rg '^QWEN_8X64_E2E_NUMERIC_VERIFY ' "${run_capture}" | tail -n 1 || true)"
        profile_line="$(rg '^QWEN_8X64_E2E_PROFILE ' "${run_capture}" | tail -n 1 || true)"
        if [ "${numeric_steps}" -ne "${generated_tokens}" ] ||
           ! line_has_fields "${numeric_verify_line}" \
               token_sequence_match=1 validation_schedule=post_inference \
               intermediate_host_copy=0 kv_cache_owner=controller PASS ||
           ! line_has_fields "${profile_line}" \
               validation_schedule=post_inference e2e_numeric_golden=1 PASS ||
           ! line_has_keys "${profile_line}" \
               total_host_elapsed_ms total_process_elapsed_ms \
               post_inference_validation_ms; then
            echo "E2E Host exited zero without the required numerical evidence" >&2
            status=65
        fi
    fi
    echo "finished_at=$(date -Is)"
    echo "exit_status=${status}"
    exit "${status}"
fi

dry_run=0
if [ "${1:-}" = "--dry-run" ]; then
    dry_run=1
    shift
fi
if [ "$#" -ne 0 ]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

for value_name in prompt_tokens generated_tokens layers block_size timeout_seconds; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "${value_name} must be a non-negative integer" >&2
        exit 2
    fi
done
if [ "${prompt_tokens}" -lt 1 ] || [ "${prompt_tokens}" -gt 2048 ]; then
    echo "VITIS_8X64_E2E_PROMPT_TOKENS must be in 1..2048" >&2
    exit 2
fi
if [ "${generated_tokens}" -lt 1 ]; then
    echo "VITIS_8X64_E2E_MAX_NEW_TOKENS must be positive" >&2
    exit 2
fi
if [ "${prompt_tokens}" -gt $((2048 - generated_tokens)) ]; then
    echo "prompt plus generated tokens exceed the 2048-token profile" >&2
    exit 2
fi
if [ "${layers}" -lt 1 ] || [ "${layers}" -gt 36 ]; then
    echo "VITIS_8X64_E2E_LAYERS must be in 1..36" >&2
    exit 2
fi
if [ "${block_size}" -lt 1 ] || [ "${block_size}" -gt 8 ]; then
    echo "VITIS_8X64_E2E_PREFILL_BLOCK_SIZE must be in 1..8" >&2
    exit 2
fi
case "${model_source}" in
    random|checkpoint) ;;
    *)
        echo "VITIS_8X64_E2E_MODEL_SOURCE must be random or checkpoint" >&2
        exit 2
        ;;
esac
if [ "${tie_embeddings}" != "0" ] && [ "${tie_embeddings}" != "1" ]; then
    echo "VITIS_8X64_E2E_TIE_EMBEDDINGS must be 0 or 1" >&2
    exit 2
fi
if [ "${verify_e2e_golden}" != "0" ] &&
   [ "${verify_e2e_golden}" != "1" ]; then
    echo "VITIS_8X64_E2E_VERIFY_GOLDEN must be 0 or 1" >&2
    exit 2
fi
if [ "${model_source}" = "checkpoint" ] && [ ! -d "${checkpoint_dir}" ]; then
    echo "Missing checkpoint directory: ${checkpoint_dir}" >&2
    exit 66
fi

prompt_blocks=$(((prompt_tokens + block_size - 1) / block_size))
decode_forwards=$((generated_tokens - 1))
expected_coarse_tasks=$((
    prompt_blocks * 2 * layers + 1 +
    decode_forwards * (2 * layers + 1)
))

if [ "${dry_run}" -eq 0 ]; then
    available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    required_kib=$((min_available_gib * 1024 * 1024))
    if [ "${available_kib}" -lt "${required_kib}" ]; then
        echo "Need at least ${min_available_gib} GiB available memory" >&2
        exit 75
    fi
    for input in "${env_script}" "${host_exe}" "${xclbin}" "${emconfig}"; do
        if [ ! -s "${input}" ]; then
            echo "Missing or empty qwen2.5-3b HW-Emu input: ${input}" >&2
            exit 66
        fi
    done
    if pgrep -af '[h]ost_qwen_8x64.*--profile qwen2.5-3b' >/dev/null; then
        echo "A qwen2.5-3b HW-Emu Host already appears to be running" >&2
        pgrep -af '[h]ost_qwen_8x64.*--profile qwen2.5-3b' >&2 || true
        exit 70
    fi
fi

if [ "${dry_run}" -eq 0 ]; then
    mkdir -p logs
fi
timestamp="$(date +%Y%m%d_%H%M%S)"
session_name="llm_qwen3b_hwemu_p${prompt_tokens}_g${generated_tokens}_l${layers}_${timestamp}"
log_path="$PWD/logs/qwen3b_e2e_hwemu_p${prompt_tokens}_g${generated_tokens}_l${layers}_${timestamp}.log"
pid_path="${log_path%.log}.pid"

# Do not rely on the tmux server environment for experiment parameters.  A
# long-lived tmux server predates most one-shot shell exports, so a launcher
# invoked with (for example) VITIS_8X64_E2E_LAYERS=1 could otherwise name the
# session L1 while the worker silently falls back to its L36 default.
worker_argv=(
    env
    "VITIS_8X64_QWEN3B_WORK_ROOT=${work_root}"
    "DEVICE=${device}"
    "VITIS_8X64_E2E_PROMPT_TOKENS=${prompt_tokens}"
    "VITIS_8X64_E2E_MAX_NEW_TOKENS=${generated_tokens}"
    "VITIS_8X64_E2E_LAYERS=${layers}"
    "VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=${block_size}"
    "VITIS_8X64_RESIDENT_SEED=${seed}"
    "VITIS_8X64_E2E_MODEL_SOURCE=${model_source}"
    "VITIS_8X64_E2E_DATA_DIR=${checkpoint_dir}"
    "VITIS_8X64_E2E_TIE_EMBEDDINGS=${tie_embeddings}"
    "VITIS_8X64_E2E_VERIFY_GOLDEN=${verify_e2e_golden}"
    "VITIS_8X64_HW_EMU_TIMEOUT=${timeout_seconds}"
    "VITIS_ENV_SCRIPT=${env_script}"
    "${script_path}"
    --worker
)
printf -v worker_command '%q ' "${worker_argv[@]}"
printf -v quoted_log_path '%q' "${log_path}"
if [ "${dry_run}" -eq 1 ]; then
    printf 'dry_run=1\nprofile=qwen2.5-3b\nprompt_tokens=%s\ngenerated_tokens=%s\nlayers=%s\nblock_size=%s\nprompt_blocks=%s\ndecode_forwards=%s\nexpected_coarse_tasks=%s\nverify_e2e_golden=%s\n' \
        "${prompt_tokens}" "${generated_tokens}" "${layers}" \
        "${block_size}" "${prompt_blocks}" "${decode_forwards}" \
        "${expected_coarse_tasks}" "${verify_e2e_golden}"
    printf 'worker_command=%s\nlog=%s\nbuild_dir=%s\n' \
        "${worker_command}" "${log_path}" "${build_dir}"
    exit 0
fi
tmux new-session -d -s "${session_name}" \
    "exec ${worker_command} >>${quoted_log_path} 2>&1"
pid="$(tmux display-message -p -t "${session_name}:0.0" '#{pane_pid}')"
printf '%s\n' "${pid}" >"${pid_path}"
printf 'pid=%s\nsession=%s\nlog=%s\npidfile=%s\nbuild_dir=%s\n' \
    "${pid}" "${session_name}" "${log_path}" "${pid_path}" "${build_dir}"
