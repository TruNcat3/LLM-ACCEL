#!/usr/bin/env bash
set -euo pipefail

script_path="$(realpath "$0")"
cd "$(dirname "${script_path}")/.."

build_dir="vitis_8x64/build.small.resident_layer.t8.d2.ii2.r0.wr33.cw1.block_prefill_q214_exact.hw_emu.xilinx_u50_gen3x16_xdma_5_202210_1"
timeout_seconds="${VITIS_8X64_HW_EMU_TIMEOUT:-43200}"

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

if [ "${1:-}" = "--worker" ]; then
    for input in \
        "${build_dir}/host_qwen_8x64.exe" \
        "${build_dir}/qwen_8x64_dual.xclbin" \
        "${build_dir}/emconfig.json"
    do
        if [ ! -s "${input}" ]; then
            echo "Missing or empty Small E2E gate input: ${input}" >&2
            exit 66
        fi
    done
    echo "started_at=$(date -Is)"
    echo "gate=small_p8_g2_l2_full_prefix_numeric"
    echo "profile=small"
    echo "prompt_tokens=8"
    echo "generated_tokens=2"
    echo "layers=2"
    echo "block_size=8"
    echo "prefill_block_size=8"
    echo "verify_e2e_golden=1"
    echo "host_exe_sha256=$(sha256sum "${build_dir}/host_qwen_8x64.exe" | awk '{print $1}')"
    echo "xclbin_sha256=$(sha256sum "${build_dir}/qwen_8x64_dual.xclbin" | awk '{print $1}')"
    echo "emconfig_sha256=$(sha256sum "${build_dir}/emconfig.json" | awk '{print $1}')"
    run_capture="$(mktemp "${TMPDIR:-/tmp}/llm-small-e2e.XXXXXX.log")"
    trap 'rm -f "${run_capture:-}"' EXIT
    set +e
    VITIS_8X64_MODEL_PROFILE=small \
    CC8_RESIDENT_TOKEN_ROWS=8 \
    VITIS_8X64_BUILD_EXACT_COMPUTE_XO=1 \
    VITIS_8X64_RESIDENT_VARIANT_TAG=block_prefill_q214_exact \
    VITIS_8X64_E2E_TOKENS=0,1,2,3,4,5,6,7 \
    VITIS_8X64_E2E_MAX_NEW_TOKENS=2 \
    VITIS_8X64_E2E_LAYERS=2 \
    VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8 \
    VITIS_8X64_E2E_VERIFY_GOLDEN=1 \
    VITIS_8X64_HW_EMU_TIMEOUT="${timeout_seconds}" \
        scripts/build_vitis_8x64_resident_layer_hwemu.sh run-generate \
        2>&1 | tee "${run_capture}"
    status="${PIPESTATUS[0]}"
    set -e
    if [ "${status}" -eq 0 ]; then
        numeric_steps="$(rg -c '^QWEN_8X64_E2E_NUMERIC_STEP .* validation_schedule=post_inference PASS$' "${run_capture}" || true)"
        progress_tasks="$(rg -c '^COARSE_TASK_PROGRESS ' "${run_capture}" || true)"
        numeric_verify_line="$(rg '^QWEN_8X64_E2E_NUMERIC_VERIFY ' "${run_capture}" | tail -n 1 || true)"
        profile_line="$(rg '^QWEN_8X64_E2E_PROFILE ' "${run_capture}" | tail -n 1 || true)"
        if [ "${numeric_steps}" -ne 2 ] ||
           [ "${progress_tasks}" -ne 10 ] ||
           ! line_has_fields "${numeric_verify_line}" \
               token_sequence_match=1 validation_schedule=post_inference \
               intermediate_host_copy=0 kv_cache_owner=controller PASS ||
           ! line_has_fields "${profile_line}" \
               validation_schedule=post_inference e2e_numeric_golden=1 PASS ||
           ! line_has_keys "${profile_line}" \
               total_host_elapsed_ms total_process_elapsed_ms \
               post_inference_validation_ms; then
            echo "Small E2E Host exited zero without the required post-inference numerical evidence" >&2
            status=65
        fi
    fi
    echo "finished_at=$(date -Is)"
    echo "exit_status=${status}"
    exit "${status}"
fi

if pgrep -af '[h]ost_qwen_8x64.*--mode generate.*--profile small' >/dev/null; then
    echo "A Small-profile generate Host already appears to be running" >&2
    pgrep -af '[h]ost_qwen_8x64.*--mode generate.*--profile small' >&2 || true
    exit 70
fi

mkdir -p logs
timestamp="$(date +%Y%m%d_%H%M%S)"
session_name="llm_small_e2e_numeric_${timestamp}"
log_path="$PWD/logs/small_e2e_numeric_p8_g2_l2_${timestamp}.log"
pid_path="${log_path%.log}.pid"

tmux new-session -d -s "${session_name}" \
    "exec '${script_path}' --worker >>'${log_path}' 2>&1"
pid="$(tmux display-message -p -t "${session_name}:0.0" '#{pane_pid}')"
printf '%s\n' "${pid}" >"${pid_path}"
printf 'pid=%s\nsession=%s\nlog=%s\npidfile=%s\n' \
    "${pid}" "${session_name}" "${log_path}" "${pid_path}"
