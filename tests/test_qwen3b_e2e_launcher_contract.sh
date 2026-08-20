#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")/.."

launcher="scripts/run_vitis_8x64_qwen3b_e2e_hwemu_tmux.sh"
work_root="/tmp/llm_accel_qwen3b_launcher_contract_missing_build"
output="$(
    VITIS_8X64_QWEN3B_WORK_ROOT="${work_root}" \
    DEVICE=contract_test_device \
    VITIS_8X64_E2E_PROMPT_TOKENS=8 \
    VITIS_8X64_E2E_MAX_NEW_TOKENS=2 \
    VITIS_8X64_E2E_LAYERS=1 \
    VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8 \
    VITIS_8X64_E2E_VERIFY_GOLDEN=1 \
        "${launcher}" --dry-run
)"

require_text() {
    local required="$1"
    if [[ "${output}" != *"${required}"* ]]; then
        echo "launcher dry-run is missing: ${required}" >&2
        exit 1
    fi
}

require_text 'dry_run=1'
require_text 'profile=qwen2.5-3b'
require_text 'prompt_tokens=8'
require_text 'generated_tokens=2'
require_text 'layers=1'
require_text 'block_size=8'
require_text 'prompt_blocks=1'
require_text 'decode_forwards=1'
require_text 'expected_coarse_tasks=6'
require_text 'verify_e2e_golden=1'
require_text 'VITIS_8X64_E2E_PROMPT_TOKENS=8'
require_text 'VITIS_8X64_E2E_MAX_NEW_TOKENS=2'
require_text 'VITIS_8X64_E2E_LAYERS=1'
require_text 'VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8'
require_text 'VITIS_8X64_E2E_VERIFY_GOLDEN=1'
require_text "VITIS_8X64_QWEN3B_WORK_ROOT=${work_root}"
require_text 'DEVICE=contract_test_device'
require_text '--worker'

full_output="$(
    VITIS_8X64_QWEN3B_WORK_ROOT="${work_root}" \
    DEVICE=contract_test_device \
    VITIS_8X64_E2E_PROMPT_TOKENS=8 \
    VITIS_8X64_E2E_MAX_NEW_TOKENS=2 \
    VITIS_8X64_E2E_LAYERS=36 \
    VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8 \
        "${launcher}" --dry-run
)"
for required in \
    'layers=36' 'prompt_blocks=1' 'decode_forwards=1' \
    'expected_coarse_tasks=146'
do
    if [[ "${full_output}" != *"${required}"* ]]; then
        echo "full launcher dry-run is missing: ${required}" >&2
        exit 1
    fi
done

multilayer_output="$(
    VITIS_8X64_QWEN3B_WORK_ROOT="${work_root}" \
    DEVICE=contract_test_device \
    VITIS_8X64_E2E_PROMPT_TOKENS=8 \
    VITIS_8X64_E2E_MAX_NEW_TOKENS=2 \
    VITIS_8X64_E2E_LAYERS=2 \
    VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8 \
    VITIS_8X64_E2E_VERIFY_GOLDEN=1 \
        "${launcher}" --dry-run
)"
for required in \
    'layers=2' 'prompt_blocks=1' 'decode_forwards=1' \
    'expected_coarse_tasks=10' 'verify_e2e_golden=1'
do
    if [[ "${multilayer_output}" != *"${required}"* ]]; then
        echo "multi-layer launcher dry-run is missing: ${required}" >&2
        exit 1
    fi
done

tail_block_output="$(
    VITIS_8X64_QWEN3B_WORK_ROOT="${work_root}" \
    DEVICE=contract_test_device \
    VITIS_8X64_E2E_PROMPT_TOKENS=11 \
    VITIS_8X64_E2E_MAX_NEW_TOKENS=1 \
    VITIS_8X64_E2E_LAYERS=2 \
    VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8 \
    VITIS_8X64_E2E_VERIFY_GOLDEN=1 \
        "${launcher}" --dry-run
)"
for required in \
    'prompt_tokens=11' 'generated_tokens=1' 'layers=2' \
    'block_size=8' 'prompt_blocks=2' 'decode_forwards=0' \
    'expected_coarse_tasks=9'
do
    if [[ "${tail_block_output}" != *"${required}"* ]]; then
        echo "tail-block launcher dry-run is missing: ${required}" >&2
        exit 1
    fi
done

set +e
VITIS_8X64_E2E_LAYERS=0 "${launcher}" --dry-run \
    >/dev/null 2>&1
invalid_layers_status="$?"
VITIS_8X64_E2E_PROMPT_TOKENS=2048 \
VITIS_8X64_E2E_MAX_NEW_TOKENS=1 \
    "${launcher}" --dry-run >/dev/null 2>&1
sequence_overflow_status="$?"
set -e

if [ "${invalid_layers_status}" -ne 2 ] ||
   [ "${sequence_overflow_status}" -ne 2 ]; then
    echo "launcher dry-run did not reject an invalid workload" >&2
    exit 1
fi

echo "QWEN3B E2E LAUNCHER CONTRACT PASS profile=qwen2.5-3b P8G2L1_tasks=6 P8G2L2_tasks=10 P8G2L36_tasks=146 P11G1L2_tasks=9 numerical_golden=1"
