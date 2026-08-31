#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")/.."

launcher="scripts/run_vitis_8x64_qwen3b_checkpoint_hwemu_tmux.sh"
host_source="host/host_qwen_8x64.cpp"

strict_output="$(
    VITIS_8X64_CHECKPOINT_LAYERS=4 \
        "${launcher}" --dry-run
)"
for expected in \
    'layers=4' \
    'tolerance=0' \
    'continue_on_failure=0' \
    'expected_checkpoints=9'
do
    if ! grep -F -x -q "${expected}" <<<"${strict_output}"; then
        echo "Strict checkpoint default is missing: ${expected}" >&2
        exit 1
    fi
done

diagnostic_output="$(
    VITIS_8X64_CHECKPOINT_LAYERS=4 \
    VITIS_8X64_CHECKPOINT_TOLERANCE=1 \
    VITIS_8X64_CHECKPOINT_CONTINUE_ON_FAILURE=1 \
        "${launcher}" --dry-run
)"
for expected in \
    'tolerance=1' \
    'continue_on_failure=1' \
    'expected_checkpoints=9'
do
    if ! grep -F -x -q "${expected}" <<<"${diagnostic_output}"; then
        echo "Diagnostic checkpoint option is missing: ${expected}" >&2
        exit 1
    fi
done
if ! grep -F -q 'VITIS_8X64_CHECKPOINT_TOLERANCE=1' \
    <<<"${diagnostic_output}" ||
   ! grep -F -q 'VITIS_8X64_CHECKPOINT_CONTINUE_ON_FAILURE=1' \
    <<<"${diagnostic_output}"; then
    echo "Checkpoint options were not propagated to the worker" >&2
    exit 1
fi

if VITIS_8X64_CHECKPOINT_TOLERANCE=-1 \
    "${launcher}" --dry-run >/dev/null 2>&1; then
    echo "Negative checkpoint tolerance was accepted" >&2
    exit 1
fi
if VITIS_8X64_CHECKPOINT_CONTINUE_ON_FAILURE=2 \
    "${launcher}" --dry-run >/dev/null 2>&1; then
    echo "Invalid continue-on-failure value was accepted" >&2
    exit 1
fi

for source_contract in \
    'int checkpoint_tolerance = 0;' \
    'bool checkpoint_continue_on_failure = false;' \
    '"rounding_within_tolerance"' \
    'all_checkpoints_pass &&'
do
    if ! rg -F -q "${source_contract}" "${host_source}"; then
        echo "Host checkpoint contract is missing: ${source_contract}" >&2
        exit 1
    fi
done

echo "CHECKPOINT TOLERANCE CONTRACT PASS strict=0 diagnostic=1-LSB continue=1"
