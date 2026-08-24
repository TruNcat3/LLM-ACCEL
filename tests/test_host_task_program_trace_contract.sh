#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")/.."

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/llm-accel-task-trace.XXXXXX")"
cleanup() {
    rm -rf "${work_dir}"
}
trap cleanup EXIT

legacy_log="${work_dir}/legacy.log"
static_log="${work_dir}/static.log"
invalid_log="${work_dir}/invalid.log"

printf '%s\n' \
    'expected_coarse_tasks=3' \
    'COARSE_TASK_PROGRESS completed=1 total=3 op=18 phase=attention layer=0 position=0 query_tokens=8' \
    'COARSE_TASK_PROGRESS completed=2 total=3 op=19 phase=ffn layer=0 position=0 query_tokens=8' \
    'COARSE_TASK_PROGRESS completed=3 total=3 op=20 phase=final_norm layer=0 position=0 query_tokens=8' \
    > "${legacy_log}"
printf '%s\n' \
    'host_task_program=static_descriptor_v1' \
    'expected_coarse_tasks=3' \
    'COARSE_TASK_PROGRESS completed=1 total=3 op=18 phase=attention layer=0 position=0 query_tokens=8 input_pair=1 output_pair=0' \
    'COARSE_TASK_PROGRESS completed=2 total=3 op=19 phase=ffn layer=0 position=0 query_tokens=8 input_pair=0 output_pair=1' \
    'COARSE_TASK_PROGRESS completed=3 total=3 op=20 phase=final_norm layer=0 position=0 query_tokens=8 input_pair=1 output_pair=0' \
    > "${static_log}"
printf '%s\n' \
    'host_task_program=static_descriptor_v1' \
    'expected_coarse_tasks=1' \
    'COARSE_TASK_PROGRESS completed=1 total=1 op=19 phase=ffn layer=0 position=0 query_tokens=8 input_pair=1 output_pair=0' \
    > "${invalid_log}"

legacy_report="$(scripts/verify_host_task_program_trace.sh "${legacy_log}")"
static_report="$(scripts/verify_host_task_program_trace.sh "${static_log}")"
if [[ "${legacy_report}" != *$'host_task_program_contract\tlegacy_equivalent_sequence'* ]] ||
   [[ "${legacy_report}" != *$'host_task_program_pair_trace_verified\t0'* ]]; then
    echo "Legacy Host task-program classification regressed" >&2
    exit 65
fi
if [[ "${static_report}" != *$'host_task_program_contract\tstatic_descriptor_v1'* ]] ||
   [[ "${static_report}" != *$'host_task_program_pair_trace_verified\t1'* ]] ||
   [[ "${static_report}" != *$'host_task_program_progress_records\t3'* ]]; then
    echo "Static Host task-program trace classification regressed" >&2
    exit 65
fi
if scripts/verify_host_task_program_trace.sh "${invalid_log}" >/dev/null 2>&1; then
    echo "Malformed Host HBM pair trace was accepted" >&2
    exit 65
fi

echo 'HOST TASK PROGRAM TRACE CONTRACT PASS static=pair_trace_verified legacy=classified invalid=rejected'
