#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 HOST_LOG" >&2
    exit 2
fi

host_log="$(realpath "$1")"
if [ ! -s "${host_log}" ]; then
    echo "Missing or empty Host task-program log: ${host_log}" >&2
    exit 66
fi

metadata() {
    local key="$1"
    awk -F= -v key="${key}" '
        $1 == key { value = substr($0, length(key) + 2) }
        END { print value }
    ' "${host_log}"
}

contract="$(metadata host_task_program)"
expected_tasks="$(metadata expected_coarse_tasks)"
progress_records="$(
    awk '/^COARSE_TASK_PROGRESS / { count++ } END { print count + 0 }' \
        "${host_log}"
)"

if [ -z "${contract}" ]; then
    printf 'field\tvalue\n'
    printf 'host_task_program_contract\tlegacy_equivalent_sequence\n'
    printf 'host_task_program_evidence\truntime_metadata_absent\n'
    printf 'host_task_program_pair_trace_verified\t0\n'
    printf 'host_task_program_progress_records\t%s\n' "${progress_records}"
    exit 0
fi

if [ "${contract}" != "static_descriptor_v1" ]; then
    echo "Unknown Host task-program contract: ${contract}" >&2
    exit 65
fi
if ! [[ "${expected_tasks}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Static Host task-program log lacks expected_coarse_tasks" >&2
    exit 65
fi
if [ "${progress_records}" -ne "${expected_tasks}" ]; then
    echo "Static Host task-program trace count mismatch: expected ${expected_tasks}, observed ${progress_records}" >&2
    exit 65
fi

if ! awk '
    function value(name,    i, prefix) {
        prefix = name "="
        for (i = 1; i <= NF; i++) {
            if (index($i, prefix) == 1) {
                return substr($i, length(prefix) + 1)
            }
        }
        return ""
    }
    /^COARSE_TASK_PROGRESS / {
        op = value("op")
        input_pair = value("input_pair")
        output_pair = value("output_pair")
        if (op == "18" || op == "20") {
            if (input_pair != "1" || output_pair != "0") invalid = 1
        } else if (op == "19") {
            if (input_pair != "0" || output_pair != "1") invalid = 1
        } else {
            invalid = 1
        }
        checked++
    }
    END { exit checked > 0 && !invalid ? 0 : 1 }
' "${host_log}"; then
    echo "Static Host task-program HBM pair trace is malformed" >&2
    exit 65
fi

printf 'field\tvalue\n'
printf 'host_task_program_contract\t%s\n' "${contract}"
printf 'host_task_program_evidence\truntime_metadata_plus_progress_pair_trace\n'
printf 'host_task_program_pair_trace_verified\t1\n'
printf 'host_task_program_progress_records\t%s\n' "${progress_records}"
