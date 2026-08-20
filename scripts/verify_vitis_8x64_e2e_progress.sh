#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "usage: $0 HOST_LOG PROMPT_TOKENS GENERATED_TOKENS LAYERS BLOCK_SIZE" >&2
    exit 2
fi

host_log="$1"
prompt="$2"
generated="$3"
layers="$4"
block="$5"
if [ ! -s "${host_log}" ]; then
    echo "Missing E2E Host log: ${host_log}" >&2
    exit 66
fi
for value_name in prompt generated layers block; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "${value_name} must be a non-negative integer" >&2
        exit 2
    fi
done
if [ "${prompt}" -lt 1 ] || [ "${generated}" -lt 1 ] ||
   [ "${layers}" -lt 1 ] || [ "${layers}" -gt 36 ] ||
   [ "${block}" -lt 1 ] || [ "${block}" -gt 8 ]; then
    echo "E2E progress shape is outside the supported contract" >&2
    exit 2
fi

prompt_blocks=$(((prompt + block - 1) / block))
decode_forwards=$((generated - 1))
expected_tasks=$((
    prompt_blocks * 2 * layers + 1 +
    decode_forwards * (2 * layers + 1)
))

awk \
    -v prompt="${prompt}" \
    -v generated="${generated}" \
    -v layers="${layers}" \
    -v block="${block}" \
    -v expected_tasks="${expected_tasks}" '
    function field(key,    i, prefix) {
        prefix = key "="
        for (i = 1; i <= NF; i++) {
            if (index($i, prefix) == 1) {
                return substr($i, length(prefix) + 1)
            }
        }
        return ""
    }
    BEGIN {
        prompt_blocks = int((prompt + block - 1) / block)
        decode_forwards = generated - 1
        expected_groups = prompt_blocks + decode_forwards
    }
    /^COARSE_TASK_PROGRESS / {
        count++
        completed_text = field("completed")
        total_text = field("total")
        op_text = field("op")
        phase = field("phase")
        layer_text = field("layer")
        position_text = field("position")
        rows_text = field("query_tokens")
        controller_ms_text = field("controller_ms")
        if (completed_text == "" || total_text == "" || op_text == "" ||
            phase == "" || layer_text == "" || position_text == "" ||
            rows_text == "" || controller_ms_text == "") {
            failed = 1
            next
        }
        completed = completed_text + 0
        if (completed == 1) group++
        if (group <= prompt_blocks) {
            expected_position = (group - 1) * block
            expected_rows = prompt - expected_position
            if (expected_rows > block) expected_rows = block
            expected_total = 2 * layers + (group == prompt_blocks ? 1 : 0)
        } else {
            decode_index = group - prompt_blocks - 1
            expected_position = prompt + decode_index
            expected_rows = 1
            expected_total = 2 * layers + 1
        }
        if (completed <= 2 * layers) {
            expected_op = completed % 2 == 1 ? 18 : 19
            expected_phase = completed % 2 == 1 ? "attention" : "ffn"
            expected_layer = int((completed - 1) / 2)
        } else {
            expected_op = 20
            expected_phase = "final_norm"
            expected_layer = 0
        }
        if (group < 1 || group > expected_groups ||
            completed < 1 || completed > expected_total ||
            total_text + 0 != expected_total ||
            op_text + 0 != expected_op || phase != expected_phase ||
            layer_text + 0 != expected_layer ||
            rows_text + 0 != expected_rows ||
            position_text + 0 != expected_position ||
            controller_ms_text + 0 <= 0) {
            print "Malformed coarse-task sequence at task " count \
                > "/dev/stderr"
            failed = 1
        }
        if (completed == expected_total) group_ends++
    }
    END {
        if (count != expected_tasks || group != expected_groups ||
            group_ends != expected_groups || failed) exit 1
    }
' "${host_log}"

printf 'E2E PROGRESS CONTRACT PASS prompt=%s generated=%s layers=%s block=%s prompt_blocks=%s decode_forwards=%s tasks=%s\n' \
    "${prompt}" "${generated}" "${layers}" "${block}" \
    "${prompt_blocks}" "${decode_forwards}" "${expected_tasks}"
