#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")/.."

for performance in results/*/performance.tsv; do
    header="$(head -n 1 "${performance}")"
    if awk -F '\t' '
        {
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^physical_efficiency/ ||
                    $i == "token_s" || $i == "tokens") exit 1
            }
        }
    ' <<<"${header}"
    then
        :
    else
        echo "Ambiguous performance schema: ${performance}" >&2
        exit 65
    fi
done

q214_header="$(head -n 1 results/q214-pd-20260811/performance.tsv)"
for required_column in \
    active_query_rows_per_block \
    modeled_interval_efficiency_percent \
    target_block_query_row_s
do
    if ! awk -F '\t' -v required="${required_column}" '
        { for (i = 1; i <= NF; i++) if ($i == required) found = 1 }
        END { exit found ? 0 : 1 }
    ' <<<"${q214_header}"; then
        echo "Q2.14 P/D schema lacks ${required_column}" >&2
        exit 65
    fi
done

profile_csv="results/block-prefill-20260817/raw/small_p8g2_profile_kernels.csv"
report="$(
    scripts/report_vitis_8x64_e2e_trace.sh \
        "${profile_csv}" small 8 2 2 8 200 300 1
)"
standard_report="$(
    scripts/report_vitis_8x64_e2e_trace.sh \
        "${profile_csv}" qwen2.5-3b 8 2 2 8 200 300 1
)"
full_stack_report="$(
    scripts/report_vitis_8x64_e2e_trace.sh \
        "${profile_csv}" qwen2.5-3b 8 2 36 8 200 300 1
)"

value() {
    local column="$1"
    awk -F '\t' -v column="${column}" '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                if ($i == column) column_index = i
            }
            next
        }
        NR == 2 && column_index { print $column_index; exit }
    ' <<<"${report}"
}

sequence_batch="$(value sequence_batch)"
prompt="$(value prompt_sequence_tokens)"
sampled="$(value sampled_output_tokens)"
decode_forwards="$(value decode_forwards)"
prefill_blocks="$(value prefill_blocks)"
layers="$(value layers)"
block_rows="$(value configured_max_active_query_rows_per_prefill_block)"
tasks="$(value expected_coarse_tasks)"
cycles="$(value xsim_cycles)"
target_us="$(value projected_target_us)"
useful_mac="$(value useful_mac)"
throughput="$(value useful_gmac_s)"
efficiency="$(value modeled_interval_efficiency_percent)"
query_row_s="$(value query_row_s)"
output_token_s="$(value request_output_token_s)"
standard_useful_mac="$(
    awk -F '\t' '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                if ($i == "useful_mac") column_index = i
            }
            next
        }
        NR == 2 && column_index { print $column_index; exit }
    ' <<<"${standard_report}"
)"
standard_numeric_tolerance=$((32 * 2))
full_stack_useful_mac="$(
    awk -F '\t' '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                if ($i == "useful_mac") mac_index = i
                if ($i == "expected_coarse_tasks") task_index = i
            }
            next
        }
        NR == 2 && mac_index && task_index {
            print $mac_index, $task_index
            exit
        }
    ' <<<"${full_stack_report}"
)"
read -r full_stack_mac full_stack_tasks <<<"${full_stack_useful_mac}"
full_stack_numeric_tolerance=$((32 * 36))

if [ "${sequence_batch}" != 1 ] || [ "${prompt}" != 8 ] ||
   [ "${sampled}" != 2 ] || [ "${decode_forwards}" != 1 ] ||
   [ "${prefill_blocks}" != 1 ] || [ "${layers}" != 2 ] ||
   [ "${block_rows}" != 8 ] || [ "${tasks}" != 10 ]; then
    echo "P8/G2/L2 workload semantics regressed" >&2
    exit 1
fi
if [ "${standard_useful_mac}" != 1387634688 ] ||
   [ "${standard_numeric_tolerance}" != 64 ]; then
    echo "Standard-shape P8/G2/L2 accounting regressed" >&2
    exit 1
fi
if [ "${full_stack_mac}" != 24977424384 ] ||
   [ "${full_stack_tasks}" != 146 ] ||
   [ "${full_stack_numeric_tolerance}" != 1152 ]; then
    echo "Standard-shape P8/G2/L36 accounting regressed" >&2
    exit 1
fi

if ! awk \
    -v observed_cycles="${cycles}" \
    -v observed_target_us="${target_us}" \
    -v observed_useful_mac="${useful_mac}" \
    -v observed_throughput="${throughput}" \
    -v observed_efficiency="${efficiency}" \
    -v observed_query_row_s="${query_row_s}" \
    -v observed_output_token_s="${output_token_s}" '
    function abs(value) { return value < 0 ? -value : value }
    BEGIN {
        hidden = 64
        intermediate = 128
        kv_channels = 32
        heads = 4
        head_dim = 16
        prompt = 8
        sampled = 2
        layers = 2
        decode_forwards = sampled - 1
        query_rows = prompt + decode_forwards
        dense_per_row = 2 * hidden * hidden + \
                        2 * hidden * kv_channels + \
                        3 * hidden * intermediate
        prompt_context_sum = prompt * (prompt + 1) / 2
        decode_context_sum = decode_forwards * \
                             ((prompt + 1) + \
                              (prompt + decode_forwards)) / 2
        attention_per_context = 2 * heads * head_dim
        expected_mac = layers * \
                       (query_rows * dense_per_row + \
                        (prompt_context_sum + decode_context_sum) * \
                        attention_per_context)
        expected_cycles = 258.503 * 300
        expected_target_us = expected_cycles / 200
        expected_throughput = 0.2 * expected_mac / expected_cycles
        expected_efficiency = 100 * expected_mac / \
                              (expected_cycles * 1024)
        expected_query_row_s = query_rows * 200000000 / expected_cycles
        expected_output_token_s = sampled * 200000000 / expected_cycles

        pass = observed_useful_mac == expected_mac && \
               abs(observed_cycles - expected_cycles) <= 0.05 && \
               abs(observed_target_us - expected_target_us) <= 0.0001 && \
               abs(observed_throughput - expected_throughput) <= 0.000001 && \
               abs(observed_efficiency - expected_efficiency) <= 0.000001 && \
               abs(observed_query_row_s - expected_query_row_s) <= 0.001 && \
               abs(observed_output_token_s - expected_output_token_s) <= 0.001
        exit pass ? 0 : 1
    }
'; then
    echo "P8/G2/L2 useful-work or throughput formula regressed" >&2
    exit 1
fi

printf 'E2E PERFORMANCE SEMANTICS PASS '
printf 'P8/G2/L2 query_rows=9 decode_forwards=1 tasks=10 '
printf 'small_useful_mac=%s standard_useful_mac=%s tolerance=%s\n' \
    "${useful_mac}" "${standard_useful_mac}" \
    "${standard_numeric_tolerance}"
printf 'E2E FULL-STACK SEMANTICS PASS '
printf 'P8/G2/L36 query_rows=9 tasks=%s useful_mac=%s tolerance=%s\n' \
    "${full_stack_tasks}" "${full_stack_mac}" \
    "${full_stack_numeric_tolerance}"
