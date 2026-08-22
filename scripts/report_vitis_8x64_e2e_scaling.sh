#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 BASE_PERFORMANCE_TSV COMPARISON_PERFORMANCE_TSV [...]" >&2
    exit 2
fi

printf '%s\n' \
    $'result\tlayers\texpected_coarse_tasks\txsim_cycles\tcycles_per_layer\tuseful_mac\tuseful_gmac_s\tefficiency_percent\tcycle_scale_vs_base\tmac_scale_vs_base\tcycles_per_layer_change_percent\tthroughput_change_percent\tefficiency_delta_pp'

base_signature=""
base_layers=""
base_cycles=""
base_mac=""
base_throughput=""
base_efficiency=""

for input in "$@"; do
    if [ ! -s "${input}" ]; then
        echo "Missing or empty E2E performance report: ${input}" >&2
        exit 66
    fi
    input="$(realpath "${input}")"
    extracted="$(
        awk -F '\t' '
            NR == 1 {
                for (i = 1; i <= NF; i++) column[$i] = i
                required = "profile sequence_batch prompt_sequence_tokens sampled_output_tokens decode_forwards prefill_blocks configured_max_active_query_rows_per_prefill_block release_nonfinal_blocks xsim_clock_mhz target_clock_mhz timed_scope host_validation numeric_validation artifact_identity host_exe_sha256 xclbin_sha256 emconfig_sha256 numeric_steps numeric_checked_values layers expected_coarse_tasks xsim_cycles useful_mac useful_gmac_s modeled_interval_efficiency_percent"
                count = split(required, name, " ")
                for (i = 1; i <= count; i++) {
                    if (!(name[i] in column)) exit 2
                }
                next
            }
            NR == 2 {
                printf "%s", $(column["profile"])
                fields = "sequence_batch prompt_sequence_tokens sampled_output_tokens decode_forwards prefill_blocks configured_max_active_query_rows_per_prefill_block release_nonfinal_blocks xsim_clock_mhz target_clock_mhz timed_scope host_validation numeric_validation artifact_identity host_exe_sha256 xclbin_sha256 emconfig_sha256 numeric_steps numeric_checked_values layers expected_coarse_tasks xsim_cycles useful_mac useful_gmac_s modeled_interval_efficiency_percent"
                count = split(fields, name, " ")
                for (i = 1; i <= count; i++) {
                    printf "\t%s", $(column[name[i]])
                }
                printf "\n"
            }
            END { if (NR != 2) exit 3 }
        ' "${input}"
    )" || {
        echo "Malformed E2E performance report: ${input}" >&2
        exit 65
    }

    IFS=$'\t' read -r \
        profile sequence_batch prompt_tokens sampled_tokens decode_forwards \
        prefill_blocks block_rows release_nonfinal xsim_mhz target_mhz \
        timed_scope host_validation numeric_validation artifact_identity \
        host_sha xclbin_sha emconfig_sha numeric_steps checked_values \
        layers tasks cycles useful_mac throughput efficiency \
        <<<"${extracted}"

    if [ "${host_validation}" != "PASS" ] ||
       [ "${numeric_validation}" != "PASS" ] ||
       [ "${artifact_identity}" != "PASS" ] ||
       ! awk -v layers="${layers}" -v tasks="${tasks}" \
           -v cycles="${cycles}" -v mac="${useful_mac}" \
           -v throughput="${throughput}" -v efficiency="${efficiency}" '
               BEGIN {
                   exit layers >= 1 && tasks >= 1 && cycles > 0 && mac > 0 &&
                        throughput > 0 && efficiency > 0 ? 0 : 1
               }
           '
    then
        echo "E2E scaling input is not a positive, fully validated result: ${input}" >&2
        exit 65
    fi

    signature="$(printf '%s|' \
        "${profile}" "${sequence_batch}" "${prompt_tokens}" \
        "${sampled_tokens}" "${decode_forwards}" "${prefill_blocks}" \
        "${block_rows}" "${release_nonfinal}" "${xsim_mhz}" \
        "${target_mhz}" "${timed_scope}" "${host_sha}" \
        "${xclbin_sha}" "${emconfig_sha}" "${numeric_steps}" \
        "${checked_values}")"
    if [ -z "${base_signature}" ]; then
        base_signature="${signature}"
        base_layers="${layers}"
        base_cycles="${cycles}"
        base_mac="${useful_mac}"
        base_throughput="${throughput}"
        base_efficiency="${efficiency}"
    elif [ "${signature}" != "${base_signature}" ]; then
        echo "E2E scaling inputs do not share one workload and artifact identity: ${input}" >&2
        exit 65
    fi

    result_name="$(basename "$(dirname "${input}")")"
    awk \
        -v result="${result_name}" \
        -v layers="${layers}" -v tasks="${tasks}" \
        -v cycles="${cycles}" -v mac="${useful_mac}" \
        -v throughput="${throughput}" -v efficiency="${efficiency}" \
        -v base_layers="${base_layers}" -v base_cycles="${base_cycles}" \
        -v base_mac="${base_mac}" -v base_throughput="${base_throughput}" \
        -v base_efficiency="${base_efficiency}" '
        BEGIN {
            OFS = "\t"
            cycles_per_layer = cycles / layers
            base_cycles_per_layer = base_cycles / base_layers
            printf "%s\t%d\t%d\t%.1f\t%.3f\t%.0f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                result, layers, tasks, cycles, cycles_per_layer, mac,
                throughput, efficiency, cycles / base_cycles,
                mac / base_mac,
                100.0 * (cycles_per_layer / base_cycles_per_layer - 1.0),
                100.0 * (throughput / base_throughput - 1.0),
                efficiency - base_efficiency
        }
    '
done
