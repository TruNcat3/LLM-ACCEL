#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 E2E_ARCHIVE_DIR" >&2
    exit 2
fi

script_path="$(realpath "$0")"
repo_root="$(dirname "$(dirname "${script_path}")")"
archive_dir="$(realpath "$1")"

for file in \
    checksums.sha256 host.raw.log manifest.tsv performance.tsv \
    profile_kernels.csv
do
    if [ ! -s "${archive_dir}/${file}" ]; then
        echo "E2E archive is missing ${file}" >&2
        exit 66
    fi
done

"${repo_root}/scripts/verify_result_checksums.sh" \
    "${archive_dir}/checksums.sha256" >/dev/null

tsv_value() {
    local file="$1"
    local column="$2"
    awk -F '\t' -v column="${column}" '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                if ($i == column) column_index = i
            }
            next
        }
        NR == 2 && column_index { print $column_index; exit }
    ' "${file}"
}

manifest_value() {
    local field="$1"
    awk -F '\t' -v field="${field}" '
        $1 == field { value = $2 }
        END { print value }
    ' "${archive_dir}/manifest.tsv"
}

performance="${archive_dir}/performance.tsv"
profile="$(tsv_value "${performance}" profile)"
sequence_batch="$(tsv_value "${performance}" sequence_batch)"
prompt_tokens="$(tsv_value "${performance}" prompt_sequence_tokens)"
sampled_tokens="$(tsv_value "${performance}" sampled_output_tokens)"
decode_forwards="$(tsv_value "${performance}" decode_forwards)"
prefill_blocks="$(tsv_value "${performance}" prefill_blocks)"
layers="$(tsv_value "${performance}" layers)"
block_rows="$(tsv_value "${performance}" configured_max_active_query_rows_per_prefill_block)"
tasks="$(tsv_value "${performance}" expected_coarse_tasks)"
evidence_source="$(tsv_value "${performance}" evidence_source)"
timed_scope="$(tsv_value "${performance}" timed_scope)"
numeric_validation="$(tsv_value "${performance}" numeric_validation)"
numeric_steps="$(tsv_value "${performance}" numeric_steps)"
checked_values="$(tsv_value "${performance}" numeric_checked_values)"
max_error="$(tsv_value "${performance}" numeric_max_raw_error)"
tolerance="$(tsv_value "${performance}" numeric_tolerance)"
xsim_us="$(tsv_value "${performance}" xsim_cu_running_us)"
xsim_mhz="$(tsv_value "${performance}" xsim_clock_mhz)"
xsim_cycles="$(tsv_value "${performance}" xsim_cycles)"
target_mhz="$(tsv_value "${performance}" target_clock_mhz)"
target_us="$(tsv_value "${performance}" projected_target_us)"
useful_mac="$(tsv_value "${performance}" useful_mac)"
useful_gmac_s="$(tsv_value "${performance}" useful_gmac_s)"
modeled_efficiency="$(
    tsv_value "${performance}" modeled_interval_efficiency_percent
)"
query_row_s="$(tsv_value "${performance}" query_row_s)"
request_output_token_s="$(
    tsv_value "${performance}" request_output_token_s
)"
host_inference_ms="$(tsv_value "${performance}" host_inference_ms)"
host_process_ms="$(tsv_value "${performance}" host_process_ms)"
host_lm_head_ms="$(tsv_value "${performance}" host_lm_head_ms)"
host_validation_ms="$(tsv_value "${performance}" host_validation_ms)"
host_exe_sha="$(tsv_value "${performance}" host_exe_sha256)"
xclbin_sha="$(tsv_value "${performance}" xclbin_sha256)"
emconfig_sha="$(tsv_value "${performance}" emconfig_sha256)"

for value in \
    "${profile}" "${sequence_batch}" "${prompt_tokens}" \
    "${sampled_tokens}" "${decode_forwards}" "${prefill_blocks}" \
    "${layers}" "${block_rows}" "${tasks}" "${evidence_source}" \
    "${timed_scope}" "${numeric_validation}" "${xsim_us}" \
    "${xsim_cycles}" "${target_mhz}" "${useful_gmac_s}" \
    "${modeled_efficiency}"
do
    if [ -z "${value}" ]; then
        echo "E2E performance row is missing a required publication field" >&2
        exit 65
    fi
done

model_source="$(
    awk -F '=' '$1 == "model_source" { value = $2 } END { print value }' \
        "${archive_dir}/host.raw.log"
)"
tie_embeddings="$(
    awk -F '=' '$1 == "tie_embeddings" { value = $2 } END { print value }' \
        "${archive_dir}/host.raw.log"
)"

printf '# %s P%s/G%s/L%s End-to-End Hardware-Emulation Evidence\n\n' \
    "${profile}" "${prompt_tokens}" "${sampled_tokens}" "${layers}"
printf 'This package freezes a completed, checksum-verified `%s` run. ' \
    "${evidence_source}"
printf 'It is modeled RTL evidence, not a physical-board measurement.\n\n'

printf '## Workload contract\n\n'
printf '| Field | Value |\n| --- | ---: |\n'
printf '| Sequence batch | %s |\n' "${sequence_batch}"
printf '| Prompt sequence tokens | %s |\n' "${prompt_tokens}"
printf '| Maximum active prefill rows per block | %s |\n' "${block_rows}"
printf '| Prefill blocks | %s |\n' "${prefill_blocks}"
printf '| Sampled output tokens | %s |\n' "${sampled_tokens}"
printf '| Real D1 decode forwards | %s |\n' "${decode_forwards}"
printf '| Decoder layers exercised | %s |\n' "${layers}"
printf '| Host-visible coarse tasks | %s |\n\n' "${tasks}"
printf '`P%s` means %s consecutive query rows from one sequence, not batch ' \
    "${block_rows}" "${block_rows}"
printf '%s and not %s generated tokens in parallel. ' \
    "${block_rows}" "${block_rows}"
printf 'The first sampled token comes from the final prefill hidden state; '
printf 'the remaining %s forward(s) are true one-row decode passes.\n\n' \
    "${decode_forwards}"

printf '## Numerical result\n\n'
printf '| Validation | Steps | Checked values | Maximum raw error | Tolerance |\n'
printf '| --- | ---: | ---: | ---: | ---: |\n'
printf '| %s | %s | %s | %s | %s |\n\n' \
    "${numeric_validation}" "${numeric_steps}" "${checked_values}" \
    "${max_error}" "${tolerance}"
printf 'The CPU fixed-point oracle executes after production inference and is '
printf 'excluded from accelerator useful work. Model source is `%s`; ' \
    "${model_source:-not_recorded}"
printf 'tied embeddings are `%s`.\n\n' "${tie_embeddings:-not_recorded}"

printf '## Modeled performance\n\n'
printf '| Evidence source | Timed scope | CU running time | XSim clock | Cycles | Target clock | Target-equivalent latency | Useful MAC | Useful GMAC/s | Modeled useful-MAC efficiency | Query rows/s | Request output tokens/s |\n'
printf '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
printf '| %s | %s | %s us | %s MHz | %s | %s MHz | %s us | %s | %s | %s%% | %s | %s |\n\n' \
    "${evidence_source}" "${timed_scope}" "${xsim_us}" "${xsim_mhz}" \
    "${xsim_cycles}" "${target_mhz}" "${target_us}" "${useful_mac}" \
    "${useful_gmac_s}" "${modeled_efficiency}" "${query_row_s}" \
    "${request_output_token_s}"
printf 'The common four-CU profiler interval does not resolve separable per-CU '
printf 'occupancy or inter-task issue gaps. Request output-token throughput '
printf 'includes prefill and must not be read as steady-state D1 throughput.\n\n'

printf 'Host wall-clock fields are retained only as HW-Emu simulator proxies:\n\n'
printf '| Inference | Whole process | LM head | Post-inference validation |\n'
printf '| ---: | ---: | ---: | ---: |\n'
printf '| %s ms | %s ms | %s ms | %s ms |\n\n' \
    "${host_inference_ms}" "${host_process_ms}" "${host_lm_head_ms}" \
    "${host_validation_ms}"

if [ -s "${archive_dir}/hls_resources.tsv" ]; then
    printf '## HLS resource estimates\n\n'
    printf '| Scope | Instances | BRAM18 | DSP | FF | LUT | URAM | Estimated period |\n'
    printf '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    awk -F '\t' '
        NR == 1 {
            for (i = 1; i <= NF; i++) column[$i] = i
            next
        }
        $column["scope"] == "controller" ||
        $column["scope"] == "compute_two" ||
        $column["scope"] == "status" ||
        $column["scope"] == "whole_system" {
            printf "| %s | %s | %s | %s | %s | %s | %s | %s ns |\n", \
                $column["scope"], $column["instances"], \
                $column["bram18k"], $column["dsp"], \
                $column["ff"], $column["lut"], $column["uram"], \
                $column["hls_estimated_ns"]
        }
    ' "${archive_dir}/hls_resources.tsv"
    printf '\nThese are local CSynth estimates, not routed utilization or timing closure.\n\n'
fi

printf '## Execution boundary\n\n'
printf -- '- Host inference compute: `%s`.\n' \
    "$(manifest_value host_inference_compute_scope)"
printf -- '- Accelerator compute: `%s`.\n' \
    "$(manifest_value accelerator_compute_scope)"
printf -- '- CPU golden: `%s`.\n' "$(manifest_value cpu_golden_scope)"
printf -- '- Intermediate hidden copy: none; KV owner: controller/HBM.\n'
printf -- '- Trace scope: `%s`; physical-board measurement: `%s`.\n\n' \
    "$(manifest_value trace_scope)" \
    "$(manifest_value physical_board_measurement)"

printf '## Provenance\n\n'
printf '| Artifact | SHA-256 |\n| --- | --- |\n'
printf '| Host executable | `%s` |\n' "${host_exe_sha}"
printf '| XCLBIN | `%s` |\n' "${xclbin_sha}"
printf '| Emulation configuration | `%s` |\n\n' "${emconfig_sha}"
printf 'The package contains the raw Host/build logs, CU profile, performance '
printf 'row, HLS resource row when applicable, archive-time source snapshot, '
if [ -s "${archive_dir}/build_source_equivalence.tsv" ]; then
    printf 'build-to-release source-equivalence table, '
fi
printf 'manifest, and checksums. Verify it from inside the package with:\n\n'
printf '```bash\nsha256sum -c checksums.sha256\n```\n'
