#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "usage: $0 HOST_LOG BUILD_DIR [ARCHIVE_DIR] [SOURCE_ROOT]" >&2
    exit 2
fi

script_path="$(realpath "$0")"
repo_root="$(dirname "$(dirname "${script_path}")")"
host_log="$(realpath "$1")"
build_dir="$(realpath "$2")"
archive_dir=""
if [ "$#" -ge 3 ]; then
    archive_dir="$(realpath "$3")"
fi
source_root="${4:-${repo_root}}"
if [ ! -d "${source_root}" ]; then
    echo "Missing release source snapshot root: ${source_root}" >&2
    exit 66
fi
source_root="$(realpath "${source_root}")"

metadata() {
    local key="$1"
    awk -F '=' -v key="${key}" '
        $1 == key { value = substr($0, length(key) + 2) }
        END { print value }
    ' "${host_log}"
}

line_field() {
    local line="$1"
    local key="$2"
    awk -v key="${key}" '
        {
            prefix = key "="
            for (i = 1; i <= NF; i++) {
                if (index($i, prefix) == 1) {
                    print substr($i, length(prefix) + 1)
                    exit
                }
            }
        }
    ' <<<"${line}"
}

require_field() {
    local line="$1"
    local required="$2"
    if [[ " ${line} " != *" ${required} "* ]]; then
        echo "Missing release field: ${required}" >&2
        exit 65
    fi
}

for input in "${host_log}" "${build_dir}"; do
    if [ ! -e "${input}" ]; then
        echo "Missing release input: ${input}" >&2
        exit 66
    fi
done

host_exe="${build_dir}/host_qwen_8x64.exe"
xclbin="${build_dir}/qwen_8x64_dual.xclbin"
emconfig="${build_dir}/emconfig.json"
for input in "${host_exe}" "${xclbin}" "${emconfig}"; do
    if [ ! -s "${input}" ]; then
        echo "Missing generated artifact: ${input}" >&2
        exit 66
    fi
done

for identity in \
    "host_exe_sha256:${host_exe}" \
    "xclbin_sha256:${xclbin}" \
    "emconfig_sha256:${emconfig}"
do
    key="${identity%%:*}"
    artifact="${identity#*:}"
    recorded="$(metadata "${key}")"
    actual="$(sha256sum "${artifact}" | awk '{ print $1 }')"
    if [ -z "${recorded}" ] || [ "${recorded}" != "${actual}" ]; then
        echo "Release artifact identity mismatch: ${key}" >&2
        exit 65
    fi
done

xclbin_info="$(xclbinutil --info --input "${xclbin}" 2>/dev/null)"
kernel_clock="$(
    awk '
        $1 == "Name:" && $2 == "KERNEL_CLK" { kernel = 1; next }
        kernel && $1 == "Frequency:" { print $2; kernel = 0 }
    ' <<<"${xclbin_info}"
)"
if [ "${kernel_clock}" != "200" ]; then
    echo "Release xclbin is not the 200 MHz HW-Emu target: ${kernel_clock}" >&2
    exit 65
fi
for required in \
    'Kernel: compute_core_8x64_unified_nk' \
    'Instance:        cc8_cu0' \
    'Instance:        cc8_cu1' \
    'Kernel: control_cache_8x64_dual_core_nk' \
    'Instance:        cc8_ctrl' \
    'Kernel: cc8_status_sink_nk' \
    'Instance:        cc8_status'
do
    if [[ "${xclbin_info}" != *"${required}"* ]]; then
        echo "Release xclbin topology is missing: ${required}" >&2
        exit 65
    fi
done
if ! rg -a -q -- '--verify-e2e-golden' "${host_exe}"; then
    echo "Release Host lacks the numerical E2E oracle option" >&2
    exit 65
fi

profile="$(metadata profile)"
prompt_tokens="$(metadata prompt_tokens)"
generated_tokens="$(metadata generated_tokens)"
layers="$(metadata layers)"
block_size="$(metadata block_size)"
recorded_prompt_blocks="$(metadata prompt_blocks)"
recorded_decode_forwards="$(metadata decode_forwards)"
recorded_tasks="$(metadata expected_coarse_tasks)"
verify_e2e_golden="$(metadata verify_e2e_golden)"
recorded_kernel_clock="$(metadata xclbin_kernel_clock_mhz)"

for value_name in \
    prompt_tokens generated_tokens layers block_size \
    recorded_prompt_blocks recorded_decode_forwards recorded_tasks
do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "Release workload has malformed ${value_name}=${value:-missing}" >&2
        exit 65
    fi
done
if [ "${profile}" != "qwen2.5-3b" ] ||
   [ "${prompt_tokens}" -lt 1 ] || [ "${prompt_tokens}" -gt 2048 ] ||
   [ "${generated_tokens}" -lt 1 ] ||
   [ "${prompt_tokens}" -gt $((2048 - generated_tokens)) ] ||
   [ "${layers}" -lt 1 ] || [ "${layers}" -gt 36 ] ||
   [ "${block_size}" -lt 1 ] || [ "${block_size}" -gt 8 ] ||
   [ "${verify_e2e_golden}" != "1" ] ||
   [ "${recorded_kernel_clock}" != "200" ]; then
    echo "Release workload is outside the Qwen2.5-3B E2E contract" >&2
    exit 65
fi
prompt_blocks=$(((prompt_tokens + block_size - 1) / block_size))
decode_forwards=$((generated_tokens - 1))
expected_tasks=$((
    prompt_blocks * 2 * layers + 1 +
    decode_forwards * (2 * layers + 1)
))
if [ "${recorded_prompt_blocks}" -ne "${prompt_blocks}" ] ||
   [ "${recorded_decode_forwards}" -ne "${decode_forwards}" ] ||
   [ "${recorded_tasks}" -ne "${expected_tasks}" ]; then
    echo "Release workload metadata does not match its derived task plan" >&2
    exit 65
fi

exit_status="$(metadata exit_status)"
if [ -z "${exit_status}" ]; then
    echo "QWEN3B_E2E_RELEASE PENDING host_run_incomplete=1" >&2
    exit 75
fi
if [ "${exit_status}" != "0" ]; then
    echo "QWEN3B_E2E_RELEASE FAIL host_exit_status=${exit_status}" >&2
    exit 65
fi

"${repo_root}/scripts/verify_vitis_8x64_e2e_progress.sh" \
    "${host_log}" "${prompt_tokens}" "${generated_tokens}" \
    "${layers}" "${block_size}" >/dev/null

numeric_summary="$(
    awk '/^QWEN_8X64_E2E_NUMERIC_VERIFY / { last = $0 } END { print last }' \
        "${host_log}"
)"
profile_summary="$(
    awk '/^QWEN_8X64_E2E_PROFILE / { last = $0 } END { print last }' \
        "${host_log}"
)"
host_pass="$(
    awk '/^QWEN_8X64_HOST PASS / { last = $0 } END { print last }' \
        "${host_log}"
)"
expected_checked_values=$((generated_tokens * 2048))
expected_tolerance=$((32 * layers))
for required in \
    "prompt_tokens=${prompt_tokens}" \
    "generated_tokens=${generated_tokens}" \
    "layers=${layers}" \
    "steps=${generated_tokens}" \
    "checked_values=${expected_checked_values}" \
    "tolerance=${expected_tolerance}" 'token_sequence_match=1' \
    'validation_schedule=post_inference' 'intermediate_host_copy=0' \
    'kv_cache_owner=controller' 'PASS'
do
    require_field "${numeric_summary}" "${required}"
done
for required in \
    "prompt_tokens=${prompt_tokens}" \
    "generated_tokens=${generated_tokens}" \
    "prompt_forward_passes=${prompt_blocks}" \
    "generated_forward_passes=${decode_forwards}" \
    'prefill_mode=coarse_block' "prefill_block_size=${block_size}" \
    "coarse_task_count=${expected_tasks}" \
    "output_materializations=${generated_tokens}" \
    "released_prompt_blocks=$((prompt_blocks - 1))" \
    'e2e_numeric_golden=1' \
    "e2e_numeric_steps=${generated_tokens}" \
    "e2e_numeric_checked_values=${expected_checked_values}" \
    "e2e_numeric_tolerance=${expected_tolerance}" \
    'validation_schedule=post_inference' 'intermediate_host_copy=0' \
    'kv_cache_owner=controller' 'PASS'
do
    require_field "${profile_summary}" "${required}"
done
if [ -z "${host_pass}" ]; then
    echo "Release Host log is missing its final PASS" >&2
    exit 65
fi
max_error="$(line_field "${numeric_summary}" max_raw_error)"
tolerance="$(line_field "${numeric_summary}" tolerance)"
if ! [[ "${max_error}" =~ ^[0-9]+$ ]] ||
   ! [[ "${tolerance}" =~ ^[0-9]+$ ]] ||
   [ "${max_error}" -gt "${tolerance}" ]; then
    echo "Release numerical error is outside tolerance" >&2
    exit 65
fi

simulation_dir="$(
    awk '
        /Path of the simulation directory[[:space:]]*:/ {
            sub(/^.*:[[:space:]]*/, "")
            path = $0
        }
        END { print path }
    ' "${host_log}"
)"
profile_csv="${simulation_dir}/profile_kernels.csv"
if [ ! -s "${profile_csv}" ]; then
    echo "Release run is missing profile_kernels.csv" >&2
    exit 66
fi
report_file="$(mktemp "${TMPDIR:-/tmp}/qwen3b-release-report.XXXXXX.tsv")"
resource_report_file="$(mktemp "${TMPDIR:-/tmp}/qwen3b-release-resources.XXXXXX.tsv")"
source_snapshot_file="$(mktemp "${TMPDIR:-/tmp}/qwen3b-release-sources.XXXXXX.tsv")"
build_source_equivalence_file="$(mktemp "${TMPDIR:-/tmp}/qwen3b-release-build-sources.XXXXXX.tsv")"
cleanup() {
    rm -f "${report_file}" "${resource_report_file}" \
        "${source_snapshot_file}" "${build_source_equivalence_file}"
}
trap cleanup EXIT
"${repo_root}/scripts/report_vitis_8x64_e2e_trace.sh" \
    "${profile_csv}" qwen2.5-3b \
    "${prompt_tokens}" "${generated_tokens}" "${layers}" \
    "${block_size}" 200 200 1 "${host_log}" \
    > "${report_file}"
if [ "$(awk 'END { print NR }' "${report_file}")" -ne 2 ]; then
    echo "Release performance report must contain one header and one row" >&2
    exit 65
fi
"${repo_root}/scripts/report_qwen3b_hls_resources.sh" \
    "$(dirname "${build_dir}")" 200 > "${resource_report_file}"
awk -F '\t' '
    NR == 1 {
        for (i = 1; i <= NF; i++) column[$i] = i
        next
    }
    $1 == "whole_system" {
        if ($(column["instances"]) != 4 ||
            $(column["link_target_mhz"]) != 200 ||
            $(column["bram18k"]) != 1308 ||
            $(column["dsp"]) != 1480 ||
            $(column["ff"]) != 829923 ||
            $(column["lut"]) != 697267 ||
            $(column["uram"]) != 0) exit 1
        found = 1
    }
    END { exit found ? 0 : 1 }
' "${resource_report_file}"
"${repo_root}/scripts/report_qwen3b_source_snapshot.sh" \
    "${source_root}" > "${source_snapshot_file}"

if [ -n "${archive_dir}" ]; then
    for file in \
        build.raw.log checksums.sha256 host.evidence.log host.raw.log manifest.tsv \
        performance.tsv profile_kernels.csv hls_resources.tsv \
        source_manifest.tsv build_source_equivalence.tsv
    do
        if [ ! -s "${archive_dir}/${file}" ]; then
            echo "Release archive is missing ${file}" >&2
            exit 66
        fi
    done
    (cd "${archive_dir}" && sha256sum --check checksums.sha256)
    archived_build_sha="$(
        awk -F '\t' '$1 == "build_log_sha256" { value = $2 } END { print value }' \
            "${archive_dir}/manifest.tsv"
    )"
    actual_build_sha="$(
        sha256sum "${archive_dir}/build.raw.log" | awk '{ print $1 }'
    )"
    archived_build_root="$(
        awk -F '=' '$1 == "work_root" { value = $2 } END { print value }' \
            "${archive_dir}/build.raw.log"
    )"
    archived_build_status="$(
        awk -F '=' '$1 == "exit_status" { value = $2 } END { print value }' \
            "${archive_dir}/build.raw.log"
    )"
    if [ "${archived_build_sha}" != "${actual_build_sha}" ] ||
       [ "${archived_build_status}" != "0" ] ||
       [ "${archived_build_root}" != "$(dirname "${build_dir}")" ] ||
       ! rg -q '^started_at=' "${archive_dir}/build.raw.log" ||
       ! rg -q '^finished_at=' "${archive_dir}/build.raw.log"; then
        echo "Archived build provenance is incomplete or inconsistent" >&2
        exit 65
    fi
    if ! cmp -s "${report_file}" "${archive_dir}/performance.tsv"; then
        echo "Archived performance row is not reproducible" >&2
        exit 65
    fi
    if ! cmp -s "${resource_report_file}" \
        "${archive_dir}/hls_resources.tsv"; then
        echo "Archived HLS resource row is not reproducible" >&2
        exit 65
    fi
    if ! cmp -s "${source_snapshot_file}" \
        "${archive_dir}/source_manifest.tsv"; then
        echo "Archived source snapshot does not match the current worktree" >&2
        exit 65
    fi
    "${repo_root}/scripts/report_qwen3b_build_source_equivalence.sh" \
        "${archive_dir}/build.raw.log" "${source_root}" \
        > "${build_source_equivalence_file}"
    if ! cmp -s "${build_source_equivalence_file}" \
        "${archive_dir}/build_source_equivalence.tsv"; then
        echo "Archived build/release source equivalence is not reproducible" >&2
        exit 65
    fi
    awk -F '\t' '
        NR == 1 {
            if ($1 != "path" || $2 != "role" ||
                $3 != "build_source_sha256" ||
                $4 != "release_source_sha256" || $5 != "result") exit 1
            next
        }
        {
            if (NF != 5 || seen[$1]++ || $3 != $4 || $5 != "MATCH") exit 1
            if ($2 != "build_input" && $2 != "build_harness") exit 1
            count++
        }
        END { exit count >= 30 ? 0 : 1 }
    ' "${archive_dir}/build_source_equivalence.tsv" || {
        echo "Archived build/release source equivalence is malformed" >&2
        exit 65
    }
    for pair in \
        'profile:qwen2.5-3b' \
        "prompt_tokens:${prompt_tokens}" \
        "generated_tokens:${generated_tokens}" \
        "layers:${layers}" "block_size:${block_size}" \
        'verify_e2e_golden:1' \
        'validation_schedule:post_inference' \
        'measurement_boundary:inference_ends_before_cpu_oracle' \
        'trace_scope:common_four_CU_running_time' \
        'per_cu_occupancy_resolved:0' \
        'inter_task_issue_gaps_resolved:0' \
        'physical_board_measurement:0' \
        'host_inference_compute_scope:embedding_plus_lm_head_argmax' \
        'accelerator_compute_scope:decoder_layers_final_norm_rope_online_attention_kv' \
        'cpu_golden_scope:post_inference_validation_only' \
        'hls_resource_report_included:1' \
        'hls_resource_scope:controller_plus_2compute_plus_status' \
        'source_snapshot_included:1' \
        'source_snapshot_scope:archive_time_release_candidate_worktree' \
        'binary_identity_scope:launch_recorded_and_archive_recomputed_sha256' \
        'build_source_equivalence_included:1' \
        'build_source_equivalence_scope:exact_build_inputs_and_build_harness_vs_release_worktree' \
        'build_log_included:1' \
        'build_log_scope:complete_HLS_XO_link_Host_stdout_stderr'
    do
        key="${pair%%:*}"
        expected="${pair#*:}"
        actual="$(
            awk -F '\t' -v key="${key}" \
                '$1 == key { value = $2 } END { print value }' \
                "${archive_dir}/manifest.tsv"
        )"
        if [ "${actual}" != "${expected}" ]; then
            echo "Release archive manifest mismatch: ${key}" >&2
            exit 65
        fi
    done
fi

echo "QWEN3B_E2E_RELEASE PASS profile=qwen2.5-3b workload=P${prompt_tokens}+G${generated_tokens} layers=${layers} tasks=${expected_tasks} numerical_full_prefix=PASS max_raw_error=${max_error} tolerance=${tolerance}"
