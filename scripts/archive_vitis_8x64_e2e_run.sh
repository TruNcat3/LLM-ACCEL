#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 8 ]; then
    echo "usage: $0 HOST_LOG BUILD_DIR OUTPUT_DIR [TARGET_MHZ] [XSIM_MHZ] [RELEASE_NONFINAL_BLOCKS] [BUILD_LOG] [SOURCE_ROOT]" >&2
    exit 2
fi

script_path="$(realpath "$0")"
repo_root="$(dirname "$(dirname "${script_path}")")"
host_log="$(realpath "$1")"
build_dir="$(realpath "$2")"
output_dir="$(realpath -m "$3")"
target_mhz="${4:-200}"
requested_xsim_mhz="${5:-}"
release_nonfinal="${6:-1}"
build_log=""
if [ "$#" -ge 7 ]; then
    build_log="$(realpath "$7")"
fi
source_root="${8:-${repo_root}}"
if [ ! -d "${source_root}" ]; then
    echo "Missing release source snapshot root: ${source_root}" >&2
    exit 66
fi
source_root="$(realpath "${source_root}")"

if [ ! -s "${host_log}" ]; then
    echo "Missing or empty completed Host log: ${host_log}" >&2
    exit 66
fi
if [ ! -d "${build_dir}" ]; then
    echo "Missing HW-Emu build directory: ${build_dir}" >&2
    exit 66
fi
if [ -n "${build_log}" ] && [ ! -s "${build_log}" ]; then
    echo "Missing or empty build provenance log: ${build_log}" >&2
    exit 66
fi
if [ -e "${output_dir}" ]; then
    echo "Refusing to overwrite archive path: ${output_dir}" >&2
    exit 73
fi
for value in "${target_mhz}" "${release_nonfinal}"; do
    if ! [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Expected a non-negative numeric argument, got: ${value}" >&2
        exit 2
    fi
done
if [ -n "${requested_xsim_mhz}" ] &&
   ! [[ "${requested_xsim_mhz}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Expected a non-negative XSim frequency, got: ${requested_xsim_mhz}" >&2
    exit 2
fi
if [ "${release_nonfinal}" != "0" ] && [ "${release_nonfinal}" != "1" ]; then
    echo "RELEASE_NONFINAL_BLOCKS must be 0 or 1" >&2
    exit 2
fi

metadata() {
    local name="$1"
    awk -F= -v name="${name}" '
        $1 == name { print substr($0, index($0, "=") + 1); exit }
    ' "${host_log}"
}

profile="$(metadata profile)"
prompt_tokens="$(metadata prompt_tokens)"
generated_tokens="$(metadata generated_tokens)"
layers="$(metadata layers)"
block_size="$(metadata block_size)"
if [ -z "${block_size}" ]; then
    block_size="$(metadata prefill_block_size)"
fi
exit_status="$(metadata exit_status)"
verify_e2e_golden="$(metadata verify_e2e_golden)"

for pair in \
    "profile:${profile}" \
    "prompt_tokens:${prompt_tokens}" \
    "generated_tokens:${generated_tokens}" \
    "layers:${layers}" \
    "block_size:${block_size}" \
    "verify_e2e_golden:${verify_e2e_golden}" \
    "exit_status:${exit_status}"
do
    name="${pair%%:*}"
    value="${pair#*:}"
    if [ -z "${value}" ]; then
        echo "Completed Host log is missing ${name}" >&2
        exit 65
    fi
done
for value in \
    "${prompt_tokens}" "${generated_tokens}" "${layers}" "${block_size}" \
    "${exit_status}"
do
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "Host metadata is not an unsigned integer: ${value}" >&2
        exit 65
    fi
done
if [ "${exit_status}" -ne 0 ]; then
    echo "Host run did not exit successfully: ${exit_status}" >&2
    exit 65
fi
if [ "${verify_e2e_golden}" != "0" ] &&
   [ "${verify_e2e_golden}" != "1" ]; then
    echo "Host metadata has invalid verify_e2e_golden=${verify_e2e_golden}" >&2
    exit 65
fi
if ! rg -q '^finished_at=' "${host_log}" ||
   ! rg -q '^QWEN_8X64_E2E_PROFILE .* PASS$' "${host_log}" ||
   ! rg -q '^QWEN_8X64_HOST PASS ' "${host_log}"; then
    echo "Host log does not contain complete E2E PASS evidence" >&2
    exit 65
fi
if [ "${verify_e2e_golden}" = "1" ]; then
    numeric_verify_line="$(rg '^QWEN_8X64_E2E_NUMERIC_VERIFY ' "${host_log}" | tail -n 1 || true)"
    profile_line="$(rg '^QWEN_8X64_E2E_PROFILE ' "${host_log}" | tail -n 1 || true)"
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
    if ! line_has_fields "${numeric_verify_line}" \
           token_sequence_match=1 validation_schedule=post_inference \
           intermediate_host_copy=0 kv_cache_owner=controller PASS ||
       ! line_has_fields "${profile_line}" \
           validation_schedule=post_inference e2e_numeric_golden=1 PASS ||
       ! line_has_keys "${profile_line}" \
           total_host_elapsed_ms total_process_elapsed_ms \
           post_inference_validation_ms; then
        echo "Host log is missing the full-prefix E2E numerical PASS" >&2
        exit 65
    fi
fi

host_exe="${build_dir}/host_qwen_8x64.exe"
xclbin="${build_dir}/qwen_8x64_dual.xclbin"
emconfig="${build_dir}/emconfig.json"
for input in "${host_exe}" "${xclbin}" "${emconfig}"; do
    if [ ! -s "${input}" ]; then
        echo "Missing or empty generated input: ${input}" >&2
        exit 66
    fi
done

build_log_sha=""
if [ -n "${build_log}" ]; then
    build_work_root="$(
        awk -F '=' '$1 == "work_root" { value = $2 } END { print value }' \
            "${build_log}"
    )"
    build_exit_status="$(
        awk -F '=' '$1 == "exit_status" { value = $2 } END { print value }' \
            "${build_log}"
    )"
    if [ "${build_exit_status}" != "0" ] ||
       [ ! -d "${build_work_root}" ] ||
       [ "$(realpath "${build_work_root}")" != "$(dirname "${build_dir}")" ] ||
       ! rg -q '^started_at=' "${build_log}" ||
       ! rg -q '^finished_at=' "${build_log}"; then
        echo "Build provenance log does not match the successful HW-Emu build" >&2
        exit 65
    fi
    build_log_sha="$(sha256sum "${build_log}" | awk '{ print $1 }')"
fi

recorded_xsim_mhz="$(metadata xclbin_kernel_clock_mhz)"
if [ -z "${recorded_xsim_mhz}" ]; then
    recorded_xsim_mhz="${target_mhz}"
fi
if ! [[ "${recorded_xsim_mhz}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Host log contains an invalid XCLBIN kernel clock: ${recorded_xsim_mhz}" >&2
    exit 65
fi
xsim_mhz="${requested_xsim_mhz:-${recorded_xsim_mhz}}"
if ! awk -v requested="${xsim_mhz}" -v recorded="${recorded_xsim_mhz}" \
    'BEGIN { difference = requested - recorded; if (difference < 0) difference = -difference; exit difference <= 0.001 ? 0 : 1 }'
then
    echo "Requested XSim frequency ${xsim_mhz} MHz does not match the logged XCLBIN KERNEL_CLK ${recorded_xsim_mhz} MHz" >&2
    exit 65
fi

simulation_dir="$(awk '
    /Path of the simulation directory[[:space:]]*:/ {
        line = $0
        sub(/^.*Path of the simulation directory[[:space:]]*:[[:space:]]*/, "", line)
        value = line
    }
    END { print value }
' "${host_log}")"
profile_csv="${simulation_dir}/profile_kernels.csv"
if [ -z "${simulation_dir}" ] || [ ! -s "${profile_csv}" ]; then
    echo "Cannot locate the run-local profile_kernels.csv from the Host log" >&2
    exit 66
fi

host_sha="$(sha256sum "${host_exe}" | awk '{print $1}')"
xclbin_sha="$(sha256sum "${xclbin}" | awk '{print $1}')"
emconfig_sha="$(sha256sum "${emconfig}" | awk '{print $1}')"
for identity in \
    "host_exe_sha256:${host_sha}" \
    "xclbin_sha256:${xclbin_sha}" \
    "emconfig_sha256:${emconfig_sha}"
do
    name="${identity%%:*}"
    actual="${identity#*:}"
    recorded="$(metadata "${name}")"
    if [ -n "${recorded}" ] && [ "${recorded}" != "${actual}" ]; then
        echo "Generated artifact changed after launch: ${name}" >&2
        exit 65
    fi
done

parent_dir="$(dirname "${output_dir}")"
mkdir -p "${parent_dir}"
temp_dir="$(mktemp -d "${parent_dir}/.e2e-archive.XXXXXX")"
cleanup() {
    if [ -d "${temp_dir}" ]; then
        rm -rf "${temp_dir}"
    fi
}
trap cleanup EXIT

cp "${host_log}" "${temp_dir}/host.raw.log"
cp "${profile_csv}" "${temp_dir}/profile_kernels.csv"
if [ -n "${build_log}" ]; then
    cp "${build_log}" "${temp_dir}/build.raw.log"
fi
{
    cat "${host_log}"
    echo "artifact_identity_source=archive_recomputed_sha256"
    echo "host_exe_sha256=${host_sha}"
    echo "xclbin_sha256=${xclbin_sha}"
    echo "emconfig_sha256=${emconfig_sha}"
} > "${temp_dir}/host.evidence.log"

"${repo_root}/scripts/report_vitis_8x64_e2e_trace.sh" \
    "${temp_dir}/profile_kernels.csv" \
    "${profile}" \
    "${prompt_tokens}" \
    "${generated_tokens}" \
    "${layers}" \
    "${block_size}" \
    "${target_mhz}" \
    "${xsim_mhz}" \
    "${release_nonfinal}" \
    "${temp_dir}/host.evidence.log" \
    > "${temp_dir}/performance.tsv"

resource_report_included=0
work_root="$(dirname "${build_dir}")"
if [ "${profile}" = "qwen2.5-3b" ] &&
   [ -d "${work_root}/hls" ] &&
   [ -x "${repo_root}/scripts/report_qwen3b_hls_resources.sh" ]; then
    "${repo_root}/scripts/report_qwen3b_hls_resources.sh" \
        "${work_root}" "${target_mhz}" \
        > "${temp_dir}/hls_resources.tsv"
    resource_report_included=1
fi

source_snapshot_included=0
if [ "${profile}" = "qwen2.5-3b" ] &&
   [ -x "${repo_root}/scripts/report_qwen3b_source_snapshot.sh" ]; then
    "${repo_root}/scripts/report_qwen3b_source_snapshot.sh" \
        "${source_root}" > "${temp_dir}/source_manifest.tsv"
    source_snapshot_included=1
fi

build_source_equivalence_included=0
if [ "${profile}" = "qwen2.5-3b" ] &&
   [ -n "${build_log}" ] &&
   [ -x "${repo_root}/scripts/report_qwen3b_build_source_equivalence.sh" ]; then
    "${repo_root}/scripts/report_qwen3b_build_source_equivalence.sh" \
        "${build_log}" "${source_root}" \
        > "${temp_dir}/build_source_equivalence.tsv"
    build_source_equivalence_included=1
fi

{
    printf 'field\tvalue\n'
    printf 'profile\t%s\n' "${profile}"
    printf 'prompt_tokens\t%s\n' "${prompt_tokens}"
    printf 'generated_tokens\t%s\n' "${generated_tokens}"
    printf 'layers\t%s\n' "${layers}"
    printf 'block_size\t%s\n' "${block_size}"
    printf 'verify_e2e_golden\t%s\n' "${verify_e2e_golden}"
    printf 'validation_schedule\t%s\n' \
        "$([ "${verify_e2e_golden}" = "1" ] && printf post_inference || printf disabled)"
    printf 'measurement_boundary\t%s\n' \
        'inference_ends_before_cpu_oracle'
    printf 'trace_scope\t%s\n' 'common_four_CU_running_time'
    printf 'per_cu_occupancy_resolved\t0\n'
    printf 'inter_task_issue_gaps_resolved\t0\n'
    printf 'physical_board_measurement\t0\n'
    printf 'host_inference_compute_scope\t%s\n' \
        'embedding_plus_lm_head_argmax'
    printf 'accelerator_compute_scope\t%s\n' \
        'decoder_layers_final_norm_rope_online_attention_kv'
    printf 'cpu_golden_scope\t%s\n' \
        'post_inference_validation_only'
    printf 'target_mhz\t%s\n' "${target_mhz}"
    printf 'xsim_mhz\t%s\n' "${xsim_mhz}"
    printf 'release_nonfinal_blocks\t%s\n' "${release_nonfinal}"
    printf 'hls_resource_report_included\t%s\n' \
        "${resource_report_included}"
    if [ "${resource_report_included}" = "1" ]; then
        printf 'hls_resource_scope\t%s\n' \
            'controller_plus_2compute_plus_status'
    fi
    printf 'source_snapshot_included\t%s\n' \
        "${source_snapshot_included}"
    if [ "${source_snapshot_included}" = "1" ]; then
        printf 'source_snapshot_scope\t%s\n' \
            'archive_time_release_candidate_worktree'
        printf 'binary_identity_scope\t%s\n' \
            'launch_recorded_and_archive_recomputed_sha256'
    fi
    printf 'build_source_equivalence_included\t%s\n' \
        "${build_source_equivalence_included}"
    if [ "${build_source_equivalence_included}" = "1" ]; then
        printf 'build_source_equivalence_scope\t%s\n' \
            'exact_build_inputs_and_build_harness_vs_release_worktree'
    fi
    printf 'build_log_included\t%s\n' \
        "$([ -n "${build_log}" ] && printf 1 || printf 0)"
    if [ -n "${build_log}" ]; then
        printf 'build_log_scope\t%s\n' 'complete_HLS_XO_link_Host_stdout_stderr'
        printf 'build_log_sha256\t%s\n' "${build_log_sha}"
    fi
    printf 'host_exe_sha256\t%s\n' "${host_sha}"
    printf 'xclbin_sha256\t%s\n' "${xclbin_sha}"
    printf 'emconfig_sha256\t%s\n' "${emconfig_sha}"
    printf 'source_host_log\t%s\n' "${host_log}"
    printf 'source_profile_csv\t%s\n' "${profile_csv}"
} > "${temp_dir}/manifest.tsv"

(
    cd "${temp_dir}"
    checksum_inputs=(
        host.raw.log host.evidence.log profile_kernels.csv
        performance.tsv manifest.tsv
    )
    if [ "${resource_report_included}" = "1" ]; then
        checksum_inputs+=(hls_resources.tsv)
    fi
    if [ "${source_snapshot_included}" = "1" ]; then
        checksum_inputs+=(source_manifest.tsv)
    fi
    if [ "${build_source_equivalence_included}" = "1" ]; then
        checksum_inputs+=(build_source_equivalence.tsv)
    fi
    if [ -n "${build_log}" ]; then
        checksum_inputs+=(build.raw.log)
    fi
    sha256sum "${checksum_inputs[@]}" > checksums.sha256
)
mv "${temp_dir}" "${output_dir}"
trap - EXIT

printf 'archive=%s\nperformance=%s\nchecksums=%s\n' \
    "${output_dir}" \
    "${output_dir}/performance.tsv" \
    "${output_dir}/checksums.sha256"
