#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 6 ]; then
    echo "usage: $0 HOST_LOG BUILD_DIR OUTPUT_DIR [TARGET_MHZ] [XSIM_MHZ] [RELEASE_NONFINAL_BLOCKS]" >&2
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

if [ ! -s "${host_log}" ]; then
    echo "Missing or empty completed Host log: ${host_log}" >&2
    exit 66
fi
if [ ! -d "${build_dir}" ]; then
    echo "Missing HW-Emu build directory: ${build_dir}" >&2
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
exit_status="$(metadata exit_status)"

for pair in \
    "profile:${profile}" \
    "prompt_tokens:${prompt_tokens}" \
    "generated_tokens:${generated_tokens}" \
    "layers:${layers}" \
    "block_size:${block_size}" \
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
if ! rg -q '^finished_at=' "${host_log}" ||
   ! rg -q '^QWEN_8X64_E2E_PROFILE .* PASS$' "${host_log}" ||
   ! rg -q '^QWEN_8X64_HOST PASS ' "${host_log}"; then
    echo "Host log does not contain complete E2E PASS evidence" >&2
    exit 65
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

{
    printf 'field\tvalue\n'
    printf 'profile\t%s\n' "${profile}"
    printf 'prompt_tokens\t%s\n' "${prompt_tokens}"
    printf 'generated_tokens\t%s\n' "${generated_tokens}"
    printf 'layers\t%s\n' "${layers}"
    printf 'block_size\t%s\n' "${block_size}"
    printf 'target_mhz\t%s\n' "${target_mhz}"
    printf 'xsim_mhz\t%s\n' "${xsim_mhz}"
    printf 'release_nonfinal_blocks\t%s\n' "${release_nonfinal}"
    printf 'host_exe_sha256\t%s\n' "${host_sha}"
    printf 'xclbin_sha256\t%s\n' "${xclbin_sha}"
    printf 'emconfig_sha256\t%s\n' "${emconfig_sha}"
    printf 'source_host_log\t%s\n' "${host_log}"
    printf 'source_profile_csv\t%s\n' "${profile_csv}"
} > "${temp_dir}/manifest.tsv"

(
    cd "${temp_dir}"
    sha256sum \
        host.raw.log host.evidence.log profile_kernels.csv \
        performance.tsv manifest.tsv \
        > checksums.sha256
)
mv "${temp_dir}" "${output_dir}"
trap - EXIT

printf 'archive=%s\nperformance=%s\nchecksums=%s\n' \
    "${output_dir}" \
    "${output_dir}/performance.tsv" \
    "${output_dir}/checksums.sha256"
