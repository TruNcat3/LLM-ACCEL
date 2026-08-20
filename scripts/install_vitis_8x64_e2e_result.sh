#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 SOURCE_ARCHIVE RESULT_NAME" >&2
    exit 2
fi

script_path="$(realpath "$0")"
repo_root="$(dirname "$(dirname "${script_path}")")"
source_archive="$(realpath "$1")"
result_name="$2"
results_root="${LLM_ACCEL_RESULTS_ROOT:-${repo_root}/results}"

if ! [[ "${result_name}" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
    echo "RESULT_NAME must be one lowercase path component" >&2
    exit 2
fi
if [ ! -d "${source_archive}" ] ||
   [ ! -s "${source_archive}/checksums.sha256" ]; then
    echo "Missing source E2E archive or checksum manifest: ${source_archive}" >&2
    exit 66
fi

mkdir -p "${results_root}"
results_root="$(realpath "${results_root}")"
destination="${results_root}/${result_name}"
if [ -e "${destination}" ]; then
    echo "Refusing to overwrite result package: ${destination}" >&2
    exit 73
fi

"${repo_root}/scripts/verify_result_checksums.sh" \
    "${source_archive}/checksums.sha256" >/dev/null

temp_dir="$(mktemp -d "${results_root}/.${result_name}.install.XXXXXX")"
readme_temp="$(mktemp "${results_root}/.${result_name}.README.XXXXXX")"
cleanup() {
    rm -rf -- "${temp_dir}"
    rm -f -- "${readme_temp}"
}
trap cleanup EXIT

cp -a -- "${source_archive}/." "${temp_dir}/"
"${repo_root}/scripts/render_vitis_8x64_e2e_result_markdown.sh" \
    "${temp_dir}" > "${readme_temp}"
mv -- "${readme_temp}" "${temp_dir}/README.md"

checksum_temp="$(mktemp "${temp_dir}/.checksums.XXXXXX")"
(
    cd "${temp_dir}"
    while IFS= read -r -d '' path; do
        sha256sum -- "${path}"
    done < <(
        find . -type f \
            ! -name checksums.sha256 \
            ! -name "$(basename "${checksum_temp}")" \
            -printf '%P\0' |
            sort -z
    )
) > "${checksum_temp}"
mv -- "${checksum_temp}" "${temp_dir}/checksums.sha256"

"${repo_root}/scripts/verify_result_checksums.sh" \
    "${temp_dir}/checksums.sha256" >/dev/null
mv -- "${temp_dir}" "${destination}"
trap - EXIT

printf 'result_package=%s\nreadme=%s\nchecksums=%s\n' \
    "${destination}" \
    "${destination}/README.md" \
    "${destination}/checksums.sha256"
