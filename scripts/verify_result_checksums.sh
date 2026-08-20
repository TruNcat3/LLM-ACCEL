#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then
    checksum_manifests=("$@")
else
    checksum_manifests=(results/*/checksums.sha256)
fi

archive_count=0
for checksum in "${checksum_manifests[@]}"; do
    if [ ! -s "${checksum}" ]; then
        echo "Missing or empty result checksum manifest: ${checksum}" >&2
        exit 66
    fi
    archive_dir="$(dirname "${checksum}")"
    first_path="$(awk 'NF >= 2 { print $2; exit }' "${checksum}")"
    if [[ "${first_path}" == results/* ]]; then
        sha256sum -c "${checksum}" >/dev/null
        checksum_scope=repository_root_relative
        archive_prefix="$(realpath --relative-to="$PWD" "${archive_dir}")/"
    else
        (
            cd "${archive_dir}"
            sha256sum -c checksums.sha256 >/dev/null
        )
        checksum_scope=archive_relative
        archive_prefix=""
    fi

    declare -A listed_paths=()
    listed_count=0
    while read -r digest path; do
        if [ -z "${digest}" ] || [ -z "${path:-}" ]; then
            echo "Malformed checksum row in ${checksum}" >&2
            exit 65
        fi
        path="${path#\*}"
        if [ "${checksum_scope}" = repository_root_relative ]; then
            if [[ "${path}" != "${archive_prefix}"* ]]; then
                echo "Checksum path escapes result archive: ${path}" >&2
                exit 65
            fi
            normalized="${path#${archive_prefix}}"
        else
            normalized="${path#./}"
            if [[ "${normalized}" = /* ]] ||
               [[ "/${normalized}/" == *"/../"* ]]; then
                echo "Checksum path escapes result archive: ${path}" >&2
                exit 65
            fi
        fi
        if [ -z "${normalized}" ] ||
           [ "${normalized}" = checksums.sha256 ] ||
           [ -n "${listed_paths[${normalized}]:-}" ]; then
            echo "Invalid or duplicate checksum path: ${path}" >&2
            exit 65
        fi
        listed_paths["${normalized}"]=1
        listed_count=$((listed_count + 1))
    done < "${checksum}"

    actual_count=0
    while IFS= read -r -d '' path; do
        if [ -z "${listed_paths[${path}]:-}" ]; then
            echo "Result file is missing from checksum manifest: ${archive_dir}/${path}" >&2
            exit 65
        fi
        actual_count=$((actual_count + 1))
    done < <(
        find "${archive_dir}" -type f ! -name checksums.sha256 -printf '%P\0' |
            sort -z
    )
    if [ "${listed_count}" -ne "${actual_count}" ]; then
        echo "Checksum manifest coverage mismatch: ${checksum}" >&2
        exit 65
    fi
    archive_count=$((archive_count + 1))
    printf 'PASS\t%s\tfiles=%s\tscope=%s\n' \
        "${archive_dir}" \
        "${listed_count}" \
        "${checksum_scope}"
done

if [ "${archive_count}" -eq 0 ]; then
    echo "No result checksum manifests found" >&2
    exit 66
fi
