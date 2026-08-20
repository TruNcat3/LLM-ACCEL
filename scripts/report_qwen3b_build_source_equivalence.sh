#!/usr/bin/env bash
set -euo pipefail

script_path="$(realpath "$0")"
repo_root="$(dirname "$(dirname "${script_path}")")"
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 BUILD_LOG [RELEASE_SOURCE_ROOT]" >&2
    exit 2
fi

build_log="$(realpath "$1")"
release_root="${2:-${repo_root}}"
if [ ! -s "${build_log}" ]; then
    echo "Missing or empty Qwen2.5-3B build log: ${build_log}" >&2
    exit 66
fi
if [ ! -d "${release_root}" ]; then
    echo "Missing release source root: ${release_root}" >&2
    exit 66
fi
release_root="$(realpath "${release_root}")"

# Vitis HLS records the source working directory before reading any design
# file.  This is the authoritative source root of the already-completed build,
# rather than an assumed sibling-repository path.
build_root="$(
    awk -F "'" '
        /INFO: \[HLS [^]]*\] In directory '\''/ {
            print $2
            exit
        }
    ' "${build_log}"
)"
if [ -z "${build_root}" ] || [ ! -d "${build_root}" ]; then
    echo "Cannot recover the HLS build source root from the build log" >&2
    exit 65
fi
build_root="$(realpath "${build_root}")"

snapshot="$(mktemp "${TMPDIR:-/tmp}/qwen3b-release-source.XXXXXX.tsv")"
cleanup() {
    rm -f "${snapshot}"
}
trap cleanup EXIT
"${release_root}/scripts/report_qwen3b_source_snapshot.sh" \
    "${release_root}" > "${snapshot}"

printf 'path\trole\tbuild_source_sha256\trelease_source_sha256\tresult\n'
matched=0
while IFS=$'\t' read -r path release_sha bytes role; do
    [ "${path}" = "path" ] && continue
    case "${role}" in
        build_input|build_harness) ;;
        *) continue ;;
    esac
    build_path="${build_root}/${path}"
    if [ ! -s "${build_path}" ]; then
        echo "Build source is missing equivalence input: ${path}" >&2
        exit 66
    fi
    build_sha="$(sha256sum "${build_path}" | awk '{ print $1 }')"
    if [ "${build_sha}" != "${release_sha}" ]; then
        echo "Build/release source mismatch: ${path}" >&2
        exit 65
    fi
    printf '%s\t%s\t%s\t%s\tMATCH\n' \
        "${path}" "${role}" "${build_sha}" "${release_sha}"
    matched=$((matched + 1))
done < "${snapshot}"

if [ "${matched}" -lt 30 ]; then
    echo "Build/release equivalence covered too few inputs: ${matched}" >&2
    exit 65
fi
