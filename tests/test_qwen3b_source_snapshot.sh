#!/usr/bin/env bash
set -euo pipefail

script_path="$(realpath "$0")"
repo_root="$(dirname "$(dirname "${script_path}")")"
cd "${repo_root}"

first="$(mktemp "${TMPDIR:-/tmp}/qwen3b-source-snapshot-a.XXXXXX.tsv")"
second="$(mktemp "${TMPDIR:-/tmp}/qwen3b-source-snapshot-b.XXXXXX.tsv")"
equivalence_log="$(mktemp "${TMPDIR:-/tmp}/qwen3b-source-equivalence.XXXXXX.log")"
equivalence_report="$(mktemp "${TMPDIR:-/tmp}/qwen3b-source-equivalence.XXXXXX.tsv")"
cleanup() {
    rm -f "${first}" "${second}" "${equivalence_log}" \
        "${equivalence_report}"
}
trap cleanup EXIT

scripts/report_qwen3b_source_snapshot.sh > "${first}"
scripts/report_qwen3b_source_snapshot.sh > "${second}"
if ! cmp -s "${first}" "${second}"; then
    echo "Qwen2.5-3B source snapshot is not deterministic" >&2
    exit 65
fi

awk -F '\t' '
    NR == 1 {
        if ($1 != "path" || $2 != "sha256" ||
            $3 != "bytes" || $4 != "role") exit 1
        next
    }
    {
        if (NF != 4 || seen[$1]++) exit 1
        if ($1 == "" || substr($1, 1, 1) == "/" ||
            $1 ~ /(^|\/)\.\.($|\/)/) exit 1
        if (length($2) != 64 || $2 !~ /^[0-9a-f]+$/) exit 1
        if ($3 !~ /^[1-9][0-9]*$/) exit 1
        if ($4 != "build_input" && $4 != "build_harness" &&
            $4 != "release_entrypoint" &&
            $4 != "validation_harness" && $4 != "validation_test") exit 1
        role_count[$4]++
        count++
    }
    END {
        if (count < 40 || role_count["build_input"] < 29 ||
            role_count["build_harness"] < 3 ||
            role_count["release_entrypoint"] < 1 ||
            role_count["validation_harness"] < 8 ||
            role_count["validation_test"] < 2) exit 1
    }
' "${first}" || {
    echo "Malformed or incomplete Qwen2.5-3B source snapshot" >&2
    exit 65
}

while IFS=$'\t' read -r path expected_sha expected_bytes role; do
    if [ "${path}" = "path" ]; then
        continue
    fi
    if [ ! -s "${path}" ]; then
        echo "Source snapshot path is missing: ${path}" >&2
        exit 66
    fi
    actual_sha="$(sha256sum "${path}" | awk '{ print $1 }')"
    actual_bytes="$(stat -c '%s' "${path}")"
    if [ "${actual_sha}" != "${expected_sha}" ] ||
       [ "${actual_bytes}" != "${expected_bytes}" ]; then
        echo "Source snapshot identity mismatch: ${path}" >&2
        exit 65
    fi
    if [ -z "${role}" ]; then
        echo "Source snapshot role is empty: ${path}" >&2
        exit 65
    fi
done < "${first}"

# Prove that the manually curated build-input list is closed over every local
# quoted include.  Without this check, adding a project header to a kernel or
# Host source could leave the release manifest internally consistent but
# incomplete.
declare -A snapshotted_paths=()
declare -A visited_sources=()
source_queue=()
while IFS=$'\t' read -r path expected_sha expected_bytes role; do
    if [ "${path}" = "path" ]; then
        continue
    fi
    snapshotted_paths["${path}"]=1
    if [ "${role}" = "build_input" ] &&
       [[ "${path}" =~ \.(c|cc|cpp|cxx|h|hh|hpp|hxx)$ ]]; then
        source_queue+=("${path}")
    fi
done < "${first}"

queue_index=0
while [ "${queue_index}" -lt "${#source_queue[@]}" ]; do
    source_path="${source_queue[$queue_index]}"
    queue_index=$((queue_index + 1))
    if [ -n "${visited_sources[${source_path}]:-}" ]; then
        continue
    fi
    visited_sources["${source_path}"]=1

    while IFS= read -r include_name; do
        [ -n "${include_name}" ] || continue
        resolved=""
        for candidate in \
            "$(dirname "${source_path}")/${include_name}" \
            "include/${include_name}" \
            "common/include/${include_name}" \
            "${include_name}"
        do
            if [ -f "${candidate}" ]; then
                resolved="$(realpath --relative-to=. "${candidate}")"
                break
            fi
        done
        if [ -z "${resolved}" ]; then
            echo "Unresolved local quoted include: ${source_path} -> ${include_name}" >&2
            exit 66
        fi
        if [ -z "${snapshotted_paths[${resolved}]:-}" ]; then
            echo "Source snapshot omits local include: ${source_path} -> ${resolved}" >&2
            exit 65
        fi
        if [[ "${resolved}" =~ \.(c|cc|cpp|cxx|h|hh|hpp|hxx)$ ]]; then
            source_queue+=("${resolved}")
        fi
    done < <(
        sed -n \
            's/^[[:space:]]*#[[:space:]]*include[[:space:]]*"\([^"]*\)".*/\1/p' \
            "${source_path}"
    )
done

printf "INFO: [HLS 200-10] In directory '%s'\n" "${repo_root}" \
    > "${equivalence_log}"
scripts/report_qwen3b_build_source_equivalence.sh \
    "${equivalence_log}" "${repo_root}" > "${equivalence_report}"
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
' "${equivalence_report}" || {
    echo "Self-contained build/release source equivalence test failed" >&2
    exit 65
}

# The build harness passes its private output root through the historical
# VITIS_8X64_BUILD_DIR name.  The publication Makefile must honor that alias;
# otherwise a clean checkout silently builds the matching Host elsewhere.
entrypoint_build_dir="/tmp/llm_accel_release_entrypoint_contract"
entrypoint_dry_run="$(
    make -n -B vitis_8x64_qwen_host \
        TARGET=hw_emu \
        DEVICE=contract_test_device \
        VITIS_8X64_MODEL_PROFILE=qwen2.5-3b \
        VITIS_8X64_BUILD_DIR="${entrypoint_build_dir}" \
        2>/dev/null
)"
if [[ "${entrypoint_dry_run}" != *"mkdir -p ${entrypoint_build_dir}"* ]] ||
   [[ "${entrypoint_dry_run}" != *"-o ${entrypoint_build_dir}/host_qwen_8x64.exe"* ]]; then
    echo "Publication Makefile ignored VITIS_8X64_BUILD_DIR" >&2
    exit 65
fi

echo "QWEN3B SOURCE SNAPSHOT PASS files=$(($(wc -l < "${first}") - 1)) local_include_closure=${#visited_sources[@]} release_entrypoint_alias=PASS scope=release_candidate_worktree"
