#!/usr/bin/env bash
set -euo pipefail

script_path="$(realpath "$0")"
repo_root="$(dirname "$(dirname "${script_path}")")"
cd "${repo_root}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Root checksum generation requires a Git worktree" >&2
    exit 69
fi

unstaged="$({ git diff --name-only; } | awk '$0 != "CHECKSUMS.sha256"')"
if [ -n "${unstaged}" ]; then
    echo "Stage every release file before regenerating root checksums:" >&2
    printf '%s\n' "${unstaged}" >&2
    exit 65
fi

untracked="$(git ls-files --others --exclude-standard)"
if [ -n "${untracked}" ]; then
    echo "Stage or remove every release-candidate file first:" >&2
    printf '%s\n' "${untracked}" >&2
    exit 65
fi

# Tool-generated raw logs and profiler CSV files are immutable experiment
# evidence.  Vitis/XSim emit trailing spaces and a final blank CSV row, so do
# not rewrite those bytes merely to satisfy Git's whitespace heuristic.  They
# remain covered by both package and root SHA-256 manifests.  All source,
# scripts, prose, and derived tables retain the strict whitespace gate.
if ! git diff --cached --check -- . \
    ':(exclude)results/**/build.raw.log' \
    ':(exclude)results/**/host.raw.log' \
    ':(exclude)results/**/host.evidence.log' \
    ':(exclude)results/**/profile_kernels.csv' \
    ':(exclude)results/**/raw/*.log'
then
    echo "The staged release candidate fails git diff --check" >&2
    exit 65
fi

tmp_manifest="$(mktemp "${repo_root}/.CHECKSUMS.sha256.XXXXXX")"
trap 'rm -f "${tmp_manifest}"' EXIT

while IFS= read -r -d '' path; do
    if [ "${path}" = "CHECKSUMS.sha256" ]; then
        continue
    fi
    if [ ! -f "${path}" ]; then
        echo "Git-indexed release file is missing from the worktree: ${path}" >&2
        exit 66
    fi
    sha256sum -- "${path}" >> "${tmp_manifest}"
done < <(git ls-files --cached -z | sort -z)

if [ ! -s "${tmp_manifest}" ]; then
    echo "Refusing to install an empty root checksum manifest" >&2
    exit 65
fi

mv "${tmp_manifest}" CHECKSUMS.sha256
trap - EXIT

sha256sum -c CHECKSUMS.sha256 >/dev/null
printf 'ROOT CHECKSUM MANIFEST PASS files=%s\n' \
    "$(wc -l < CHECKSUMS.sha256)"
printf 'Run: git add CHECKSUMS.sha256 && '
printf 'VERIFY_ROOT_CHECKSUMS=1 make test_publication_tree\n'
