#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

required_files=(
    README.md
    CITATION.cff
    docs/architecture.md
    docs/design-space.md
    docs/coarse-task-runtime.md
    docs/usage.md
    docs/experiments.md
    results/README.md
)
for path in "${required_files[@]}"; do
    if [ ! -s "${path}" ]; then
        echo "Missing publication document: ${path}" >&2
        exit 66
    fi
done

required_executables=(
    scripts/archive_vitis_8x64_e2e_run.sh
    scripts/install_vitis_8x64_e2e_result.sh
    scripts/regenerate_root_checksums.sh
    scripts/render_vitis_8x64_e2e_result_markdown.sh
    scripts/report_qwen3b_build_source_equivalence.sh
    scripts/report_vitis_8x64_e2e_trace.sh
    scripts/run_vitis_8x64_qwen3b_e2e_hwemu_tmux.sh
    scripts/verify_q214_pd_release.sh
    scripts/verify_qwen3b_e2e_release.sh
    scripts/verify_vitis_8x64_e2e_progress.sh
    scripts/verify_result_checksums.sh
)
for path in "${required_executables[@]}"; do
    if [ ! -x "${path}" ]; then
        echo "Missing or non-executable publication tool: ${path}" >&2
        exit 66
    fi
done

for heading in \
    '## Research contributions' \
    '## Architecture at a glance' \
    '## Key results' \
    '## Reproduce the core validation' \
    '## Citation'
do
    if ! rg -F -q "${heading}" README.md; then
        echo "README is missing required section: ${heading}" >&2
        exit 65
    fi
done

for field in 'cff-version:' 'message:' 'title:' 'authors:' 'version:' 'date-released:'; do
    if ! rg -q "^${field}" CITATION.cff; then
        echo "CITATION.cff is missing field: ${field}" >&2
        exit 65
    fi
done

mapfile -t markdown_files < <(
    find . -type f -name '*.md' -not -path './.git/*' -print | sort
)
if rg -n -P '\p{Han}' "${markdown_files[@]}" CITATION.cff; then
    echo "Publication prose must remain English" >&2
    exit 65
fi

for path in "${markdown_files[@]}"; do
    fence_count="$(awk '/^```/ { count++ } END { print count + 0 }' "${path}")"
    if [ $((fence_count % 2)) -ne 0 ]; then
        echo "Unbalanced Markdown code fences: ${path}" >&2
        exit 65
    fi
done

link_failure=0
while IFS=$'\t' read -r source target; do
    case "${target}" in
        http://*|https://*|mailto:*|\#*) continue ;;
    esac
    target="${target%%#*}"
    target="${target#<}"
    target="${target%>}"
    if [ -z "${target}" ]; then
        continue
    fi
    path="$(dirname "${source}")/${target}"
    if [ ! -e "${path}" ]; then
        printf 'Broken local link: %s -> %s\n' "${source}" "${target}" >&2
        link_failure=1
    fi
done < <(
    perl -ne \
        'while (/\[[^]]*\]\(([^)]+)\)/g) { print "$ARGV\t$1\n" }' \
        "${markdown_files[@]}"
)
if [ "${link_failure}" -ne 0 ]; then
    exit 65
fi

if rg -n '^(<<<<<<<|=======|>>>>>>>)' \
    README.md CITATION.cff docs Makefile host include kernel scripts tcl tests
then
    echo "Publication tree contains a merge-conflict marker" >&2
    exit 65
fi

mapfile -t result_archives < <(
    find results -mindepth 1 -maxdepth 1 -type d -print | sort
)
for archive in "${result_archives[@]}"; do
    if [ ! -s "${archive}/README.md" ] ||
       [ ! -s "${archive}/checksums.sha256" ]; then
        echo "Result archive lacks README or checksums: ${archive}" >&2
        exit 66
    fi
    archive_name="$(basename "${archive}")"
    if ! rg -F -q "(${archive_name}/)" results/README.md; then
        echo "Result archive is absent from the evidence index: ${archive}" >&2
        exit 65
    fi
done

scripts/verify_result_checksums.sh >/dev/null
tests/test_e2e_performance_semantics.sh >/dev/null

if [ "${VERIFY_ROOT_CHECKSUMS:-0}" = "1" ]; then
    sha256sum -c CHECKSUMS.sha256 >/dev/null
fi

git diff --check

printf 'PUBLICATION TREE PASS markdown=%d result_archives=%d root_checksums=%s\n' \
    "${#markdown_files[@]}" \
    "${#result_archives[@]}" \
    "${VERIFY_ROOT_CHECKSUMS:-0}"
