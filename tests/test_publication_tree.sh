#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

required_files=(
    README.md
    CITATION.cff
    LICENSE
    LICENSES/CC-BY-NC-4.0.md
    docs/README.md
    docs/environment.md
    docs/architecture.md
    docs/design-space.md
    docs/coarse-task-runtime.md
    docs/usage.md
    docs/experiments.md
    results/README.md
    docs/assets/results-overview.svg
    docs/assets/e2e-scaling.svg
    docs/assets/pd-efficiency.svg
    docs/assets/resource-utilization.svg
)
for path in "${required_files[@]}"; do
    if [ ! -s "${path}" ]; then
        echo "Missing publication document: ${path}" >&2
        exit 66
    fi
done

required_executables=(
    scripts/check_environment.sh
    scripts/setup_environment.sh
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
    tests/test_environment_contract.sh
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
    '## Implementation variants and evidence map' \
    '## Key results' \
    '## Reproduce the core validation' \
    '## Citation' \
    '## License'
do
    if ! rg -F -q "${heading}" README.md; then
        echo "README is missing required section: ${heading}" >&2
        exit 65
    fi
done

for design_id in R1 D1 P1 S1; do
    if ! rg -q "\\*\\*${design_id} —" README.md; then
        echo "README is missing implementation ID: ${design_id}" >&2
        exit 65
    fi
done

if ! rg -F -q 'Selected R1 results · current mainline overview' docs/assets/results-overview.svg ||
   ! rg -F -q '[R1] P8/G2' docs/assets/e2e-scaling.svg ||
   ! rg -F -q '[D1] Query-block' docs/assets/pd-efficiency.svg ||
   ! rg -F -q '[R1] Whole-system' docs/assets/resource-utilization.svg; then
    echo "Root figures are not mapped to implementation IDs" >&2
    exit 65
fi

readme_lines="$(wc -l < README.md)"
if [ "${readme_lines}" -gt 260 ]; then
    echo "Root README exceeded the concise publication budget: ${readme_lines} lines" >&2
    exit 65
fi

for figure in \
    docs/assets/results-overview.svg \
    docs/assets/e2e-scaling.svg \
    docs/assets/pd-efficiency.svg \
    docs/assets/resource-utilization.svg
do
    if ! rg -q '<title id="title">.+</title>' "${figure}" ||
       ! rg -q '<desc id="desc">.+</desc>' "${figure}"; then
        echo "Publication figure lacks accessible title/description: ${figure}" >&2
        exit 65
    fi
done

if [ "$(rg -c '^!\[' README.md)" -ne 1 ] ||
   ! rg -F -q '](docs/assets/results-overview.svg)' README.md; then
    echo "Root README must present exactly one consolidated result figure" >&2
    exit 65
fi

if [ "$(rg -o '>R1<' docs/assets/results-overview.svg | wc -l)" -ne 4 ] ||
   rg -q '>D1<' docs/assets/results-overview.svg; then
    echo "Consolidated root figure must contain exactly four R1 panels" >&2
    exit 65
fi

for metric in '119.652' '58.424%' '189.285' '92.424%' '1.948x' '+2.67%' '+1.520 pp' '80.0%'; do
    if ! rg -F -q "${metric}" docs/assets/results-overview.svg; then
        echo "Consolidated result figure is missing released metric: ${metric}" >&2
        exit 65
    fi
done

if ! rg -F -q '@software{wang2026llmaccel,' README.md; then
    echo "README is missing the canonical BibTeX tag" >&2
    exit 65
fi

for field in 'cff-version:' 'message:' 'title:' 'authors:' 'version:' 'date-released:' 'license:'; do
    if ! rg -q "^${field}" CITATION.cff; then
        echo "CITATION.cff is missing field: ${field}" >&2
        exit 65
    fi
done

for identity in \
    'given-names: "Teng"' \
    'family-names: "Wang"' \
    'High Efficient Intelligent Computing Lab, Suzhou Institute for Advanced Research of USTC, Suzhou, China'
do
    if ! rg -F -q "${identity}" CITATION.cff; then
        echo "CITATION.cff is missing author identity: ${identity}" >&2
        exit 65
    fi
done

if ! rg -F -q 'PolyForm Noncommercial License 1.0.0' LICENSE ||
   ! rg -F -q 'CC BY-NC 4.0' LICENSES/CC-BY-NC-4.0.md; then
    echo "Noncommercial license metadata is incomplete" >&2
    exit 65
fi

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
