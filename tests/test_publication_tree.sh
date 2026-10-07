#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

required_files=(
    README.md
    CITATION.cff
    LICENSE
    LICENSES/CC-BY-NC-4.0.md
    docs/README.md
    docs/implementations.md
    docs/repository-map.md
    docs/release-workflow.md
    docs/environment.md
    docs/architecture.md
    docs/design-space.md
    docs/coarse-task-runtime.md
    docs/coarse-task-runtime-history.md
    docs/usage.md
    docs/reproduction-resident.md
    docs/reproduction-diagnostics.md
    docs/experiments.md
    docs/experiment-details.md
    cases/README.md
    cases/streaming-split/README.md
    cases/quantized-block/README.md
    results/README.md
    docs/assets/results-overview.svg
    docs/assets/e2e-scaling.svg
    docs/assets/pd-efficiency.svg
    docs/assets/resource-utilization.svg
    include/host_coarse_task_program.hpp
    tests/coarse_task_program_tb.cpp
    designs/catalog.json
    scripts/cowave.py
    docs/designs/fix16.md
    docs/designs/quantized.md
    docs/designs/streaming-split.md
    cases/quantized-layer/README.md
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
    tests/test_checkpoint_tolerance_contract.sh
)
for path in "${required_executables[@]}"; do
    if [ ! -x "${path}" ]; then
        echo "Missing or non-executable publication tool: ${path}" >&2
        exit 66
    fi
done

for heading in \
    '## Architecture' \
    '## Choose a design' \
    '## Selected results' \
    '## Get started' \
    '## Citation' \
    '## License'
do
    if ! rg -F -q "${heading}" README.md; then
        echo "README is missing required section: ${heading}" >&2
        exit 65
    fi
done

for public_name in \
    'cowave-fix16-2-8-64' \
    'cowave-int4-4-8-128' \
    'cowave-int8-4-4-128'
do
    if ! rg -F -q "${public_name}" README.md ||
       ! rg -F -q "${public_name}" docs/implementations.md; then
        echo "README/catalog is missing implementation family: ${public_name}" >&2
        exit 65
    fi
done

for evidence_scope in 'Fix16 operator diagnostics' 'Small-shape protocol tests'; do
    if ! rg -F -q "**${evidence_scope}**" docs/implementations.md; then
        echo "Catalog is missing resident evidence scope: ${evidence_scope}" >&2
        exit 65
    fi
done

if ! rg -F -q 'cowave-int4-4-8-128' docs/assets/results-overview.svg ||
   ! rg -F -q 'cowave-int8-4-4-128' docs/assets/results-overview.svg ||
   ! rg -F -q 'cowave-fix16-2-8-64' docs/assets/results-overview.svg ||
   ! rg -F -q 'Fix16 resident P8/G2' docs/assets/e2e-scaling.svg ||
   ! rg -F -q 'Fix16 operator diagnostics · query-block' docs/assets/pd-efficiency.svg ||
   ! rg -F -q 'Fix16 resident · whole-system' docs/assets/resource-utilization.svg; then
    echo "Root figures are not mapped to public implementation names" >&2
    exit 65
fi

if ! rg -F -q '(docs/implementations.md)' README.md ||
   ! rg -F -q '(implementations.md)' docs/README.md ||
   ! rg -F -q '## Legacy Label Migration' docs/implementations.md; then
    echo "Canonical implementation catalog is not linked or complete" >&2
    exit 65
fi

for legacy_alias in R1 D1 P1 S1 Q1; do
    if ! rg -F -q "| \`${legacy_alias}\` |" docs/implementations.md; then
        echo "Canonical implementation catalog is missing legacy alias: ${legacy_alias}" >&2
        exit 65
    fi
done

if rg -F -q '[R1]' docs/assets/e2e-scaling.svg docs/assets/resource-utilization.svg ||
   rg -F -q '[D1]' docs/assets/pd-efficiency.svg ||
   rg -q '>R1<' docs/assets/results-overview.svg; then
    echo "Public figures still expose legacy implementation badges" >&2
    exit 65
fi

readme_lines="$(wc -l < README.md)"
if [ "${readme_lines}" -gt 160 ]; then
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

if rg -q '>R1<' docs/assets/results-overview.svg ||
   ! rg -F -q 'workload D1' docs/assets/pd-efficiency.svg; then
    echo "Figures must distinguish design names from workload D1" >&2
    exit 65
fi

for metric in '119.652' '58.424%' '189.285' '92.424%'; do
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
    'email: "wangt635@ustc.edu.cn"' \
    'High Efficient Intelligent Computing Lab, Suzhou Institute for Advanced Research of USTC, Suzhou, China'
do
    if ! rg -F -q "${identity}" CITATION.cff; then
        echo "CITATION.cff is missing author identity: ${identity}" >&2
        exit 65
    fi
done

if ! rg -F -q 'email        = {wangt635@ustc.edu.cn},' README.md; then
    echo "README BibTeX is missing the correspondence email" >&2
    exit 65
fi

if ! rg -F -q 'PolyForm Noncommercial License 1.0.0' LICENSE ||
   ! rg -F -q 'CC BY-NC 4.0' LICENSES/CC-BY-NC-4.0.md; then
    echo "Noncommercial license metadata is incomplete" >&2
    exit 65
fi

mapfile -t markdown_files < <(
    find . -type f -name '*.md' -not -path './.git/*' -print | sort
)
# Match the Han script itself. New PCRE Unicode tables include the middle-dot
# punctuation used in English navigation in Han's script extensions.
if rg -n -P '\p{sc=Han}' "${markdown_files[@]}" CITATION.cff; then
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
tests/test_checkpoint_tolerance_contract.sh >/dev/null

if [ "${VERIFY_ROOT_CHECKSUMS:-0}" = "1" ]; then
    sha256sum -c CHECKSUMS.sha256 >/dev/null
fi

git diff --check

printf 'PUBLICATION TREE PASS markdown=%d result_archives=%d root_checksums=%s\n' \
    "${#markdown_files[@]}" \
    "${#result_archives[@]}" \
    "${VERIFY_ROOT_CHECKSUMS:-0}"
