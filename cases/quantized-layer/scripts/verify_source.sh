#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"

required_dirs=(include kernel host common tests tcl scripts config provenance)
for directory in "${required_dirs[@]}"; do
    [[ -d "${directory}" ]] || {
        echo "Missing release directory: ${directory}" >&2
        exit 66
    }
done

required_files=(
    Makefile
    scripts/cowave_quantized_layer.sh
    scripts/run_native_regression.sh
    provenance/source.sha256
    provenance/source_origins.tsv
    provenance/verification.tsv
)
for file in "${required_files[@]}"; do
    [[ -s "${file}" ]] || {
        echo "Missing release file: ${file}" >&2
        exit 66
    }
done

sha256sum -c provenance/source.sha256 >/dev/null

public_paths=(include kernel host common tests tcl scripts config Makefile README.md)
forbidden_paths=(
    "/home""/hepc"
    "/home""/wt"
    "/tools""/Xilinx"
    "/opt""/xilinx"
)
for forbidden in "${forbidden_paths[@]}"; do
    if rg -n --fixed-strings --hidden "${forbidden}" "${public_paths[@]}"; then
        echo "Forbidden machine-local path found in public release: ${forbidden}" >&2
        exit 65
    fi
done

for script in scripts/*.sh; do
    bash -n "${script}"
done

while IFS= read -r python_file; do
    python3 - "${python_file}" <<'PY'
import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY
done < <(rg --files scripts | rg '\.py$' | LC_ALL=C sort)

for precision in w4 w8; do
    alignment=()
    if [[ "${precision}" == w8 ]]; then
        alignment=(QUANTIZED_ALIGNMENT_W8=1)
    fi
    env \
        QUANTIZED_LAYER_PROFILE=integrated \
        QUANTIZED_LAYER_SILU_LANES=4 \
        QUANTIZED_LAYER_RMS_LANES=2 \
        QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING=32 \
        QUANTIZED_LAYER_ATTENTION_WAVE=0 \
        QUANTIZED_LAYER_PREFILL_FFN_OVERLAP=1 \
        "${alignment[@]}" \
        bash -c 'set -euo pipefail
            source scripts/quantized_layer_profiles.sh
            quantized_layer_profile_apply "$1"
            [[ "${QUANTIZED_LAYER_PROFILE}" == integrated ]]
            [[ "${CU_NL_LANES_CONFIG}" == 4 ]]
            [[ "${CU_RMS_LANES_CONFIG}" == 2 ]]
            [[ "${QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING}" == 32 ]]
            [[ "${QUANTIZED_PREFILL_FFN_OVERLAP}" == 1 ]]' \
        _ "${precision}"
done

echo "quantized-layer source check: PASS"
