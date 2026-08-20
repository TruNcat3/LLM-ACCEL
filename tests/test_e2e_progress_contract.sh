#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")/.."

log_file="$(mktemp "${TMPDIR:-/tmp}/e2e-progress-contract.XXXXXX.log")"
bad_log="$(mktemp "${TMPDIR:-/tmp}/e2e-progress-contract-bad.XXXXXX.log")"
cleanup() {
    rm -f -- "${log_file}" "${bad_log}"
}
trap cleanup EXIT

emit_group() {
    local total="$1"
    local layers="$2"
    local position="$3"
    local rows="$4"
    local completed op phase layer
    for ((completed = 1; completed <= total; completed++)); do
        if [ "${completed}" -le $((2 * layers)) ]; then
            if [ $((completed % 2)) -eq 1 ]; then
                op=18
                phase=attention
            else
                op=19
                phase=ffn
            fi
            layer=$(((completed - 1) / 2))
        else
            op=20
            phase=final_norm
            layer=0
        fi
        printf 'COARSE_TASK_PROGRESS completed=%s total=%s op=%s phase=%s layer=%s position=%s query_tokens=%s controller_ms=1\n' \
            "${completed}" "${total}" "${op}" "${phase}" "${layer}" \
            "${position}" "${rows}" >> "${log_file}"
    done
}

# One P8 block plus one real D1 forward through two layers: 5 + 5 tasks.
emit_group 5 2 0 8
emit_group 5 2 8 1
scripts/verify_vitis_8x64_e2e_progress.sh \
    "${log_file}" 8 2 2 8 >/dev/null

# Two prompt blocks, including a three-row tail. The non-final block omits
# Task 20; the tail block owns the sole final normalization.
: > "${log_file}"
emit_group 4 2 0 8
emit_group 5 2 8 3
scripts/verify_vitis_8x64_e2e_progress.sh \
    "${log_file}" 11 1 2 8 >/dev/null

# A wrong tail position must be rejected.
sed 's/position=8 query_tokens=3/position=9 query_tokens=3/' \
    "${log_file}" > "${bad_log}"
if scripts/verify_vitis_8x64_e2e_progress.sh \
    "${bad_log}" 11 1 2 8 >/dev/null 2>&1
then
    echo "E2E progress verifier accepted a malformed tail block" >&2
    exit 1
fi

echo "E2E PROGRESS CONTRACT TEST PASS P8G2L2_tasks=10 P11G1L2_tasks=9"
