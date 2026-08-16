#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

prefix="${1:-}"
interval_seconds="${2:-3600}"
if [ -z "${prefix}" ]; then
    echo "usage: $0 PREFIX [INTERVAL_SECONDS]" >&2
    exit 2
fi
if ! [[ "${interval_seconds}" =~ ^[1-9][0-9]*$ ]]; then
    echo "INTERVAL_SECONDS must be a positive integer" >&2
    exit 2
fi

mkdir -p logs
status_log="logs/${prefix}_watch.log"
performance_tsv="logs/${prefix}_performance.tsv"
precision_tsv="logs/${prefix}_precision.tsv"

list_sessions() {
    tmux list-sessions -F '#{session_name}' 2>/dev/null \
        | grep -E "^${prefix}_(p|d)(64|256|512|1024)$" || true
}

while :; do
    sessions="$(list_sessions)"
    count="$(printf '%s\n' "${sessions}" | sed '/^$/d' | wc -l)"
    printf 'checked_at=%s active_sessions=%s\n' "$(date -Is)" "${count}" \
        | tee -a "${status_log}"
    if [ "${count}" -eq 0 ]; then
        break
    fi
    printf '%s\n' "${sessions}" | sed '/^$/d;s/^/  /' | tee -a "${status_log}"
    sleep "${interval_seconds}"
done

mapfile -t result_files < <(
    sed -n 's/^results_tsv=\([^ ]*\).*/\1/p' logs/${prefix}_{p,d}{64,256,512,1024}.log \
        | sort -u
)

if [ "${#result_files[@]}" -ne 8 ]; then
    echo "Expected 8 result files, found ${#result_files[@]}" | tee -a "${status_log}" >&2
    exit 66
fi
for result in "${result_files[@]}"; do
    if [ ! -s "${result}" ]; then
        echo "Missing result file: ${result}" | tee -a "${status_log}" >&2
        exit 66
    fi
done

scripts/report_vitis_8x64_pd_sweep.sh "${result_files[@]}" \
    | tee "${performance_tsv}"
scripts/report_vitis_8x64_pd_precision.sh "${result_files[@]}" \
    | tee "${precision_tsv}"

failed="$(awk -F '\t' 'FNR > 1 && $6 != 0 { n++ } END { print n + 0 }' "${result_files[@]}")"
printf 'completed_at=%s results=%s failed_cases=%s performance=%s precision=%s\n' \
    "$(date -Is)" "${#result_files[@]}" "${failed}" "${performance_tsv}" "${precision_tsv}" \
    | tee -a "${status_log}"
exit "$((failed != 0))"
