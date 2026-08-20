#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 RESULTS_TSV [RESULTS_TSV ...]" >&2
    exit 2
fi

printf 'phase\tcontext_len\tactive_query_rows_per_block\tq214_checks\tq214_failures\tq214_max_raw_error\treference_checks\treference_failures\treference_max_raw_error\n'

for results_tsv in "$@"; do
    if [ ! -s "${results_tsv}" ]; then
        echo "Missing or empty results TSV: ${results_tsv}" >&2
        exit 66
    fi

    while IFS=$'\t' read -r phase context _start _end tokens status _active _cycles host_log _profile; do
        if [ "${phase}" = "phase" ]; then
            continue
        fi
        if [ "${status}" != "0" ] || [ ! -s "${host_log}" ]; then
            printf '%s\t%s\t%s\tNA\tNA\tNA\tNA\tNA\tNA\n' \
                "${phase}" "${context}" "${tokens}"
            continue
        fi

        awk -v phase="${phase}" -v context="${context}" -v tokens="${tokens}" '
        BEGIN {
            qchecks=0; qfail=0; qmax=0
            rchecks=0; rfail=0; rmax=0
        }
        $2 ~ /_port[01]_q2_14_hw$/ && ($3 == "PASS" || $3 == "FAIL") {
            qchecks++
            if ($3 == "FAIL") qfail++
            for (i=4; i<=NF; i++) {
                if ($i ~ /^max_raw_error=/) {
                    split($i, a, "=")
                    if ((a[2] + 0) > qmax) qmax=a[2] + 0
                }
            }
        }
        $2 ~ /_port[01]_reference$/ && ($3 == "PASS" || $3 == "FAIL") {
            rchecks++
            if ($3 == "FAIL") rfail++
            for (i=4; i<=NF; i++) {
                if ($i ~ /^max_raw_error=/) {
                    split($i, a, "=")
                    if ((a[2] + 0) > rmax) rmax=a[2] + 0
                }
            }
        }
        END {
            printf "%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n", \
                phase, context, tokens, qchecks, qfail, qmax, \
                rchecks, rfail, rmax
        }
        ' "${host_log}"
    done < "${results_tsv}"
done
