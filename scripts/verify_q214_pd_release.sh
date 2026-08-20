#!/usr/bin/env bash
set -euo pipefail

script_path="$(realpath "$0")"
repo_root="$(dirname "$(dirname "${script_path}")")"
cd "${repo_root}"

artifact_dir="${1:-results/q214-pd-20260811}"
for relative in README.md performance.tsv precision.tsv checksums.sha256; do
    if [ ! -s "${artifact_dir}/${relative}" ]; then
        echo "Q2.14 P/D artifact is missing ${relative}" >&2
        exit 66
    fi
done
scripts/verify_result_checksums.sh \
    "${artifact_dir}/checksums.sha256" >/dev/null

results_temp="$(mktemp "${TMPDIR:-/tmp}/q214-pd-results.XXXXXX.tsv")"
performance_temp="$(mktemp "${TMPDIR:-/tmp}/q214-pd-performance.XXXXXX.tsv")"
precision_temp="$(mktemp "${TMPDIR:-/tmp}/q214-pd-precision.XXXXXX.tsv")"
cleanup() {
    rm -f "${results_temp}" "${performance_temp}" "${precision_temp}"
}
trap cleanup EXIT

printf 'phase\tcontext_len\tprefill_start\tprefill_len\ttokens\texit_status\tactive_us\tcycles_200mhz\thost_log\tprofile_csv\n' \
    > "${results_temp}"

case_count=0
for context in 64 256 512 1024; do
    for phase in p d; do
        host_log="$(realpath "${artifact_dir}/raw/${phase}${context}_host.txt")"
        profile_csv="$(realpath "${artifact_dir}/raw/${phase}${context}_profile_kernels.csv")"
        if [ ! -s "${host_log}" ] || [ ! -s "${profile_csv}" ]; then
            echo "Q2.14 P/D raw case is incomplete: ${phase}${context}" >&2
            exit 66
        fi

        if [ "${phase}" = p ]; then
            tokens=8
            start=$((context - tokens))
            end="${context}"
        else
            tokens=1
            start="${context}"
            end=$((context + 1))
        fi
        case_line="case=${phase} context_len=${context} prefill_start=${start} prefill_len=${end} tokens=${tokens}"
        if ! grep -Fqx "${case_line}" "${host_log}" ||
           ! grep -Fqx 'exit_status=0' "${host_log}" ||
           ! rg -q '^PREFILL_PROFILE .* PASS$' "${host_log}" ||
           ! rg -q '^completed_at=.* overall_status=0$' "${host_log}"; then
            echo "Q2.14 P/D Host evidence did not pass: ${phase}${context}" >&2
            exit 65
        fi

        cu_summary="$(awk -F, '
            NF == 5 &&
            ($1 == "cc8_ctrl" || $1 == "cc8_cu0" ||
             $1 == "cc8_cu1" || $1 == "cc8_status") {
                count[$1]++
                # Preserve the CSV decimal string. Converting to numeric here
                # and printing with AWK defaults can discard the third decimal
                # before the later cycle-rounding step.
                time[$1] = $2
            }
            END {
                print count["cc8_ctrl"] + 0,
                      count["cc8_cu0"] + 0,
                      count["cc8_cu1"] + 0,
                      count["cc8_status"] + 0,
                      time["cc8_ctrl"], time["cc8_cu0"],
                      time["cc8_cu1"], time["cc8_status"]
            }
        ' "${profile_csv}")"
        read -r ctrl_count cu0_count cu1_count status_count \
            ctrl_us cu0_us cu1_us status_us <<<"${cu_summary}"
        if [ "${ctrl_count}" -ne 1 ] || [ "${cu0_count}" -ne 1 ] ||
           [ "${cu1_count}" -ne 1 ] || [ "${status_count}" -ne 1 ] ||
           ! awk -v a="${ctrl_us}" -v b="${cu0_us}" \
                 -v c="${cu1_us}" -v d="${status_us}" '
                function abs(value) { return value < 0 ? -value : value }
                BEGIN {
                    exit abs(a-b) <= 0.001 && abs(a-c) <= 0.001 &&
                         abs(a-d) <= 0.001 ? 0 : 1
                }
           '; then
            echo "Q2.14 P/D profile is not one common four-CU interval: ${phase}${context}" >&2
            exit 65
        fi
        cycles="$(awk -v us="${ctrl_us}" 'BEGIN { printf "%d", us * 200 + 0.5 }')"
        printf '%s\t%s\t%s\t%s\t%s\t0\t%s\t%s\t%s\t%s\n' \
            "${phase}" "${context}" "${start}" "${end}" "${tokens}" \
            "${ctrl_us}" "${cycles}" "${host_log}" "${profile_csv}" \
            >> "${results_temp}"
        case_count=$((case_count + 1))
    done
done

scripts/report_vitis_8x64_pd_sweep.sh "${results_temp}" \
    > "${performance_temp}"
scripts/report_vitis_8x64_pd_precision.sh "${results_temp}" \
    > "${precision_temp}"
if ! cmp -s "${performance_temp}" "${artifact_dir}/performance.tsv"; then
    echo "Q2.14 P/D performance table is not reproducible from raw evidence" >&2
    diff -u "${artifact_dir}/performance.tsv" "${performance_temp}" >&2 || true
    exit 65
fi
if ! cmp -s "${precision_temp}" "${artifact_dir}/precision.tsv"; then
    echo "Q2.14 P/D precision table is not reproducible from raw evidence" >&2
    diff -u "${artifact_dir}/precision.tsv" "${precision_temp}" >&2 || true
    exit 65
fi

echo "Q214 P/D RELEASE VERIFY PASS cases=${case_count} performance=reproduced precision=reproduced trace_scope=common_four_CU_running_time"
