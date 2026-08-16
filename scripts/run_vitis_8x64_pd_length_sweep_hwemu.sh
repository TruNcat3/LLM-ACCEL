#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

build_dir="vitis_8x64/build.qwen_layer_long.prefill_layer.d2.ii2.wr33.cw1.scratch.hw_emu.xilinx_u50_gen3x16_xdma_5_202210_1"
profile="qwen-layer-long"
lengths="64,256,512,1024"
phase="pd"
seed="20260722"
timeout_seconds="604800"
dry_run=0

usage() {
    echo "Usage: $0 [options]"
    echo "  --build-dir DIR       Existing prefill-layer hw_emu build"
    echo "  --profile NAME        Host/kernel profile (default: qwen-layer-long)"
    echo "  --lengths CSV         Context lengths (default: 64,256,512,1024)"
    echo "  --phase p|d|pd        Run final 8-token P block, next-token D, or both"
    echo "  --seed N              Random model/KV seed"
    echo "  --timeout SECONDS     Per-case host timeout"
    echo "  --dry-run             Print the generated cases without running"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --build-dir) build_dir="$2"; shift 2 ;;
        --profile) profile="$2"; shift 2 ;;
        --lengths) lengths="$2"; shift 2 ;;
        --phase) phase="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --timeout) timeout_seconds="$2"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${phase}" in p|d|pd) ;; *) echo "--phase must be p, d, or pd" >&2; exit 2 ;; esac
if [ ! -d "${build_dir}" ]; then
    echo "Missing build directory: ${build_dir}" >&2
    exit 66
fi

IFS=',' read -r -a length_array <<< "${lengths}"
if [ "${#length_array[@]}" -eq 0 ]; then
    echo "--lengths must contain at least one context length" >&2
    exit 2
fi

mkdir -p logs
timestamp="$(date +%Y%m%d_%H%M%S_%N)_${BASHPID}"
summary="logs/vitis_8x64_pd_length_sweep_${timestamp}.log"
artifact_dir="logs/vitis_8x64_pd_length_sweep_${timestamp}"
results_tsv="${artifact_dir}/results.tsv"
mkdir -p "${artifact_dir}"
printf 'phase\tcontext_len\tprefill_start\tprefill_len\ttokens\texit_status\tactive_us\tcycles_200mhz\thost_log\tprofile_csv\n' \
    > "${results_tsv}"

run_case() {
    local kind="$1"
    local context_len="$2"
    local start end tokens
    if [ "${kind}" = "p" ]; then
        tokens=8
        if [ "${context_len}" -lt "${tokens}" ]; then
            tokens="${context_len}"
        fi
        start=$((context_len - tokens))
        end="${context_len}"
    else
        tokens=1
        start="${context_len}"
        end=$((context_len + 1))
    fi

    printf 'case=%s context_len=%s prefill_start=%s prefill_len=%s tokens=%s\n' \
        "${kind}" "${context_len}" "${start}" "${end}" "${tokens}" | tee -a "${summary}"
    if [ "${dry_run}" -eq 1 ]; then
        return 0
    fi

    local run_started_at
    run_started_at="$(date +%s)"
    set +e
    scripts/run_vitis_8x64_qwen_profile_hwemu.sh \
        --build-dir "${build_dir}" \
        --profile "${profile}" \
        --mode profile-prefill-block \
        --prefill-start "${start}" \
        --prefill-len "${end}" \
        --seed "${seed}" \
        --random-model \
        --timeout "${timeout_seconds}"
    local status=$?
    set -e
    printf 'case=%s context_len=%s exit_status=%s finished_at=%s\n' \
        "${kind}" "${context_len}" "${status}" "$(date -Is)" | tee -a "${summary}"

    local profile_tag case_log cycle_report profile_csv copied_profile active_us cycles run_id
    profile_tag="${profile//[^A-Za-z0-9]/_}"
    case_log="$({
        while IFS= read -r candidate; do
            if grep -Fqx \
                    "prefill_stage mode=profile-prefill-block prefill_start=${start} prefill_len=${end}" \
                    "${candidate}" && \
               grep -Fqx "exit_status=${status}" "${candidate}"; then
                stat -c '%Y %n' "${candidate}"
            fi
        done < <(find logs -maxdepth 1 -type f \
            -name "vitis_8x64_hwemu_${profile_tag}_profile-prefill-block_*.log" \
            -newermt "@${run_started_at}" -print 2>/dev/null || true)
    } | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [ -z "${case_log}" ] || [ ! -s "${case_log}" ]; then
        echo "Cannot identify completed host log for ${kind}@${context_len}" >&2
        printf '%s\t%s\t%s\t%s\t%s\t%s\tNA\tNA\tNA\tNA\n' \
            "${kind}" "${context_len}" "${start}" "${end}" "${tokens}" "${status}" \
            >> "${results_tsv}"
        if [ "${status}" -ne 0 ]; then return "${status}"; else return 66; fi
    fi
    printf 'case=%s context_len=%s host_log=%s\n' \
        "${kind}" "${context_len}" "${case_log}" | tee -a "${summary}"
    run_id="$(sed -n \
        's#.*[/]\.run/\([0-9][0-9]*\)/hw_em/.*#\1#p' "${case_log}" | tail -n 1)"
    if ! [[ "${run_id}" =~ ^[0-9]+$ ]]; then
        echo "Cannot extract hw_emu run id from ${case_log}" >&2
        printf '%s\t%s\t%s\t%s\t%s\t%s\tNA\tNA\t%s\tNA\n' \
            "${kind}" "${context_len}" "${start}" "${end}" "${tokens}" "${status}" \
            "${case_log}" >> "${results_tsv}"
        if [ "${status}" -ne 0 ]; then return "${status}"; else return 66; fi
    fi
    profile_csv="${build_dir}/.run/${run_id}/hw_em/device0/binary_0/behav_waveform/xsim/profile_kernels.csv"
    if [ -z "${profile_csv}" ] || [ ! -s "${profile_csv}" ]; then
        echo "No new profile_kernels.csv found for ${kind}@${context_len}" >&2
        printf '%s\t%s\t%s\t%s\t%s\t%s\tNA\tNA\t%s\tNA\n' \
            "${kind}" "${context_len}" "${start}" "${end}" "${tokens}" "${status}" \
            "${case_log}" >> "${results_tsv}"
        if [ "${status}" -ne 0 ]; then return "${status}"; else return 66; fi
    fi
    copied_profile="${artifact_dir}/${kind}_${context_len}_profile_kernels.csv"
    cp -p "${profile_csv}" "${copied_profile}"
    active_us="$(awk -F, '$1 == "cc8_ctrl" { print $2; exit }' "${copied_profile}")"
    if ! [[ "${active_us}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Cannot extract cc8_ctrl active time from ${copied_profile}" >&2
        return 65
    fi
    cycles="$(awk -v us="${active_us}" 'BEGIN { printf "%d", us * 200 + 0.5 }')"
    cycle_report="logs/vitis_8x64_pd_${kind}${context_len}_cycles_${timestamp}.txt"
    scripts/report_hwemu_kernel_cycles.sh "${build_dir}" 200 "${copied_profile}" \
        | tee "${cycle_report}" \
        | sed "s/^/case=${kind} context_len=${context_len} /" \
        | tee -a "${summary}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${kind}" "${context_len}" "${start}" "${end}" "${tokens}" "${status}" \
        "${active_us}" "${cycles}" "${case_log}" "${copied_profile}" \
        >> "${results_tsv}"
    return "${status}"
}

printf 'started_at=%s build_dir=%s profile=%s lengths=%s phase=%s seed=%s\n' \
    "$(date -Is)" "$(realpath "${build_dir}")" "${profile}" "${lengths}" "${phase}" "${seed}" | tee "${summary}"
printf 'results_tsv=%s artifact_dir=%s\n' "${results_tsv}" "${artifact_dir}" | tee -a "${summary}"

overall_status=0
for context_len in "${length_array[@]}"; do
    if ! [[ "${context_len}" =~ ^[0-9]+$ ]] || [ "${context_len}" -lt 1 ] || [ "${context_len}" -gt 1024 ]; then
        echo "Context lengths must be integers in 1..1024: ${context_len}" >&2
        exit 2
    fi
    if [ "${phase}" = "p" ] || [ "${phase}" = "pd" ]; then
        if ! run_case p "${context_len}"; then
            overall_status=1
        fi
    fi
    if [ "${phase}" = "d" ] || [ "${phase}" = "pd" ]; then
        if ! run_case d "${context_len}"; then
            overall_status=1
        fi
    fi
done

printf 'completed_at=%s overall_status=%s\n' \
    "$(date -Is)" "${overall_status}" | tee -a "${summary}"
exit "${overall_status}"
