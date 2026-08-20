#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

artifact_dir="${1:-results/q214-resident-fix-20260818}"
verify_current_source="${Q214_VERIFY_CURRENT_SOURCE:-0}"
if [ "${verify_current_source}" != "0" ] &&
   [ "${verify_current_source}" != "1" ]; then
    echo "Q214_VERIFY_CURRENT_SOURCE must be 0 or 1" >&2
    exit 2
fi
required=(
    README.md
    validation.tsv
    performance.tsv
    resource.tsv
    artifact_manifest.tsv
    source_manifest.tsv
    checksums.sha256
    raw/full_p8_host.log
    raw/full_p8_profile_kernels.csv
)

for relative in "${required[@]}"; do
    if [ ! -s "${artifact_dir}/${relative}" ]; then
        echo "Q2.14 release artifact is missing: ${artifact_dir}/${relative}" >&2
        exit 66
    fi
done

awk -F '\t' '
    NR == 1 {
        if ($1 != "path" || $2 != "sha256" || $3 != "role") exit 1
        next
    }
    {
        if (NF != 3 || seen[$1]++) exit 1
        if ($1 == "" || length($2) != 64 || $2 !~ /^[0-9a-f]+$/ || $3 == "") exit 1
        count++
    }
    END { if (count < 10) exit 1 }
' "${artifact_dir}/source_manifest.tsv" || {
    echo "Malformed or incomplete source manifest" >&2
    exit 65
}

if [ "${verify_current_source}" = "1" ]; then
    while IFS=$'\t' read -r path expected role; do
        if [ "${path}" = "path" ]; then
            continue
        fi
        if [ ! -s "${path}" ]; then
            echo "Source manifest path is missing: ${path}" >&2
            exit 66
        fi
        actual="$(sha256sum "${path}" | awk '{print $1}')"
        if [ "${actual}" != "${expected}" ]; then
            echo "Historical source snapshot mismatch: ${path}" >&2
            echo "  expected=${expected}" >&2
            echo "  actual=${actual}" >&2
            exit 65
        fi
        if [ -z "${role}" ]; then
            echo "Source manifest has an empty role: ${path}" >&2
            exit 65
        fi
    done < "${artifact_dir}/source_manifest.tsv"
fi

awk -F '\t' '
    NR == 1 {
        if ($1 != "artifact" || $2 != "sha256" || $3 != "role") exit 1
        next
    }
    {
        if (seen[$1]++) exit 1
        if (length($2) != 64 || $2 !~ /^[0-9a-f]+$/ || $3 == "") exit 1
        count++
    }
    END { if (count < 6) exit 1 }
' "${artifact_dir}/artifact_manifest.tsv" || {
    echo "Malformed or incomplete artifact manifest" >&2
    exit 65
}

full_log="${artifact_dir}/raw/full_p8_host.log"
manifest_hash() {
    local artifact="$1"
    awk -F '\t' -v artifact="${artifact}" \
        '$1 == artifact { print $2; exit }' \
        "${artifact_dir}/artifact_manifest.tsv"
}
log_hash() {
    local field="$1"
    awk -F '=' -v field="${field}" \
        '$1 == field { print $2; exit }' \
        "${full_log}"
}
for identity in \
    host_qwen_8x64.exe:host_exe_sha256 \
    qwen_8x64_dual.xclbin:xclbin_sha256 \
    emconfig.json:emconfig_sha256
do
    artifact="${identity%%:*}"
    field="${identity#*:}"
    expected="$(manifest_hash "${artifact}")"
    actual="$(log_hash "${field}")"
    if [ -z "${expected}" ] || [ "${actual}" != "${expected}" ]; then
        echo "Artifact identity mismatch: ${artifact}" >&2
        echo "  manifest=${expected:-missing}" >&2
        echo "  host_log=${actual:-missing}" >&2
        exit 65
    fi
done

for contract in \
    'gate=standard_qwen_exact_p8' \
    'target=hw_emu' \
    'profile=qwen-layer' \
    'resident_token_rows=8' \
    'frequency=200' \
    'build_exact_compute_xo=1'
do
    if ! grep -Fxq "${contract}" "${full_log}"; then
        echo "Full P8 Host log is missing release contract: ${contract}" >&2
        exit 65
    fi
done

if [ "$(rg -c '^COARSE_TASK_PROGRESS ' "${full_log}")" -ne 3 ]; then
    echo "Full P8 Host log does not contain exactly three coarse-task completions" >&2
    exit 65
fi
rg -q '^COARSE_TASK_PROGRESS completed=1 total=3 op=18 phase=attention layer=0 position=0 query_tokens=8 controller_ms=[0-9.e+]+$' "${full_log}"
rg -q '^COARSE_TASK_PROGRESS completed=2 total=3 op=19 phase=ffn layer=0 position=0 query_tokens=8 controller_ms=[0-9.e+]+$' "${full_log}"
rg -q '^COARSE_TASK_PROGRESS completed=3 total=3 op=20 phase=final_norm layer=0 position=0 query_tokens=8 controller_ms=[0-9.e+]+$' "${full_log}"

awk -F '\t' '
    NR == 1 {
        for (i = 1; i <= NF; i++) column[$i] = i
        next
    }
    $(column["evidence"]) == "HW_Emu" &&
    $(column["workload"]) == "Task18_Task19_Task20_active_query_rows_8" &&
    $(column["result"]) == "PASS" &&
    $(column["detail"]) ~ /16384_values/ &&
    $(column["detail"]) ~ /intermediate_host_copy_0/ {
        found = 1
    }
    END { exit found ? 0 : 1 }
' "${artifact_dir}/validation.tsv" || {
    echo "Missing passing full P8 Task-18/19/20 validation row" >&2
    exit 65
}

awk -F '\t' '
    NR == 1 {
        for (i = 1; i <= NF; i++) column[$i] = i
        next
    }
    $(column["evidence_source"]) == "HW_Emu_CU_trace" &&
    $(column["scope"]) == "Task18_Task19_Task20" &&
    $(column["active_query_rows_per_block"]) == 8 &&
    $(column["layers"]) == 1 &&
    $(column["tasks"]) == 3 &&
    $(column["checked_values"]) == 16384 &&
    $(column["intermediate_host_copy"]) == 0 &&
    $(column["result"]) == "PASS" {
        # Three coarse tasks each write an 8x2048 Fix16 hidden tensor. Even an
        # ideal 512-bit port requires 1536 cycles, so this also rejects the
        # known 651-cycle trace fragment from the obsolete-oracle failure.
        if ($(column["xsim_cycles"]) < 1536 ||
            $(column["useful_mac"]) <= 0 ||
            $(column["useful_gmac_s"]) <= 0 ||
            $(column["modeled_interval_efficiency_percent"]) <= 0 ||
            $(column["modeled_interval_efficiency_percent"]) > 100) exit 2
        found = 1
    }
    END { exit found ? 0 : 1 }
' "${artifact_dir}/performance.tsv" || {
    echo "Missing or physically implausible full P8 CU-trace row" >&2
    exit 65
}

rg -q '^RANDOM task_composed_prefill_block PASS values=16384 ' "${full_log}"
rg -q '^TASK_COMPOSED_PREFILL_BLOCK_VERIFY .* intermediate_host_copy=0 .* PASS$' "${full_log}"
rg -q '^exit_status=0$' "${full_log}"

raw_active_us="$(awk -F ',' '$1 == "cc8_ctrl" { print $2; exit }' \
    "${artifact_dir}/raw/full_p8_profile_kernels.csv")"
read -r table_clock_mhz table_cycles table_active_us < <(
    awk -F '\t' '
        NR == 1 {
            for (i = 1; i <= NF; i++) column[$i] = i
            next
        }
        $(column["scope"]) == "Task18_Task19_Task20" {
            print $(column["xsim_clock_mhz"]),
                  $(column["xsim_cycles"]),
                  $(column["latency_us_at_200mhz"])
            exit
        }
    ' "${artifact_dir}/performance.tsv"
)
if [ -z "${raw_active_us}" ] || [ -z "${table_active_us:-}" ]; then
    echo "Cannot reconcile full P8 raw profile and performance row" >&2
    exit 65
fi
awk \
    -v raw_us="${raw_active_us}" \
    -v table_us="${table_active_us}" \
    -v mhz="${table_clock_mhz}" \
    -v cycles="${table_cycles}" '
        function abs(value) { return value < 0 ? -value : value }
        BEGIN {
            if (abs(raw_us - table_us) > 0.001) exit 1
            if (abs(raw_us * mhz - cycles) > 1.0) exit 1
        }
    ' || {
        echo "Full P8 performance row does not reproduce the raw CU trace" >&2
        exit 65
    }

(
    cd "$(dirname "${artifact_dir}")/.."
    sha256sum -c "$(basename "$(dirname "${artifact_dir}")")/$(basename "${artifact_dir}")/checksums.sha256"
)

echo "Q2.14 RESIDENT RELEASE VERIFY PASS artifact=${artifact_dir} source_scope=$([ "${verify_current_source}" = "1" ] && printf historical_snapshot_matches_current_worktree || printf historical_snapshot_manifest_only)"
