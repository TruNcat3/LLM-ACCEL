#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/llm-e2e-scaling.XXXXXX")"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

report="$(
    scripts/report_vitis_8x64_e2e_scaling.sh \
        results/qwen3b-e2e-20260820/performance.tsv \
        results/qwen3b-e2e-l2-20260821/performance.tsv
)"

if ! awk -F '\t' '
    function abs(value) { return value < 0 ? -value : value }
    NR == 1 {
        for (i = 1; i <= NF; i++) column[$i] = i
        next
    }
    NR == 2 {
        if ($(column["result"]) != "qwen3b-e2e-20260820" ||
            $(column["layers"]) != 1 ||
            abs($(column["cycle_scale_vs_base"]) - 1) > 0.000001 ||
            abs($(column["mac_scale_vs_base"]) - 1) > 0.000001 ||
            abs($(column["cycles_per_layer_change_percent"])) > 0.000001) exit 1
        baseline = 1
    }
    NR == 3 {
        if ($(column["result"]) != "qwen3b-e2e-l2-20260821" ||
            $(column["layers"]) != 2 ||
            $(column["expected_coarse_tasks"]) != 10 ||
            abs($(column["cycle_scale_vs_base"]) - 1.947976) > 0.000001 ||
            abs($(column["mac_scale_vs_base"]) - 2) > 0.000001 ||
            abs($(column["cycles_per_layer_change_percent"]) + 2.601199) > 0.000001 ||
            abs($(column["throughput_change_percent"]) - 2.670668) > 0.000001 ||
            abs($(column["efficiency_delta_pp"]) - 1.519726) > 0.000001) exit 1
        comparison = 1
    }
    END { exit NR == 3 && baseline && comparison ? 0 : 1 }
' <<<"${report}"; then
    echo "E2E scaling report regression" >&2
    printf '%s\n' "${report}" >&2
    exit 65
fi

awk -F '\t' 'BEGIN { OFS = "\t" }
    NR == 1 {
        for (i = 1; i <= NF; i++) {
            if ($i == "xclbin_sha256") xclbin_column = i
        }
        print
        next
    }
    NR == 2 {
        $xclbin_column = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        print
    }
' results/qwen3b-e2e-l2-20260821/performance.tsv \
    > "${tmp_dir}/performance.tsv"
set +e
scripts/report_vitis_8x64_e2e_scaling.sh \
    results/qwen3b-e2e-20260820/performance.tsv \
    "${tmp_dir}/performance.tsv" \
    > "${tmp_dir}/mismatch.stdout" 2> "${tmp_dir}/mismatch.stderr"
mismatch_status="$?"
set -e
if [ "${mismatch_status}" -ne 65 ] ||
   ! grep -q 'do not share one workload and artifact identity' \
       "${tmp_dir}/mismatch.stderr"; then
    echo "E2E scaling report accepted a different xclbin identity" >&2
    exit 65
fi

printf '%s\n' \
    'E2E SCALING REPORT PASS L1_to_L2 cycles=1.947976x MAC=2x cycles_per_layer=-2.601199% throughput=+2.670668% efficiency=+1.519726pp mismatched_xclbin=rejected'
