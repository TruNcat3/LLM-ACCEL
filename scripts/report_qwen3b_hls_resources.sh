#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 2 ]; then
    echo "usage: $0 [QWEN3B_WORK_ROOT] [LINK_TARGET_MHZ]" >&2
    exit 2
fi

work_root="${1:-/tmp/llm_accel_qwen3b_q214_resident_fix}"
link_target_mhz="${2:-200}"
hls_root="${work_root}/hls"
if [ ! -d "${hls_root}" ]; then
    echo "Missing Qwen2.5-3B HLS root: ${hls_root}" >&2
    exit 66
fi
if ! [[ "${link_target_mhz}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "LINK_TARGET_MHZ must be numeric" >&2
    exit 2
fi

find_top_report() {
    local top="$1"
    local -a matches=()
    mapfile -t matches < <(
        find "${hls_root}" -type f \
            -path "*/solution1/syn/report/${top}_csynth.rpt" \
            ! -path '*/impl/*' ! -path '*/.autopilot/*' |
        sort
    )
    if [ "${#matches[@]}" -ne 1 ]; then
        echo "Expected one top-level ${top} report, found ${#matches[@]}" >&2
        exit 66
    fi
    printf '%s\n' "${matches[0]}"
}

timing_fields() {
    local report="$1"
    awk -F '|' '
        function trim(value) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            return value
        }
        $2 ~ /ap_clk/ {
            target = trim($3)
            estimate = trim($4)
            sub(/[[:space:]]*ns$/, "", target)
            sub(/[[:space:]]*ns$/, "", estimate)
            print target, estimate
            exit
        }
    ' "${report}"
}

resource_fields() {
    local report="$1"
    awk -F '|' '
        function trim(value) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            return value
        }
        trim($2) == "Total" {
            print trim($3), trim($4), trim($5), trim($6), trim($7)
            exit
        }
    ' "${report}"
}

compute_report="$(find_top_report compute_core_8x64_unified_nk)"
controller_report="$(find_top_report control_cache_8x64_dual_core_nk)"
status_report="$(find_top_report cc8_status_sink_nk)"
compute_evidence="${compute_report#${work_root}/}"
controller_evidence="${controller_report#${work_root}/}"
status_evidence="${status_report#${work_root}/}"

read -r compute_target compute_estimate < <(timing_fields "${compute_report}")
read -r controller_target controller_estimate < <(timing_fields "${controller_report}")
read -r status_target status_estimate < <(timing_fields "${status_report}")
read -r compute_bram compute_dsp compute_ff compute_lut compute_uram \
    < <(resource_fields "${compute_report}")
read -r controller_bram controller_dsp controller_ff controller_lut controller_uram \
    < <(resource_fields "${controller_report}")
read -r status_bram status_dsp status_ff status_lut status_uram \
    < <(resource_fields "${status_report}")

for value in \
    "${compute_target}" "${compute_estimate}" \
    "${controller_target}" "${controller_estimate}" \
    "${status_target}" "${status_estimate}" \
    "${compute_bram}" "${compute_dsp}" "${compute_ff}" \
    "${compute_lut}" "${compute_uram}" \
    "${controller_bram}" "${controller_dsp}" "${controller_ff}" \
    "${controller_lut}" "${controller_uram}" \
    "${status_bram}" "${status_dsp}" "${status_ff}" \
    "${status_lut}" "${status_uram}"
do
    if ! [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Malformed HLS report value: ${value:-missing}" >&2
        exit 65
    fi
done

system_bram=$((controller_bram + 2 * compute_bram + status_bram))
system_dsp=$((controller_dsp + 2 * compute_dsp + status_dsp))
system_ff=$((controller_ff + 2 * compute_ff + status_ff))
system_lut=$((controller_lut + 2 * compute_lut + status_lut))
system_uram=$((controller_uram + 2 * compute_uram + status_uram))
system_estimate="$(
    awk -v a="${compute_estimate}" -v b="${controller_estimate}" \
        -v c="${status_estimate}" \
        'BEGIN { maximum = a; if (b > maximum) maximum = b; if (c > maximum) maximum = c; printf "%.3f", maximum }'
)"
system_target="$(
    awk -v a="${compute_target}" -v b="${controller_target}" \
        -v c="${status_target}" \
        'BEGIN { maximum = a; if (b > maximum) maximum = b; if (c > maximum) maximum = c; printf "%.2f", maximum }'
)"

# Xilinx U50 xcu50-fsvh2104 whole-device capacities reported by Vitis HLS.
available_bram=2688
available_dsp=5952
available_ff=1743360
available_lut=871680
available_uram=640

print_row() {
    local scope="$1"
    local instances="$2"
    local target_ns="$3"
    local estimate_ns="$4"
    local bram="$5"
    local dsp="$6"
    local ff="$7"
    local lut="$8"
    local uram="$9"
    local evidence="${10}"
    awk \
        -v scope="${scope}" -v instances="${instances}" \
        -v target_ns="${target_ns}" -v estimate_ns="${estimate_ns}" \
        -v link_mhz="${link_target_mhz}" \
        -v bram="${bram}" -v available_bram="${available_bram}" \
        -v dsp="${dsp}" -v available_dsp="${available_dsp}" \
        -v ff="${ff}" -v available_ff="${available_ff}" \
        -v lut="${lut}" -v available_lut="${available_lut}" \
        -v uram="${uram}" -v available_uram="${available_uram}" \
        -v evidence="${evidence}" '
        BEGIN {
            printf "%s\t%d\t%.2f\t%.3f\t%.3f\t%.3f\t%d\t%.3f\t%d\t%.3f\t%d\t%.3f\t%d\t%.3f\t%d\t%.3f\t%s\n",
                scope, instances, target_ns, estimate_ns,
                1000.0 / estimate_ns, link_mhz,
                bram, 100.0 * bram / available_bram,
                dsp, 100.0 * dsp / available_dsp,
                ff, 100.0 * ff / available_ff,
                lut, 100.0 * lut / available_lut,
                uram, 100.0 * uram / available_uram,
                evidence
        }
    '
}

printf 'scope\tinstances\thls_target_ns\thls_estimated_ns\thls_estimated_fmax_mhz\tlink_target_mhz\tbram18k\tbram18k_percent\tdsp\tdsp_percent\tff\tff_percent\tlut\tlut_percent\turam\turam_percent\tevidence\n'
print_row controller 1 "${controller_target}" "${controller_estimate}" \
    "${controller_bram}" "${controller_dsp}" "${controller_ff}" \
    "${controller_lut}" "${controller_uram}" "${controller_evidence}"
print_row compute_one 1 "${compute_target}" "${compute_estimate}" \
    "${compute_bram}" "${compute_dsp}" "${compute_ff}" \
    "${compute_lut}" "${compute_uram}" "${compute_evidence}"
print_row compute_two 2 "${compute_target}" "${compute_estimate}" \
    "$((2 * compute_bram))" "$((2 * compute_dsp))" \
    "$((2 * compute_ff))" "$((2 * compute_lut))" \
    "$((2 * compute_uram))" "2x:${compute_evidence}"
print_row status 1 "${status_target}" "${status_estimate}" \
    "${status_bram}" "${status_dsp}" "${status_ff}" \
    "${status_lut}" "${status_uram}" "${status_evidence}"
print_row whole_system 4 "${system_target}" "${system_estimate}" \
    "${system_bram}" "${system_dsp}" "${system_ff}" \
    "${system_lut}" "${system_uram}" derived_controller_plus_2compute_plus_status
