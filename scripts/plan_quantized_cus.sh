#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/plan_quantized_cus.sh <w4a4|w8a8> [auto|COUNT]

Select a quantized compute-CU replication count from HLS resource estimates.
The default resident-r1 profile reserves the published controller and status
sink resources before applying an 85% whole-device utilization cap.

Environment overrides:
  QUANT_FIXED_PROFILE=resident-r1|none  Fixed non-compute resource profile
  QUANT_RESOURCE_CAP_PCT=1..100         Whole-device budget (default: 85)
  QUANT_DEVICE_BRAM=2688                U50 BRAM_18K capacity
  QUANT_DEVICE_DSP=5952                 U50 DSP capacity
  QUANT_DEVICE_FF=1743360               U50 FF capacity
  QUANT_DEVICE_LUT=871680               U50 LUT capacity
  QUANT_FIXED_{BRAM,DSP,FF,LUT}=N       Override fixed-profile resources
  QUANT_MAX_CUS=N                       Optional replication cap; 0 disables
  QUANT_INPUT_BITS_PER_CYCLE_CAP=N      Optional aggregate input-wire cap

The emitted nk_line is only the CU replication declaration. Every CU still
requires a unique task/activation/weight/output stream set from the controller.
EOF
}

die() {
    echo "quantized CU planner: $*" >&2
    exit 2
}

is_uint() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

fit_count() {
    local available="$1"
    local per_cu="$2"
    if [ "${per_cu}" -eq 0 ]; then
        echo 1000000000
    elif [ "${available}" -le 0 ]; then
        echo 0
    else
        echo $((available / per_cu))
    fi
}

percent() {
    awk -v used="$1" -v total="$2" 'BEGIN { printf "%.2f", 100.0 * used / total }'
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

candidate="${1:-}"
requested_count="${2:-auto}"

case "${candidate}" in
    w4a4)
        kernel_name=compute_mm_stream_8x64_int4x4_block_nk
        instance_prefix=qmm_w4a4_
        tile=8x64
        per_bram=0
        per_dsp=128
        per_ff=59417
        per_lut=192809
        products_per_cycle=512
        estimated_fmax_mhz=440.53
        input_bits_per_cycle=288
        stream_sets_per_cu=4
        ;;
    w8a8)
        kernel_name=compute_mm_stream_4x128_int8x8_block_nk
        instance_prefix=qmm_w8a8_
        tile=4x128
        per_bram=0
        per_dsp=512
        per_ff=76459
        per_lut=225343
        products_per_cycle=512
        estimated_fmax_mhz=521.69
        input_bits_per_cycle=1056
        stream_sets_per_cu=7
        ;;
    *)
        usage >&2
        die "candidate must be w4a4 or w8a8"
        ;;
esac

if [ "${requested_count}" != auto ] && ! is_uint "${requested_count}"; then
    die "COUNT must be auto or a positive integer"
fi
if [ "${requested_count}" = 0 ]; then
    die "COUNT must be greater than zero"
fi

device_bram="${QUANT_DEVICE_BRAM:-2688}"
device_dsp="${QUANT_DEVICE_DSP:-5952}"
device_ff="${QUANT_DEVICE_FF:-1743360}"
device_lut="${QUANT_DEVICE_LUT:-871680}"
cap_pct="${QUANT_RESOURCE_CAP_PCT:-85}"
fixed_profile="${QUANT_FIXED_PROFILE:-resident-r1}"
max_cus="${QUANT_MAX_CUS:-0}"
input_cap="${QUANT_INPUT_BITS_PER_CYCLE_CAP:-0}"

for value in "${device_bram}" "${device_dsp}" "${device_ff}" \
             "${device_lut}" "${cap_pct}" "${max_cus}" "${input_cap}"; do
    is_uint "${value}" || die "resource values must be unsigned integers"
done
if [ "${cap_pct}" -lt 1 ] || [ "${cap_pct}" -gt 100 ]; then
    die "QUANT_RESOURCE_CAP_PCT must be in 1..100"
fi

case "${fixed_profile}" in
    resident-r1)
        profile_bram=1212
        profile_dsp=123
        profile_ff=473348
        profile_lut=462019
        ;;
    none)
        profile_bram=0
        profile_dsp=0
        profile_ff=0
        profile_lut=0
        ;;
    *)
        die "QUANT_FIXED_PROFILE must be resident-r1 or none"
        ;;
esac

fixed_bram="${QUANT_FIXED_BRAM:-${profile_bram}}"
fixed_dsp="${QUANT_FIXED_DSP:-${profile_dsp}}"
fixed_ff="${QUANT_FIXED_FF:-${profile_ff}}"
fixed_lut="${QUANT_FIXED_LUT:-${profile_lut}}"
for value in "${fixed_bram}" "${fixed_dsp}" "${fixed_ff}" "${fixed_lut}"; do
    is_uint "${value}" || die "fixed resource values must be unsigned integers"
done

cap_bram=$((device_bram * cap_pct / 100))
cap_dsp=$((device_dsp * cap_pct / 100))
cap_ff=$((device_ff * cap_pct / 100))
cap_lut=$((device_lut * cap_pct / 100))

available_bram=$((cap_bram - fixed_bram))
available_dsp=$((cap_dsp - fixed_dsp))
available_ff=$((cap_ff - fixed_ff))
available_lut=$((cap_lut - fixed_lut))

max_by_bram="$(fit_count "${available_bram}" "${per_bram}")"
max_by_dsp="$(fit_count "${available_dsp}" "${per_dsp}")"
max_by_ff="$(fit_count "${available_ff}" "${per_ff}")"
max_by_lut="$(fit_count "${available_lut}" "${per_lut}")"
max_by_stream=1000000000
if [ "${input_cap}" -gt 0 ]; then
    max_by_stream=$((input_cap / input_bits_per_cycle))
fi

resource_max_cus="${max_by_bram}"
resource_limit=bram
for pair in \
    "dsp:${max_by_dsp}" \
    "ff:${max_by_ff}" \
    "lut:${max_by_lut}" \
    "input_bits_per_cycle:${max_by_stream}"
do
    name="${pair%%:*}"
    value="${pair#*:}"
    if [ "${value}" -lt "${resource_max_cus}" ]; then
        resource_max_cus="${value}"
        resource_limit="${name}"
    fi
done
if [ "${max_cus}" -gt 0 ] && [ "${max_cus}" -lt "${resource_max_cus}" ]; then
    resource_max_cus="${max_cus}"
    resource_limit=replication_cap
fi

if [ "${requested_count}" = auto ]; then
    selected_cus="${resource_max_cus}"
else
    selected_cus="${requested_count}"
    if [ "${selected_cus}" -gt "${resource_max_cus}" ]; then
        die "requested ${selected_cus} CUs exceed the modeled maximum ${resource_max_cus} (${resource_limit})"
    fi
fi
if [ "${selected_cus}" -lt 1 ]; then
    die "no ${candidate} CU fits the selected fixed-resource profile and cap"
fi

total_bram=$((fixed_bram + selected_cus * per_bram))
total_dsp=$((fixed_dsp + selected_cus * per_dsp))
total_ff=$((fixed_ff + selected_cus * per_ff))
total_lut=$((fixed_lut + selected_cus * per_lut))
total_input_bits=$((selected_cus * input_bits_per_cycle))
total_stream_sets=$((selected_cus * stream_sets_per_cu))
roofline_gmac_s="$(
    awk -v products="${products_per_cycle}" \
        -v mhz="${estimated_fmax_mhz}" -v count="${selected_cus}" \
        'BEGIN { printf "%.3f", products * mhz * count / 1000.0 }'
)"

instances=
for ((cu = 0; cu < selected_cus; cu++)); do
    if [ -n "${instances}" ]; then
        instances+="."
    fi
    instances+="${instance_prefix}${cu}"
done

echo "candidate=${candidate}"
echo "tile=${tile}"
echo "fixed_profile=${fixed_profile}"
echo "resource_cap_pct=${cap_pct}"
echo "per_cu_bram18=${per_bram}"
echo "per_cu_dsp=${per_dsp}"
echo "per_cu_ff=${per_ff}"
echo "per_cu_lut=${per_lut}"
echo "per_cu_input_bits_per_cycle=${input_bits_per_cycle}"
echo "max_cus_by_bram=${max_by_bram}"
echo "max_cus_by_dsp=${max_by_dsp}"
echo "max_cus_by_ff=${max_by_ff}"
echo "max_cus_by_lut=${max_by_lut}"
if [ "${input_cap}" -gt 0 ]; then
    echo "max_cus_by_input_bits_per_cycle=${max_by_stream}"
else
    echo "max_cus_by_input_bits_per_cycle=unbounded"
fi
echo "resource_limit=${resource_limit}"
echo "modeled_max_cus=${resource_max_cus}"
echo "selected_cus=${selected_cus}"
echo "aggregate_input_bits_per_cycle=${total_input_bits}"
echo "required_stream_port_sets=${total_stream_sets}"
echo "arithmetic_roofline_gmac_s=${roofline_gmac_s}"
echo "modeled_total_bram18=${total_bram}"
echo "modeled_total_dsp=${total_dsp}"
echo "modeled_total_ff=${total_ff}"
echo "modeled_total_lut=${total_lut}"
echo "modeled_total_bram_percent=$(percent "${total_bram}" "${device_bram}")"
echo "modeled_total_dsp_percent=$(percent "${total_dsp}" "${device_dsp}")"
echo "modeled_total_ff_percent=$(percent "${total_ff}" "${device_ff}")"
echo "modeled_total_lut_percent=$(percent "${total_lut}" "${device_lut}")"
echo "nk_line=nk=${kernel_name}:${selected_cus}:${instances}"
echo "wiring_gate=one_unique_stream_set_per_cu_required"

if [ "${cap_pct}" -gt 90 ]; then
    echo "warning=resource_cap_above_90_percent_has_little_shell_and_routing_margin"
fi
