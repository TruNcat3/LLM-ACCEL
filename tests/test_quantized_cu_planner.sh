#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

assert_field() {
    local output="$1"
    local expected="$2"
    if ! grep -qxF "${expected}" <<<"${output}"; then
        echo "missing planner field: ${expected}" >&2
        exit 1
    fi
}

resident_w4="$(scripts/plan_quantized_cus.sh w4a4 auto)"
assert_field "${resident_w4}" "fixed_profile=resident-r1"
assert_field "${resident_w4}" "accum_impl=single"
assert_field "${resident_w4}" "narrow_accum=1"
assert_field "${resident_w4}" "resource_limit=replication_cap"
assert_field "${resident_w4}" "modeled_max_cus=4"
assert_field "${resident_w4}" "selected_cus=4"
assert_field "${resident_w4}" "per_cu_lut=43970"
assert_field "${resident_w4}" \
    "nk_line=nk=compute_mm_stream_8x64_int4x4_block_nk:4:qmm_w4a4_0.qmm_w4a4_1.qmm_w4a4_2.qmm_w4a4_3"

resident_w8="$(scripts/plan_quantized_cus.sh w8a8 auto)"
assert_field "${resident_w8}" "resource_limit=replication_cap"
assert_field "${resident_w8}" "modeled_max_cus=4"
assert_field "${resident_w8}" "selected_cus=4"
assert_field "${resident_w8}" "per_cu_lut=35720"
assert_field "${resident_w8}" "aggregate_input_bits_per_cycle=4224"

resident_w4_dsp="$(
    QUANT_ACCUM_IMPL=dsp QUANT_NARROW_ACCUM=0 \
        scripts/plan_quantized_cus.sh w4a4 auto
)"
assert_field "${resident_w4_dsp}" "accum_impl=dsp"
assert_field "${resident_w4_dsp}" "resource_limit=dsp"
assert_field "${resident_w4_dsp}" "modeled_max_cus=2"
assert_field "${resident_w4_dsp}" "selected_cus=2"
assert_field "${resident_w4_dsp}" "per_cu_dsp=2176"

resident_w8_dsp="$(
    QUANT_ACCUM_IMPL=dsp QUANT_NARROW_ACCUM=1 \
        QUANT_RESOURCE_CAP_PCT=90 scripts/plan_quantized_cus.sh w8a8 auto
)"
assert_field "${resident_w8_dsp}" "narrow_accum=1"
assert_field "${resident_w8_dsp}" "resource_limit=dsp"
assert_field "${resident_w8_dsp}" "modeled_max_cus=2"
assert_field "${resident_w8_dsp}" "selected_cus=2"
assert_field "${resident_w8_dsp}" "per_cu_lut=128975"

isolated_w4="$(QUANT_FIXED_PROFILE=none scripts/plan_quantized_cus.sh w4a4 auto)"
assert_field "${isolated_w4}" "modeled_max_cus=4"
assert_field "${isolated_w4}" "selected_cus=4"
assert_field "${isolated_w4}" "arithmetic_roofline_gmac_s=902.205"

bandwidth_w8="$(
    QUANT_FIXED_PROFILE=none \
    QUANT_INPUT_BITS_PER_CYCLE_CAP=2112 \
        scripts/plan_quantized_cus.sh w8a8 auto
)"
assert_field "${bandwidth_w8}" "resource_limit=input_bits_per_cycle"
assert_field "${bandwidth_w8}" "selected_cus=2"

if scripts/plan_quantized_cus.sh w8a8 5 >/dev/null 2>&1; then
    echo "planner accepted an over-budget resident W8A8 count" >&2
    exit 1
fi

if scripts/plan_quantized_cus.sh invalid auto >/dev/null 2>&1; then
    echo "planner accepted an invalid candidate" >&2
    exit 1
fi

echo "QUANTIZED CU PLANNER PASS resident_w4=4 resident_w8=4 resident_w4_dsp=2 resident_w8_dsp=2 isolated_w4=4 bandwidth_w8=2"
