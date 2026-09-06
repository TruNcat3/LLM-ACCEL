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
assert_field "${resident_w4}" "resource_limit=lut"
assert_field "${resident_w4}" "modeled_max_cus=1"
assert_field "${resident_w4}" "selected_cus=1"
assert_field "${resident_w4}" \
    "nk_line=nk=compute_mm_stream_8x64_int4x4_block_nk:1:qmm_w4a4_0"

resident_w8="$(scripts/plan_quantized_cus.sh w8a8 auto)"
assert_field "${resident_w8}" "resource_limit=lut"
assert_field "${resident_w8}" "modeled_max_cus=1"
assert_field "${resident_w8}" "selected_cus=1"
assert_field "${resident_w8}" "aggregate_input_bits_per_cycle=1056"

isolated_w4="$(QUANT_FIXED_PROFILE=none scripts/plan_quantized_cus.sh w4a4 auto)"
assert_field "${isolated_w4}" "modeled_max_cus=3"
assert_field "${isolated_w4}" "selected_cus=3"
assert_field "${isolated_w4}" "arithmetic_roofline_gmac_s=676.654"

bandwidth_w8="$(
    QUANT_FIXED_PROFILE=none \
    QUANT_INPUT_BITS_PER_CYCLE_CAP=2112 \
        scripts/plan_quantized_cus.sh w8a8 auto
)"
assert_field "${bandwidth_w8}" "resource_limit=input_bits_per_cycle"
assert_field "${bandwidth_w8}" "selected_cus=2"

if scripts/plan_quantized_cus.sh w8a8 2 >/dev/null 2>&1; then
    echo "planner accepted an over-budget resident W8A8 count" >&2
    exit 1
fi

if scripts/plan_quantized_cus.sh invalid auto >/dev/null 2>&1; then
    echo "planner accepted an invalid candidate" >&2
    exit 1
fi

echo "QUANTIZED CU PLANNER PASS resident_w4=1 resident_w8=1 isolated_w4=3 bandwidth_w8=2"
