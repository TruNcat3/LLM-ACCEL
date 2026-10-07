#ifndef LLM_FPGA_QUANTIZED_NUMERIC_HPP
#define LLM_FPGA_QUANTIZED_NUMERIC_HPP

#include "datatypes.hpp"

#include <ap_fixed.h>
#include <ap_int.h>

using quantized_w8_accum_t = ap_int<32>;
using quantized_w4_accum_t = ap_int<24>;
using quantized_dequant_wide_t = ap_fixed<56, 32, AP_RND, AP_SAT>;
using quantized_dequant_scale_t = ap_ufixed<32, 8, AP_RND, AP_SAT>;
using quantized_dequant_product_t = ap_fixed<64, 40, AP_RND, AP_SAT>;
using quantized_requant_wide_t = ap_fixed<40, 24, AP_RND, AP_SAT>;

struct quantized_symmetric_scale_t {
    quant_scale_t scale;
    quant_inverse_scale_t inverse_scale;
};

inline quantized_symmetric_scale_t make_quantized_symmetric_scale(
    fm_t max_abs) {
    #pragma HLS inline
    quantized_symmetric_scale_t result{};
    if (max_abs <= fm_t(0)) {
        result.scale = quant_scale_t(1);
        result.inverse_scale = quant_inverse_scale_t(1);
        return result;
    }
    // Division is kept at this semantic boundary.  A reciprocal LUT/Newton
    // implementation can replace it without changing the stream ABI.
    result.scale = quant_scale_t(max_abs / fm_t(127));
    if (result.scale == quant_scale_t(0)) {
        result.scale = quant_scale_t(1.0 / 4096.0);
    }
    result.inverse_scale = quant_inverse_scale_t(fm_t(1) / result.scale);
    return result;
}

inline quantized_symmetric_scale_t make_quantized_symmetric_scale_w4(
    fm_t max_abs) {
    #pragma HLS inline
    quantized_symmetric_scale_t result{};
    if (max_abs <= fm_t(0)) {
        result.scale = quant_scale_t(1);
        result.inverse_scale = quant_inverse_scale_t(1);
        return result;
    }
    result.scale = quant_scale_t(max_abs / fm_t(7));
    if (result.scale == quant_scale_t(0)) {
        result.scale = quant_scale_t(1.0 / 4096.0);
    }
    result.inverse_scale = quant_inverse_scale_t(fm_t(1) / result.scale);
    return result;
}

inline ap_int<8> quantize_symmetric_int8(
    fm_t value,
    quant_inverse_scale_t inverse_scale) {
    #pragma HLS inline
    const quantized_requant_wide_t scaled =
        quantized_requant_wide_t(value) *
        quantized_requant_wide_t(inverse_scale);
    if (scaled > quantized_requant_wide_t(127)) return ap_int<8>(127);
    if (scaled < quantized_requant_wide_t(-127)) return ap_int<8>(-127);
    return ap_int<8>(scaled);
}

inline ap_int<4> quantize_symmetric_int4(
    fm_t value,
    quant_inverse_scale_t inverse_scale) {
    #pragma HLS inline
    const quantized_requant_wide_t scaled =
        quantized_requant_wide_t(value) *
        quantized_requant_wide_t(inverse_scale);
    if (scaled > quantized_requant_wide_t(7)) return ap_int<4>(7);
    if (scaled < quantized_requant_wide_t(-7)) return ap_int<4>(-7);
    return ap_int<4>(scaled);
}

inline quantized_dequant_scale_t combine_quantized_dequant_scales(
    quant_scale_t activation_scale,
    quant_scale_t weight_scale) {
    #pragma HLS inline
    return quantized_dequant_scale_t(activation_scale * weight_scale);
}

inline fm_t dequantize_w8_accumulator(
    quantized_w8_accum_t accumulator,
    quantized_dequant_scale_t combined_scale) {
    #pragma HLS inline
    const quantized_dequant_product_t value = accumulator * combined_scale;
    return fm_t(value);
}

inline fm_t dequantize_w8_accumulator(
    quantized_w8_accum_t accumulator,
    quant_scale_t activation_scale,
    quant_scale_t weight_scale) {
    #pragma HLS inline
    return dequantize_w8_accumulator(
        accumulator,
        combine_quantized_dequant_scales(activation_scale, weight_scale));
}

inline fm_t dequantize_w4_accumulator(
    quantized_w4_accum_t accumulator,
    quantized_dequant_scale_t combined_scale) {
    #pragma HLS inline
    const quantized_dequant_product_t value = accumulator * combined_scale;
    return fm_t(value);
}

inline fm_t dequantize_w4_accumulator(
    quantized_w4_accum_t accumulator,
    quant_scale_t activation_scale,
    quant_scale_t weight_scale) {
    #pragma HLS inline
    return dequantize_w4_accumulator(
        accumulator,
        combine_quantized_dequant_scales(activation_scale, weight_scale));
}

#endif
