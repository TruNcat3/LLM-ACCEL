#ifndef LLM_FPGA_QUANTIZED_DECODE_ROWS_HPP
#define LLM_FPGA_QUANTIZED_DECODE_ROWS_HPP

#include "mm_stream_quantized_nk.hpp"
#include <hls_stream.h>

// Experimental component ABI, not interchangeable with the production XO.
// Task bit 126 selects a single-query, output-major Decode mapping.
constexpr unsigned int QDR_MODE_BIT = 126;
constexpr unsigned int QDR_PREFILL_COLS = 128;
constexpr unsigned int QDR_WEIGHT_BITS = 4096;
#ifndef QDR_INGRESS_BITS
#define QDR_INGRESS_BITS 4096
#endif
#ifndef QDR_COMPACT
#define QDR_COMPACT 0
#endif
#ifndef QDR_BLOCK_OPS
#define QDR_BLOCK_OPS 0
#endif
#ifndef QDR_SHARED_STATE
#define QDR_SHARED_STATE 0
#endif
#ifndef QDR_DIRECT_WIDE
#define QDR_DIRECT_WIDE 0
#endif
static_assert(QDR_DIRECT_WIDE == 0 || QDR_DIRECT_WIDE == 1,
              "QDR_DIRECT_WIDE must be 0 or 1");
static_assert(!QDR_DIRECT_WIDE || QDR_COMPACT,
              "QDR_DIRECT_WIDE requires QDR_COMPACT");
static_assert(QDR_SHARED_STATE == 0 || QDR_SHARED_STATE == 1,
              "QDR_SHARED_STATE must be 0 or 1");
static_assert(!QDR_SHARED_STATE || QDR_COMPACT,
              "QDR_SHARED_STATE requires QDR_COMPACT");
static_assert(QDR_BLOCK_OPS == 0 || QDR_BLOCK_OPS == 1,
              "QDR_BLOCK_OPS must be 0 or 1");
static_assert(QDR_COMPACT == 0 || QDR_COMPACT == 1,
              "QDR_COMPACT must be 0 or 1");
static_assert(QDR_INGRESS_BITS == 512 || QDR_INGRESS_BITS == 1024 ||
              QDR_INGRESS_BITS == 2048 || QDR_INGRESS_BITS == 4096,
              "weight ingress must contain whole 512-bit segments");
#ifdef QUANTIZED_ALIGNMENT_W8
constexpr unsigned int QDR_BITS = 8;
constexpr unsigned int QDR_ROWS = 4;
constexpr unsigned int QDR_DECODE_COLS = 512;
constexpr unsigned int QDR_DSPS = 512;
constexpr unsigned int QDR_ACCUM_BITS = 32;
using qdr_output_word_t = ap_uint<576>;
#else
constexpr unsigned int QDR_BITS = 4;
constexpr unsigned int QDR_ROWS = 8;
constexpr unsigned int QDR_DECODE_COLS = 1024;
constexpr unsigned int QDR_DSPS = 256;
constexpr unsigned int QDR_ACCUM_BITS = 24;
using qdr_output_word_t = ap_uint<448>;
#endif
using qdr_activation_word_t = ap_uint<32>;
using qdr_weight_word_t = ap_uint<QDR_INGRESS_BITS>;

// An INT4 magnitude is at most 8, so its product needs seven unsigned bits.
// Four weights spaced by seven bits fit in 25 bits on the DSP multiplier A
// input; the common activation fits on its B input. Unlike the old 2x2 outer
// product, all four products are useful for a single query.
inline ap_uint<4> qdr_abs4(ap_int<4> value) {
    #pragma HLS inline
    ap_int<5> extended = value;
    return extended < 0 ? ap_uint<4>(-extended) : ap_uint<4>(extended);
}

struct qdr_w4_quad_t { ap_int<8> value[4]; };

template <unsigned int Lane>
inline ap_int<8> qdr_w4_product_lane(ap_uint<32> product,
                                      bool decode, bool negative) {
    #pragma HLS inline
    // Fixed slices are intentional: no array index or run-time shift enters
    // the four-product transport path.
    const ap_uint<7> magnitude = decode ?
        ap_uint<7>(product.range(Lane * 7 + 6, Lane * 7)) :
        ap_uint<7>(product.range(Lane * 8 + 6, Lane * 8));
    const ap_int<8> signed_magnitude = magnitude;
    return negative ? ap_int<8>(-signed_magnitude) : signed_magnitude;
}

inline ap_uint<32> qdr_w4_products_from_magnitudes(
    bool decode, ap_uint<8> activation_magnitudes, ap_uint<2> activation_signs,
    ap_uint<16> weight_magnitudes, ap_uint<4> weight_signs) {
    #pragma HLS inline
    ap_uint<25> operand_a = 0;
    ap_uint<12> operand_b = 0;
    if (decode) {
        operand_a.range(3, 0) = weight_magnitudes.range(3, 0);
        operand_a.range(10, 7) = weight_magnitudes.range(7, 4);
        operand_a.range(17, 14) = weight_magnitudes.range(11, 8);
        operand_a.range(24, 21) = weight_magnitudes.range(15, 12);
        operand_b.range(3, 0) = activation_magnitudes.range(3, 0);
    } else {
        operand_a.range(3, 0) = activation_magnitudes.range(3, 0);
        operand_a.range(19, 16) = activation_magnitudes.range(7, 4);
        operand_b.range(3, 0) = weight_magnitudes.range(3, 0);
        operand_b.range(11, 8) = weight_magnitudes.range(7, 4);
    }
    // Valid P operands produce less than 2^31; valid D operands less than
    // 2^28. Keep the 25x12 multiplier but discard its unused high result bits
    // immediately, before the four-lane extraction/fanout.
    ap_uint<32> product;
    #pragma HLS bind_op variable=product op=mul impl=dsp
    product = operand_a * operand_b;
    ap_uint<32> result;
    result.range(7, 0) = qdr_w4_product_lane<0>(product, decode, activation_signs[0] != weight_signs[0]);
    result.range(15, 8) = qdr_w4_product_lane<1>(product, decode, activation_signs[0] != weight_signs[1]);
    result.range(23, 16) = qdr_w4_product_lane<2>(
        product, decode, decode ? activation_signs[0] != weight_signs[2] : activation_signs[1] != weight_signs[0]);
    result.range(31, 24) = qdr_w4_product_lane<3>(
        product, decode, decode ? activation_signs[0] != weight_signs[3] : activation_signs[1] != weight_signs[1]);
    return result;
}

inline ap_uint<32> qdr_w4_products_block(
    bool decode, ap_int<4> a0, ap_int<4> a1,
    ap_int<4> w0, ap_int<4> w1, ap_int<4> w2, ap_int<4> w3) {
    #pragma HLS inline
    ap_uint<8> activation_magnitudes;
    activation_magnitudes.range(3, 0) = qdr_abs4(a0);
    activation_magnitudes.range(7, 4) = qdr_abs4(a1);
    ap_uint<2> activation_signs;
    activation_signs[0] = a0[3]; activation_signs[1] = a1[3];
    ap_uint<16> weight_magnitudes;
    weight_magnitudes.range(3, 0) = qdr_abs4(w0);
    weight_magnitudes.range(7, 4) = qdr_abs4(w1);
    weight_magnitudes.range(11, 8) = qdr_abs4(w2);
    weight_magnitudes.range(15, 12) = qdr_abs4(w3);
    ap_uint<4> weight_signs;
    weight_signs[0] = w0[3]; weight_signs[1] = w1[3];
    weight_signs[2] = w2[3]; weight_signs[3] = w3[3];
    return qdr_w4_products_from_magnitudes(decode, activation_magnitudes,
        activation_signs, weight_magnitudes, weight_signs);
}

inline qdr_w4_quad_t qdr_w4_products(
    bool decode, ap_int<4> a0, ap_int<4> a1,
    ap_int<4> w0, ap_int<4> w1, ap_int<4> w2, ap_int<4> w3) {
    #pragma HLS inline
    ap_uint<25> operand_a = 0;
    ap_uint<12> operand_b = 0;
    if (decode) {
        operand_a.range(3, 0) = qdr_abs4(w0);
        operand_a.range(10, 7) = qdr_abs4(w1);
        operand_a.range(17, 14) = qdr_abs4(w2);
        operand_a.range(24, 21) = qdr_abs4(w3);
        operand_b.range(3, 0) = qdr_abs4(a0);
    } else {
        operand_a.range(3, 0) = qdr_abs4(a0);
        operand_a.range(19, 16) = qdr_abs4(a1);
        operand_b.range(3, 0) = qdr_abs4(w0);
        operand_b.range(11, 8) = qdr_abs4(w1);
    }
    // One multiply call site serves both modes. HLS resource reports must
    // establish whether the requested 25x12 product uses one physical DSP.
    ap_uint<37> packed_product;
    #pragma HLS bind_op variable=packed_product op=mul impl=dsp
    packed_product = operand_a * operand_b;
    qdr_w4_quad_t result;
    #pragma HLS array_partition variable=result.value complete
    const ap_int<4> weights[4] = {w0, w1, w2, w3};
    #pragma HLS array_partition variable=weights complete
    for (unsigned int p = 0; p < 4; ++p) {
        #pragma HLS unroll
        const ap_int<4> activation = decode || p < 2 ? a0 : a1;
        const ap_int<4> weight = decode ? weights[p] : weights[p % 2];
        ap_int<8> magnitude = decode ?
            ap_int<8>(ap_uint<7>(packed_product.range(p * 7 + 6, p * 7))) :
            ap_int<8>(ap_uint<8>(packed_product.range(p * 8 + 7, p * 8)));
        result.value[p] = (activation < 0) != (weight < 0) ?
            ap_int<8>(-magnitude) : magnitude;
    }
    return result;
}

void compute_quantized_decode_rows(
    hls::stream<qdr_output_word_t>& output,
    hls::stream<mm_stream_quantized_task_word_t>& tasks,
    hls::stream<qdr_activation_word_t>& activations,
    hls::stream<qdr_weight_word_t>& weights,
    unsigned int task_count = 1);

#endif
