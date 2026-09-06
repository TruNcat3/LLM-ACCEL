#ifndef LLM_FPGA_MM_STREAM_8X64_INT4X4_BLOCK_HPP
#define LLM_FPGA_MM_STREAM_8X64_INT4X4_BLOCK_HPP

#include "mm_stream_quantized_nk.hpp"

#include <hls_stream.h>

constexpr unsigned int MM_STREAM_8X64_INT4X4_TOKENS = 8;
constexpr unsigned int MM_STREAM_8X64_INT4X4_OUTPUTS = 64;
constexpr unsigned int MM_STREAM_8X64_INT4X4_TOKEN_PAIRS = 4;
constexpr unsigned int MM_STREAM_8X64_INT4X4_OUTPUT_PAIRS = 32;
constexpr unsigned int MM_STREAM_8X64_INT4X4_PHYSICAL_DSPS =
    MM_STREAM_8X64_INT4X4_TOKEN_PAIRS *
    MM_STREAM_8X64_INT4X4_OUTPUT_PAIRS;
constexpr unsigned int MM_STREAM_8X64_INT4X4_LOGICAL_PRODUCTS_PER_K =
    MM_STREAM_8X64_INT4X4_TOKENS * MM_STREAM_8X64_INT4X4_OUTPUTS;
constexpr unsigned int MM_STREAM_8X64_INT4X4_PACKING_FACTOR = 4;
constexpr unsigned int MM_STREAM_8X64_INT4X4_ACCUM_BITS = 24;
constexpr unsigned int MM_STREAM_8X64_INT4X4_GROUPS = 4;
constexpr unsigned int MM_STREAM_8X64_INT4X4_LANES_PER_GROUP = 16;

using mm_stream_8x64_int4x4_activation_word_t = ap_uint<32>;
using mm_stream_8x64_int4x4_weight_word_t = ap_uint<256>;
using mm_stream_8x64_int4x4_output_word_t = ap_uint<448>;

static_assert(MM_STREAM_8X64_INT4X4_PHYSICAL_DSPS == 128,
              "W4A4 8x64 needs 128 four-product DSP lanes");
static_assert(MM_STREAM_8X64_INT4X4_LOGICAL_PRODUCTS_PER_K ==
                  MM_STREAM_8X64_INT4X4_PHYSICAL_DSPS *
                      MM_STREAM_8X64_INT4X4_PACKING_FACTOR,
              "physical and logical W4A4 widths must agree");

inline ap_int<4> unpack_mm_stream_8x64_int4x4_activation(
    const mm_stream_8x64_int4x4_activation_word_t& word,
    unsigned int token) {
    #pragma HLS inline
    return ap_int<4>(word.range(token * 4 + 3, token * 4));
}

inline ap_int<4> unpack_mm_stream_8x64_int4x4_weight(
    const mm_stream_8x64_int4x4_weight_word_t& word,
    unsigned int output) {
    #pragma HLS inline
    return ap_int<4>(word.range(output * 4 + 3, output * 4));
}

inline ap_int<MM_STREAM_8X64_INT4X4_ACCUM_BITS>
unpack_mm_stream_8x64_int4x4_output(
    const mm_stream_8x64_int4x4_output_word_t& word,
    unsigned int lane) {
    #pragma HLS inline
    const unsigned int low = lane * MM_STREAM_8X64_INT4X4_ACCUM_BITS;
    return ap_int<MM_STREAM_8X64_INT4X4_ACCUM_BITS>(
        word.range(low + MM_STREAM_8X64_INT4X4_ACCUM_BITS - 1, low));
}

void compute_mm_stream_8x64_int4x4_block_nk(
    hls::stream<mm_stream_8x64_int4x4_output_word_t>& out_stream,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream,
    hls::stream<mm_stream_8x64_int4x4_activation_word_t>& activation_stream,
    hls::stream<mm_stream_8x64_int4x4_weight_word_t>& weight_stream);

#endif
