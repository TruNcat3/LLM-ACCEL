#ifndef LLM_FPGA_MM_STREAM_4X128_INT8X8_BLOCK_HPP
#define LLM_FPGA_MM_STREAM_4X128_INT8X8_BLOCK_HPP

#include "mm_stream_quantized_nk.hpp"

#include <hls_stream.h>

constexpr unsigned int MM_STREAM_4X128_INT8X8_TOKENS = 4;
constexpr unsigned int MM_STREAM_4X128_INT8X8_OUTPUTS = 128;
constexpr unsigned int MM_STREAM_4X128_INT8X8_WEIGHT_STREAMS = 4;
constexpr unsigned int MM_STREAM_4X128_INT8X8_WEIGHTS_PER_STREAM = 32;
constexpr unsigned int MM_STREAM_4X128_INT8X8_OUTPUT_GROUPS = 8;
constexpr unsigned int MM_STREAM_4X128_INT8X8_LANES_PER_GROUP = 16;
constexpr unsigned int MM_STREAM_4X128_INT8X8_PHYSICAL_DSPS =
    MM_STREAM_4X128_INT8X8_TOKENS * MM_STREAM_4X128_INT8X8_OUTPUTS;
constexpr unsigned int MM_STREAM_4X128_INT8X8_LOGICAL_PRODUCTS_PER_K =
    MM_STREAM_4X128_INT8X8_PHYSICAL_DSPS;
constexpr unsigned int MM_STREAM_4X128_INT8X8_ACCUM_BITS = 32;

using mm_stream_4x128_int8x8_activation_word_t = ap_uint<32>;
using mm_stream_4x128_int8x8_weight_word_t = ap_uint<256>;
using mm_stream_4x128_int8x8_output_word_t = ap_uint<576>;

static_assert(MM_STREAM_4X128_INT8X8_PHYSICAL_DSPS == 512,
              "W8A8 4x128 exposes 512 physical products per K");
static_assert(MM_STREAM_4X128_INT8X8_WEIGHT_STREAMS *
                      MM_STREAM_4X128_INT8X8_WEIGHTS_PER_STREAM ==
                  MM_STREAM_4X128_INT8X8_OUTPUTS,
              "four block streams must cover all 128 output weights");

inline ap_int<8> unpack_mm_stream_4x128_int8x8_activation(
    const mm_stream_4x128_int8x8_activation_word_t& word,
    unsigned int token) {
    #pragma HLS inline
    return ap_int<8>(word.range(token * 8 + 7, token * 8));
}

inline ap_int<MM_STREAM_4X128_INT8X8_ACCUM_BITS>
unpack_mm_stream_4x128_int8x8_output(
    const mm_stream_4x128_int8x8_output_word_t& word,
    unsigned int lane) {
    #pragma HLS inline
    const unsigned int low = lane * MM_STREAM_4X128_INT8X8_ACCUM_BITS;
    return ap_int<MM_STREAM_4X128_INT8X8_ACCUM_BITS>(
        word.range(low + MM_STREAM_4X128_INT8X8_ACCUM_BITS - 1, low));
}

void compute_mm_stream_4x128_int8x8_block_nk(
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& out_stream,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream,
    hls::stream<mm_stream_4x128_int8x8_activation_word_t>& activation_stream,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream0,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream1,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream2,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream3,
    unsigned int task_count = 1);

#endif
