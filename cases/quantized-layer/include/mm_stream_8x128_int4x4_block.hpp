#ifndef LLM_FPGA_MM_STREAM_8X128_INT4X4_BLOCK_HPP
#define LLM_FPGA_MM_STREAM_8X128_INT4X4_BLOCK_HPP

#include "mm_stream_quantized_nk.hpp"

#include <hls_stream.h>

constexpr unsigned int MM_STREAM_8X128_INT4X4_TOKENS = 8;
constexpr unsigned int MM_STREAM_8X128_INT4X4_OUTPUTS = 128;
constexpr unsigned int MM_STREAM_8X128_INT4X4_TOKEN_PAIRS = 4;
#ifndef MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE_CONFIG
#define MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE_CONFIG 128
#endif
#ifndef MM_STREAM_8X128_INT4X4_REUSE_BANK_CONFIG
#define MM_STREAM_8X128_INT4X4_REUSE_BANK_CONFIG 0
#endif
#ifndef MM_STREAM_8X128_INT4X4_PACKED_ACCUM_CONFIG
#define MM_STREAM_8X128_INT4X4_PACKED_ACCUM_CONFIG 0
#endif
constexpr unsigned int MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE =
    MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE_CONFIG;
constexpr unsigned int MM_STREAM_8X128_INT4X4_OUTPUT_PAIRS =
    MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE / 2;
constexpr unsigned int MM_STREAM_8X128_INT4X4_WAVES_PER_K =
    MM_STREAM_8X128_INT4X4_OUTPUTS /
    MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE;
constexpr unsigned int MM_STREAM_8X128_INT4X4_WEIGHTS_PER_STREAM =
    MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE / 2;
constexpr unsigned int MM_STREAM_8X128_INT4X4_PHYSICAL_DSPS =
    MM_STREAM_8X128_INT4X4_TOKEN_PAIRS *
    MM_STREAM_8X128_INT4X4_OUTPUT_PAIRS;
constexpr unsigned int MM_STREAM_8X128_INT4X4_LOGICAL_PRODUCTS_PER_K =
    MM_STREAM_8X128_INT4X4_TOKENS * MM_STREAM_8X128_INT4X4_OUTPUTS;
constexpr unsigned int MM_STREAM_8X128_INT4X4_PACKING_FACTOR = 4;
constexpr unsigned int MM_STREAM_8X128_INT4X4_ACCUM_BITS = 24;
constexpr unsigned int MM_STREAM_8X128_INT4X4_GROUPS = 8;
constexpr unsigned int MM_STREAM_8X128_INT4X4_OUTPUT_GROUPS =
    MM_STREAM_8X128_INT4X4_GROUPS;
constexpr unsigned int MM_STREAM_8X128_INT4X4_LANES_PER_GROUP = 16;

using mm_stream_8x128_int4x4_activation_word_t = ap_uint<32>;
using mm_stream_8x128_int4x4_weight_word_t = ap_uint<
    MM_STREAM_8X128_INT4X4_WEIGHTS_PER_STREAM * 4>;
using mm_stream_8x128_int4x4_output_word_t = ap_uint<448>;

static_assert(MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE == 64 ||
                  MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE == 128,
              "W4A4 8x128 supports 64- or 128-output waves");
static_assert(MM_STREAM_8X128_INT4X4_REUSE_BANK_CONFIG == 0 ||
                  MM_STREAM_8X128_INT4X4_REUSE_BANK_CONFIG == 1,
              "bank reuse must be disabled or enabled");
static_assert(MM_STREAM_8X128_INT4X4_PACKED_ACCUM_CONFIG == 0 ||
                  MM_STREAM_8X128_INT4X4_PACKED_ACCUM_CONFIG == 1,
              "packed accumulator must be disabled or enabled");
static_assert(MM_STREAM_8X128_INT4X4_OUTPUTS %
                  MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE == 0,
              "wave width must divide the complete output tile");
static_assert(MM_STREAM_8X128_INT4X4_LOGICAL_PRODUCTS_PER_K ==
                  MM_STREAM_8X128_INT4X4_PHYSICAL_DSPS *
                      MM_STREAM_8X128_INT4X4_PACKING_FACTOR *
                      MM_STREAM_8X128_INT4X4_WAVES_PER_K,
              "physical and logical W4A4 widths must agree");

inline ap_int<MM_STREAM_8X128_INT4X4_ACCUM_BITS>
unpack_mm_stream_8x128_int4x4_output(
    const mm_stream_8x128_int4x4_output_word_t& word,
    unsigned int lane) {
    #pragma HLS inline
    const unsigned int low = lane * MM_STREAM_8X128_INT4X4_ACCUM_BITS;
    return ap_int<MM_STREAM_8X128_INT4X4_ACCUM_BITS>(
        word.range(low + MM_STREAM_8X128_INT4X4_ACCUM_BITS - 1, low));
}

void compute_mm_stream_8x128_int4x4_block_nk(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activation_stream,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream0,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream1,
    unsigned int task_count = 1);

#endif
