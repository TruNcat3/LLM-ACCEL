#ifndef LLM_FPGA_MM_STREAM_8X64_INT4_PACKED_BLOCK_HPP
#define LLM_FPGA_MM_STREAM_8X64_INT4_PACKED_BLOCK_HPP

#include "datatypes.hpp"
#include <ap_int.h>
#include <hls_stream.h>

// A matrix-shaped INT4 block contract. One physical lane carries one token
// activation and two adjacent output weights. The full 8x64 logical tile is
// covered by one or more waves, decoupling tile parallelism from DSP count.
#ifndef MM_STREAM_8X64_INT4_PACKED_PHYSICAL_LANES_CONFIG
#define MM_STREAM_8X64_INT4_PACKED_PHYSICAL_LANES_CONFIG 256
#endif

constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_TOKENS = 8;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_OUTPUTS = 64;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_FACTOR = 2;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_PHYSICAL_LANES =
    MM_STREAM_8X64_INT4_PACKED_PHYSICAL_LANES_CONFIG;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_TILE_LOGICAL_PRODUCTS =
    MM_STREAM_8X64_INT4_PACKED_TOKENS *
    MM_STREAM_8X64_INT4_PACKED_OUTPUTS;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_LOGICAL_PRODUCTS_PER_WAVE =
    MM_STREAM_8X64_INT4_PACKED_PHYSICAL_LANES *
    MM_STREAM_8X64_INT4_PACKED_FACTOR;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_WAVES_PER_K =
    MM_STREAM_8X64_INT4_PACKED_TILE_LOGICAL_PRODUCTS /
    MM_STREAM_8X64_INT4_PACKED_LOGICAL_PRODUCTS_PER_WAVE;
// Compatibility name: this is the logical work of the complete tile, not
// the number of products physically issued in one wave.
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_LOGICAL_PRODUCTS =
    MM_STREAM_8X64_INT4_PACKED_TILE_LOGICAL_PRODUCTS;
// A signed 24-bit tile value covers the worst-case 4096-term INT8xINT4
// accumulation (including the INT4 zero-point correction) while keeping the
// text-based HLS co-simulation record below its 4096-byte line limit.
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_OUTPUT_ACCUM_BITS = 24;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_OUTPUT_DATA_BITS =
    MM_STREAM_8X64_INT4_PACKED_TOKENS * MM_STREAM_8X64_INT4_PACKED_OUTPUTS *
    MM_STREAM_8X64_INT4_PACKED_OUTPUT_ACCUM_BITS;
// Keep each direct Vitis AXIS connection below the system-link width limit.
// Eight 1536-bit words preserve the 12288-bit logical tile and keep every
// output value wholly inside one word. 1536-bit direct AXIS links are accepted
// by the Vitis system linker while remaining aligned to the 24-bit values.
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_OUTPUT_WORD_BITS = 1536;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_OUTPUT_WORDS =
    (MM_STREAM_8X64_INT4_PACKED_OUTPUT_DATA_BITS +
     MM_STREAM_8X64_INT4_PACKED_OUTPUT_WORD_BITS - 1) /
    MM_STREAM_8X64_INT4_PACKED_OUTPUT_WORD_BITS;

static_assert(MM_STREAM_8X64_INT4_PACKED_PHYSICAL_LANES > 0,
              "at least one packed physical lane is required");
static_assert(MM_STREAM_8X64_INT4_PACKED_FACTOR == 2,
              "the block contract maps two INT4 weights per physical lane");
static_assert(
    MM_STREAM_8X64_INT4_PACKED_TILE_LOGICAL_PRODUCTS %
            MM_STREAM_8X64_INT4_PACKED_LOGICAL_PRODUCTS_PER_WAVE ==
        0,
    "physical lane count must divide the complete 8x64 logical tile");
static_assert(MM_STREAM_8X64_INT4_PACKED_OUTPUT_DATA_BITS %
                      MM_STREAM_8X64_INT4_PACKED_OUTPUT_WORD_BITS ==
                  0,
              "output words must divide the complete packed tile");

struct mm_stream_8x64_int4_packed_block_task_t {
    unsigned int k_count;
    unsigned int elem_base;
    unsigned int block_id;
    bool last_stream;
    fm_t activation_scale;
    fm_t weight_scale;
    ap_int<4> weight_zero_point;
};

struct mm_stream_8x64_int4_packed_block_input_t {
    ap_int<8> activation[MM_STREAM_8X64_INT4_PACKED_PHYSICAL_LANES];
    ap_int<4> weight0[MM_STREAM_8X64_INT4_PACKED_PHYSICAL_LANES];
    ap_int<4> weight1[MM_STREAM_8X64_INT4_PACKED_PHYSICAL_LANES];
};

// Integer tile output keeps the block scales off the K-cycle critical path.
// A following converter applies the metadata once per output. The data words
// are separate streams because Vitis system-link only supports bounded AXIS
// widths for direct kernel connections.
typedef ap_uint<MM_STREAM_8X64_INT4_PACKED_OUTPUT_WORD_BITS>
    mm_stream_8x64_int4_packed_block_word_t;

constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_INPUT_ACTIVATION_BITS =
    MM_STREAM_8X64_INT4_PACKED_PHYSICAL_LANES * 8;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_INPUT_WEIGHT_BITS =
    MM_STREAM_8X64_INT4_PACKED_PHYSICAL_LANES * 4;
typedef ap_uint<MM_STREAM_8X64_INT4_PACKED_INPUT_ACTIVATION_BITS>
    mm_stream_8x64_int4_packed_block_activation_word_t;
typedef ap_uint<MM_STREAM_8X64_INT4_PACKED_INPUT_WEIGHT_BITS>
    mm_stream_8x64_int4_packed_block_weight_word_t;
constexpr unsigned int MM_STREAM_8X64_INT4_PACKED_OUTPUT_VALUES_PER_WORD =
    MM_STREAM_8X64_INT4_PACKED_OUTPUT_WORD_BITS /
    MM_STREAM_8X64_INT4_PACKED_OUTPUT_ACCUM_BITS;

struct mm_stream_8x64_int4_packed_block_output_meta_t {
    unsigned int elem_base;
    unsigned int block_id;
    bool last_block;
    bool last_stream;
    fm_t activation_scale;
    fm_t weight_scale;
    ap_int<4> weight_zero_point;
};

inline void set_mm_stream_8x64_int4_packed_block_output(
    mm_stream_8x64_int4_packed_block_word_t& output, unsigned int token,
    unsigned int out,
    ap_int<MM_STREAM_8X64_INT4_PACKED_OUTPUT_ACCUM_BITS> value) {
    #pragma HLS inline
    const unsigned int index = token * MM_STREAM_8X64_INT4_PACKED_OUTPUTS + out;
    const unsigned int local_index =
        index % MM_STREAM_8X64_INT4_PACKED_OUTPUT_VALUES_PER_WORD;
    const unsigned int low =
        local_index * MM_STREAM_8X64_INT4_PACKED_OUTPUT_ACCUM_BITS;
    output.range(
        low + MM_STREAM_8X64_INT4_PACKED_OUTPUT_ACCUM_BITS - 1, low) = value;
}

inline ap_int<MM_STREAM_8X64_INT4_PACKED_OUTPUT_ACCUM_BITS>
get_mm_stream_8x64_int4_packed_block_output(
    const mm_stream_8x64_int4_packed_block_word_t& output,
    unsigned int token, unsigned int out) {
    #pragma HLS inline
    const unsigned int index = token * MM_STREAM_8X64_INT4_PACKED_OUTPUTS + out;
    const unsigned int local_index =
        index % MM_STREAM_8X64_INT4_PACKED_OUTPUT_VALUES_PER_WORD;
    const unsigned int low =
        local_index * MM_STREAM_8X64_INT4_PACKED_OUTPUT_ACCUM_BITS;
    return ap_int<MM_STREAM_8X64_INT4_PACKED_OUTPUT_ACCUM_BITS>(
        output.range(
            low + MM_STREAM_8X64_INT4_PACKED_OUTPUT_ACCUM_BITS - 1, low));
}

void compute_mm_stream_8x64_int4_packed_block(
    hls::stream<mm_stream_8x64_int4_packed_block_word_t>& out_stream0,
    hls::stream<mm_stream_8x64_int4_packed_block_word_t>& out_stream1,
    hls::stream<mm_stream_8x64_int4_packed_block_word_t>& out_stream2,
    hls::stream<mm_stream_8x64_int4_packed_block_word_t>& out_stream3,
    hls::stream<mm_stream_8x64_int4_packed_block_word_t>& out_stream4,
    hls::stream<mm_stream_8x64_int4_packed_block_word_t>& out_stream5,
    hls::stream<mm_stream_8x64_int4_packed_block_word_t>& out_stream6,
    hls::stream<mm_stream_8x64_int4_packed_block_word_t>& out_stream7,
    hls::stream<mm_stream_8x64_int4_packed_block_output_meta_t>& meta_stream,
    hls::stream<mm_stream_8x64_int4_packed_block_task_t>& task_stream,
    hls::stream<mm_stream_8x64_int4_packed_block_activation_word_t>&
        activation_stream,
    hls::stream<mm_stream_8x64_int4_packed_block_weight_word_t>& weight0_stream,
    hls::stream<mm_stream_8x64_int4_packed_block_weight_word_t>& weight1_stream);

#endif
