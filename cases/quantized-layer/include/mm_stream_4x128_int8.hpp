#ifndef LLM_FPGA_MM_STREAM_4X128_INT8_HPP
#define LLM_FPGA_MM_STREAM_4X128_INT8_HPP

#include "compute_stream.hpp"

// Decode-oriented INT8 probe: four token rows and eight output groups.  The
// logical arithmetic width remains 512 products per K step, while the packet
// shape halves token waves and doubles the output span relative to 4x64.
constexpr unsigned int MM_STREAM_4X128_INT8_TOKENS = 4;
constexpr unsigned int MM_STREAM_4X128_INT8_OUTPUTS = 128;
constexpr unsigned int MM_STREAM_4X128_INT8_WEIGHT_GROUPS =
    MM_STREAM_4X128_INT8_OUTPUTS / CU_VEC_LANES;
constexpr unsigned int MM_STREAM_4X128_INT8_ACCUM_BANKS = 4;
constexpr unsigned int MM_STREAM_4X128_INT8_PRODUCTS_PER_CYCLE =
    MM_STREAM_4X128_INT8_TOKENS * MM_STREAM_4X128_INT8_OUTPUTS;

static_assert(MM_STREAM_4X128_INT8_WEIGHT_GROUPS == 8,
              "4x128 INT8 core expects eight 16-lane weight streams");
static_assert(MM_STREAM_4X128_INT8_PRODUCTS_PER_CYCLE == 512,
              "4x128 INT8 core must preserve the 512-product arithmetic width");

struct mm_stream_4x128_int8_task_t {
    unsigned int k_count;
    unsigned int elem_base;
    unsigned int block_id;
    bool last_stream;
};

struct mm_stream_4x128_int8_activation_packet_t {
    fm_t data[MM_STREAM_4X128_INT8_TOKENS];
};

struct mm_stream_4x128_int8_weight_packet_t {
    ap_int<8> data[CU_VEC_LANES];
};

void compute_mm_stream_4x128_int8_core(
    hls::stream<cu_accum16_packet_t>& out_stream,
    hls::stream<mm_stream_4x128_int8_task_t>& task_stream,
    hls::stream<mm_stream_4x128_int8_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_4x128_int8_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_4x128_int8_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_4x128_int8_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_4x128_int8_weight_packet_t>& weight_stream3,
    hls::stream<mm_stream_4x128_int8_weight_packet_t>& weight_stream4,
    hls::stream<mm_stream_4x128_int8_weight_packet_t>& weight_stream5,
    hls::stream<mm_stream_4x128_int8_weight_packet_t>& weight_stream6,
    hls::stream<mm_stream_4x128_int8_weight_packet_t>& weight_stream7
);

#endif
