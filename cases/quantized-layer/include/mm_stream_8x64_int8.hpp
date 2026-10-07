#ifndef LLM_FPGA_MM_STREAM_8X64_INT8_HPP
#define LLM_FPGA_MM_STREAM_8X64_INT8_HPP

#include "compute_stream.hpp"

// Isolated INT8 weight-only A/B engine.  The activation and accumulation
// formats intentionally remain identical to the production Fix16 path.
#ifndef MM_STREAM_8X64_INT8_TOKENS_CONFIG
#define MM_STREAM_8X64_INT8_TOKENS_CONFIG 8
#endif
constexpr unsigned int MM_STREAM_8X64_INT8_TOKENS =
    MM_STREAM_8X64_INT8_TOKENS_CONFIG;
constexpr unsigned int MM_STREAM_8X64_INT8_OUTPUTS = 64;
constexpr unsigned int MM_STREAM_8X64_INT8_WEIGHT_GROUPS =
    MM_STREAM_8X64_INT8_OUTPUTS / CU_VEC_LANES;
constexpr unsigned int MM_STREAM_8X64_INT8_ACCUM_BANKS = 4;
constexpr unsigned int MM_STREAM_8X64_INT8_PACKETS_PER_BLOCK =
    MM_STREAM_8X64_INT8_TOKENS * MM_STREAM_8X64_INT8_WEIGHT_GROUPS;

static_assert(MM_STREAM_8X64_INT8_WEIGHT_GROUPS == 4,
              "8x64 INT8 core expects four 16-lane weight streams");

struct mm_stream_8x64_int8_task_t {
    unsigned int k_count;
    unsigned int elem_base;
    unsigned int block_id;
    bool last_stream;
};

struct mm_stream_8x64_int8_activation_packet_t {
    fm_t data[MM_STREAM_8X64_INT8_TOKENS];
};

struct mm_stream_8x64_int8_weight_packet_t {
    ap_int<8> data[CU_VEC_LANES];
};

void compute_mm_stream_8x64_int8_core(
    hls::stream<cu_accum16_packet_t>& out_stream,
    hls::stream<mm_stream_8x64_int8_task_t>& task_stream,
    hls::stream<mm_stream_8x64_int8_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_int8_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_int8_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_int8_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_int8_weight_packet_t>& weight_stream3
);

#endif
