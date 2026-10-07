#ifndef LLM_FPGA_MM_STREAM_8X64_INT4_HPP
#define LLM_FPGA_MM_STREAM_8X64_INT4_HPP

#include "compute_stream.hpp"

// Isolated INT4 weight-only A/B engine. Activations and accumulators retain
// the production formats so only the linear weight representation changes.
#ifndef MM_STREAM_8X64_INT4_TOKENS_CONFIG
#define MM_STREAM_8X64_INT4_TOKENS_CONFIG 8
#endif
constexpr unsigned int MM_STREAM_8X64_INT4_TOKENS =
    MM_STREAM_8X64_INT4_TOKENS_CONFIG;
constexpr unsigned int MM_STREAM_8X64_INT4_OUTPUTS = 64;
constexpr unsigned int MM_STREAM_8X64_INT4_WEIGHT_GROUPS =
    MM_STREAM_8X64_INT4_OUTPUTS / CU_VEC_LANES;
constexpr unsigned int MM_STREAM_8X64_INT4_ACCUM_BANKS = 4;

static_assert(MM_STREAM_8X64_INT4_WEIGHT_GROUPS == 4,
              "8x64 INT4 core expects four 16-lane weight streams");

struct mm_stream_8x64_int4_task_t {
    unsigned int k_count;
    unsigned int elem_base;
    unsigned int block_id;
    bool last_stream;
};

struct mm_stream_8x64_int4_activation_packet_t {
    fm_t data[MM_STREAM_8X64_INT4_TOKENS];
};

struct mm_stream_8x64_int4_weight_packet_t {
    ap_int<4> data[CU_VEC_LANES];
};

void compute_mm_stream_8x64_int4_core(
    hls::stream<cu_accum16_packet_t>& out_stream,
    hls::stream<mm_stream_8x64_int4_task_t>& task_stream,
    hls::stream<mm_stream_8x64_int4_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_int4_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_int4_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_int4_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_int4_weight_packet_t>& weight_stream3
);

#endif
