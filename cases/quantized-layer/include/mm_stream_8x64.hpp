#ifndef LLM_FPGA_MM_STREAM_8X64_HPP
#define LLM_FPGA_MM_STREAM_8X64_HPP

#include "compute_stream.hpp"

constexpr unsigned int MM_STREAM_8X64_TOKENS = 8;
constexpr unsigned int MM_STREAM_8X64_OUTPUTS = 64;
constexpr unsigned int MM_STREAM_8X64_WEIGHT_GROUPS =
    MM_STREAM_8X64_OUTPUTS / CU_VEC_LANES;
constexpr unsigned int MM_STREAM_8X64_ACCUM_BANKS = 4;
constexpr unsigned int MM_STREAM_8X64_PACKETS_PER_BLOCK =
    MM_STREAM_8X64_TOKENS * MM_STREAM_8X64_WEIGHT_GROUPS;

static_assert(MM_STREAM_8X64_WEIGHT_GROUPS == 4, "8x64 core expects four 16-lane weight streams");

struct mm_stream_8x64_task_t {
    unsigned int k_count;
    unsigned int elem_base;
    unsigned int block_id;
    bool last_stream;
};

struct mm_stream_8x64_activation_packet_t {
    fm_t data[MM_STREAM_8X64_TOKENS];
};

struct mm_stream_8x64_weight_packet_t {
    wt_linear_t data[CU_VEC_LANES];
};

void compute_mm_stream_8x64_engine(
    hls::stream<cu_accum16_packet_t>& out_stream,
    const mm_stream_8x64_task_t& task,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3
);

void compute_mm_stream_8x64_fused_mac_engine(
    hls::stream<cu_accum16_packet_t>& out_stream,
    const mm_stream_8x64_task_t& task,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3
);

void compute_mm_stream_8x64_core(
    hls::stream<cu_accum16_packet_t>& out_stream,
    hls::stream<mm_stream_8x64_task_t>& task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3
);

void compute_mm_stream_8x64_rotating_core(
    hls::stream<cu_accum16_packet_t>& out_stream,
    hls::stream<mm_stream_8x64_task_t>& task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3
);

void compute_mm_stream_8x64_rotating_flat_emit_core(
    hls::stream<cu_accum16_packet_t>& out_stream,
    hls::stream<mm_stream_8x64_task_t>& task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3
);

void compute_mm_stream_8x64_fused_mac_core(
    hls::stream<cu_accum16_packet_t>& out_stream,
    hls::stream<mm_stream_8x64_task_t>& task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3
);

#endif
