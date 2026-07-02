#ifndef LLM_FPGA_COMPUTE_CORE_8X64_UNIFIED_HPP
#define LLM_FPGA_COMPUTE_CORE_8X64_UNIFIED_HPP

#include "mm_stream_8x64.hpp"

constexpr unsigned int CU8_MAX_TASKS_PER_LAUNCH = 256;
constexpr unsigned int CU8_MAX_MM_REPEATS = 256;

enum cu8_mode_t {
    CU8_MODE_MM = 0,
    CU8_MODE_MM_SILU = 1,
    CU8_MODE_MM_SCALE = 2,
    CU8_MODE_SILU = 3,
    CU8_MODE_SILU_MUL = 4,
    CU8_MODE_RMSNORM = 5,
    CU8_MODE_RESIDUAL_ADD = 6,
    CU8_MODE_SOFTMAX = 7,
    CU8_MODE_STOP = 8
};

struct cu8_task_t {
    cu8_mode_t mode;
    unsigned int k_count;
    unsigned int token_count;
    unsigned int elem_count;
    unsigned int packet_count;
    unsigned int elem_base;
    unsigned int block_id;
    unsigned int repeat_count;
    unsigned int elem_stride;
    unsigned int block_stride;
    fm_t output_scale;
    bool last_task;
};

void compute_core_8x64_unified(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu8_task_t>& task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3,
    hls::stream<cu_vec16_packet_t>& vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& vector_input1_stream
);

#endif
