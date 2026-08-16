#ifndef LLM_FPGA_COMPUTE_CORE_8X64_UNIFIED_HPP
#define LLM_FPGA_COMPUTE_CORE_8X64_UNIFIED_HPP

#include "mm_stream_8x64.hpp"
#include "stream_depth_config.hpp"

// A prefill-attention launch emits one QK task and HEAD_DIM/64 PV tasks for
// every GQA head and sequence tile.  This is a real RTL loop bound: keeping
// the legacy value of 256 after increasing MAX_SEQ_LEN causes the compute CU
// to stop reading its task stream before the controller emits last_task.
constexpr unsigned int CU8_ATTN_TASKS_PER_TILE =
    GQA_GROUP_SIZE *
    (1 + ceildiv_size(HEAD_DIM, MM_STREAM_8X64_OUTPUTS));
constexpr unsigned int CU8_MAX_ATTN_TASKS_PER_LAUNCH =
    ATTENTION_NUM_TILES * CU8_ATTN_TASKS_PER_TILE;
constexpr unsigned int CU8_MAX_TASKS_PER_LAUNCH =
    CU8_MAX_ATTN_TASKS_PER_LAUNCH > 256 ?
    CU8_MAX_ATTN_TASKS_PER_LAUNCH : 256;
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

enum cu8_result_policy_t {
    CU8_RESULT_RELEASE = 0,
    CU8_RESULT_HOLD = 1,
    CU8_RESULT_EMIT = 2,
    CU8_RESULT_BYPASS = 3
};

struct cu8_task_t {
    cu8_mode_t mode;
    cu8_result_policy_t result_policy;
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
