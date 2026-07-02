#ifndef LLM_FPGA_CONTROL_CACHE_8X64_HPP
#define LLM_FPGA_CONTROL_CACHE_8X64_HPP

#include "compute_core_8x64_unified.hpp"
#include "mm_controller.hpp"

constexpr unsigned int CC8_MM_CORE_COUNT = 2;
constexpr unsigned int CC8_OUTPUTS_PER_WAVE =
    CC8_MM_CORE_COUNT * MM_STREAM_8X64_OUTPUTS;
constexpr unsigned int CC8_OUTPUT_TILES_PER_WAVE =
    CC8_OUTPUTS_PER_WAVE / MM_PE_OUT;
constexpr unsigned int CC8_TOKENS_PER_DATA_PORT =
    ceildiv(LINEAR_TOKEN_TILE_ACTIVE, 2u);
constexpr unsigned int CC8_FEATURE_WORDS_PER_TOKEN =
    ceildiv(MAX_LINEAR_IN_DIM, FM_BLOCK_SIZE);
constexpr unsigned int CC8_FEATURE_WORDS_PER_PORT =
    CC8_TOKENS_PER_DATA_PORT * CC8_FEATURE_WORDS_PER_TOKEN;
constexpr unsigned int CC8_DATA_PORT_WORDS =
    CC8_FEATURE_WORDS_PER_PORT > CU_MV_CACHE_FM_WORDS ?
    CC8_FEATURE_WORDS_PER_PORT :
    CU_MV_CACHE_FM_WORDS;
constexpr unsigned int CC8_GBUF_BLOCKS = MAX_LINEAR_IN_BLOCKS;
constexpr unsigned int CC8_ATTN_TILE = CU_SOFTMAX_ELEMS;
constexpr unsigned int CC8_HEAD_WORDS =
    ceildiv(HEAD_DIM, FM_BLOCK_SIZE);
constexpr unsigned int CC8_ATTN_BUFFER_ELEMS =
    HEAD_DIM > CC8_ATTN_TILE ? HEAD_DIM : CC8_ATTN_TILE;
constexpr unsigned int CC8_ATTN_BUFFER_BLOCKS =
    ceildiv(CC8_ATTN_BUFFER_ELEMS, MM_PE_IN);
constexpr unsigned int CC8_ATTN_BUFFER_WORDS =
    ceildiv(CC8_ATTN_BUFFER_ELEMS, FM_BLOCK_SIZE);

static_assert(CC8_MM_CORE_COUNT == 2, "v1 control/cache core drives two MM cores");
static_assert(CC8_OUTPUTS_PER_WAVE == 128, "two 8x64 cores produce one 8x128 wave");
static_assert(FM_BLOCK_SIZE == 2 * CU_VEC_LANES, "one 512-bit word holds two vec16 packets");
static_assert(NUM_KEY_VALUE_HEADS == CC8_MM_CORE_COUNT, "two unified cores map the two Qwen KV groups");
static_assert(GQA_GROUP_SIZE <= MM_STREAM_8X64_TOKENS, "one core must hold one GQA group");

struct cc8_attention_panel_t {
    fm_word_t word[CC8_ATTN_TILE][CC8_HEAD_WORDS];
};

struct cc8_attention_buffer_t {
    mm_input_block_t block
        [MM_STREAM_8X64_TOKENS]
        [CC8_ATTN_BUFFER_BLOCKS];
};

enum cc8_operator_t {
    CC8_OP_NOP = 0,
    CC8_OP_Q_PROJECTION = 1,
    CC8_OP_K_PROJECTION = 2,
    CC8_OP_V_PROJECTION = 3,
    CC8_OP_O_PROJECTION = 4,
    CC8_OP_FFN_GATE = 5,
    CC8_OP_FFN_UP = 6,
    CC8_OP_FFN_DOWN = 7,
    CC8_OP_RMSNORM = 8,
    CC8_OP_SILU_MUL = 9,
    CC8_OP_RESIDUAL_ADD = 10,
    CC8_OP_ATTN_QK = 11,
    CC8_OP_ATTN_PV = 12,
    CC8_OP_SOFTMAX = 13
};

enum cc8_status_code_t {
    CC8_STATUS_OK = 0,
    CC8_STATUS_BAD_OPERATOR = 1,
    CC8_STATUS_BAD_TOKEN_COUNT = 2,
    CC8_STATUS_BAD_TILE_LENGTH = 3
};

struct cc8_operator_spec_t {
    bool uses_mm;
    bool uses_vector;
    mm_projection_kind_t projection;
    cu8_mode_t compute_mode;
    unsigned int in_dim;
    unsigned int out_dim;
    unsigned int vector_inputs;
};

struct cc8_status_packet_t {
    cc8_operator_t op;
    cc8_status_code_t status;
    unsigned int token_count;
    unsigned int output_waves;
    unsigned int dispatched_mm_tasks;
    unsigned int dispatched_vector_tasks;
    unsigned int completed_output_packets;
    bool last_task;
};

struct cc8_global_buffer_t {
    mm_input_block_t block[LINEAR_TOKEN_TILE_ACTIVE][CC8_GBUF_BLOCKS];
};

void cc8_status_sink(
    hls::stream<cc8_status_packet_t>& status_stream,
    fm_word_t status_output[1]
);

bool cc8_get_operator_spec(
    cc8_operator_t op,
    cc8_operator_spec_t& spec
);

void control_cache_8x64_dual_core(
    hls::stream<cu8_task_t>& core0_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core0_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core0_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core0_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu8_task_t>& core1_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core1_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core1_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core1_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    hls::stream<cc8_status_packet_t>& status_stream,
    fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT],
    const fm_word_t input_port0[CC8_DATA_PORT_WORDS],
    const fm_word_t input_port1[CC8_DATA_PORT_WORDS],
    const fm_word_t aux_port0[CC8_DATA_PORT_WORDS],
    const fm_word_t aux_port1[CC8_DATA_PORT_WORDS],
    unsigned int operator_kind,
    unsigned int layer_id,
    unsigned int token_count,
    unsigned int position,
    unsigned int tile_len,
    QWEN_WEIGHT_SHARD_PARAMS
);

#endif
