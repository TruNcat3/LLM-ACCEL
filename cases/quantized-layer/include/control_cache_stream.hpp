#ifndef LLM_FPGA_CONTROL_CACHE_STREAM_HPP
#define LLM_FPGA_CONTROL_CACHE_STREAM_HPP

#include "compute_stream.hpp"
#include "mm_controller.hpp"

constexpr unsigned int CC_STREAM_VERSION = 1;
constexpr unsigned int CC_GBUF_COUNT = MM_GLOBAL_BUFFER_COUNT;
constexpr unsigned int CC_WEIGHT_WORDS_PER_TILE = MM_TILE_WEIGHT_BLOCKS;
constexpr unsigned int CC_EXTERNAL_DATA_PORTS = 2;
constexpr unsigned int CC_EXTERNAL_PORT_BIT_WIDTH = HBM_PORT_BIT_WIDTH;
constexpr unsigned int CC_TOKENS_PER_INPUT_PORT =
    ceildiv(LINEAR_TOKEN_TILE_ACTIVE, CC_EXTERNAL_DATA_PORTS);
constexpr unsigned int CC_INPUT_WORDS_PER_TOKEN =
    ceildiv(MAX_LINEAR_IN_DIM, FM_BLOCK_SIZE);
constexpr unsigned int CC_INPUT_WORDS_PER_PORT =
    CC_TOKENS_PER_INPUT_PORT * CC_INPUT_WORDS_PER_TOKEN;
constexpr unsigned int CC_OUTPUT_WORDS_PER_PORT =
    ceildiv(LINEAR_TOKEN_TILE_ACTIVE, FM_BLOCK_SIZE);

enum cc_operator_t {
    CC_OP_NOP = 0,
    CC_OP_LOAD_HIDDEN = 1,
    CC_OP_STORE_HIDDEN = 2,
    CC_OP_LINEAR_MM = 3,
    CC_OP_FFN_GATE_UP = 4,
    CC_OP_FFN_DOWN = 5,
    CC_OP_RMSNORM = 6,
    CC_OP_RESIDUAL_ADD = 7,
    CC_OP_ATTENTION_MV = 8,
    CC_OP_DECODER_LAYER = 9
};

enum cc_buffer_id_t {
    CC_GBUF_0 = 0,
    CC_GBUF_1 = 1,
    CC_GBUF_AUX = 2,
    CC_GBUF_INVALID = 255
};

enum cc_task_flag_t {
    CC_TASK_FLAG_NONE = 0,
    CC_TASK_FLAG_ACCUM_INPUT = 1,
    CC_TASK_FLAG_ACCUM_OUTPUT = 2,
    CC_TASK_FLAG_TOKEN_TAIL = 4,
    CC_TASK_FLAG_ELEM_TAIL = 8
};

struct cc_task_packet_t {
    cc_operator_t op;
    cu_stream_op_t stream_op;
    mm_controller_mode_t mm_mode;
    weight_addr_t weight_base;
    unsigned int layer_id;
    unsigned int position;
    unsigned int token_count;
    unsigned int in_dim;
    unsigned int out_dim;
    unsigned int block_id;
    unsigned int elem_base;
    unsigned int packet_count;
    ap_uint<8> src_buffer;
    ap_uint<8> rhs_buffer;
    ap_uint<8> dst_buffer;
    ap_uint<16> flags;
    bool last_task;
};

struct cc_weight_packet_t {
    wt_block_t data;
    unsigned int block_id;
    unsigned int in_tile;
    unsigned int out_tile;
    unsigned int tile_word;
    ap_uint<CC_WEIGHT_WORDS_PER_TILE> valid_words;
    bool last_tile;
    bool last_block;
    bool last_stream;
};

struct cc_status_packet_t {
    cc_operator_t op;
    unsigned int block_id;
    unsigned int completed_packets;
    ap_uint<16> status;
    bool last_task;
};

unsigned int cc_vec_packet_count(unsigned int elem_count);

unsigned int cc_weight_packet_count(
    const mm_controller_task_t& task,
    unsigned int output_block
);

void control_cache_emit_linear_block_streams(
    hls::stream<cc_task_packet_t>& task_stream,
    hls::stream<cu_vec16_packet_t>& input_stream,
    hls::stream<cc_weight_packet_t>& weight_stream,
    hls::stream<cc_status_packet_t>& status_stream,
    fm_t input_tokens[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_DIM],
    mm_projection_kind_t projection,
    unsigned int token_count,
    unsigned int output_block,
    weight_addr_t layer_base,
    QWEN_WEIGHT_SHARD_PARAMS
);

void control_cache_stream_adapter_wide_synth_stub(
    fm_word_t sink_words[CC_OUTPUT_WORDS_PER_PORT],
    const fm_word_t input_port0[CC_INPUT_WORDS_PER_PORT],
    const fm_word_t input_port1[CC_INPUT_WORDS_PER_PORT],
    unsigned int projection_kind,
    unsigned int token_count,
    unsigned int output_block,
    QWEN_WEIGHT_SHARD_PARAMS
);

void control_cache_stream_adapter_synth_stub(
    fm_t sink[LINEAR_TOKEN_TILE_ACTIVE],
    fm_t input_tokens[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_DIM],
    QWEN_WEIGHT_SHARD_PARAMS
);

#endif
