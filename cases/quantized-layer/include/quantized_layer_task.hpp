#ifndef LLM_FPGA_QUANTIZED_LAYER_TASK_HPP
#define LLM_FPGA_QUANTIZED_LAYER_TASK_HPP

#include "model_config.hpp"

#include <ap_int.h>

// Physical token tiles are backend properties, not request-level sequence
// limits.  Keep the legacy name as the W4/default value for source and packed
// descriptor compatibility.
constexpr unsigned int QUANTIZED_LAYER_W4_TOKEN_BLOCK = 8;
constexpr unsigned int QUANTIZED_LAYER_W8_TOKEN_BLOCK = 4;
constexpr unsigned int QUANTIZED_LAYER_DEFAULT_TOKEN_BLOCK =
    QUANTIZED_LAYER_W4_TOKEN_BLOCK;
constexpr unsigned int QUANTIZED_LAYER_MAX_TOKEN_BLOCK = 8;
constexpr unsigned int QUANTIZED_LAYER_TOKEN_BLOCK =
    QUANTIZED_LAYER_DEFAULT_TOKEN_BLOCK;
constexpr unsigned int QUANTIZED_LAYER_TASK_BITS = 256;
using quantized_layer_task_word_t = ap_uint<QUANTIZED_LAYER_TASK_BITS>;

enum quantized_layer_op_t {
    QUANTIZED_LAYER_OP_PREFILL = 0,
    QUANTIZED_LAYER_OP_DECODE = 1,
    QUANTIZED_LAYER_OP_ATTENTION = 2,
    QUANTIZED_LAYER_OP_FFN = 3,
    QUANTIZED_LAYER_OP_FINAL_NORM = 4,
    QUANTIZED_LAYER_OP_STOP = 15
};

enum quantized_layer_phase_t {
    QUANTIZED_LAYER_PHASE_PREFILL = 0,
    QUANTIZED_LAYER_PHASE_DECODE = 1,
    QUANTIZED_LAYER_PHASE_INTERNAL = 2
};

enum quantized_activation_format_t {
    QUANTIZED_ACTIVATION_FP16 = 0,
    QUANTIZED_ACTIVATION_INT4 = 1,
    QUANTIZED_ACTIVATION_INT8 = 2
};

enum quantized_weight_format_t {
    QUANTIZED_WEIGHT_INT4 = 0,
    QUANTIZED_WEIGHT_INT8 = 1
};

// This descriptor is request/block level.  It is deliberately separate from
// mm_stream_quantized_task_t, whose 128-bit ABI is retained for block-MM
// bring-up and existing xclbins.
struct quantized_layer_task_t {
    quantized_layer_op_t op;
    quantized_layer_phase_t phase;
    unsigned int layer;
    unsigned int position;
    unsigned int query_tokens;
    unsigned int sequence_length;
    unsigned int kv_context_length;
    unsigned int block_index;
    unsigned int block_count;
    unsigned int input_pair;
    unsigned int output_pair;
    quantized_activation_format_t activation_format;
    quantized_weight_format_t weight_format;
    unsigned int activation_scale_id;
    unsigned int weight_scale_id;
    unsigned int physical_block_size;
    bool last_block;
    bool last_task;
};

constexpr unsigned int quantized_layer_effective_block_size(
    unsigned int physical_block_size) {
    // Zero is the legacy packed representation, which used an implicit
    // eight-row W4 tile.
    return physical_block_size == 0 ?
        QUANTIZED_LAYER_DEFAULT_TOKEN_BLOCK : physical_block_size;
}

constexpr unsigned int quantized_layer_block_count(
    unsigned int sequence_length,
    unsigned int block_size = QUANTIZED_LAYER_DEFAULT_TOKEN_BLOCK) {
    return block_size == 0 ? 0 :
        (sequence_length + block_size - 1) / block_size;
}

constexpr unsigned int quantized_layer_source_token_begin(
    const quantized_layer_task_t& task) {
    return task.block_index *
        quantized_layer_effective_block_size(task.physical_block_size);
}

inline bool quantized_layer_task_shape_valid(
    const quantized_layer_task_t& task,
    unsigned int max_sequence_length = MAX_SEQ_LEN,
    unsigned int max_layers = NUM_LAYERS) {
    const unsigned int block_size =
        quantized_layer_effective_block_size(task.physical_block_size);
    if (task.layer >= max_layers || task.query_tokens == 0 ||
        block_size > QUANTIZED_LAYER_MAX_TOKEN_BLOCK ||
        task.query_tokens > block_size ||
        task.sequence_length == 0 ||
        task.sequence_length > max_sequence_length ||
        task.position + task.query_tokens > max_sequence_length ||
        task.block_count == 0 ||
        task.block_index >= task.block_count ||
        task.block_count !=
            quantized_layer_block_count(task.sequence_length, block_size) ||
        task.input_pair > 1 || task.output_pair > 1 ||
        task.activation_format > QUANTIZED_ACTIVATION_INT8 ||
        task.weight_format > QUANTIZED_WEIGHT_INT8) {
        return false;
    }
    if (task.phase == QUANTIZED_LAYER_PHASE_DECODE) {
        return task.query_tokens == 1 && task.sequence_length == 1 &&
            task.kv_context_length <= max_sequence_length;
    }
    if (task.phase == QUANTIZED_LAYER_PHASE_PREFILL) {
        return task.kv_context_length == task.position;
    }
    return task.phase == QUANTIZED_LAYER_PHASE_INTERNAL;
}

inline quantized_layer_task_t make_quantized_prefill_block_task(
    unsigned int layer,
    unsigned int request_position,
    unsigned int sequence_length,
    unsigned int block_index,
    unsigned int input_pair,
    unsigned int output_pair,
    quantized_activation_format_t activation_format,
    quantized_weight_format_t weight_format,
    unsigned int activation_scale_id = 0,
    unsigned int weight_scale_id = 0,
    unsigned int max_sequence_length = MAX_SEQ_LEN,
    unsigned int physical_block_size =
        QUANTIZED_LAYER_DEFAULT_TOKEN_BLOCK) {
    quantized_layer_task_t task{};
    const unsigned int block_size =
        quantized_layer_effective_block_size(physical_block_size);
    task.op = QUANTIZED_LAYER_OP_PREFILL;
    task.phase = QUANTIZED_LAYER_PHASE_PREFILL;
    task.layer = layer;
    task.sequence_length = sequence_length;
    task.block_count = quantized_layer_block_count(sequence_length, block_size);
    task.block_index = block_index;
    task.position = request_position + block_index * block_size;
    task.query_tokens =
        sequence_length - block_index * block_size;
    if (task.query_tokens > block_size) {
        task.query_tokens = block_size;
    }
    task.kv_context_length = task.position;
    task.input_pair = input_pair;
    task.output_pair = output_pair;
    task.activation_format = activation_format;
    task.weight_format = weight_format;
    task.activation_scale_id = activation_scale_id;
    task.weight_scale_id = weight_scale_id;
    task.physical_block_size = block_size;
    task.last_block = block_index + 1 == task.block_count;
    task.last_task = task.last_block;
    (void)max_sequence_length;
    return task;
}

inline quantized_layer_task_t make_quantized_decode_task(
    unsigned int layer,
    unsigned int position,
    unsigned int kv_context_length,
    unsigned int input_pair,
    unsigned int output_pair,
    quantized_activation_format_t activation_format,
    quantized_weight_format_t weight_format,
    unsigned int activation_scale_id = 0,
    unsigned int weight_scale_id = 0,
    unsigned int physical_block_size =
        QUANTIZED_LAYER_DEFAULT_TOKEN_BLOCK) {
    quantized_layer_task_t task{};
    task.op = QUANTIZED_LAYER_OP_DECODE;
    task.phase = QUANTIZED_LAYER_PHASE_DECODE;
    task.layer = layer;
    task.position = position;
    task.query_tokens = 1;
    task.sequence_length = 1;
    task.kv_context_length = kv_context_length;
    task.block_index = 0;
    task.block_count = 1;
    task.input_pair = input_pair;
    task.output_pair = output_pair;
    task.activation_format = activation_format;
    task.weight_format = weight_format;
    task.activation_scale_id = activation_scale_id;
    task.weight_scale_id = weight_scale_id;
    task.physical_block_size =
        quantized_layer_effective_block_size(physical_block_size);
    task.last_block = true;
    task.last_task = true;
    return task;
}

inline quantized_layer_task_word_t pack_quantized_layer_task(
    const quantized_layer_task_t& task) {
    #pragma HLS inline
    quantized_layer_task_word_t word = 0;
    word.range(3, 0) = static_cast<unsigned int>(task.op);
    word.range(5, 4) = static_cast<unsigned int>(task.phase);
    word.range(15, 6) = task.layer;
    word.range(31, 16) = task.position;
    word.range(35, 32) = task.query_tokens;
    word.range(51, 36) = task.sequence_length;
    word.range(67, 52) = task.kv_context_length;
    word.range(83, 68) = task.block_index;
    word.range(99, 84) = task.block_count;
    word.range(101, 100) = task.input_pair;
    word.range(103, 102) = task.output_pair;
    word.range(105, 104) = static_cast<unsigned int>(task.activation_format);
    word.range(107, 106) = static_cast<unsigned int>(task.weight_format);
    word.range(123, 108) = task.activation_scale_id;
    word.range(139, 124) = task.weight_scale_id;
    word[140] = task.last_block;
    word[141] = task.last_task;
    word.range(145, 142) = task.physical_block_size;
    return word;
}

inline quantized_layer_task_t unpack_quantized_layer_task(
    const quantized_layer_task_word_t& word) {
    #pragma HLS inline
    quantized_layer_task_t task{};
    task.op = static_cast<quantized_layer_op_t>(word.range(3, 0).to_uint());
    task.phase = static_cast<quantized_layer_phase_t>(
        word.range(5, 4).to_uint());
    task.layer = word.range(15, 6).to_uint();
    task.position = word.range(31, 16).to_uint();
    task.query_tokens = word.range(35, 32).to_uint();
    task.sequence_length = word.range(51, 36).to_uint();
    task.kv_context_length = word.range(67, 52).to_uint();
    task.block_index = word.range(83, 68).to_uint();
    task.block_count = word.range(99, 84).to_uint();
    task.input_pair = word.range(101, 100).to_uint();
    task.output_pair = word.range(103, 102).to_uint();
    task.activation_format = static_cast<quantized_activation_format_t>(
        word.range(105, 104).to_uint());
    task.weight_format = static_cast<quantized_weight_format_t>(
        word.range(107, 106).to_uint());
    task.activation_scale_id = word.range(123, 108).to_uint();
    task.weight_scale_id = word.range(139, 124).to_uint();
    task.last_block = word[140];
    task.last_task = word[141];
    task.physical_block_size = quantized_layer_effective_block_size(
        word.range(145, 142).to_uint());
    return task;
}

#endif
