#ifndef LLM_FPGA_QUANTIZED_LAYER_INTEGRATION_W8_CLOSED_LOOP_HPP
#define LLM_FPGA_QUANTIZED_LAYER_INTEGRATION_W8_CLOSED_LOOP_HPP

#include "control_cache_quantized_w8_layer.hpp"

// A bounded, memory-only ABI around the production W8 controller.  The
// controller's AXIS channels remain internal to this fixture so the C model
// and RTL exercise the same four-CU graph.
unsigned int quantized_layer_integration_w8_closed_loop(
    const mm_input_block_t* hidden_input,
    mm_input_block_t* hidden_output,
    mm_input_block_t* key_cache,
    mm_input_block_t* value_cache,
    const mm_input_block_t* norm_memory,
    const mm_input_block_t* rope_memory,
    const quantized_w8_scale_word_t* scale_memory,
    const quantized_w8_weight_word_t* weight_mem0a,
    const quantized_w8_weight_word_t* weight_mem0b,
    const quantized_w8_weight_word_t* weight_mem0c,
    const quantized_w8_weight_word_t* weight_mem0d,
    const quantized_w8_weight_word_t* weight_mem1a,
    const quantized_w8_weight_word_t* weight_mem1b,
    const quantized_w8_weight_word_t* weight_mem1c,
    const quantized_w8_weight_word_t* weight_mem1d,
    const quantized_w8_weight_word_t* weight_mem2a,
    const quantized_w8_weight_word_t* weight_mem2b,
    const quantized_w8_weight_word_t* weight_mem2c,
    const quantized_w8_weight_word_t* weight_mem2d,
    const quantized_w8_weight_word_t* weight_mem3a,
    const quantized_w8_weight_word_t* weight_mem3b,
    const quantized_w8_weight_word_t* weight_mem3c,
    const quantized_w8_weight_word_t* weight_mem3d,
    unsigned int layer,
    unsigned int sequence_length,
    unsigned int request_position,
    unsigned int kv_context_length,
    unsigned int request_op,
    unsigned int prefill_block_size);

#endif

