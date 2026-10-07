#ifndef LLM_FPGA_QUANTIZED_BATCH_DECODE_CLOSED_LOOP_HPP
#define LLM_FPGA_QUANTIZED_BATCH_DECODE_CLOSED_LOOP_HPP

#include "quantized_batch_decode_runtime.hpp"
#include "quantized_batch_decode_task.hpp"
#include "quantized_weight_axi_config.hpp"

using quantized_batch_decode_scale_word_t = ap_uint<128>;
constexpr unsigned int QUANTIZED_BATCH_DECODE_GUARD_WORDS = 8;
constexpr unsigned int QUANTIZED_BATCH_DECODE_HIDDEN_DEPTH =
    QUANTIZED_BATCH_DECODE_MAX * QUANTIZED_BATCH_DECODE_HIDDEN_WORDS +
    QUANTIZED_BATCH_DECODE_GUARD_WORDS;
constexpr unsigned int QUANTIZED_BATCH_DECODE_KV_DEPTH =
    QUANTIZED_BATCH_DECODE_MAX * QUANTIZED_BATCH_DECODE_FULL_KV_WORDS +
    QUANTIZED_BATCH_DECODE_GUARD_WORDS;

// The stream graph is internal to the fixture.  Keeping only bounded memory
// arrays and descriptors on the kernel ABI matches the production closed-loop
// fixtures and lets HLS size every controller/CU FIFO explicitly.
unsigned int quantized_batch_decode_closed_loop(
    const quantized_batch_decode_word_t descriptors[
        QUANTIZED_BATCH_DECODE_MAX],
    unsigned int batch_count,
    unsigned int layer,
    const mm_input_block_t* hidden_input,
    mm_input_block_t* hidden_output,
    mm_input_block_t* key_cache,
    mm_input_block_t* value_cache,
    const mm_input_block_t* norm_memory,
    const mm_input_block_t* rope_memory,
    const quantized_batch_decode_scale_word_t* scale_memory,
    const qbd_weight_t* weight_mem0a,
    const qbd_weight_t* weight_mem0b,
#ifdef QUANTIZED_ALIGNMENT_W8
    const qbd_weight_t* weight_mem0c,
    const qbd_weight_t* weight_mem0d,
#endif
    const qbd_weight_t* weight_mem1a,
    const qbd_weight_t* weight_mem1b,
#ifdef QUANTIZED_ALIGNMENT_W8
    const qbd_weight_t* weight_mem1c,
    const qbd_weight_t* weight_mem1d,
#endif
    const qbd_weight_t* weight_mem2a,
    const qbd_weight_t* weight_mem2b,
#ifdef QUANTIZED_ALIGNMENT_W8
    const qbd_weight_t* weight_mem2c,
    const qbd_weight_t* weight_mem2d,
#endif
    const qbd_weight_t* weight_mem3a,
    const qbd_weight_t* weight_mem3b,
#ifdef QUANTIZED_ALIGNMENT_W8
    const qbd_weight_t* weight_mem3c,
    const qbd_weight_t* weight_mem3d,
#endif
    unsigned int input_words,
    unsigned int output_words,
    unsigned int key_words,
    unsigned int value_words);

#endif
