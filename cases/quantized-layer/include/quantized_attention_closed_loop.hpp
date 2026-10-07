#ifndef LLM_FPGA_QUANTIZED_ATTENTION_CLOSED_LOOP_HPP
#define LLM_FPGA_QUANTIZED_ATTENTION_CLOSED_LOOP_HPP
#ifdef QUANTIZED_ALIGNMENT_W8
#include "quantized_w8_attention_engine.hpp"
using attention_feedback_policy = quantized_w8_attention_policy;
#else
#include "quantized_w4_attention_engine.hpp"
using attention_feedback_policy = quantized_w4_attention_policy;
#endif
constexpr unsigned int ATTENTION_QUERY_WORDS = attention_feedback_policy::token_rows * HIDDEN_SIZE / 16;
constexpr unsigned int ATTENTION_CACHE_WORDS = NUM_LAYERS * MAX_SEQ_LEN * KV_CHANNELS / 16;
void quantized_attention_closed_loop(
    const mm_input_block_t query[ATTENTION_QUERY_WORDS],
    const mm_input_block_t key[ATTENTION_CACHE_WORDS],
    const mm_input_block_t value[ATTENTION_CACHE_WORDS],
    mm_input_block_t output[ATTENTION_QUERY_WORDS],
    unsigned int rows, unsigned int attended_length, unsigned int decode, unsigned int layer);
#endif
