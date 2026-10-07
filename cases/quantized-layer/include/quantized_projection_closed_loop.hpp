#ifndef LLM_FPGA_QUANTIZED_PROJECTION_CLOSED_LOOP_HPP
#define LLM_FPGA_QUANTIZED_PROJECTION_CLOSED_LOOP_HPP

#ifdef QUANTIZED_ALIGNMENT_W8
#include "quantized_w8_projection_engine.hpp"
using alignment_policy_t = quantized_w8_projection_policy;
using alignment_buffer_t = quantized_w8_wide_buffer_t;
constexpr unsigned int ALIGNMENT_ROWS = QUANTIZED_W8_RESIDENT_TOKEN_ROWS;
constexpr unsigned int ALIGNMENT_BITS = 8;
constexpr unsigned int ALIGNMENT_WEIGHT_DEPTH = QUANTIZED_W8_WEIGHT_MEMORY_DEPTH;
#else
#include "quantized_w4_projection_engine.hpp"
using alignment_policy_t = quantized_w4_projection_policy;
using alignment_buffer_t = quantized_w4_wide_buffer_t;
constexpr unsigned int ALIGNMENT_ROWS = QUANTIZED_W4_RESIDENT_TOKEN_ROWS;
constexpr unsigned int ALIGNMENT_BITS = 4;
constexpr unsigned int ALIGNMENT_WEIGHT_DEPTH = QUANTIZED_W4_WEIGHT_MEMORY_DEPTH;
#endif
constexpr unsigned int ALIGNMENT_BUFFER_WORDS =
    ALIGNMENT_ROWS * alignment_buffer_t::kBlockCount;
constexpr unsigned int ALIGNMENT_WEIGHT_PORTS = alignment_policy_t::weight_ports;

void quantized_projection_closed_loop(
    const mm_input_block_t input[ALIGNMENT_BUFFER_WORDS],
    mm_input_block_t output[ALIGNMENT_BUFFER_WORDS],
    const alignment_policy_t::weight_word_t
        weight_memory[4][ALIGNMENT_WEIGHT_PORTS][ALIGNMENT_WEIGHT_DEPTH],
    unsigned int projection, unsigned int rows, unsigned int layer);

#endif
