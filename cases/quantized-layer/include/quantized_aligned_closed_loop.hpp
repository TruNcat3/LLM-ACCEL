#ifndef LLM_FPGA_QUANTIZED_ALIGNED_CLOSED_LOOP_HPP
#define LLM_FPGA_QUANTIZED_ALIGNED_CLOSED_LOOP_HPP

#include "quantized_projection_closed_loop.hpp"
#include "quantized_vector_engine.hpp"
#ifdef QUANTIZED_ALIGNMENT_W8
#include "compute_core_quantized_w8_unified.hpp"
#else
#include "compute_core_quantized_w4_unified.hpp"
#endif

void quantized_aligned_closed_loop(
    const mm_input_block_t input[ALIGNMENT_BUFFER_WORDS],
    const mm_input_block_t rhs_input[ALIGNMENT_BUFFER_WORDS],
    mm_input_block_t output[ALIGNMENT_BUFFER_WORDS],
    const alignment_policy_t::weight_word_t
        weight_memory[4][ALIGNMENT_WEIGHT_PORTS][ALIGNMENT_WEIGHT_DEPTH],
    unsigned int projection, unsigned int rows, unsigned int layer,
    unsigned int vector_mode, unsigned int elements);

#endif
