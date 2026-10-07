#ifndef LLM_FPGA_QUANTIZED_VECTOR_CLOSED_LOOP_HPP
#define LLM_FPGA_QUANTIZED_VECTOR_CLOSED_LOOP_HPP

#include "quantized_projection_closed_loop.hpp"
#include "quantized_vector_engine.hpp"
#ifdef QUANTIZED_ALIGNMENT_W8
#include "compute_core_quantized_w8_unified.hpp"
#else
#include "compute_core_quantized_w4_unified.hpp"
#endif
#include <cassert>

void quantized_vector_closed_loop(
    const mm_input_block_t input0[ALIGNMENT_BUFFER_WORDS],
    const mm_input_block_t input1[ALIGNMENT_BUFFER_WORDS],
    mm_input_block_t output[ALIGNMENT_BUFFER_WORDS],
    unsigned int mode, unsigned int elements, unsigned int rows);

#endif

