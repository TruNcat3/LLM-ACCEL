#ifndef LLM_FPGA_QUANTIZED_VECTOR_INPLACE_CLOSED_LOOP_HPP
#define LLM_FPGA_QUANTIZED_VECTOR_INPLACE_CLOSED_LOOP_HPP

#include "quantized_vector_closed_loop.hpp"

#ifdef QUANTIZED_ALIGNMENT_W8
using alignment_inplace_hidden_buffer_t = quantized_w8_hidden_buffer_t;
#else
using alignment_inplace_hidden_buffer_t = quantized_w4_hidden_buffer_t;
#endif

void quantized_vector_inplace_closed_loop(
    const mm_input_block_t input0[ALIGNMENT_BUFFER_WORDS],
    const mm_input_block_t input1[ALIGNMENT_BUFFER_WORDS],
    mm_input_block_t output[ALIGNMENT_BUFFER_WORDS],
    unsigned int mode, unsigned int elements, unsigned int rows,
    unsigned int projection);

#endif
