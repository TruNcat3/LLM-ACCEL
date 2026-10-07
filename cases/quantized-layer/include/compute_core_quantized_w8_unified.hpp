#ifndef LLM_FPGA_COMPUTE_CORE_QUANTIZED_W8_UNIFIED_HPP
#define LLM_FPGA_COMPUTE_CORE_QUANTIZED_W8_UNIFIED_HPP

#include "mm_stream_4x128_int8x8_block.hpp"
#include "quantized_compute_vector.hpp"

void compute_core_quantized_w8_unified_nk(
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& out_stream,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream,
    hls::stream<mm_stream_4x128_int8x8_activation_word_t>& activation_stream,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream0,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream1,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream2,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream3,
    hls::stream<quantized_vector_word_t>& vector_input0_stream,
    hls::stream<quantized_vector_word_t>& vector_input1_stream,
    unsigned int task_count);

#endif

