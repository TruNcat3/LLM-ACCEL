#ifndef LLM_FPGA_COMPUTE_CORE_QUANTIZED_W4_UNIFIED_HPP
#define LLM_FPGA_COMPUTE_CORE_QUANTIZED_W4_UNIFIED_HPP

#include "mm_stream_8x128_int4x4_block.hpp"
#include "quantized_compute_vector.hpp"

void compute_core_quantized_w4_unified_nk(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activation_stream,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream0,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream1,
    hls::stream<quantized_vector_word_t>& vector_input0_stream,
    hls::stream<quantized_vector_word_t>& vector_input1_stream,
    unsigned int task_count);

#endif

