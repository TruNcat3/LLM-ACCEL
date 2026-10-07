#ifndef LLM_FPGA_MM_STREAM_QUANTIZED_W4_QUAD_HPP
#define LLM_FPGA_MM_STREAM_QUANTIZED_W4_QUAD_HPP

#include "mm_stream_8x64_int4x4_block.hpp"

#include <hls_stream.h>

constexpr unsigned int MM_STREAM_QUANTIZED_W4_QUAD_CUS = 4;
constexpr unsigned int MM_STREAM_QUANTIZED_W4_QUAD_PACKETS_PER_WAVE_CU =
    MM_STREAM_8X64_INT4X4_TOKENS * MM_STREAM_8X64_INT4X4_GROUPS;
constexpr unsigned int MM_STREAM_QUANTIZED_W4_QUAD_RESULT_BITS = 512;
using mm_stream_quantized_w4_quad_result_word_t =
    ap_uint<MM_STREAM_QUANTIZED_W4_QUAD_RESULT_BITS>;

void mm_stream_quantized_w4_quad_source(
    hls::stream<mm_stream_quantized_task_word_t>& task_stream0,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream1,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream2,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream3,
    hls::stream<mm_stream_8x64_int4x4_activation_word_t>& activation_stream0,
    hls::stream<mm_stream_8x64_int4x4_activation_word_t>& activation_stream1,
    hls::stream<mm_stream_8x64_int4x4_activation_word_t>& activation_stream2,
    hls::stream<mm_stream_8x64_int4x4_activation_word_t>& activation_stream3,
    hls::stream<mm_stream_8x64_int4x4_weight_word_t>& weight_stream0,
    hls::stream<mm_stream_8x64_int4x4_weight_word_t>& weight_stream1,
    hls::stream<mm_stream_8x64_int4x4_weight_word_t>& weight_stream2,
    hls::stream<mm_stream_8x64_int4x4_weight_word_t>& weight_stream3,
    const mm_stream_8x64_int4x4_activation_word_t* activation_mem,
    const mm_stream_8x64_int4x4_weight_word_t* weight_mem0,
    const mm_stream_8x64_int4x4_weight_word_t* weight_mem1,
    const mm_stream_8x64_int4x4_weight_word_t* weight_mem2,
    const mm_stream_8x64_int4x4_weight_word_t* weight_mem3,
    unsigned int k_count,
    unsigned int wave_count);

void mm_stream_quantized_w4_quad_sink(
    hls::stream<mm_stream_8x64_int4x4_output_word_t>& out_stream0,
    hls::stream<mm_stream_8x64_int4x4_output_word_t>& out_stream1,
    hls::stream<mm_stream_8x64_int4x4_output_word_t>& out_stream2,
    hls::stream<mm_stream_8x64_int4x4_output_word_t>& out_stream3,
    mm_stream_quantized_w4_quad_result_word_t* result_mem0,
    mm_stream_quantized_w4_quad_result_word_t* result_mem1,
    mm_stream_quantized_w4_quad_result_word_t* result_mem2,
    mm_stream_quantized_w4_quad_result_word_t* result_mem3,
    unsigned int wave_count);

#endif
