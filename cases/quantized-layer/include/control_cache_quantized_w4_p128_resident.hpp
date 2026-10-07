#ifndef LLM_FPGA_CONTROL_CACHE_QUANTIZED_W4_P128_RESIDENT_HPP
#define LLM_FPGA_CONTROL_CACHE_QUANTIZED_W4_P128_RESIDENT_HPP

#include "control_cache_quantized_w4_p128.hpp"

// Request-level block scheduler for the W4A4 P128 backend.  Unlike the
// original block-MM controller, sequence_length may exceed eight.  The
// scheduler emits ceil(sequence_length / block_size) physical blocks and
// stores padded tail rows in the corresponding result range.
void control_cache_quantized_w4_p128_resident(
    hls::stream<mm_stream_quantized_task_word_t>& task_stream0,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream1,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream2,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream3,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activation_stream0,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activation_stream1,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activation_stream2,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activation_stream3,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream0a,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream0b,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream1a,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream1b,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream2a,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream2b,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream3a,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream3b,
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream0,
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream1,
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream2,
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream3,
    const mm_stream_8x128_int4x4_activation_word_t* activation_mem,
    const mm_stream_8x128_int4x4_weight_word_t* weight_mem0a,
    const mm_stream_8x128_int4x4_weight_word_t* weight_mem0b,
    const mm_stream_8x128_int4x4_weight_word_t* weight_mem1a,
    const mm_stream_8x128_int4x4_weight_word_t* weight_mem1b,
    const mm_stream_8x128_int4x4_weight_word_t* weight_mem2a,
    const mm_stream_8x128_int4x4_weight_word_t* weight_mem2b,
    const mm_stream_8x128_int4x4_weight_word_t* weight_mem3a,
    const mm_stream_8x128_int4x4_weight_word_t* weight_mem3b,
    mm_stream_quantized_w4_p128_quad_result_word_t* result_mem0,
    mm_stream_quantized_w4_p128_quad_result_word_t* result_mem1,
    mm_stream_quantized_w4_p128_quad_result_word_t* result_mem2,
    mm_stream_quantized_w4_p128_quad_result_word_t* result_mem3,
    unsigned int k_count,
    unsigned int wave_count,
    unsigned int sequence_length,
    unsigned int block_size,
    unsigned int request_position,
    unsigned int kv_context_length);

#endif
