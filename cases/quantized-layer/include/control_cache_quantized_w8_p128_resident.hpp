#ifndef LLM_FPGA_CONTROL_CACHE_QUANTIZED_W8_P128_RESIDENT_HPP
#define LLM_FPGA_CONTROL_CACHE_QUANTIZED_W8_P128_RESIDENT_HPP

#include "mm_stream_4x128_int8x8_block.hpp"
#include "quantized_layer_task.hpp"

#include <hls_stream.h>

// Request-level resident scheduler for the W8A8 4x128 backend.  The
// physical compute tile consumes four token rows per task; sequence_length
// may be larger and is split into consecutive four-row blocks.
void control_cache_quantized_w8_p128_resident(
    hls::stream<mm_stream_quantized_task_word_t>& task_stream0,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream1,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream2,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream3,
    hls::stream<mm_stream_4x128_int8x8_activation_word_t>& activation_stream0,
    hls::stream<mm_stream_4x128_int8x8_activation_word_t>& activation_stream1,
    hls::stream<mm_stream_4x128_int8x8_activation_word_t>& activation_stream2,
    hls::stream<mm_stream_4x128_int8x8_activation_word_t>& activation_stream3,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream0a,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream0b,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream0c,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream0d,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream1a,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream1b,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream1c,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream1d,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream2a,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream2b,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream2c,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream2d,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream3a,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream3b,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream3c,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream3d,
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& out_stream0,
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& out_stream1,
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& out_stream2,
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& out_stream3,
    const mm_stream_4x128_int8x8_activation_word_t* activation_mem,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem0a,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem0b,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem0c,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem0d,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem1a,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem1b,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem1c,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem1d,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem2a,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem2b,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem2c,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem2d,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem3a,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem3b,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem3c,
    const mm_stream_4x128_int8x8_weight_word_t* weight_mem3d,
    mm_stream_4x128_int8x8_output_word_t* result_mem0,
    mm_stream_4x128_int8x8_output_word_t* result_mem1,
    mm_stream_4x128_int8x8_output_word_t* result_mem2,
    mm_stream_4x128_int8x8_output_word_t* result_mem3,
    unsigned int k_count,
    unsigned int wave_count,
    unsigned int sequence_length,
    unsigned int block_size,
    unsigned int request_position,
    unsigned int kv_context_length,
    unsigned int request_op);

#endif
