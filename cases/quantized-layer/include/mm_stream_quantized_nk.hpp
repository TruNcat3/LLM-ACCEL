#ifndef LLM_FPGA_MM_STREAM_QUANTIZED_NK_HPP
#define LLM_FPGA_MM_STREAM_QUANTIZED_NK_HPP

#include "datatypes.hpp"

#include <ap_int.h>

constexpr unsigned int MM_STREAM_QUANTIZED_TASK_BITS = 128;
using mm_stream_quantized_task_word_t =
    ap_uint<MM_STREAM_QUANTIZED_TASK_BITS>;

enum mm_stream_quantized_compute_mode_t {
    MM_STREAM_QUANTIZED_MODE_LINEAR = 0,
    MM_STREAM_QUANTIZED_MODE_ATTENTION_QK = 1,
    MM_STREAM_QUANTIZED_MODE_ATTENTION_PV = 2,
    MM_STREAM_QUANTIZED_MODE_BYPASS = 3,
    MM_STREAM_QUANTIZED_MODE_RMSNORM = 4,
    MM_STREAM_QUANTIZED_MODE_SILU_MUL = 5,
    MM_STREAM_QUANTIZED_MODE_RESIDUAL_ADD = 6,
    MM_STREAM_QUANTIZED_MODE_SILU_ONLY = 7,
    MM_STREAM_QUANTIZED_MODE_MUL = 8
};

struct mm_stream_quantized_task_t {
    unsigned int k_count;
    unsigned int elem_base;
    unsigned int block_id;
    bool last_stream;
    quant_scale_t activation_scale;
    quant_scale_t weight_scale;
    unsigned int request_position;
    unsigned int valid_tokens;
    unsigned int phase;
    unsigned int kv_context_length;
    unsigned int projection;
    mm_stream_quantized_compute_mode_t compute_mode;
};

inline mm_stream_quantized_task_word_t pack_mm_stream_quantized_task(
    const mm_stream_quantized_task_t& task) {
    #pragma HLS inline
    static_assert(quant_scale_t::width == 16,
                  "quantized task ABI reserves 16 bits per scale");
    mm_stream_quantized_task_word_t word = 0;
    word.range(15, 0) = task.k_count;
    word.range(31, 16) = task.elem_base;
    word.range(47, 32) = task.block_id;
    word[48] = task.last_stream;
    word.range(64, 49) =
        task.activation_scale.range(quant_scale_t::width - 1, 0);
    word.range(80, 65) =
        task.weight_scale.range(quant_scale_t::width - 1, 0);
    word.range(96, 81) = task.request_position;
    word.range(100, 97) = task.valid_tokens;
    word.range(102, 101) = task.phase;
    word.range(118, 103) = task.kv_context_length;
    word.range(121, 119) = task.projection;
    // Vector operations use two formerly reserved bits; the word stays 128-bit.
    // Existing matrix task encodings remain unchanged.
    word.range(125, 122) = static_cast<unsigned int>(task.compute_mode);
    return word;
}

inline mm_stream_quantized_task_t unpack_mm_stream_quantized_task(
    const mm_stream_quantized_task_word_t& word) {
    #pragma HLS inline
    mm_stream_quantized_task_t task;
    task.k_count = word.range(15, 0).to_uint();
    task.elem_base = word.range(31, 16).to_uint();
    task.block_id = word.range(47, 32).to_uint();
    task.last_stream = word[48];
    task.activation_scale.range(quant_scale_t::width - 1, 0) =
        word.range(64, 49);
    task.weight_scale.range(quant_scale_t::width - 1, 0) =
        word.range(80, 65);
    task.request_position = word.range(96, 81).to_uint();
    task.valid_tokens = word.range(100, 97).to_uint();
    task.phase = word.range(102, 101).to_uint();
    task.kv_context_length = word.range(118, 103).to_uint();
    task.projection = word.range(121, 119).to_uint();
    task.compute_mode = static_cast<mm_stream_quantized_compute_mode_t>(
        word.range(125, 122).to_uint());
    return task;
}

#endif
