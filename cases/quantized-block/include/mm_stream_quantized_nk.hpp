#ifndef LLM_FPGA_MM_STREAM_QUANTIZED_NK_HPP
#define LLM_FPGA_MM_STREAM_QUANTIZED_NK_HPP

#include "datatypes.hpp"

#include <ap_int.h>

constexpr unsigned int MM_STREAM_QUANTIZED_TASK_BITS = 128;
using mm_stream_quantized_task_word_t =
    ap_uint<MM_STREAM_QUANTIZED_TASK_BITS>;

struct mm_stream_quantized_task_t {
    unsigned int k_count;
    unsigned int elem_base;
    unsigned int block_id;
    bool last_stream;
    fm_t activation_scale;
    fm_t weight_scale;
};

inline mm_stream_quantized_task_word_t pack_mm_stream_quantized_task(
    const mm_stream_quantized_task_t& task) {
    #pragma HLS inline
    static_assert(fm_t::width == 16,
                  "quantized task ABI reserves 16 bits per scale");
    mm_stream_quantized_task_word_t word = 0;
    word.range(15, 0) = task.k_count;
    word.range(31, 16) = task.elem_base;
    word.range(47, 32) = task.block_id;
    word[48] = task.last_stream;
    word.range(64, 49) =
        task.activation_scale.range(fm_t::width - 1, 0);
    word.range(80, 65) = task.weight_scale.range(fm_t::width - 1, 0);
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
    task.activation_scale.range(fm_t::width - 1, 0) = word.range(64, 49);
    task.weight_scale.range(fm_t::width - 1, 0) = word.range(80, 65);
    return task;
}

#endif
