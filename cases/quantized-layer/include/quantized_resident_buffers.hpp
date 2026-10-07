#ifndef LLM_FPGA_QUANTIZED_RESIDENT_BUFFERS_HPP
#define LLM_FPGA_QUANTIZED_RESIDENT_BUFFERS_HPP

#include "hardware.hpp"
#include "mm_stream_4x128_int8x8_block.hpp"
#include "quantized_layer_schedule.hpp"
#include "quantized_numeric.hpp"

constexpr unsigned int QUANTIZED_W8_RESIDENT_TOKEN_ROWS =
    MM_STREAM_4X128_INT8X8_TOKENS;
constexpr unsigned int QUANTIZED_W8_RESIDENT_LANES_PER_WORD =
    MM_INPUT_BLOCK_BIT_WIDTH / fm_t::width;
constexpr unsigned int QUANTIZED_W8_HIDDEN_BLOCKS =
    (HIDDEN_SIZE + QUANTIZED_W8_RESIDENT_LANES_PER_WORD - 1) /
        QUANTIZED_W8_RESIDENT_LANES_PER_WORD;
constexpr unsigned int QUANTIZED_W8_KV_BLOCKS =
    (KV_CHANNELS + QUANTIZED_W8_RESIDENT_LANES_PER_WORD - 1) /
        QUANTIZED_W8_RESIDENT_LANES_PER_WORD;
constexpr unsigned int QUANTIZED_W8_WIDE_BLOCKS =
    (INTERMEDIATE_SIZE + QUANTIZED_W8_RESIDENT_LANES_PER_WORD - 1) /
        QUANTIZED_W8_RESIDENT_LANES_PER_WORD;

// Keep the AXI memory depths consistent with the selected model profile.
// Hard-coded full-Qwen depths make Vitis' CoSim wrapper copy past small-model
// test arrays before the DUT starts, which appears as a wrapper segfault.
constexpr unsigned int QUANTIZED_W8_HIDDEN_MEMORY_DEPTH =
    MAX_SEQ_LEN * QUANTIZED_W8_HIDDEN_BLOCKS;
constexpr unsigned int QUANTIZED_W8_KV_MEMORY_DEPTH =
    NUM_LAYERS * MAX_SEQ_LEN * QUANTIZED_W8_KV_BLOCKS;
constexpr unsigned int QUANTIZED_W8_NORM_MEMORY_DEPTH =
    NUM_LAYERS * 2 * QUANTIZED_W8_HIDDEN_BLOCKS;
constexpr unsigned int QUANTIZED_W8_ROPE_MEMORY_DEPTH =
    MAX_SEQ_LEN * 2 * ((HEAD_DIM / 2 +
        QUANTIZED_W8_RESIDENT_LANES_PER_WORD - 1) /
        QUANTIZED_W8_RESIDENT_LANES_PER_WORD);
constexpr unsigned int QUANTIZED_W8_SCALE_MEMORY_DEPTH = NUM_LAYERS;
constexpr unsigned int QUANTIZED_W8_WEIGHT_MEMORY_DEPTH =
    NUM_LAYERS * (
        2 * ((HIDDEN_SIZE + QUANTIZED_LAYER_OUTPUTS_PER_WAVE - 1) /
             QUANTIZED_LAYER_OUTPUTS_PER_WAVE) * HIDDEN_SIZE +
        2 * ((KV_CHANNELS + QUANTIZED_LAYER_OUTPUTS_PER_WAVE - 1) /
             QUANTIZED_LAYER_OUTPUTS_PER_WAVE) * HIDDEN_SIZE +
        2 * ((INTERMEDIATE_SIZE + QUANTIZED_LAYER_OUTPUTS_PER_WAVE - 1) /
             QUANTIZED_LAYER_OUTPUTS_PER_WAVE) * HIDDEN_SIZE +
        ((HIDDEN_SIZE + QUANTIZED_LAYER_OUTPUTS_PER_WAVE - 1) /
             QUANTIZED_LAYER_OUTPUTS_PER_WAVE) * INTERMEDIATE_SIZE);

template <unsigned int BLOCK_COUNT>
struct quantized_w8_feature_buffer_t {
    static constexpr unsigned int kBlockCount = BLOCK_COUNT;
    mm_input_block_t block[QUANTIZED_W8_RESIDENT_TOKEN_ROWS][BLOCK_COUNT];
};

using quantized_w8_hidden_buffer_t =
    quantized_w8_feature_buffer_t<QUANTIZED_W8_HIDDEN_BLOCKS>;
using quantized_w8_kv_buffer_t =
    quantized_w8_feature_buffer_t<QUANTIZED_W8_KV_BLOCKS>;
using quantized_w8_wide_buffer_t =
    quantized_w8_feature_buffer_t<QUANTIZED_W8_WIDE_BLOCKS>;

template <unsigned int BLOCK_COUNT>
inline fm_t quantized_w8_feature_get(
    const quantized_w8_feature_buffer_t<BLOCK_COUNT>& buffer,
    unsigned int token,
    unsigned int element) {
    #pragma HLS inline
    fm_t value = 0;
    if (token < QUANTIZED_W8_RESIDENT_TOKEN_ROWS &&
        element < BLOCK_COUNT * QUANTIZED_W8_RESIDENT_LANES_PER_WORD) {
        const unsigned int block =
            element / QUANTIZED_W8_RESIDENT_LANES_PER_WORD;
        const unsigned int lane =
            element % QUANTIZED_W8_RESIDENT_LANES_PER_WORD;
        value.range(fm_t::width - 1, 0) = buffer.block[token][block].range(
            (lane + 1) * fm_t::width - 1, lane * fm_t::width);
    }
    return value;
}

template <unsigned int BLOCK_COUNT>
inline void quantized_w8_feature_set(
    quantized_w8_feature_buffer_t<BLOCK_COUNT>& buffer,
    unsigned int token,
    unsigned int element,
    fm_t value) {
    #pragma HLS inline
    if (token < QUANTIZED_W8_RESIDENT_TOKEN_ROWS &&
        element < BLOCK_COUNT * QUANTIZED_W8_RESIDENT_LANES_PER_WORD) {
        const unsigned int block =
            element / QUANTIZED_W8_RESIDENT_LANES_PER_WORD;
        const unsigned int lane =
            element % QUANTIZED_W8_RESIDENT_LANES_PER_WORD;
        buffer.block[token][block].range(
            (lane + 1) * fm_t::width - 1, lane * fm_t::width) =
            value.range(fm_t::width - 1, 0);
    }
}

template <unsigned int BLOCK_COUNT>
inline mm_stream_4x128_int8x8_activation_word_t
pack_quantized_w8_activation_word(
    const quantized_w8_feature_buffer_t<BLOCK_COUNT>& source,
    unsigned int input_element,
    unsigned int valid_tokens,
    quant_inverse_scale_t inverse_scale) {
    #pragma HLS inline
    mm_stream_4x128_int8x8_activation_word_t packed = 0;
    for (unsigned int token = 0;
         token < QUANTIZED_W8_RESIDENT_TOKEN_ROWS; ++token) {
        #pragma HLS unroll
        const ap_int<8> value = token < valid_tokens ?
            quantize_symmetric_int8(
                quantized_w8_feature_get(source, token, input_element),
                inverse_scale) : ap_int<8>(0);
        packed.range((token + 1) * 8 - 1, token * 8) = value;
    }
    return packed;
}

template <unsigned int BLOCK_COUNT>
inline void store_quantized_w8_output_word(
    quantized_w8_feature_buffer_t<BLOCK_COUNT>& destination,
    const mm_stream_4x128_int8x8_output_word_t& output,
    unsigned int valid_tokens,
    unsigned int output_dim,
    quant_scale_t activation_scale,
    quant_scale_t weight_scale) {
    #pragma HLS inline
    const unsigned int token = output.range(535, 528).to_uint();
    const unsigned int elem_base = output.range(551, 536).to_uint();
    if (token >= valid_tokens) return;
    for (unsigned int lane = 0;
         lane < MM_STREAM_4X128_INT8X8_LANES_PER_GROUP; ++lane) {
        #pragma HLS unroll
        const unsigned int element = elem_base + lane;
        if (element < output_dim) {
            const quantized_w8_accum_t accumulator =
                unpack_mm_stream_4x128_int8x8_output(output, lane);
            quantized_w8_feature_set(
                destination, token, element,
                dequantize_w8_accumulator(
                    accumulator, activation_scale, weight_scale));
        }
    }
}

template <unsigned int BLOCK_COUNT>
inline void store_quantized_w8_aligned_output_word(
    quantized_w8_feature_buffer_t<BLOCK_COUNT>& destination,
    const mm_stream_4x128_int8x8_output_word_t& output,
    unsigned int valid_tokens,
    unsigned int output_dim,
    quant_scale_t activation_scale,
    quant_scale_t weight_scale) {
    #pragma HLS inline
    const unsigned int token = output.range(535, 528).to_uint();
    const unsigned int elem_base = output.range(551, 536).to_uint();
    if (token >= valid_tokens || elem_base >= output_dim) return;
    const quantized_dequant_scale_t combined_scale =
        combine_quantized_dequant_scales(activation_scale, weight_scale);

    // Compute outputs are groups of 16 aligned elements, exactly one resident
    // 256-bit word.  Commit the whole group at once so the packet loop has no
    // BRAM read-modify-write dependence.
    mm_input_block_t packed = 0;
    for (unsigned int lane = 0;
         lane < MM_STREAM_4X128_INT8X8_LANES_PER_GROUP; ++lane) {
        #pragma HLS unroll
        const unsigned int element = elem_base + lane;
        if (element < output_dim) {
            const quantized_w8_accum_t accumulator =
                unpack_mm_stream_4x128_int8x8_output(output, lane);
            const fm_t value =
                dequantize_w8_accumulator(accumulator, combined_scale);
            packed.range((lane + 1) * fm_t::width - 1,
                         lane * fm_t::width) =
                value.range(fm_t::width - 1, 0);
        }
    }
    const unsigned int block =
        elem_base / QUANTIZED_W8_RESIDENT_LANES_PER_WORD;
    if (block < BLOCK_COUNT) destination.block[token][block] = packed;
}

#endif
