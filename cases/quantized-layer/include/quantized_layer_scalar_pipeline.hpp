#ifndef LLM_FPGA_QUANTIZED_LAYER_SCALAR_PIPELINE_HPP
#define LLM_FPGA_QUANTIZED_LAYER_SCALAR_PIPELINE_HPP

#include "hardware.hpp"
#include "quantized_numeric.hpp"
#include <hls_math.h>

using quantized_norm_accum_t = ap_fixed<40, 20, AP_RND, AP_SAT>;
constexpr unsigned int QUANTIZED_SCALAR_WORD_LANES =
    MM_INPUT_BLOCK_BIT_WIDTH / fm_t::width;

// Share the nonlinear pipeline while preserving all matrix-engine lanes.
#ifndef QUANTIZED_SCALAR_SILU_INSTANCES
#define QUANTIZED_SCALAR_SILU_INSTANCES 4
#endif
static_assert(QUANTIZED_SCALAR_SILU_INSTANCES > 0 &&
              QUANTIZED_SCALAR_SILU_INSTANCES <= QUANTIZED_SCALAR_WORD_LANES,
              "SiLU pipeline count must fit a resident word");

inline fm_t quantized_scalar_abs(fm_t value) {
    #pragma HLS inline
    // Retain the original saturating fm_t absolute-value semantics, including
    // the most negative representable input.
    return value < fm_t(0) ? fm_t(-value) : value;
}

template <unsigned int ROWS, unsigned int BLOCKS>
void quantized_clear_inactive_rows(
    mm_input_block_t (&buffer)[ROWS][BLOCKS],
    unsigned int valid_tokens,
    unsigned int element_count) {
    #pragma HLS inline
    const unsigned int words =
        (element_count + QUANTIZED_SCALAR_WORD_LANES - 1) /
        QUANTIZED_SCALAR_WORD_LANES;
    for (unsigned int token = valid_tokens; token < ROWS; ++token) {
        for (unsigned int block = 0; block < words; ++block) {
            #pragma HLS pipeline II=1
            // A partial final word preserves elements outside element_count.
            const bool full_word = (block + 1) * QUANTIZED_SCALAR_WORD_LANES <=
                element_count;
            mm_input_block_t word = full_word ? mm_input_block_t(0) :
                buffer[token][block];
            for (unsigned int lane = 0; lane < QUANTIZED_SCALAR_WORD_LANES;
                 ++lane) {
                #pragma HLS unroll
                if (block * QUANTIZED_SCALAR_WORD_LANES + lane < element_count)
                    word.range((lane + 1) * fm_t::width - 1,
                               lane * fm_t::width) = 0;
            }
            buffer[token][block] = word;
        }
    }
}

// The sum keeps its original scalar order and fixed-point assignment points.
// Only active rows execute normalization; scale reduction is kept outside the
// critical arithmetic pipeline.
template <unsigned int ROWS, unsigned int BLOCKS>
fm_t quantized_rmsnorm_blocks(
    const mm_input_block_t (&source)[ROWS][BLOCKS],
    const mm_input_block_t (&weight)[ROWS][BLOCKS],
    mm_input_block_t (&destination)[ROWS][BLOCKS],
    unsigned int valid_tokens,
    unsigned int element_count) {
    #pragma HLS inline off
    const unsigned int rows = valid_tokens < ROWS ? valid_tokens : ROWS;
    for (unsigned int token = 0; token < rows; ++token) {
        quantized_norm_accum_t square_sum = 0;
        mm_input_block_t source_word = 0;
        for (unsigned int element = 0; element < element_count; ++element) {
            #pragma HLS pipeline II=1
            const unsigned int lane = element % QUANTIZED_SCALAR_WORD_LANES;
            if (lane == 0)
                source_word = source[token][element / QUANTIZED_SCALAR_WORD_LANES];
            fm_t value;
            value.range(fm_t::width - 1, 0) = source_word.range(
                (lane + 1) * fm_t::width - 1, lane * fm_t::width);
            const quantized_norm_accum_t wide = value;
            square_sum += wide * wide;
        }
        quantized_norm_accum_t mean = 0;
        if (element_count != 0)
            mean = quantized_norm_accum_t(
                square_sum / quantized_norm_accum_t(element_count));
        const quantized_norm_accum_t denominator = hls::sqrt(
            mean + quantized_norm_accum_t(RMS_NORM_EPS));
        quantized_norm_accum_t inverse = 0;
        if (denominator != 0)
            inverse = quantized_norm_accum_t(
                quantized_norm_accum_t(1) / denominator);
        mm_input_block_t weight_word = 0;
        mm_input_block_t output_word = 0;
        for (unsigned int element = 0; element < element_count; ++element) {
            #pragma HLS pipeline II=1
            const unsigned int lane = element % QUANTIZED_SCALAR_WORD_LANES;
            const unsigned int block = element / QUANTIZED_SCALAR_WORD_LANES;
            if (lane == 0) {
                source_word = source[token][block];
                weight_word = weight[0][block];
                output_word = destination[token][block];
            }
            fm_t input, norm_weight;
            input.range(fm_t::width - 1, 0) = source_word.range(
                (lane + 1) * fm_t::width - 1, lane * fm_t::width);
            norm_weight.range(fm_t::width - 1, 0) = weight_word.range(
                (lane + 1) * fm_t::width - 1, lane * fm_t::width);
            const fm_t value = fm_t(quantized_norm_accum_t(input) *
                inverse * quantized_norm_accum_t(norm_weight));
            output_word.range((lane + 1) * fm_t::width - 1,
                              lane * fm_t::width) = value.range(fm_t::width - 1, 0);
            if (lane + 1 == QUANTIZED_SCALAR_WORD_LANES ||
                element + 1 == element_count)
                destination[token][block] = output_word;
        }
    }
    quantized_clear_inactive_rows(destination, rows, element_count);
    return fm_t(0);
}

inline fm_t quantized_word_max_abs(mm_input_block_t word,
                                 unsigned int valid_lanes) {
    #pragma HLS inline
    static_assert(QUANTIZED_SCALAR_WORD_LANES == 16,
                  "The reduction tree expects sixteen resident lanes");
    fm_t tree[2 * QUANTIZED_SCALAR_WORD_LANES];
    #pragma HLS array_partition variable=tree complete
    for (unsigned int lane = 0; lane < QUANTIZED_SCALAR_WORD_LANES; ++lane) {
        #pragma HLS unroll
        fm_t value;
        value.range(fm_t::width - 1, 0) = word.range(
            (lane + 1) * fm_t::width - 1, lane * fm_t::width);
        tree[QUANTIZED_SCALAR_WORD_LANES + lane] =
            lane < valid_lanes ? quantized_scalar_abs(value) : fm_t(0);
    }
    for (int node = QUANTIZED_SCALAR_WORD_LANES - 1; node > 0; --node) {
        #pragma HLS unroll
        tree[node] = tree[2 * node] > tree[2 * node + 1] ?
            tree[2 * node] : tree[2 * node + 1];
    }
    return tree[1];
}

template <unsigned int ROWS, unsigned int BLOCKS>
fm_t quantized_buffer_max_abs(
    const mm_input_block_t (&source)[ROWS][BLOCKS],
    unsigned int valid_tokens,
    unsigned int element_count) {
    #pragma HLS inline off
    const unsigned int rows = valid_tokens < ROWS ? valid_tokens : ROWS;
    const unsigned int words = (element_count + QUANTIZED_SCALAR_WORD_LANES - 1) /
        QUANTIZED_SCALAR_WORD_LANES;
    fm_t maximum = 0;
    for (unsigned int token = 0; token < rows; ++token) {
        for (unsigned int block = 0; block < words; ++block) {
            #pragma HLS pipeline II=1
            const unsigned int remaining = element_count -
                block * QUANTIZED_SCALAR_WORD_LANES;
            const fm_t magnitude = quantized_word_max_abs(source[token][block],
                remaining < QUANTIZED_SCALAR_WORD_LANES ?
                    remaining : QUANTIZED_SCALAR_WORD_LANES);
            if (magnitude > maximum) maximum = magnitude;
        }
    }
    return maximum;
}

// One pipeline spans every packed word in a row. ProductOp retains each
// precision's existing arithmetic. This supports destination == gate: a word
// is read once before its first lane and committed only after its last lane.
template <typename ProductOp, unsigned int ROWS, unsigned int BLOCKS>
fm_t quantized_silu_blocks(
    const mm_input_block_t (&gate)[ROWS][BLOCKS],
    const mm_input_block_t (&up)[ROWS][BLOCKS],
    mm_input_block_t (&destination)[ROWS][BLOCKS],
    unsigned int valid_tokens) {
    #pragma HLS inline off
    const unsigned int rows = valid_tokens < ROWS ? valid_tokens : ROWS;
    fm_t maximum = 0;
    for (unsigned int token = 0; token < rows; ++token) {
        mm_input_block_t gate_word = 0, up_word = 0, product_word = 0;
        for (unsigned int element = 0;
             element < BLOCKS * QUANTIZED_SCALAR_WORD_LANES; ++element) {
            #pragma HLS pipeline II=1
            const unsigned int lane = element % QUANTIZED_SCALAR_WORD_LANES;
            const unsigned int block = element / QUANTIZED_SCALAR_WORD_LANES;
            if (lane == 0) {
                gate_word = gate[token][block];
                up_word = up[token][block];
            }
            fm_t gate_value, up_value;
            gate_value.range(fm_t::width - 1, 0) = gate_word.range(
                (lane + 1) * fm_t::width - 1, lane * fm_t::width);
            up_value.range(fm_t::width - 1, 0) = up_word.range(
                (lane + 1) * fm_t::width - 1, lane * fm_t::width);
            const fm_t value = ProductOp::eval(gate_value, up_value);
            product_word.range((lane + 1) * fm_t::width - 1,
                               lane * fm_t::width) = value.range(fm_t::width - 1, 0);
            if (lane + 1 == QUANTIZED_SCALAR_WORD_LANES)
                destination[token][block] = product_word;
        }
    }
    quantized_clear_inactive_rows(destination, rows,
                                  BLOCKS * QUANTIZED_SCALAR_WORD_LANES);
    return fm_t(0);
}

#endif
