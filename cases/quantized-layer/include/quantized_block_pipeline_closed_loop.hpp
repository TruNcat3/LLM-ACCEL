#ifndef LLM_FPGA_QUANTIZED_BLOCK_PIPELINE_CLOSED_LOOP_HPP
#define LLM_FPGA_QUANTIZED_BLOCK_PIPELINE_CLOSED_LOOP_HPP

#include "quantized_block_pipeline.hpp"
#include "quantized_layer_task.hpp"
#ifdef QUANTIZED_ALIGNMENT_W8
#include "mm_stream_8x128_int4x4_block.hpp"
#include "mm_stream_4x128_int8x8_block.hpp"
#include "quantized_w8_projection_engine.hpp"
#else
#include "mm_stream_8x128_int4x4_block.hpp"
#include "quantized_w4_projection_engine.hpp"
#endif

#ifdef QUANTIZED_ALIGNMENT_W8
using quantized_block_pipeline_policy_t = quantized_w8_projection_policy;
using quantized_block_pipeline_hidden_t = quantized_w8_hidden_buffer_t;
using quantized_block_pipeline_wide_t = quantized_w8_wide_buffer_t;
using quantized_block_pipeline_weight_t = quantized_w8_weight_word_t;
constexpr unsigned int QUANTIZED_BLOCK_PIPELINE_HIDDEN_BLOCKS =
    QUANTIZED_W8_HIDDEN_BLOCKS;
constexpr unsigned int QUANTIZED_BLOCK_PIPELINE_WIDE_BLOCKS =
    QUANTIZED_W8_WIDE_BLOCKS;
constexpr unsigned int QUANTIZED_BLOCK_PIPELINE_TOKEN_ROWS =
    QUANTIZED_W8_RESIDENT_TOKEN_ROWS;
constexpr unsigned int QUANTIZED_BLOCK_PIPELINE_BITS = 8;
constexpr unsigned int QUANTIZED_BLOCK_PIPELINE_WEIGHT_DEPTH =
    QUANTIZED_W8_WEIGHT_MEMORY_DEPTH;
#else
using quantized_block_pipeline_policy_t = quantized_w4_projection_policy;
using quantized_block_pipeline_hidden_t = quantized_w4_hidden_buffer_t;
using quantized_block_pipeline_wide_t = quantized_w4_wide_buffer_t;
using quantized_block_pipeline_weight_t = quantized_w4_weight_word_t;
constexpr unsigned int QUANTIZED_BLOCK_PIPELINE_HIDDEN_BLOCKS =
    QUANTIZED_W4_HIDDEN_BLOCKS;
constexpr unsigned int QUANTIZED_BLOCK_PIPELINE_WIDE_BLOCKS =
    QUANTIZED_W4_WIDE_BLOCKS;
constexpr unsigned int QUANTIZED_BLOCK_PIPELINE_TOKEN_ROWS =
    QUANTIZED_W4_RESIDENT_TOKEN_ROWS;
constexpr unsigned int QUANTIZED_BLOCK_PIPELINE_BITS = 4;
constexpr unsigned int QUANTIZED_BLOCK_PIPELINE_WEIGHT_DEPTH =
    QUANTIZED_W4_WEIGHT_MEMORY_DEPTH;
#endif

constexpr unsigned int QUANTIZED_BLOCK_PIPELINE_HIDDEN_WORDS =
    QUANTIZED_BLOCK_PIPELINE_TOKEN_ROWS *
    QUANTIZED_BLOCK_PIPELINE_HIDDEN_BLOCKS;
constexpr unsigned int QUANTIZED_BLOCK_PIPELINE_WIDE_WORDS =
    QUANTIZED_BLOCK_PIPELINE_TOKEN_ROWS *
    QUANTIZED_BLOCK_PIPELINE_WIDE_BLOCKS;
constexpr unsigned int QUANTIZED_BLOCK_PIPELINE_WEIGHT_PORTS =
    quantized_block_pipeline_policy_t::weight_ports;

// A single closed-loop invocation represents one physical block.  The
// projection writes the current block to projection_output; when enabled,
// RMSNorm of next_hidden_input is issued ahead of the matrix waves and lands
// in next_normalized_output.  The output arrays are deliberately sized to
// the resident W4/W8 tile rather than the full model, keeping this fixture a
// bounded CoSim contract.
void quantized_block_pipeline_closed_loop(
    const mm_input_block_t hidden_input[QUANTIZED_BLOCK_PIPELINE_HIDDEN_WORDS],
    const mm_input_block_t wide_input[QUANTIZED_BLOCK_PIPELINE_WIDE_WORDS],
    const mm_input_block_t next_hidden_input[QUANTIZED_BLOCK_PIPELINE_HIDDEN_WORDS],
    const mm_input_block_t norm_weight_input[QUANTIZED_BLOCK_PIPELINE_HIDDEN_WORDS],
    mm_input_block_t projection_output[QUANTIZED_BLOCK_PIPELINE_WIDE_WORDS],
    mm_input_block_t next_normalized_output[QUANTIZED_BLOCK_PIPELINE_HIDDEN_WORDS],
    const quantized_block_pipeline_weight_t
        weight_memory[4][QUANTIZED_BLOCK_PIPELINE_WEIGHT_PORTS]
                     [QUANTIZED_BLOCK_PIPELINE_WEIGHT_DEPTH],
    unsigned int projection, unsigned int rows, unsigned int next_rows,
    unsigned int enable_lookahead, unsigned int layer);

#endif
