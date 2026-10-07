#ifndef LLM_FPGA_TOKEN_CACHE_HPP
#define LLM_FPGA_TOKEN_CACHE_HPP

#include "linear.hpp"

#define TOKEN_TILE_OUTPUT_PORT_PARAMS \
    fm_t* output_token0, \
    fm_t* output_token1, \
    fm_t* output_token2, \
    fm_t* output_token3, \
    fm_t* output_token4, \
    fm_t* output_token5, \
    fm_t* output_token6, \
    fm_t* output_token7

#define TOKEN_TILE_INPUT_PORT_PARAMS \
    const fm_t* input_token0, \
    const fm_t* input_token1, \
    const fm_t* input_token2, \
    const fm_t* input_token3, \
    const fm_t* input_token4, \
    const fm_t* input_token5, \
    const fm_t* input_token6, \
    const fm_t* input_token7

void pack_token_tile_input_blocks(
    linear_in_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    TOKEN_TILE_INPUT_PORT_PARAMS,
    unsigned int in_dim
);

void pack_token_tile_mm_input_blocks(
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    TOKEN_TILE_INPUT_PORT_PARAMS,
    unsigned int in_dim
);

void write_token_tile_outputs(
    TOKEN_TILE_OUTPUT_PORT_PARAMS,
    fm_t output_local[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM],
    unsigned int token_count,
    unsigned int out_dim
);

#endif
