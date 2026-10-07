#ifndef LLM_FPGA_LINEAR_HPP
#define LLM_FPGA_LINEAR_HPP

#include "datatypes.hpp"
#include "hardware.hpp"
#include <hls_stream.h>

#define QWEN_WEIGHT_SHARD_PARAMS \
    const wt_block_t* layer_weights_shard0, \
    const wt_block_t* layer_weights_shard1, \
    const wt_block_t* layer_weights_shard2, \
    const wt_block_t* layer_weights_shard3, \
    const wt_block_t* layer_weights_shard4, \
    const wt_block_t* layer_weights_shard5, \
    const wt_block_t* layer_weights_shard6, \
    const wt_block_t* layer_weights_shard7, \
    const wt_block_t* layer_weights_shard8, \
    const wt_block_t* layer_weights_shard9, \
    const wt_block_t* layer_weights_shard10, \
    const wt_block_t* layer_weights_shard11, \
    const wt_block_t* layer_weights_shard12, \
    const wt_block_t* layer_weights_shard13, \
    const wt_block_t* layer_weights_shard14, \
    const wt_block_t* layer_weights_shard15

#define QWEN_WEIGHT_SHARD_ARGS \
    layer_weights_shard0, \
    layer_weights_shard1, \
    layer_weights_shard2, \
    layer_weights_shard3, \
    layer_weights_shard4, \
    layer_weights_shard5, \
    layer_weights_shard6, \
    layer_weights_shard7, \
    layer_weights_shard8, \
    layer_weights_shard9, \
    layer_weights_shard10, \
    layer_weights_shard11, \
    layer_weights_shard12, \
    layer_weights_shard13, \
    layer_weights_shard14, \
    layer_weights_shard15

void read_linear_input_stream(
    hls::stream<linear_in_t>& in_stream,
    const fm_t input[],
    unsigned int in_dim
);

void compute_linear_on_stream(
    fm_t output[],
    hls::stream<linear_in_t>& in_stream,
    weight_addr_t weight_base,
    unsigned int out_dim,
    unsigned int in_dim,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_linear_from_shards(
    fm_t output[],
    const fm_t input[],
    weight_addr_t weight_base,
    unsigned int out_dim,
    unsigned int in_dim,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_linear_token_tile_from_shards(
    fm_t output[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM],
    const fm_t input[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_DIM],
    unsigned int token_count,
    weight_addr_t weight_base,
    unsigned int out_dim,
    unsigned int in_dim,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_linear_token_tile_from_cached_blocks(
    fm_t output[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM],
    linear_in_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    unsigned int token_count,
    weight_addr_t weight_base,
    unsigned int out_dim,
    unsigned int in_dim,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_linear_token_tile_full_from_cached_blocks(
    fm_t output[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM],
    linear_in_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    weight_addr_t weight_base,
    unsigned int out_dim,
    unsigned int in_dim,
    QWEN_WEIGHT_SHARD_PARAMS
);

#endif
