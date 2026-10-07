#ifndef LLM_FPGA_TOKEN_PIPELINE_HPP
#define LLM_FPGA_TOKEN_PIPELINE_HPP

#include "mm_global_cache.hpp"
#include "token_cache.hpp"

constexpr unsigned int QKV_TILE_Q_OFFSET = 0;
constexpr unsigned int QKV_TILE_K_OFFSET = HIDDEN_SIZE;
constexpr unsigned int QKV_TILE_V_OFFSET = HIDDEN_SIZE + KV_CHANNELS;
constexpr unsigned int QKV_TILE_OUT_DIM = HIDDEN_SIZE + 2 * KV_CHANNELS;

void compute_qkv_token_tile_from_cached_blocks(
    fm_t qkv_output[LINEAR_TOKEN_TILE_ACTIVE][QKV_TILE_OUT_DIM],
    linear_in_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    unsigned int token_count,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_qkv_token_tile_full_from_cached_blocks(
    fm_t qkv_output[LINEAR_TOKEN_TILE_ACTIVE][QKV_TILE_OUT_DIM],
    linear_in_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_qkv_token_chunks_with_mm_global_cache(
    fm_t qkv_chunks[MM_GLOBAL_CACHE_MAX_CHUNKS][LINEAR_TOKEN_TILE_ACTIVE][QKV_TILE_OUT_DIM],
    mm_input_block_t input_chunks[MM_GLOBAL_CACHE_MAX_CHUNKS][LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    unsigned int total_tokens,
    QWEN_WEIGHT_SHARD_PARAMS
);

void write_qkv_tile_outputs(
    TOKEN_TILE_OUTPUT_PORT_PARAMS,
    fm_t qkv_output[LINEAR_TOKEN_TILE_ACTIVE][QKV_TILE_OUT_DIM],
    unsigned int token_count
);

void write_qkv_tile_outputs_full(
    TOKEN_TILE_OUTPUT_PORT_PARAMS,
    fm_t qkv_output[LINEAR_TOKEN_TILE_ACTIVE][QKV_TILE_OUT_DIM]
);

#endif
