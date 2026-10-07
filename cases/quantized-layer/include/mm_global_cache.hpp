#ifndef LLM_FPGA_MM_GLOBAL_CACHE_HPP
#define LLM_FPGA_MM_GLOBAL_CACHE_HPP

#include "mm_cache.hpp"

struct mm_global_cache_slot_t {
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS];
    fm_t output[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM];
};

struct mm_global_cache_pair_t {
    mm_global_cache_slot_t slot[MM_GLOBAL_BUFFER_COUNT];
};

unsigned int get_mm_token_chunk_count(unsigned int total_tokens);

unsigned int get_mm_chunk_token_count(
    unsigned int total_tokens,
    unsigned int chunk
);

void pack_mm_global_cache_input_chunks(
    mm_input_block_t input_chunks[MM_GLOBAL_CACHE_MAX_CHUNKS][LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    fm_t input_tokens[MM_GLOBAL_CACHE_TOKEN_SLOTS][MAX_LINEAR_IN_DIM],
    unsigned int total_tokens,
    unsigned int in_dim
);

void load_mm_global_cache_slot_from_chunks(
    mm_global_cache_slot_t& slot,
    mm_input_block_t input_chunks[MM_GLOBAL_CACHE_MAX_CHUNKS][LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    unsigned int chunk,
    unsigned int chunk_tokens,
    unsigned int in_tile_count
);

void compute_mm_global_cache_slot(
    mm_global_cache_slot_t& slot,
    mm_controller_mode_t mode,
    unsigned int chunk_tokens,
    weight_addr_t weight_base,
    unsigned int out_dim,
    unsigned int in_dim,
    QWEN_WEIGHT_SHARD_PARAMS
);

void store_mm_global_cache_slot_to_chunks(
    fm_t output_chunks[MM_GLOBAL_CACHE_MAX_CHUNKS][LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM],
    const mm_global_cache_slot_t& slot,
    unsigned int chunk,
    unsigned int chunk_tokens,
    unsigned int out_dim
);

void compute_mm_double_buffer_pipeline_cached_chunks(
    fm_t output_chunks[MM_GLOBAL_CACHE_MAX_CHUNKS][LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM],
    mm_input_block_t input_chunks[MM_GLOBAL_CACHE_MAX_CHUNKS][LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    unsigned int total_tokens,
    mm_controller_mode_t mode,
    weight_addr_t weight_base,
    unsigned int out_dim,
    unsigned int in_dim,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_mm_double_buffer_pipeline_cached_chunks_banked(
    fm_t output_chunks[MM_GLOBAL_CACHE_MAX_CHUNKS][LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM],
    mm_input_block_t input_chunks[MM_GLOBAL_CACHE_MAX_CHUNKS][LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    unsigned int total_tokens,
    mm_controller_mode_t mode,
    weight_addr_t weight_base,
    unsigned int out_dim,
    unsigned int in_dim,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_mm_projection_chunks_with_global_cache(
    fm_t output_chunks[MM_GLOBAL_CACHE_MAX_CHUNKS][LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM],
    mm_input_block_t input_chunks[MM_GLOBAL_CACHE_MAX_CHUNKS][LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    unsigned int total_tokens,
    mm_projection_kind_t projection,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_mm_global_cache_local_synth_stub(
    fm_t sink[MM_GLOBAL_CACHE_TOKEN_SLOTS],
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_mm_global_cache_banked_local_synth_stub(
    fm_t sink[MM_GLOBAL_CACHE_TOKEN_SLOTS],
    QWEN_WEIGHT_SHARD_PARAMS
);

#endif
