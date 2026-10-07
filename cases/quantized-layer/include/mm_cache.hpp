#ifndef LLM_FPGA_MM_CACHE_HPP
#define LLM_FPGA_MM_CACHE_HPP

#include "mm_controller.hpp"

struct mm_input_wave_t {
    linear_in_t lane[MM_CONTROLLER_TOKEN_LANES];
};

struct mm_weight_wave_t {
    wt_linear_block_t slot[MM_CONTROLLER_OUT_TILE_PARALLEL];
};

struct mm_accum_wave_t {
    fm_accum_t value[MM_CONTROLLER_TOKEN_LANES][MM_CONTROLLER_OUT_TILE_PARALLEL][MM_PE_OUT];
};

constexpr unsigned int MM_ACCUM_RING_BANKS = 8;
static_assert((MM_ACCUM_RING_BANKS & (MM_ACCUM_RING_BANKS - 1)) == 0, "accum ring bank count must be power-of-two");

struct mm_accum_ring_t {
    fm_accum_t value[MM_ACCUM_RING_BANKS][MM_CONTROLLER_TOKEN_LANES][MM_CONTROLLER_OUT_TILE_PARALLEL][MM_PE_OUT];
};

void clear_mm_accum_wave(mm_accum_wave_t& accum);

void clear_mm_accum_ring(mm_accum_ring_t& ring);

void load_mm_input_wave_from_cached_blocks(
    mm_input_wave_t& wave,
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    unsigned int in_tile,
    unsigned int token_count
);

void load_mm_weight_wave_from_shards(
    mm_weight_wave_t& wave,
    const mm_controller_task_t& task,
    unsigned int output_wave,
    unsigned int in_tile,
    QWEN_WEIGHT_SHARD_PARAMS
);

void load_mm_weight_blocks_from_shards(
    wt_block_t cached_weight_blocks[MM_CONTROLLER_OUT_TILE_PARALLEL][MM_TILE_WEIGHT_BLOCKS],
    const mm_controller_task_t& task,
    unsigned int output_wave,
    unsigned int in_tile,
    QWEN_WEIGHT_SHARD_PARAMS
);

void load_mm_weight_wave_from_cached_blocks(
    mm_weight_wave_t& wave,
    wt_block_t cached_weight_blocks[MM_CONTROLLER_OUT_TILE_PARALLEL][MM_TILE_WEIGHT_BLOCKS],
    const mm_controller_task_t& task,
    unsigned int output_wave,
    unsigned int in_tile
);

void compute_mm_wave_16core(
    mm_accum_wave_t& accum,
    const mm_input_wave_t& input_wave,
    const mm_weight_wave_t& weight_wave,
    const mm_controller_task_t& task,
    unsigned int output_wave
);

void reduce_mm_accum_ring(
    mm_accum_wave_t& accum,
    const mm_accum_ring_t& ring
);

void store_mm_accum_wave_to_output(
    fm_t output[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM],
    const mm_accum_wave_t& accum,
    const mm_controller_task_t& task,
    unsigned int output_wave
);

void store_mm_accum_ring_to_output(
    fm_t output[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM],
    const mm_accum_ring_t& ring,
    const mm_controller_task_t& task,
    unsigned int output_wave
);

void compute_mm_cache_pipeline_from_cached_blocks(
    fm_t output[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM],
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    mm_controller_mode_t mode,
    unsigned int token_count,
    weight_addr_t weight_base,
    unsigned int out_dim,
    unsigned int in_dim,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_mm_cache_pipeline_full_from_cached_blocks(
    fm_t output[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM],
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    mm_controller_mode_t mode,
    weight_addr_t weight_base,
    unsigned int out_dim,
    unsigned int in_dim,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_mm_cache_pipeline_local_synth_stub(
    fm_t sink[LINEAR_TOKEN_TILE_ACTIVE],
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_mm_cache_pipeline_full_local_synth_stub(
    fm_t sink[LINEAR_TOKEN_TILE_ACTIVE],
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_mm_cache_ring_synth_stub(
    fm_t sink[LINEAR_TOKEN_TILE_ACTIVE],
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    wt_block_t weight_blocks[MM_CONTROLLER_OUT_TILE_PARALLEL][MM_TILE_WEIGHT_BLOCKS]
);

#endif
