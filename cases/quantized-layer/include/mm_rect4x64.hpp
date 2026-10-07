#ifndef LLM_FPGA_MM_RECT4X64_HPP
#define LLM_FPGA_MM_RECT4X64_HPP

#include "compute_stream.hpp"
#include "mm_controller.hpp"

constexpr unsigned int MM_RECT_K = 4;
constexpr unsigned int MM_RECT_OUT = 64;
constexpr unsigned int MM_RECT_TOKEN_GROUPS = 2;
constexpr unsigned int MM_RECT_OUT_GROUPS = 2;
constexpr unsigned int MM_RECT_OUTPUT_BLOCK = MM_RECT_OUT * MM_RECT_OUT_GROUPS;
constexpr unsigned int MM_RECT_TOKENS_PER_ISLAND = 4;
constexpr unsigned int MM_RECT_OUTPUT_TILES_PER_BLOCK = MM_RECT_OUTPUT_BLOCK / MM_PE_OUT;
constexpr unsigned int MM_RECT_PACKETS_PER_TOKEN = MM_RECT_OUTPUT_BLOCK / CU_VEC_LANES;
constexpr unsigned int MM_RECT_PACKETS_PER_BLOCK = MM_CONTROLLER_TOKEN_LANES * MM_RECT_PACKETS_PER_TOKEN;

static_assert(MM_CONTROLLER_TOKEN_LANES == MM_RECT_TOKEN_GROUPS * MM_RECT_TOKENS_PER_ISLAND,
              "rect4x64 v1 maps 8 token lanes into two 4-token groups");
static_assert(MM_RECT_OUTPUT_BLOCK == 128, "rect4x64 v1 emits 128 output columns per block");
static_assert(MM_RECT_OUTPUT_TILES_PER_BLOCK == 8, "rect4x64 v1 decodes eight 16x16 tiles per block");

struct mm_rect_input_micro_wave_t {
    fm_t token[MM_CONTROLLER_TOKEN_LANES][MM_RECT_K];
};

struct mm_rect_weight_micro_wave_t {
    wt_linear_t value[MM_RECT_OUT_GROUPS][MM_RECT_OUT][MM_RECT_K];
};

struct mm_rect_partial_wave_t {
    fm_accum_t value[MM_CONTROLLER_TOKEN_LANES][MM_RECT_OUTPUT_BLOCK];
};

struct mm_rect_accum_bank_t {
    fm_accum_t value[MM_CONTROLLER_TOKEN_LANES][MM_RECT_OUTPUT_BLOCK];
};

struct mm_rect_packed_weight_tiles_t {
    wt_block_t block[MM_RECT_OUTPUT_TILES_PER_BLOCK][MM_TILE_WEIGHT_BLOCKS];
};

void compute_mm_rect4x64_block_stream(
    hls::stream<cu_accum16_packet_t>& out_stream,
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    const mm_controller_task_t& task,
    unsigned int output_block,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_mm_rect4x64_block_stream_lean_full(
    hls::stream<cu_accum16_packet_t>& out_stream,
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    const mm_controller_task_t& task,
    unsigned int output_block,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_mm_rect4x64_island_synth_stub(
    fm_t sink[LINEAR_TOKEN_TILE_ACTIVE],
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE],
    wt_block_t weight_blocks[MM_RECT_OUTPUT_TILES_PER_BLOCK][MM_TILE_WEIGHT_BLOCKS]
);

void compute_mm_rect4x64_block_perf_stub(
    fm_t sink[LINEAR_TOKEN_TILE_ACTIVE],
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    unsigned int token_count,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int output_block,
    weight_addr_t weight_base,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_mm_rect4x64_block_lean_full_synth_stub(
    fm_t sink[LINEAR_TOKEN_TILE_ACTIVE],
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    unsigned int in_dim,
    unsigned int output_block,
    weight_addr_t weight_base,
    QWEN_WEIGHT_SHARD_PARAMS
);

#ifndef QWEN_MM_RECT4X64_BLOCK_ONLY

void compute_ffn_stream_pipeline_local(
    fm_t down_output[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_OUT_DIM],
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    unsigned int token_count,
    weight_addr_t layer_base,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_ffn_stream_pipeline_local_synth_stub(
    fm_t sink[LINEAR_TOKEN_TILE_ACTIVE],
    QWEN_WEIGHT_SHARD_PARAMS
);

#endif

#endif
