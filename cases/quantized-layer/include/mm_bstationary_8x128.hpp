#ifndef LLM_FPGA_MM_BSTATIONARY_8X128_HPP
#define LLM_FPGA_MM_BSTATIONARY_8X128_HPP

#include "compute_stream.hpp"
#include "linear.hpp"

constexpr unsigned int MM_BS_TOKENS = LINEAR_TOKEN_TILE_ACTIVE;
constexpr unsigned int MM_BS_K_TILE = MM_PE_IN;
constexpr unsigned int MM_BS_OUTPUT_BLOCK = 128;
constexpr unsigned int MM_BS_OUTPUT_TILES_PER_BLOCK = MM_BS_OUTPUT_BLOCK / MM_PE_OUT;
#ifndef MM_BS_ACCUM_BANKS
#define MM_BS_ACCUM_BANKS 4
#endif
constexpr unsigned int MM_BS_PACKETS_PER_TOKEN = MM_BS_OUTPUT_BLOCK / CU_VEC_LANES;
constexpr unsigned int MM_BS_PACKETS_PER_BLOCK = MM_BS_TOKENS * MM_BS_PACKETS_PER_TOKEN;

static_assert(MM_BS_TOKENS == 8, "bstationary v1 keeps the full 8-token block");
static_assert(MM_BS_OUTPUT_BLOCK == 128, "bstationary v1 keeps the full 128-output block");
static_assert(MM_BS_OUTPUT_TILES_PER_BLOCK == 8, "bstationary v1 maps eight 16-output tiles");
static_assert(MM_BS_ACCUM_BANKS == 2 || MM_BS_ACCUM_BANKS == 4, "bstationary v1 supports 2 or 4 accumulator banks");

struct mm_bs_input_tile_t {
    fm_t value[MM_BS_TOKENS][MM_BS_K_TILE];
};

struct mm_bs_weight_panel_t {
    wt_linear_t value[MM_BS_K_TILE][MM_BS_OUTPUT_BLOCK];
};

struct mm_bs_accum_bank_t {
    fm_accum_t value[MM_BS_TOKENS][MM_BS_OUTPUT_BLOCK];
};

void compute_mm_bstationary_8x128_block_stream(
    hls::stream<cu_accum16_packet_t>& out_stream,
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    unsigned int in_dim,
    unsigned int output_block,
    weight_addr_t weight_base,
    QWEN_WEIGHT_SHARD_PARAMS
);

void compute_mm_bstationary_8x128_core_synth_stub(
    fm_t sink[LINEAR_TOKEN_TILE_ACTIVE],
    mm_input_block_t input_blocks[LINEAR_TOKEN_TILE_ACTIVE][MAX_LINEAR_IN_BLOCKS],
    unsigned int in_dim,
    unsigned int output_block,
    weight_addr_t weight_base,
    QWEN_WEIGHT_SHARD_PARAMS
);

#endif
