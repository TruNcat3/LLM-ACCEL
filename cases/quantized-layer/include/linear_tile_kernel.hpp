#ifndef LLM_FPGA_LINEAR_TILE_KERNEL_HPP
#define LLM_FPGA_LINEAR_TILE_KERNEL_HPP

#include "token_cache.hpp"

#define LINEAR_TILE_OUTPUT_PORT_PARAMS TOKEN_TILE_OUTPUT_PORT_PARAMS
#define LINEAR_TILE_INPUT_PORT_PARAMS TOKEN_TILE_INPUT_PORT_PARAMS

extern "C" void linear_tile_kernel(
    LINEAR_TILE_OUTPUT_PORT_PARAMS,
    LINEAR_TILE_INPUT_PORT_PARAMS,
    unsigned int token_count,
    weight_addr_t weight_base,
    unsigned int out_dim,
    unsigned int in_dim,
    QWEN_WEIGHT_SHARD_PARAMS
);

#endif
