#ifndef LLM_FPGA_QKV_TILE_KERNEL_HPP
#define LLM_FPGA_QKV_TILE_KERNEL_HPP

#include "token_pipeline.hpp"

extern "C" void qkv_tile_kernel(
    TOKEN_TILE_OUTPUT_PORT_PARAMS,
    TOKEN_TILE_INPUT_PORT_PARAMS,
    unsigned int token_count,
    QWEN_WEIGHT_SHARD_PARAMS
);

#endif
