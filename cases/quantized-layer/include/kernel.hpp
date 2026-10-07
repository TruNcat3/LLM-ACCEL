#ifndef LLM_FPGA_KERNEL_HPP
#define LLM_FPGA_KERNEL_HPP

#include "linear.hpp"

constexpr unsigned int QWEN_DEBUG_NONE = 0;
constexpr unsigned int QWEN_DEBUG_EMBEDDING = 1;
constexpr unsigned int QWEN_DEBUG_PREFILL = 2;
constexpr unsigned int QWEN_DEBUG_LAYER_BASE = 100;

extern "C" {
void qwen_kernel(
    fm_t output_hidden[HIDDEN_SIZE],
    unsigned int token_id,
    unsigned int position,
    unsigned int num_layers,
    unsigned int debug_id,
    const wt_linear_t token_embedding[TOKEN_EMBEDDING_ELEMS],
    QWEN_WEIGHT_SHARD_PARAMS,
    const wt_norm_t norm_weights[NORM_WEIGHT_ELEMS],
    fm_t kv_cache_k[KV_CACHE_ELEMS],
    fm_t kv_cache_v[KV_CACHE_ELEMS],
    fm_t scratch[SCRATCH_ELEMS]
);
}

#endif
