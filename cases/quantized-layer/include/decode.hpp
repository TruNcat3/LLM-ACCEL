#ifndef LLM_FPGA_DECODE_HPP
#define LLM_FPGA_DECODE_HPP

#include "linear.hpp"

void load_token_embedding(
    fm_t hidden[HIDDEN_SIZE],
    unsigned int token_id,
    const wt_linear_t token_embedding[TOKEN_EMBEDDING_ELEMS]
);

void write_hidden_output(
    fm_t output_hidden[HIDDEN_SIZE],
    const fm_t hidden[HIDDEN_SIZE]
);

void run_decoder_layer(
    fm_t hidden[HIDDEN_SIZE],
    unsigned int layer,
    unsigned int position,
    const wt_norm_t norm_weights[NORM_WEIGHT_ELEMS],
    fm_t kv_cache_k[KV_CACHE_ELEMS],
    fm_t kv_cache_v[KV_CACHE_ELEMS],
    QWEN_WEIGHT_SHARD_PARAMS
);

#endif
