#ifndef LLM_FPGA_CLOSED_LOOP_8X64_COSIM_HPP
#define LLM_FPGA_CLOSED_LOOP_8X64_COSIM_HPP

#include "control_cache_8x64.hpp"

void cc8_closed_loop_inner_cosim(
    fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t status_output[1],
    const fm_word_t input_port0[CC8_DATA_PORT_WORDS],
    const fm_word_t input_port1[CC8_DATA_PORT_WORDS],
    const fm_word_t aux_port0[CC8_DATA_PORT_WORDS],
    const fm_word_t aux_port1[CC8_DATA_PORT_WORDS],
    unsigned int operator_kind,
    unsigned int layer_id,
    unsigned int token_count,
    unsigned int position,
    unsigned int tile_len,
    QWEN_WEIGHT_SHARD_PARAMS,
    fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS],
    fm_word_t kv_cache_v[CC8_KV_CACHE_WORDS]
);

void cc8_closed_loop_nk_cosim(
    fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t status_output[1],
    const fm_word_t input_port0[CC8_DATA_PORT_WORDS],
    const fm_word_t input_port1[CC8_DATA_PORT_WORDS],
    const fm_word_t aux_port0[CC8_DATA_PORT_WORDS],
    const fm_word_t aux_port1[CC8_DATA_PORT_WORDS],
    unsigned int operator_kind,
    unsigned int layer_id,
    unsigned int token_count,
    unsigned int position,
    unsigned int tile_len,
    QWEN_WEIGHT_SHARD_PARAMS,
    fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS],
    fm_word_t kv_cache_v[CC8_KV_CACHE_WORDS]
);

#endif
