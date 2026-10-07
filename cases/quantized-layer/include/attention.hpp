#ifndef LLM_FPGA_ATTENTION_HPP
#define LLM_FPGA_ATTENTION_HPP

#include "datatypes.hpp"

void store_kv_cache(
    fm_t kv_cache_k[KV_CACHE_ELEMS],
    fm_t kv_cache_v[KV_CACHE_ELEMS],
    const fm_t k[KV_CHANNELS],
    const fm_t v[KV_CHANNELS],
    unsigned int layer,
    unsigned int position
);

void compute_gqa_attention(
    fm_t output[HIDDEN_SIZE],
    const fm_t q[HIDDEN_SIZE],
    const fm_t kv_cache_k[KV_CACHE_ELEMS],
    const fm_t kv_cache_v[KV_CACHE_ELEMS],
    unsigned int layer,
    unsigned int position
);

#ifndef __SYNTHESIS__
struct attention_debug_counters_t {
    unsigned long long calls;
    unsigned long long tile_loads;
    unsigned long long last_call_tiles;
    unsigned long long max_call_tiles;
};

void reset_attention_debug_counters();
attention_debug_counters_t get_attention_debug_counters();
unsigned int attention_expected_tiles_for_position(unsigned int position);
#endif

#endif
