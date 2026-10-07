#include "quantized_w8_layer_math.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool close_to(fm_t actual, double expected, double tolerance = 0.02) {
    return std::abs(static_cast<double>(actual) - expected) <= tolerance;
}

}  // namespace

int main() {
    assert(quantized_w8_kv_cache_bytes() == 37748736ULL);
    const std::size_t test_cache_words =
        quantized_w8_kv_word_index(2, 11, QUANTIZED_W8_KV_BLOCKS - 1) + 1;
    std::vector<mm_input_block_t> key_cache(test_cache_words);
    std::vector<mm_input_block_t> value_cache(test_cache_words);
    quantized_w8_kv_buffer_t key{};
    quantized_w8_kv_buffer_t value{};
    quantized_w8_feature_set(key, 0, 17, fm_t(3.5));
    quantized_w8_feature_set(value, 0, 129, fm_t(-2.25));
    store_quantized_w8_current_kv(
        key_cache.data(), value_cache.data(), key, value, 2, 11, 1);
    assert(load_quantized_w8_kv_value(
               key_cache.data(), 2, 11, 17) == fm_t(3.5));
    assert(load_quantized_w8_kv_value(
               value_cache.data(), 2, 11, 129) == fm_t(-2.25));

    quantized_w8_hidden_buffer_t query{};
    fm_t cosine[HEAD_DIM / 2];
    fm_t sine[HEAD_DIM / 2];
    for (unsigned int i = 0; i < HEAD_DIM / 2; ++i) {
        cosine[i] = fm_t(0);
        sine[i] = fm_t(1);
    }
    quantized_w8_feature_set(query, 0, 0, fm_t(2));
    quantized_w8_feature_set(query, 0, HEAD_DIM / 2, fm_t(3));
    quantized_w8_feature_set(query, 0, 15, fm_t(-2));
    quantized_w8_feature_set(query, 0, HEAD_DIM / 2 + 15, fm_t(7));
    quantized_w8_feature_set(key, 0, 0, fm_t(4));
    quantized_w8_feature_set(key, 0, HEAD_DIM / 2, fm_t(5));
    quantized_w8_feature_set(key, 0, 15, fm_t(6));
    quantized_w8_feature_set(key, 0, HEAD_DIM / 2 + 15, fm_t(-1));
    apply_quantized_w8_rope(query, key, 0, cosine, sine);
    assert(quantized_w8_feature_get(query, 0, 0) == fm_t(-3));
    assert(quantized_w8_feature_get(query, 0, HEAD_DIM / 2) == fm_t(2));
    assert(quantized_w8_feature_get(query, 0, 15) == fm_t(-7));
    assert(quantized_w8_feature_get(query, 0, HEAD_DIM / 2 + 15) == fm_t(-2));
    assert(quantized_w8_feature_get(key, 0, 0) == fm_t(-5));
    assert(quantized_w8_feature_get(key, 0, HEAD_DIM / 2) == fm_t(4));
    assert(quantized_w8_feature_get(key, 0, 15) == fm_t(1));
    assert(quantized_w8_feature_get(key, 0, HEAD_DIM / 2 + 15) == fm_t(6));

    quantized_w8_online_attention_state_t state{};
    quantized_w8_attention_score_tile_t scores{};
    quantized_w8_attention_probability_tile_t probabilities{};
    attention_prob_t old_scale[QUANTIZED_W8_RESIDENT_TOKEN_ROWS]
                                  [NUM_ATTENTION_HEADS]{};
    init_quantized_w8_online_attention(state);
    const quantized_layer_task_t attention_task = make_quantized_decode_task(
        0, 128, 128, 0, 1, QUANTIZED_ACTIVATION_INT8,
        QUANTIZED_WEIGHT_INT8, 0, 0, QUANTIZED_LAYER_W8_TOKEN_BLOCK);
    for (unsigned int column = 0; column < 128; ++column) {
        quantized_w8_score_set(scores, 0, 0, column, fm_t(0));
    }
    update_quantized_w8_online_probabilities(
        state, probabilities, old_scale, scores, attention_task, 0, 128);
    assert(old_scale[0][0] == attention_prob_t(0));
    assert(std::abs(static_cast<double>(state.sum[0][0]) - 128.0) <= 0.1);
    quantized_w8_hidden_buffer_t first_tile_context{};
    quantized_w8_feature_set(first_tile_context, 0, 0, fm_t(64));
    merge_quantized_w8_attention_tile(
        state, first_tile_context, old_scale, 1);

    quantized_w8_score_set(scores, 0, 0, 0, fm_t(0));
    update_quantized_w8_online_probabilities(
        state, probabilities, old_scale, scores, attention_task, 128, 1);
    assert(close_to(old_scale[0][0], 1.0));
    assert(std::abs(static_cast<double>(state.sum[0][0]) - 129.0) <= 0.1);
    quantized_w8_hidden_buffer_t second_tile_context{};
    quantized_w8_feature_set(second_tile_context, 0, 0, fm_t(2));
    merge_quantized_w8_attention_tile(
        state, second_tile_context, old_scale, 1);
    quantized_w8_hidden_buffer_t attention_output{};
    finalize_quantized_w8_online_attention(attention_output, state, 1);
    assert(close_to(
        quantized_w8_feature_get(attention_output, 0, 0),
        66.0 / 129.0, 0.03));

    quantized_w8_hidden_buffer_t norm_source{};
    quantized_w8_hidden_buffer_t norm_weight{};
    quantized_w8_hidden_buffer_t norm_output{};
    for (unsigned int element = 0; element < HIDDEN_SIZE; ++element) {
        quantized_w8_feature_set(norm_source, 0, element, fm_t(2));
        quantized_w8_feature_set(norm_weight, 0, element, fm_t(1));
    }
    quantized_w8_rmsnorm(
        norm_source, norm_weight, norm_output, 1, HIDDEN_SIZE);
    assert(close_to(quantized_w8_feature_get(norm_output, 0, 0), 1.0));

    assert(close_to(quantized_w8_silu(fm_t(0)), 0.0, 0.01));
    assert(close_to(
        quantized_w8_silu(fm_t(1)), 1.0 / (1.0 + std::exp(-1.0)), 0.02));
    assert(close_to(
        quantized_w8_silu(fm_t(-1)), -1.0 / (1.0 + std::exp(1.0)), 0.02));

    quantized_w8_wide_buffer_t gate_product{};
    quantized_w8_wide_buffer_t up{};
    for (unsigned int lane = 0;
         lane < QUANTIZED_W8_RESIDENT_LANES_PER_WORD; ++lane) {
        quantized_w8_feature_set(
            gate_product, 0, lane, lane & 1 ? fm_t(-1) : fm_t(1));
        quantized_w8_feature_set(up, 0, lane, fm_t(2));
    }
    quantized_w8_silu_multiply_inplace(gate_product, up, 1);
    assert(close_to(
        quantized_w8_feature_get(gate_product, 0, 0),
        2.0 / (1.0 + std::exp(-1.0)), 0.04));
    assert(close_to(
        quantized_w8_feature_get(gate_product, 0, 15),
        -2.0 / (1.0 + std::exp(1.0)), 0.04));
    assert(quantized_w8_feature_get(gate_product, 1, 0) == fm_t(0));

    std::cout << "QUANTIZED W8 LAYER MATH PASS "
              << "kv_bytes=" << quantized_w8_kv_cache_bytes()
              << " online_tiles=2 rmsnorm=1\n";
    return 0;
}
