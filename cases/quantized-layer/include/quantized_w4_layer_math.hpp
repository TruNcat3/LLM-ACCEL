#ifndef LLM_FPGA_QUANTIZED_W4_LAYER_MATH_HPP
#define LLM_FPGA_QUANTIZED_W4_LAYER_MATH_HPP

#include "quantized_w4_resident_buffers.hpp"
#include "quantized_w4_attention_schedule.hpp"

#include "quantized_layer_scalar_pipeline.hpp"

struct quantized_w4_attention_score_tile_t {
    static constexpr unsigned int kLanesPerWord =
        MM_INPUT_BLOCK_BIT_WIDTH / fm_t::width;
    static constexpr unsigned int kBlocks =
        (QUANTIZED_W4_ATTENTION_POSITION_TILE + kLanesPerWord - 1) /
            kLanesPerWord;
    mm_input_block_t block[QUANTIZED_W4_RESIDENT_TOKEN_ROWS]
                          [NUM_ATTENTION_HEADS][kBlocks];
};

struct quantized_w4_attention_probability_tile_t {
    attention_prob_t value[QUANTIZED_W4_RESIDENT_TOKEN_ROWS]
                              [NUM_ATTENTION_HEADS]
                              [QUANTIZED_W4_ATTENTION_POSITION_TILE];
};

struct quantized_w4_online_attention_state_t {
    fm_t maximum[QUANTIZED_W4_RESIDENT_TOKEN_ROWS][NUM_ATTENTION_HEADS];
    fm_accum_t sum[QUANTIZED_W4_RESIDENT_TOKEN_ROWS][NUM_ATTENTION_HEADS];
    fm_accum_t context[QUANTIZED_W4_RESIDENT_TOKEN_ROWS]
                      [NUM_ATTENTION_HEADS][HEAD_DIM];
};

inline fm_t quantized_w4_clamp(fm_t value, fm_t low, fm_t high) {
    #pragma HLS inline
    return value < low ? low : (value > high ? high : value);
}

inline fm_t quantized_w4_score_get(
    const quantized_w4_attention_score_tile_t& score,
    unsigned int token,
    unsigned int head,
    unsigned int position) {
    #pragma HLS inline
    fm_t value = 0;
    const unsigned int block =
        position / quantized_w4_attention_score_tile_t::kLanesPerWord;
    const unsigned int lane =
        position % quantized_w4_attention_score_tile_t::kLanesPerWord;
    value.range(fm_t::width - 1, 0) = score.block[token][head][block].range(
        (lane + 1) * fm_t::width - 1, lane * fm_t::width);
    return value;
}

inline void quantized_w4_score_set(
    quantized_w4_attention_score_tile_t& score,
    unsigned int token,
    unsigned int head,
    unsigned int position,
    fm_t value) {
    #pragma HLS inline
    const unsigned int block =
        position / quantized_w4_attention_score_tile_t::kLanesPerWord;
    const unsigned int lane =
        position % quantized_w4_attention_score_tile_t::kLanesPerWord;
    score.block[token][head][block].range(
        (lane + 1) * fm_t::width - 1, lane * fm_t::width) =
        value.range(fm_t::width - 1, 0);
}

inline attention_prob_t quantized_w4_exp_probability(fm_t value) {
    #pragma HLS inline off
    #pragma HLS pipeline II=1
    const fm_t clamped = quantized_w4_clamp(value, fm_t(-8), fm_t(0));
    return attention_prob_t(hls::exp(ap_fixed<18, 4>(clamped)));
}

inline fm_t quantized_w4_reciprocal_one_to_two(fm_t value) {
    #pragma HLS inline
    // Stable SiLU evaluation keeps the denominator in [1, 2].  An affine
    // seed plus two Newton steps avoids synthesizing a generic divider.
    fm_t reciprocal = fm_t(1.5) - fm_t(0.5) * value;
    for (unsigned int iteration = 0; iteration < 2; ++iteration) {
        #pragma HLS unroll
        reciprocal = fm_t(
            reciprocal * (fm_t(2) - value * reciprocal));
    }
    return reciprocal;
}

inline fm_t quantized_w4_silu(fm_t value) {
    #pragma HLS inline off
    #pragma HLS pipeline II=1
    const fm_t clamped = quantized_w4_clamp(value, fm_t(-8), fm_t(8));
    const fm_t magnitude = clamped < fm_t(0) ?
        fm_t(-clamped) : clamped;
    const fm_t exponential =
        fm_t(hls::exp(ap_fixed<16, 8>(-magnitude)));
    const fm_t reciprocal = quantized_w4_reciprocal_one_to_two(
        fm_t(1) + exponential);
    const fm_t sigmoid = clamped < fm_t(0) ?
        fm_t(exponential * reciprocal) : reciprocal;
    return fm_t(clamped * sigmoid);
}

using quantized_w4_rope_pair_t = ap_uint<2 * fm_t::width>;

inline quantized_w4_rope_pair_t quantized_w4_rotate_pair(
    fm_t low,
    fm_t high,
    fm_t cosine,
    fm_t sine) {
    #pragma HLS inline off
    #pragma HLS pipeline II=1
    const fm_t rotated_low = fm_t(low * cosine - high * sine);
    const fm_t rotated_high = fm_t(high * cosine + low * sine);
    quantized_w4_rope_pair_t packed = 0;
    packed.range(fm_t::width - 1, 0) =
        rotated_low.range(fm_t::width - 1, 0);
    packed.range(2 * fm_t::width - 1, fm_t::width) =
        rotated_high.range(fm_t::width - 1, 0);
    return packed;
}

inline fm_t quantized_w4_silu_product(fm_t gate, fm_t up) {
    #pragma HLS inline off
    #pragma HLS pipeline II=1
    return fm_t(quantized_w4_silu(gate) * up);
}

inline fm_t quantized_w4_reciprocal_sum(fm_accum_t value) {
    #pragma HLS inline
    const fm_accum_t minimum = fm_accum_t(0.000244140625);
    return fm_t(fm_accum_t(1) / (value < minimum ? minimum : value));
}

inline std::size_t quantized_w4_kv_word_index(
    unsigned int layer,
    unsigned int position,
    unsigned int block) {
    return (static_cast<std::size_t>(layer) * MAX_SEQ_LEN + position) *
        QUANTIZED_W4_KV_BLOCKS + block;
}

inline std::size_t quantized_w4_kv_cache_words() {
    return static_cast<std::size_t>(NUM_LAYERS) * MAX_SEQ_LEN *
        QUANTIZED_W4_KV_BLOCKS;
}

inline std::size_t quantized_w4_kv_cache_bytes() {
    return quantized_w4_kv_cache_words() * sizeof(mm_input_block_t);
}

inline void store_quantized_w4_current_kv(
    mm_input_block_t* key_cache,
    mm_input_block_t* value_cache,
    const quantized_w4_kv_buffer_t& key,
    const quantized_w4_kv_buffer_t& value,
    unsigned int layer,
    unsigned int position,
    unsigned int valid_tokens) {
    #pragma HLS inline off
    for (unsigned int token = 0;
         token < QUANTIZED_W4_RESIDENT_TOKEN_ROWS; ++token) {
        for (unsigned int block = 0; block < QUANTIZED_W4_KV_BLOCKS; ++block) {
            #pragma HLS pipeline II=1
            if (token < valid_tokens && position + token < MAX_SEQ_LEN) {
                const std::size_t index = quantized_w4_kv_word_index(
                    layer, position + token, block);
                key_cache[index] = key.block[token][block];
                value_cache[index] = value.block[token][block];
            }
        }
    }
}

inline fm_t load_quantized_w4_kv_value(
    const mm_input_block_t* cache,
    unsigned int layer,
    unsigned int position,
    unsigned int element) {
    #pragma HLS inline
    const unsigned int block =
        element / QUANTIZED_W4_RESIDENT_LANES_PER_WORD;
    const unsigned int lane =
        element % QUANTIZED_W4_RESIDENT_LANES_PER_WORD;
    return unpack_mm_input_block_lane(
        cache[quantized_w4_kv_word_index(layer, position, block)], lane);
}

template <unsigned int BLOCK_COUNT>
inline void apply_quantized_w4_rope_words(
    quantized_w4_feature_buffer_t<BLOCK_COUNT>& buffer,
    unsigned int token,
    unsigned int head_count,
    const fm_t cosine[HEAD_DIM / 2],
    const fm_t sine[HEAD_DIM / 2]) {
    #pragma HLS inline
#if defined(QWEN_TEST_SMALL)
    static_assert(
        HEAD_DIM == QUANTIZED_W4_RESIDENT_LANES_PER_WORD,
        "small-model RoPE expects one resident word per head");
    for (unsigned int head = 0; head < head_count; ++head) {
        const unsigned int block = head;
        const mm_input_block_t input_word = buffer.block[token][block];
        fm_t input_lane[QUANTIZED_W4_RESIDENT_LANES_PER_WORD];
        quantized_w4_rope_pair_t rotated[HEAD_DIM / 2];
        #pragma HLS array_partition variable=input_lane complete
        #pragma HLS array_partition variable=rotated complete
        for (unsigned int lane = 0;
             lane < QUANTIZED_W4_RESIDENT_LANES_PER_WORD; ++lane) {
            #pragma HLS unroll
            input_lane[lane].range(fm_t::width - 1, 0) = input_word.range(
                (lane + 1) * fm_t::width - 1, lane * fm_t::width);
        }
        for (unsigned int pair = 0; pair < HEAD_DIM / 2; ++pair) {
            #pragma HLS pipeline II=1
            rotated[pair] = quantized_w4_rotate_pair(
                input_lane[pair], input_lane[pair + HEAD_DIM / 2],
                cosine[pair], sine[pair]);
        }
        mm_input_block_t output_word = 0;
        for (unsigned int pair = 0; pair < HEAD_DIM / 2; ++pair) {
            #pragma HLS unroll
            output_word.range(
                (pair + 1) * fm_t::width - 1, pair * fm_t::width) =
                rotated[pair].range(fm_t::width - 1, 0);
            output_word.range(
                (pair + HEAD_DIM / 2 + 1) * fm_t::width - 1,
                (pair + HEAD_DIM / 2) * fm_t::width) =
                rotated[pair].range(2 * fm_t::width - 1, fm_t::width);
        }
        buffer.block[token][block] = output_word;
    }
#else
    static_assert(
        (HEAD_DIM / 2) % QUANTIZED_W4_RESIDENT_LANES_PER_WORD == 0,
        "production RoPE half-head must align to resident words");
    constexpr unsigned int kHalfHeadBlocks =
        (HEAD_DIM / 2) / QUANTIZED_W4_RESIDENT_LANES_PER_WORD;

    for (unsigned int head = 0; head < head_count; ++head) {
        for (unsigned int word = 0; word < kHalfHeadBlocks; ++word) {
            const unsigned int low_block =
                head * (2 * kHalfHeadBlocks) + word;
            const unsigned int high_block = low_block + kHalfHeadBlocks;
            const mm_input_block_t low_word = buffer.block[token][low_block];
            const mm_input_block_t high_word = buffer.block[token][high_block];
            fm_t low_lane[QUANTIZED_W4_RESIDENT_LANES_PER_WORD];
            fm_t high_lane[QUANTIZED_W4_RESIDENT_LANES_PER_WORD];
            quantized_w4_rope_pair_t
                rotated[QUANTIZED_W4_RESIDENT_LANES_PER_WORD];
            #pragma HLS array_partition variable=low_lane complete
            #pragma HLS array_partition variable=high_lane complete
            #pragma HLS array_partition variable=rotated complete

            for (unsigned int lane = 0;
                 lane < QUANTIZED_W4_RESIDENT_LANES_PER_WORD; ++lane) {
                #pragma HLS unroll
                low_lane[lane].range(fm_t::width - 1, 0) = low_word.range(
                    (lane + 1) * fm_t::width - 1, lane * fm_t::width);
                high_lane[lane].range(fm_t::width - 1, 0) = high_word.range(
                    (lane + 1) * fm_t::width - 1, lane * fm_t::width);
            }
            for (unsigned int lane = 0;
                 lane < QUANTIZED_W4_RESIDENT_LANES_PER_WORD; ++lane) {
                #pragma HLS pipeline II=1
                const unsigned int pair =
                    word * QUANTIZED_W4_RESIDENT_LANES_PER_WORD + lane;
                rotated[lane] = quantized_w4_rotate_pair(
                    low_lane[lane], high_lane[lane],
                    cosine[pair], sine[pair]);
            }

            mm_input_block_t rotated_low_word = 0;
            mm_input_block_t rotated_high_word = 0;
            for (unsigned int lane = 0;
                 lane < QUANTIZED_W4_RESIDENT_LANES_PER_WORD; ++lane) {
                #pragma HLS unroll
                rotated_low_word.range(
                    (lane + 1) * fm_t::width - 1,
                    lane * fm_t::width) =
                    rotated[lane].range(fm_t::width - 1, 0);
                rotated_high_word.range(
                    (lane + 1) * fm_t::width - 1,
                    lane * fm_t::width) =
                    rotated[lane].range(
                        2 * fm_t::width - 1, fm_t::width);
            }
            buffer.block[token][low_block] = rotated_low_word;
            buffer.block[token][high_block] = rotated_high_word;
        }
    }
#endif
}

inline void apply_quantized_w4_rope(
    quantized_w4_hidden_buffer_t& query,
    quantized_w4_kv_buffer_t& key,
    unsigned int token,
    const fm_t cosine[HEAD_DIM / 2],
    const fm_t sine[HEAD_DIM / 2]) {
    #pragma HLS inline off
    apply_quantized_w4_rope_words(
        query, token, NUM_ATTENTION_HEADS, cosine, sine);
    apply_quantized_w4_rope_words(
        key, token, NUM_KEY_VALUE_HEADS, cosine, sine);
}

inline void init_quantized_w4_online_attention(
    quantized_w4_online_attention_state_t& state) {
    #pragma HLS inline off
    for (unsigned int token = 0;
         token < QUANTIZED_W4_RESIDENT_TOKEN_ROWS; ++token) {
        for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
            state.maximum[token][head] = fm_t(-128);
            state.sum[token][head] = fm_accum_t(0);
            for (unsigned int element = 0; element < HEAD_DIM; ++element) {
                #pragma HLS pipeline II=1
                state.context[token][head][element] = fm_accum_t(0);
            }
        }
    }
}

inline void update_quantized_w4_online_probabilities(
    quantized_w4_online_attention_state_t& state,
    quantized_w4_attention_probability_tile_t& probability,
    attention_prob_t old_scale[QUANTIZED_W4_RESIDENT_TOKEN_ROWS]
                                  [NUM_ATTENTION_HEADS],
    const quantized_w4_attention_score_tile_t& score,
    const quantized_layer_task_t& layer_task,
    unsigned int tile_begin,
    unsigned int tile_length) {
    #pragma HLS inline off
    for (unsigned int token = 0;
         token < QUANTIZED_W4_RESIDENT_TOKEN_ROWS; ++token) {
        for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
            fm_t tile_maximum = fm_t(-128);
            const unsigned int query_position = layer_task.position + token;
            for (unsigned int column = 0;
                 column < QUANTIZED_W4_ATTENTION_POSITION_TILE; ++column) {
                #pragma HLS pipeline II=1
                const bool valid = token < layer_task.query_tokens &&
                    column < tile_length &&
                    tile_begin + column <= query_position;
                const fm_t score_value = quantized_w4_score_get(
                    score, token, head, column);
                if (valid && score_value > tile_maximum) {
                    tile_maximum = score_value;
                }
            }
            const fm_t previous_maximum = state.maximum[token][head];
            const bool has_previous = state.sum[token][head] != fm_accum_t(0);
            fm_t new_maximum = tile_maximum;
            if (has_previous && previous_maximum > new_maximum) {
                new_maximum = previous_maximum;
            }
            const attention_prob_t previous_scale = has_previous ?
                quantized_w4_exp_probability(previous_maximum - new_maximum) :
                attention_prob_t(0);
            old_scale[token][head] = previous_scale;
            fm_accum_t tile_sum = 0;
            for (unsigned int column = 0;
                 column < QUANTIZED_W4_ATTENTION_POSITION_TILE; ++column) {
                #pragma HLS pipeline II=1
                const bool valid = token < layer_task.query_tokens &&
                    column < tile_length &&
                    tile_begin + column <= query_position;
                const attention_prob_t value = valid ?
                    quantized_w4_exp_probability(
                        quantized_w4_score_get(score, token, head, column) -
                            new_maximum) :
                    attention_prob_t(0);
                probability.value[token][head][column] = value;
                tile_sum += fm_accum_t(value);
            }
            state.maximum[token][head] = new_maximum;
            state.sum[token][head] =
                state.sum[token][head] * fm_accum_t(previous_scale) + tile_sum;
        }
    }
}

inline void merge_quantized_w4_attention_tile(
    quantized_w4_online_attention_state_t& state,
    const quantized_w4_hidden_buffer_t& tile_context,
    const attention_prob_t old_scale[QUANTIZED_W4_RESIDENT_TOKEN_ROWS]
                                      [NUM_ATTENTION_HEADS],
    unsigned int valid_tokens) {
    #pragma HLS inline off
    for (unsigned int token = 0;
         token < QUANTIZED_W4_RESIDENT_TOKEN_ROWS; ++token) {
        for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
            for (unsigned int element = 0; element < HEAD_DIM; ++element) {
                #pragma HLS pipeline II=1
                if (token < valid_tokens) {
                    const unsigned int hidden_element = head * HEAD_DIM + element;
                    state.context[token][head][element] =
                        state.context[token][head][element] *
                            fm_accum_t(old_scale[token][head]) +
                        fm_accum_t(quantized_w4_feature_get(
                            tile_context, token, hidden_element));
                }
            }
        }
    }
}

inline void finalize_quantized_w4_online_attention(
    quantized_w4_hidden_buffer_t& destination,
    const quantized_w4_online_attention_state_t& state,
    unsigned int valid_tokens) {
    #pragma HLS inline off
    for (unsigned int token = 0;
         token < QUANTIZED_W4_RESIDENT_TOKEN_ROWS; ++token) {
        for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
            const fm_t inverse_sum = token < valid_tokens ?
                quantized_w4_reciprocal_sum(state.sum[token][head]) : fm_t(0);
            for (unsigned int element = 0; element < HEAD_DIM; ++element) {
                #pragma HLS pipeline II=1
                quantized_w4_feature_set(
                    destination, token, head * HEAD_DIM + element,
                    token < valid_tokens ?
                        fm_t(state.context[token][head][element] *
                             fm_accum_t(inverse_sum)) : fm_t(0));
            }
        }
    }
}

template <unsigned int BLOCKS>
void quantized_w4_rmsnorm(
    const quantized_w4_feature_buffer_t<BLOCKS>& source,
    const quantized_w4_feature_buffer_t<BLOCKS>& weight,
    quantized_w4_feature_buffer_t<BLOCKS>& destination,
    unsigned int valid_tokens,
    unsigned int element_count) {
    #pragma HLS inline off
    for (unsigned int token = 0; token < valid_tokens; ++token) {
        quantized_norm_accum_t square_sum = 0;
        for (unsigned int element = 0; element < element_count; ++element) {
            #pragma HLS pipeline II=1
            const quantized_norm_accum_t value =
                quantized_w4_feature_get(source, token, element);
            square_sum += value * value;
        }
        const quantized_norm_accum_t mean = element_count == 0 ?
            quantized_norm_accum_t(0) :
            quantized_norm_accum_t(square_sum / quantized_norm_accum_t(element_count));
        const quantized_norm_accum_t denominator =
            hls::sqrt(mean + quantized_norm_accum_t(RMS_NORM_EPS));
        const quantized_norm_accum_t inverse = denominator != 0 ?
            quantized_norm_accum_t(quantized_norm_accum_t(1) / denominator) :
            quantized_norm_accum_t(0);
        for (unsigned int element = 0; element < element_count; ++element) {
            #pragma HLS pipeline II=1
            quantized_w4_feature_set(destination, token, element,
                fm_t(quantized_norm_accum_t(
                    quantized_w4_feature_get(source, token, element)) * inverse *
                    quantized_norm_accum_t(
                        quantized_w4_feature_get(weight, 0, element))));
        }
    }
    quantized_clear_inactive_rows(destination.block, valid_tokens, element_count);
}

template <unsigned int BLOCKS>
inline quantized_symmetric_scale_t quantized_w4_buffer_scale(
    const quantized_w4_feature_buffer_t<BLOCKS>& source,
    unsigned int valid_tokens,
    unsigned int element_count) {
    #pragma HLS inline off
    const fm_t maximum = quantized_buffer_max_abs(
        source.block, valid_tokens, element_count);
    return make_quantized_symmetric_scale_w4(maximum);
}

template <unsigned int BLOCKS>
void quantized_w4_residual_add(
    quantized_w4_feature_buffer_t<BLOCKS>& residual,
    const quantized_w4_feature_buffer_t<BLOCKS>& update,
    unsigned int valid_tokens,
    unsigned int element_count) {
    #pragma HLS inline off
    const unsigned int block_count =
        (element_count + QUANTIZED_W4_RESIDENT_LANES_PER_WORD - 1) /
        QUANTIZED_W4_RESIDENT_LANES_PER_WORD;
    for (unsigned int token = 0;
         token < QUANTIZED_W4_RESIDENT_TOKEN_ROWS; ++token) {
        for (unsigned int block = 0; block < block_count; ++block) {
            #pragma HLS pipeline II=1
            if (token < valid_tokens) {
                const mm_input_block_t residual_word =
                    residual.block[token][block];
                const mm_input_block_t update_word = update.block[token][block];
                mm_input_block_t output_word = residual_word;
                for (unsigned int lane = 0;
                     lane < QUANTIZED_W4_RESIDENT_LANES_PER_WORD; ++lane) {
                    #pragma HLS unroll
                    const unsigned int element =
                        block * QUANTIZED_W4_RESIDENT_LANES_PER_WORD + lane;
                    if (element < element_count) {
                        fm_t residual_value;
                        fm_t update_value;
                        residual_value.range(fm_t::width - 1, 0) =
                            residual_word.range(
                                (lane + 1) * fm_t::width - 1,
                                lane * fm_t::width);
                        update_value.range(fm_t::width - 1, 0) =
                            update_word.range(
                                (lane + 1) * fm_t::width - 1,
                                lane * fm_t::width);
                        const fm_t sum = fm_t(residual_value + update_value);
                        output_word.range(
                            (lane + 1) * fm_t::width - 1,
                            lane * fm_t::width) =
                            sum.range(fm_t::width - 1, 0);
                    }
                }
                residual.block[token][block] = output_word;
            }
        }
    }
}

struct quantized_w4_silu_product_op {
    static fm_t eval(fm_t gate, fm_t up) {
        #pragma HLS inline
        return quantized_w4_silu_product(gate, up);
    }
};

inline void quantized_w4_silu_multiply(
    const quantized_w4_wide_buffer_t& gate,
    const quantized_w4_wide_buffer_t& up,
    quantized_w4_wide_buffer_t& product,
    unsigned int valid_tokens) {
    #pragma HLS inline off
    #pragma HLS allocation function instances=quantized_w4_silu_product limit=QUANTIZED_SCALAR_SILU_INSTANCES
    for (unsigned int token = 0; token < valid_tokens; ++token)
        for (unsigned int block = 0; block < QUANTIZED_W4_WIDE_BLOCKS; ++block) {
            #pragma HLS pipeline II=1
            const mm_input_block_t gate_word = gate.block[token][block];
            const mm_input_block_t up_word = up.block[token][block];
            mm_input_block_t product_word = 0;
            for (unsigned int lane = 0;
                 lane < QUANTIZED_W4_RESIDENT_LANES_PER_WORD; ++lane) {
                #pragma HLS unroll
                fm_t gate_value, up_value;
                gate_value.range(fm_t::width - 1, 0) = gate_word.range(
                    (lane + 1) * fm_t::width - 1, lane * fm_t::width);
                up_value.range(fm_t::width - 1, 0) = up_word.range(
                    (lane + 1) * fm_t::width - 1, lane * fm_t::width);
                const fm_t value = quantized_w4_silu_product(gate_value, up_value);
                product_word.range((lane + 1) * fm_t::width - 1,
                    lane * fm_t::width) = value.range(fm_t::width - 1, 0);
            }
            product.block[token][block] = product_word;
        }
    quantized_clear_inactive_rows(product.block, valid_tokens,
        QUANTIZED_W4_WIDE_BLOCKS * QUANTIZED_W4_RESIDENT_LANES_PER_WORD);
}

inline void quantized_w4_silu_multiply_inplace(
    quantized_w4_wide_buffer_t& gate_product,
    const quantized_w4_wide_buffer_t& up,
    unsigned int valid_tokens) {
    #pragma HLS inline
    quantized_w4_silu_multiply(
        gate_product, up, gate_product, valid_tokens);
}

#endif
