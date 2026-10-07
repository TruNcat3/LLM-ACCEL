#ifndef LLM_FPGA_QUANTIZED_W8_ATTENTION_ENGINE_HPP
#define LLM_FPGA_QUANTIZED_W8_ATTENTION_ENGINE_HPP

#include "quantized_w8_layer_math.hpp"
#include "quantized_w8_projection_engine.hpp"
#include "quantized_attention_config.hpp"

inline fm_t quantized_w8_abs(fm_t value) {
    #pragma HLS inline
    return value < fm_t(0) ? fm_t(-value) : value;
}

constexpr unsigned int QUANTIZED_W8_KV_HEAD_BLOCKS =
    (HEAD_DIM + QUANTIZED_W8_RESIDENT_LANES_PER_WORD - 1) /
        QUANTIZED_W8_RESIDENT_LANES_PER_WORD;

struct quantized_w8_kv_head_tile_t {
    mm_input_block_t block[QUANTIZED_W8_ATTENTION_POSITION_TILE]
                          [QUANTIZED_W8_KV_HEAD_BLOCKS];
};

struct quantized_w8_attention_weight_tile_t {
    quantized_w8_weight_word_t word[QUANTIZED_W8_ATTENTION_POSITION_TILE]
                                     [QUANTIZED_W8_WEIGHT_STREAMS_PER_CU];
};

inline void load_quantized_w8_kv_head_tile(
    quantized_w8_kv_head_tile_t& tile,
    const mm_input_block_t* cache,
    unsigned int layer,
    unsigned int kv_head,
    unsigned int tile_begin,
    unsigned int tile_length) {
    #pragma HLS inline off
    // Scale and QK packing read only valid positions. The unused scratch
    // tail may retain the preceding tile; QK still explicitly pads columns.
    const unsigned int positions = quantized_attention_prepared_positions<
        QUANTIZED_W8_ATTENTION_POSITION_TILE>(tile_length);
    for (unsigned int position = 0;
         position < positions; ++position) {
        #pragma HLS loop_tripcount min=0 max=128
        for (unsigned int block = 0;
             block < QUANTIZED_W8_KV_HEAD_BLOCKS; ++block) {
            #pragma HLS pipeline II=1
            tile.block[position][block] = position < tile_length ?
                cache[quantized_w8_kv_word_index(
                    layer, tile_begin + position,
                    kv_head * QUANTIZED_W8_KV_HEAD_BLOCKS + block)] :
                mm_input_block_t(0);
        }
    }
}

inline quantized_symmetric_scale_t quantized_w8_kv_head_tile_scale(
    const quantized_w8_kv_head_tile_t& tile,
    unsigned int tile_length) {
    #pragma HLS inline off
    fm_t maximum = 0;
    for (unsigned int position = 0;
         position < tile_length && position < QUANTIZED_W8_ATTENTION_POSITION_TILE; ++position) {
        #pragma HLS loop_tripcount min=0 max=128
        for (unsigned int block = 0;
             block < QUANTIZED_W8_KV_HEAD_BLOCKS; ++block) {
            #pragma HLS pipeline II=1
            const mm_input_block_t packed = tile.block[position][block];
            fm_t word_maximum = 0;
            for (unsigned int lane = 0;
                 lane < QUANTIZED_W8_RESIDENT_LANES_PER_WORD; ++lane) {
                #pragma HLS unroll
                const unsigned int element =
                    block * QUANTIZED_W8_RESIDENT_LANES_PER_WORD + lane;
                const fm_t value = position < tile_length && element < HEAD_DIM ?
                    quantized_w8_abs(unpack_mm_input_block_lane(packed, lane)) :
                    fm_t(0);
                if (value > word_maximum) word_maximum = value;
            }
            if (word_maximum > maximum) maximum = word_maximum;
        }
    }
    return make_quantized_symmetric_scale(maximum);
}

inline void pack_quantized_w8_qk_tile(
    quantized_w8_attention_weight_tile_t& destination,
    const quantized_w8_kv_head_tile_t& source,
    quant_inverse_scale_t inverse_scale,
    unsigned int tile_length) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=destination.word complete dim=2
    // Transpose one group of 16 head elements in registers, then write full
    // destination words. Partial writes to 16 BRAM addresses per input word
    // previously serialized this loop to II=24.
    for (unsigned int block = 0;
         block < QUANTIZED_W8_KV_HEAD_BLOCKS; ++block) {
        for (unsigned int port = 0;
             port < QUANTIZED_W8_WEIGHT_STREAMS_PER_CU; ++port) {
            quantized_w8_weight_word_t words[QUANTIZED_W8_RESIDENT_LANES_PER_WORD];
            #pragma HLS array_partition variable=words complete
            for (unsigned int lane = 0;
                 lane < QUANTIZED_W8_RESIDENT_LANES_PER_WORD; ++lane) {
                #pragma HLS unroll
                words[lane] = 0;
            }
            const unsigned int begin = port * QUANTIZED_W8_WEIGHTS_PER_STREAM_WORD;
            const unsigned int available = tile_length > begin ? tile_length - begin : 0;
            const unsigned int count = available < QUANTIZED_W8_WEIGHTS_PER_STREAM_WORD ?
                available : QUANTIZED_W8_WEIGHTS_PER_STREAM_WORD;
            // Descending positions allow a constant shift and low-slice insert.
            // Unused high positions stay zero, including after a shorter tile.
            for (unsigned int remaining = count; remaining != 0; --remaining) {
                #pragma HLS pipeline II=1
                #pragma HLS loop_tripcount min=0 max=32
                const mm_input_block_t packed = source.block[begin + remaining - 1][block];
                for (unsigned int lane = 0;
                     lane < QUANTIZED_W8_RESIDENT_LANES_PER_WORD; ++lane) {
                    #pragma HLS unroll
                    const ap_int<8> value = quantize_symmetric_int8(
                        unpack_mm_input_block_lane(packed, lane), inverse_scale);
                    words[lane] <<= 8;
                    words[lane].range(7, 0) = value;
                }
            }
            for (unsigned int lane = 0;
                 lane < QUANTIZED_W8_RESIDENT_LANES_PER_WORD; ++lane) {
                #pragma HLS pipeline II=1
                const unsigned int element = block * QUANTIZED_W8_RESIDENT_LANES_PER_WORD + lane;
                if (element < HEAD_DIM) destination.word[element][port] = words[lane];
            }
        }
    }
}

inline void pack_quantized_w8_pv_tile(
    quantized_w8_attention_weight_tile_t& destination,
    const quantized_w8_kv_head_tile_t& source,
    unsigned int tile_length,
    quant_inverse_scale_t inverse_scale) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=destination.word complete dim=2
    // PV issues k_count=tile_length and never sends these unused rows.
    // Keep padding within every valid packed word, including HEAD_DIM tails.
    const unsigned int positions = quantized_attention_prepared_positions<
        QUANTIZED_W8_ATTENTION_POSITION_TILE>(tile_length);
    for (unsigned int position = 0;
         position < positions; ++position) {
        #pragma HLS loop_tripcount min=0 max=128
        for (unsigned int port = 0;
             port < QUANTIZED_W8_WEIGHT_STREAMS_PER_CU; ++port) {
            #pragma HLS pipeline II=1
            quantized_w8_weight_word_t packed = 0;
            for (unsigned int lane = 0;
                 lane < QUANTIZED_W8_WEIGHTS_PER_STREAM_WORD; ++lane) {
                #pragma HLS unroll
                const unsigned int element =
                    port * QUANTIZED_W8_WEIGHTS_PER_STREAM_WORD + lane;
                if (position < tile_length && element < HEAD_DIM) {
                    const unsigned int block =
                        element / QUANTIZED_W8_RESIDENT_LANES_PER_WORD;
                    const unsigned int block_lane =
                        element % QUANTIZED_W8_RESIDENT_LANES_PER_WORD;
                    const ap_int<8> value = quantize_symmetric_int8(
                        unpack_mm_input_block_lane(
                            source.block[position][block], block_lane),
                        inverse_scale);
                    packed.range((lane + 1) * 8 - 1, lane * 8) = value;
                }
            }
            destination.word[position][port] = packed;
        }
    }
}

inline quantized_symmetric_scale_t quantized_w8_query_head_scale(
    const quantized_w8_hidden_buffer_t& query,
    unsigned int query_head,
    unsigned int valid_tokens) {
    #pragma HLS inline off
    fm_t maximum = 0;
    for (unsigned int token = 0;
         token < valid_tokens && token < QUANTIZED_W8_RESIDENT_TOKEN_ROWS; ++token) {
        #pragma HLS loop_tripcount min=0 max=4
        for (unsigned int element = 0; element < HEAD_DIM; ++element) {
            #pragma HLS pipeline II=1
            const fm_t value = token < valid_tokens ?
                quantized_w8_abs(quantized_w8_feature_get(
                    query, token, query_head * HEAD_DIM + element)) : fm_t(0);
            if (value > maximum) maximum = value;
        }
    }
    return make_quantized_symmetric_scale(maximum);
}

inline quantized_symmetric_scale_t quantized_w8_kv_tile_scale(
    const mm_input_block_t* cache,
    unsigned int layer,
    unsigned int kv_head,
    unsigned int tile_begin,
    unsigned int tile_length) {
    #pragma HLS inline
    fm_t maximum = 0;
    for (unsigned int position = 0;
         position < tile_length && position < QUANTIZED_W8_ATTENTION_POSITION_TILE; ++position) {
        #pragma HLS loop_tripcount min=0 max=128
        for (unsigned int element = 0; element < HEAD_DIM; ++element) {
            #pragma HLS pipeline II=1
            const fm_t value = position < tile_length ?
                quantized_w8_abs(load_quantized_w8_kv_value(
                    cache, layer, tile_begin + position,
                    kv_head * HEAD_DIM + element)) : fm_t(0);
            if (value > maximum) maximum = value;
        }
    }
    return make_quantized_symmetric_scale(maximum);
}

inline quantized_symmetric_scale_t quantized_w8_probability_scale(
    const quantized_w8_attention_probability_tile_t& probability,
    unsigned int query_head,
    unsigned int valid_tokens,
    unsigned int tile_length) {
    #pragma HLS inline off
    fm_t maximum = 0;
    for (unsigned int token = 0;
         token < valid_tokens && token < QUANTIZED_W8_RESIDENT_TOKEN_ROWS; ++token) {
        #pragma HLS loop_tripcount min=0 max=4
        for (unsigned int position = 0;
             position < tile_length && position < QUANTIZED_W8_ATTENTION_POSITION_TILE; ++position) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=0 max=128
            const fm_t value = token < valid_tokens && position < tile_length ?
                fm_t(probability.value[token][query_head][position]) : fm_t(0);
            if (value > maximum) maximum = value;
        }
    }
    return make_quantized_symmetric_scale(maximum);
}

inline quantized_w8_activation_word_t pack_quantized_w8_qk_activation(
    const quantized_w8_hidden_buffer_t& query,
    unsigned int query_head,
    unsigned int element,
    unsigned int valid_tokens,
    quant_inverse_scale_t inverse_scale) {
    #pragma HLS inline
    quantized_w8_activation_word_t packed = 0;
    for (unsigned int token = 0;
         token < QUANTIZED_W8_RESIDENT_TOKEN_ROWS; ++token) {
        #pragma HLS unroll
        const fm_t value = token < valid_tokens ?
            quantized_w8_feature_get(
                query, token, query_head * HEAD_DIM + element) : fm_t(0);
        packed.range((token + 1) * 8 - 1, token * 8) =
            quantize_symmetric_int8(value, inverse_scale);
    }
    return packed;
}

inline quantized_w8_weight_word_t pack_quantized_w8_qk_weight(
    const mm_input_block_t* key_cache,
    unsigned int layer,
    unsigned int kv_head,
    unsigned int tile_begin,
    unsigned int tile_length,
    unsigned int port,
    unsigned int head_element,
    quant_inverse_scale_t inverse_scale) {
    #pragma HLS inline
    quantized_w8_weight_word_t packed = 0;
    for (unsigned int lane = 0;
         lane < QUANTIZED_W8_WEIGHTS_PER_STREAM_WORD; ++lane) {
        #pragma HLS unroll
        const unsigned int tile_position =
            port * QUANTIZED_W8_WEIGHTS_PER_STREAM_WORD + lane;
        const fm_t value = tile_position < tile_length ?
            load_quantized_w8_kv_value(
                key_cache, layer, tile_begin + tile_position,
                kv_head * HEAD_DIM + head_element) : fm_t(0);
        packed.range((lane + 1) * 8 - 1, lane * 8) =
            quantize_symmetric_int8(value, inverse_scale);
    }
    return packed;
}

inline void store_quantized_w8_qk_output(
    quantized_w8_attention_score_tile_t& score,
    const quantized_w8_output_word_t& output,
    unsigned int query_head,
    unsigned int tile_begin,
    unsigned int valid_tokens,
    unsigned int tile_length,
    quant_scale_t query_scale,
    quant_scale_t key_scale) {
    #pragma HLS inline
    const unsigned int token = output.range(535, 528).to_uint();
    const unsigned int elem_base = output.range(551, 536).to_uint();
    if (token >= valid_tokens || elem_base < tile_begin) return;
    const unsigned int tile_position_base = elem_base - tile_begin;
    if (tile_position_base >= tile_length) return;

    mm_input_block_t packed = 0;
    for (unsigned int lane = 0;
         lane < MM_STREAM_4X128_INT8X8_LANES_PER_GROUP; ++lane) {
        #pragma HLS unroll
        const unsigned int tile_position = tile_position_base + lane;
        if (tile_position < tile_length) {
            const fm_t value = fm_t(dequantize_w8_accumulator(
                unpack_mm_stream_4x128_int8x8_output(output, lane),
                query_scale, key_scale) * fm_t(ATTENTION_SCALE));
            packed.range((lane + 1) * fm_t::width - 1, lane * fm_t::width) =
                value.range(fm_t::width - 1, 0);
        }
    }
    const unsigned int block = tile_position_base /
        quantized_w8_attention_score_tile_t::kLanesPerWord;
    if (block < quantized_w8_attention_score_tile_t::kBlocks) {
        score.block[token][query_head][block] = packed;
    }
}

inline quantized_w8_activation_word_t pack_quantized_w8_pv_activation(
    const quantized_w8_attention_probability_tile_t& probability,
    unsigned int query_head,
    unsigned int tile_position,
    unsigned int valid_tokens,
    quant_inverse_scale_t inverse_scale) {
    #pragma HLS inline
    quantized_w8_activation_word_t packed = 0;
    for (unsigned int token = 0;
         token < QUANTIZED_W8_RESIDENT_TOKEN_ROWS; ++token) {
        #pragma HLS unroll
        const fm_t value = token < valid_tokens ?
            fm_t(probability.value[token][query_head][tile_position]) : fm_t(0);
        packed.range((token + 1) * 8 - 1, token * 8) =
            quantize_symmetric_int8(value, inverse_scale);
    }
    return packed;
}

inline quantized_w8_weight_word_t pack_quantized_w8_pv_weight(
    const mm_input_block_t* value_cache,
    unsigned int layer,
    unsigned int kv_head,
    unsigned int absolute_position,
    unsigned int port,
    quant_inverse_scale_t inverse_scale) {
    #pragma HLS inline
    quantized_w8_weight_word_t packed = 0;
    for (unsigned int lane = 0;
         lane < QUANTIZED_W8_WEIGHTS_PER_STREAM_WORD; ++lane) {
        #pragma HLS unroll
        const unsigned int head_element =
            port * QUANTIZED_W8_WEIGHTS_PER_STREAM_WORD + lane;
        const fm_t value = load_quantized_w8_kv_value(
            value_cache, layer, absolute_position,
            kv_head * HEAD_DIM + head_element);
        packed.range((lane + 1) * 8 - 1, lane * 8) =
            quantize_symmetric_int8(value, inverse_scale);
    }
    return packed;
}

inline void store_quantized_w8_pv_output(
    quantized_w8_hidden_buffer_t& tile_context,
    const quantized_w8_output_word_t& output,
    unsigned int valid_tokens,
    quant_scale_t probability_scale,
    quant_scale_t value_scale) {
    #pragma HLS inline
    store_quantized_w8_aligned_output_word(
        tile_context, output, valid_tokens, HIDDEN_SIZE,
        probability_scale, value_scale);
}

inline void run_quantized_w8_qk_head(
    const quantized_layer_task_t& layer_task,
    unsigned int tile,
    unsigned int query_head,
    const quantized_w8_hidden_buffer_t& query,
    const quantized_w8_attention_weight_tile_t& key_weight,
    quant_scale_t key_scale,
    quantized_w8_attention_score_tile_t& score,
    quantized_w8_projection_streams_t& streams) {
    // Keep the HBM pointer in the flattened top-level call graph. Vitis HLS
    // rejects selecting a pointer passed through a child function here.
    #pragma HLS inline
    const quantized_w8_attention_dispatch_t dispatch =
        make_quantized_w8_attention_dispatch(
            layer_task, MM_STREAM_QUANTIZED_MODE_ATTENTION_QK,
            tile, query_head);
    const quantized_symmetric_scale_t query_quant =
        quantized_w8_query_head_scale(
            query, query_head, layer_task.query_tokens);
    const unsigned int cu = dispatch.cu;
    quantized_w8_write_task(streams, cu, pack_mm_stream_quantized_task(
        make_quantized_w8_attention_task(
            layer_task, dispatch, query_quant.scale, key_scale)));
    for (unsigned int element = 0; element < HEAD_DIM; ++element) {
        #pragma HLS pipeline II=1
        quantized_w8_write_activation(streams, cu, pack_quantized_w8_qk_activation(
            query, query_head, element, layer_task.query_tokens,
            query_quant.inverse_scale));
        for (unsigned int port = 0;
             port < QUANTIZED_W8_WEIGHT_STREAMS_PER_CU; ++port) {
            #pragma HLS unroll
            quantized_w8_write_weight(
                streams, cu, port, key_weight.word[element][port]);
        }
    }
    for (unsigned int packet = 0;
         packet < MM_STREAM_4X128_INT8X8_TOKENS *
                      MM_STREAM_4X128_INT8X8_OUTPUT_GROUPS; ++packet) {
        #pragma HLS pipeline II=1
        store_quantized_w8_qk_output(
            score, quantized_w8_read_output(streams, cu), query_head,
            dispatch.tile_begin, layer_task.query_tokens,
            dispatch.tile_length, query_quant.scale, key_scale);
    }
}

inline void run_quantized_w8_pv_head(
    const quantized_layer_task_t& layer_task,
    unsigned int tile,
    unsigned int query_head,
    const quantized_w8_attention_probability_tile_t& probability,
    const quantized_w8_attention_weight_tile_t& value_weight,
    quant_scale_t value_scale,
    quantized_w8_hidden_buffer_t& tile_context,
    quantized_w8_projection_streams_t& streams) {
    #pragma HLS inline
    const quantized_w8_attention_dispatch_t dispatch =
        make_quantized_w8_attention_dispatch(
            layer_task, MM_STREAM_QUANTIZED_MODE_ATTENTION_PV,
            tile, query_head);
    const quantized_symmetric_scale_t probability_quant =
        quantized_w8_probability_scale(
            probability, query_head, layer_task.query_tokens,
            dispatch.tile_length);
    const unsigned int cu = dispatch.cu;
    quantized_w8_write_task(streams, cu, pack_mm_stream_quantized_task(
        make_quantized_w8_attention_task(
            layer_task, dispatch, probability_quant.scale,
            value_scale)));
    for (unsigned int position = 0; position < dispatch.tile_length; ++position) {
        #pragma HLS pipeline II=1
        quantized_w8_write_activation(streams, cu, pack_quantized_w8_pv_activation(
            probability, query_head, position, layer_task.query_tokens,
            probability_quant.inverse_scale));
        for (unsigned int port = 0;
             port < QUANTIZED_W8_WEIGHT_STREAMS_PER_CU; ++port) {
            #pragma HLS unroll
            quantized_w8_write_weight(
                streams, cu, port, value_weight.word[position][port]);
        }
    }
    for (unsigned int packet = 0;
         packet < MM_STREAM_4X128_INT8X8_TOKENS *
                      MM_STREAM_4X128_INT8X8_OUTPUT_GROUPS; ++packet) {
        #pragma HLS pipeline II=1
        store_quantized_w8_pv_output(
            tile_context, quantized_w8_read_output(streams, cu),
            layer_task.query_tokens, probability_quant.scale,
            value_scale);
    }
}

// Legacy serial numerical reference; production dispatch uses the aligned
// block/GQA-row schedule below.
inline void run_quantized_w8_online_attention_serial_reference(
    const quantized_layer_task_t& layer_task,
    const quantized_w8_hidden_buffer_t& query,
    const mm_input_block_t* key_cache,
    const mm_input_block_t* value_cache,
    quantized_w8_hidden_buffer_t& destination,
    quantized_w8_projection_streams_t& streams) {
    #pragma HLS inline
    quantized_w8_online_attention_state_t state;
    quantized_w8_attention_score_tile_t score;
    quantized_w8_attention_probability_tile_t probability;
    quantized_w8_hidden_buffer_t tile_context;
    quantized_w8_kv_head_tile_t kv_tile;
    quantized_w8_attention_weight_tile_t weight_tile;
    attention_prob_t old_scale[QUANTIZED_W8_RESIDENT_TOKEN_ROWS]
                                  [NUM_ATTENTION_HEADS];
    #pragma HLS bind_storage variable=state.context type=ram_2p impl=bram
    #pragma HLS bind_storage variable=score.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=probability.value type=ram_2p impl=bram
    #pragma HLS bind_storage variable=tile_context.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=kv_tile.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=kv_tile.block complete dim=2
    #pragma HLS bind_storage variable=weight_tile.word type=ram_2p impl=bram
    #pragma HLS array_partition variable=weight_tile.word complete dim=2
    #pragma HLS array_partition variable=weight_tile.word cyclic factor=16 dim=1
    init_quantized_w8_online_attention(state);
    const unsigned int tile_count = quantized_w8_attention_tile_count(
        quantized_w8_attention_context_length(layer_task));
    for (unsigned int tile = 0; tile < tile_count; ++tile) {
        #pragma HLS loop_tripcount min=1 max=16 avg=4
        for (unsigned int kv_head = 0;
             kv_head < NUM_KEY_VALUE_HEADS; ++kv_head) {
            const unsigned int tile_begin =
                tile * QUANTIZED_W8_ATTENTION_POSITION_TILE;
            unsigned int tile_length =
                quantized_w8_attention_context_length(layer_task) - tile_begin;
            if (tile_length > QUANTIZED_W8_ATTENTION_POSITION_TILE) {
                tile_length = QUANTIZED_W8_ATTENTION_POSITION_TILE;
            }
            load_quantized_w8_kv_head_tile(
                kv_tile, key_cache, layer_task.layer, kv_head,
                tile_begin, tile_length);
            const quantized_symmetric_scale_t key_quant =
                quantized_w8_kv_head_tile_scale(kv_tile, tile_length);
            pack_quantized_w8_qk_tile(
                weight_tile, kv_tile, key_quant.inverse_scale, tile_length);
            for (unsigned int group_head = 0;
                 group_head < GQA_GROUP_SIZE; ++group_head) {
                const unsigned int head = kv_head * GQA_GROUP_SIZE + group_head;
                run_quantized_w8_qk_head(
                    layer_task, tile, head, query, weight_tile,
                    key_quant.scale, score, streams);
            }
        }
        const unsigned int tile_begin =
            tile * QUANTIZED_W8_ATTENTION_POSITION_TILE;
        unsigned int tile_length =
            quantized_w8_attention_context_length(layer_task) - tile_begin;
        if (tile_length > QUANTIZED_W8_ATTENTION_POSITION_TILE) {
            tile_length = QUANTIZED_W8_ATTENTION_POSITION_TILE;
        }
        update_quantized_w8_online_probabilities(
            state, probability, old_scale, score, layer_task,
            tile_begin, tile_length);
        for (unsigned int kv_head = 0;
             kv_head < NUM_KEY_VALUE_HEADS; ++kv_head) {
            load_quantized_w8_kv_head_tile(
                kv_tile, value_cache, layer_task.layer, kv_head,
                tile_begin, tile_length);
            const quantized_symmetric_scale_t value_quant =
                quantized_w8_kv_head_tile_scale(kv_tile, tile_length);
            pack_quantized_w8_pv_tile(
                weight_tile, kv_tile, tile_length,
                value_quant.inverse_scale);
            for (unsigned int group_head = 0;
                 group_head < GQA_GROUP_SIZE; ++group_head) {
                const unsigned int head = kv_head * GQA_GROUP_SIZE + group_head;
                run_quantized_w8_pv_head(
                    layer_task, tile, head, probability, weight_tile,
                    value_quant.scale, tile_context, streams);
            }
        }
        merge_quantized_w8_attention_tile(
            state, tile_context, old_scale, layer_task.query_tokens);
    }
    finalize_quantized_w8_online_attention(
        destination, state, layer_task.query_tokens);
}

#include "quantized_attention_dataflow.hpp"

struct quantized_w8_attention_policy : quantized_w8_projection_policy {
    static constexpr unsigned int bits = 8;
    using element_t = ap_int<8>;
    using hidden_t = quantized_w8_hidden_buffer_t;
    using score_t = quantized_w8_attention_score_tile_t;
    using probability_t = quantized_w8_attention_probability_tile_t;
    using state_t = quantized_w8_online_attention_state_t;
    using kv_tile_t = quantized_w8_kv_head_tile_t;
    using weight_tile_t = quantized_w8_attention_weight_tile_t;
    static element_t quantize(fm_t value, quant_inverse_scale_t inverse) {
        #pragma HLS inline
        return quantize_symmetric_int8(value, inverse);
    }
    static fm_t feature_get(const hidden_t& data, unsigned int row, unsigned int element) {
        #pragma HLS inline
        return quantized_w8_feature_get(data, row, element);
    }
    static fm_t score_get(const score_t& data, unsigned int row, unsigned int head, unsigned int col) {
        #pragma HLS inline
        return quantized_w8_score_get(data, row, head, col);
    }
    static fm_t dequantize(output_word_t word, unsigned int lane, quant_scale_t a, quant_scale_t w) {
        #pragma HLS inline
        return dequantize_w8_accumulator(unpack_mm_stream_4x128_int8x8_output(word, lane), a, w);
    }
    static attention_prob_t exp(fm_t value) {
        #pragma HLS inline
        return quantized_w8_exp_probability(value);
    }
    static fm_t reciprocal(fm_accum_t value) {
        #pragma HLS inline
        return quantized_w8_reciprocal_sum(value);
    }
    static quantized_symmetric_scale_t query_scale(const hidden_t& query, unsigned int head, unsigned int rows) {
        #pragma HLS inline
        return quantized_w8_query_head_scale(query, head, rows);
    }
    static quantized_symmetric_scale_t probability_scale(const probability_t& p, unsigned int head,
                                                         unsigned int rows, unsigned int length) {
        #pragma HLS inline
        return quantized_w8_probability_scale(p, head, rows, length);
    }
    static quantized_symmetric_scale_t scale_from_maximum(fm_t maximum) {
        #pragma HLS inline
        return make_quantized_symmetric_scale(maximum);
    }
    static void load_kv(kv_tile_t& tile, const mm_input_block_t* cache, unsigned int layer,
                        unsigned int head, unsigned int begin, unsigned int length) {
        #pragma HLS inline
        load_quantized_w8_kv_head_tile(tile, cache, layer, head, begin, length);
    }
    static quantized_symmetric_scale_t kv_scale(const kv_tile_t& tile, unsigned int length) {
        #pragma HLS inline
        return quantized_w8_kv_head_tile_scale(tile, length);
    }
    static void pack_key(weight_tile_t& weight, const kv_tile_t& tile,
                         quant_inverse_scale_t inverse, unsigned int length) {
        #pragma HLS inline
        pack_quantized_w8_qk_tile(weight, tile, inverse, length);
    }
    static void pack_value(weight_tile_t& weight, const kv_tile_t& tile, unsigned int length,
                            quant_inverse_scale_t inverse) {
        #pragma HLS inline
        pack_quantized_w8_pv_tile(weight, tile, length, inverse);
    }
#ifndef __SYNTHESIS__
    static void service_csim(streams_t&, const quantized_layer_task_t&, unsigned int) {}
#endif
};

inline void run_quantized_w8_online_attention(
    const quantized_layer_task_t& task, const quantized_w8_hidden_buffer_t& query,
    const mm_input_block_t* key_cache, const mm_input_block_t* value_cache,
    quantized_w8_hidden_buffer_t& destination, quantized_w8_projection_streams_t& streams) {
    #pragma HLS inline
    run_quantized_attention_aligned<quantized_w8_attention_policy>(
        task, query, key_cache, value_cache, destination, streams);
}

#endif
