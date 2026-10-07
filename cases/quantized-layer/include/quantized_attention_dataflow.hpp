#ifndef LLM_FPGA_QUANTIZED_ATTENTION_DATAFLOW_HPP
#define LLM_FPGA_QUANTIZED_ATTENTION_DATAFLOW_HPP

#include "quantized_attention_mapping.hpp"

template <typename Policy>
struct quantized_attention_activation_panel_t {
    typename Policy::activation_word_t word[QUANTIZED_LAYER_COMPUTE_CUS][128];
};

template <typename Policy>
struct quantized_attention_result_panel_t {
    // The reference drains into separate CU-local result banks. A logical
    // head-indexed destination makes HLS serialize potentially-aliasing CU
    // writes even when the scheduler assigns distinct heads.
    mm_input_block_t block[QUANTIZED_LAYER_COMPUTE_CUS][Policy::token_rows][8];
};

template <typename Policy>
inline mm_input_block_t quantized_attention_result_word(
    const quantized_attention_result_panel_t<Policy>& result,
    const quantized_layer_task_t& task, unsigned int token,
    unsigned int head, unsigned int block) {
    #pragma HLS inline
    const bool decode = task.phase == QUANTIZED_LAYER_PHASE_DECODE;
    const unsigned int heads_per_unit = decode ? Policy::token_rows : 1;
    const unsigned int unit = head / GQA_GROUP_SIZE +
        (head % GQA_GROUP_SIZE / heads_per_unit) * NUM_KEY_VALUE_HEADS;
    const unsigned int row = decode ? head % GQA_GROUP_SIZE % Policy::token_rows : token;
    return result.block[unit % QUANTIZED_LAYER_COMPUTE_CUS][row][block];
}

template <typename Policy>
inline fm_t quantized_attention_result_value(
    const quantized_attention_result_panel_t<Policy>& result,
    const quantized_layer_task_t& task, unsigned int token,
    unsigned int head, unsigned int column) {
    #pragma HLS inline
    return unpack_mm_input_block_lane(
        quantized_attention_result_word<Policy>(result, task, token, head, column / 16), column % 16);
}

template <typename Policy>
void prepare_quantized_attention_activation_panel(
    quantized_attention_activation_panel_t<Policy>& panel,
    const quantized_layer_task_t& task, unsigned int wave, bool pv,
    unsigned int length, const typename Policy::hidden_t& query,
    const typename Policy::probability_t& probability,
    const quantized_symmetric_scale_t scales[NUM_ATTENTION_HEADS]) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=panel.word complete dim=1
#if QUANTIZED_ATTENTION_HEAD_LANES > 1
    #pragma HLS array_partition variable=probability.value cyclic factor=QUANTIZED_ATTENTION_HEAD_LANES dim=2
#endif
#if QUANTIZED_ATTENTION_PACK_LANES > 1
    #pragma HLS array_partition variable=probability.value cyclic factor=QUANTIZED_ATTENTION_PACK_LANES dim=3
#endif
    // Assemble a bounded group in registers and write each complete word once.
    // Updating row slices directly in panel BRAM creates a read/mux/write
    // recurrence, particularly when a short PV tile has only one K position.
    constexpr unsigned int group_width = 16;
    for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
        const auto unit = quantized_attention_unit<Policy::token_rows>(task, wave, cu);
        if (!unit.active) continue;
        for (unsigned int begin = 0; begin < length; begin += group_width) {
            #pragma HLS loop_tripcount min=0 max=8
            typename Policy::activation_word_t words[group_width];
            #pragma HLS array_partition variable=words complete
            for (unsigned int lane = 0; lane < group_width; ++lane) {
                #pragma HLS unroll
                words[lane] = 0;
            }
            const unsigned int remaining = length - begin;
            const unsigned int count = remaining < group_width ? remaining : group_width;
            // Descending rows permit a constant shift/insert, retaining zero
            // high lanes for Prefill tails and the actual GQA rows in Decode.
            for (unsigned int rows = unit.rows; rows != 0; --rows) {
                #pragma HLS loop_tripcount min=1 max=8
                const unsigned int row = rows - 1;
                const unsigned int token = task.phase == QUANTIZED_LAYER_PHASE_DECODE ? 0 : row;
                const unsigned int head = unit.head_begin +
                    (task.phase == QUANTIZED_LAYER_PHASE_DECODE ? row : 0);
#if QUANTIZED_ATTENTION_PACK_LANES > 1
                for (unsigned int group = 0; group < count; group += QUANTIZED_ATTENTION_PACK_LANES) {
                    #pragma HLS pipeline II=1
                    #pragma HLS loop_tripcount min=1 max=16
                    for (unsigned int port = 0; port < QUANTIZED_ATTENTION_PACK_LANES; ++port) {
                        #pragma HLS unroll
                        const unsigned int lane = group + port;
                        if (lane < count) {
                            const unsigned int k = begin + lane;
                            const fm_t input = pv ? fm_t(probability.value[token][quantized_attention_probability_head(head)][k]) :
                                Policy::feature_get(query, token, head * HEAD_DIM + k);
                            const typename Policy::element_t value = Policy::quantize(input, scales[head].inverse_scale);
                            words[lane] <<= Policy::bits;
                            words[lane].range(Policy::bits - 1, 0) = value;
                        }
                    }
                }
#else
                for (unsigned int lane = 0; lane < count; ++lane) {
                    #pragma HLS pipeline II=1
                    #pragma HLS loop_tripcount min=1 max=16
                    const unsigned int k = begin + lane;
                    const fm_t input = pv ? fm_t(probability.value[token][quantized_attention_probability_head(head)][k]) :
                        Policy::feature_get(query, token, head * HEAD_DIM + k);
                    const typename Policy::element_t value =
                        Policy::quantize(input, scales[head].inverse_scale);
                    words[lane] <<= Policy::bits;
                    words[lane].range(Policy::bits - 1, 0) = value;
                }
#endif
            }
            for (unsigned int lane = 0; lane < count; ++lane) {
                #pragma HLS pipeline II=1
                #pragma HLS loop_tripcount min=1 max=16
                panel.word[cu][begin + lane] = words[lane];
            }
        }
    }
}

template <typename Policy>
void emit_quantized_attention_wave(
    typename Policy::streams_t& streams, const quantized_layer_task_t& task,
    unsigned int tile, unsigned int wave, bool pv, unsigned int tile_length,
    const quantized_attention_activation_panel_t<Policy>& activation,
    const typename Policy::weight_tile_t weights[NUM_KEY_VALUE_HEADS]) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=activation.word complete dim=1
    #pragma HLS array_partition variable=weights complete dim=1
    for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
        #pragma HLS unroll
        const auto unit = quantized_attention_unit<Policy::token_rows>(task, wave, cu);
        if (unit.active) {
            mm_stream_quantized_task_t compute{};
            compute.k_count = pv ? tile_length : HEAD_DIM;
            compute.elem_base = pv ? 0 : tile * 128;
            compute.block_id = quantized_projection_waves_per_block() * 4 +
                (tile * quantized_attention_waves<Policy::token_rows>(task) + wave) * 8 +
                (pv ? 4 : 0) + cu;
            // Matrix engines return raw integer accumulators. Scale is per
            // logical head and applied during drain, also for mixed-head rows.
            compute.activation_scale = quant_scale_t(1);
            compute.weight_scale = quant_scale_t(1);
            compute.valid_tokens = unit.rows;
            compute.request_position = task.position;
            compute.kv_context_length = task.kv_context_length;
            compute.phase = static_cast<unsigned int>(task.phase);
            compute.compute_mode = pv ? MM_STREAM_QUANTIZED_MODE_ATTENTION_PV :
                MM_STREAM_QUANTIZED_MODE_ATTENTION_QK;
            Policy::write_task(streams, cu, pack_mm_stream_quantized_task(compute));
        }
    }
    const unsigned int length = pv ? tile_length : HEAD_DIM;
    for (unsigned int k = 0; k < length; ++k) {
        #pragma HLS pipeline II=1
        // Read each shared KV word once, then broadcast registers to its GQA
        // consumers. Repeating dynamic memory reads per CU creates false port
        // contention even though the logical payloads are identical.
        typename Policy::weight_word_t kv_word[NUM_KEY_VALUE_HEADS][Policy::weight_ports];
        #pragma HLS array_partition variable=kv_word complete dim=0
        for (unsigned int kv = 0; kv < NUM_KEY_VALUE_HEADS; ++kv) {
            #pragma HLS unroll
            for (unsigned int port = 0; port < Policy::weight_ports; ++port) {
                #pragma HLS unroll
                kv_word[kv][port] = weights[kv].word[k][port];
            }
        }
        for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
            #pragma HLS unroll
            const auto unit = quantized_attention_unit<Policy::token_rows>(task, wave, cu);
            if (unit.active) {
                Policy::write_activation(streams, cu, activation.word[cu][k]);
                for (unsigned int port = 0; port < Policy::weight_ports; ++port) {
                    #pragma HLS unroll
                    Policy::write_weight(streams, cu, port, kv_word[unit.kv_head][port]);
                }
            }
        }
    }
}

template <typename Policy>
void drain_quantized_attention_wave(
    typename Policy::streams_t& streams, const quantized_layer_task_t& task,
    unsigned int tile, unsigned int wave, bool pv, unsigned int tile_length,
    const quantized_symmetric_scale_t activation_scales[NUM_ATTENTION_HEADS],
    const quantized_symmetric_scale_t weight_scales[NUM_KEY_VALUE_HEADS],
    quantized_attention_result_panel_t<Policy>& result) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=result.block complete dim=1
    constexpr unsigned int payload = Policy::output_word_t::width == 576 ? 512 : 384;
    for (unsigned int packet = 0; packet < Policy::token_rows * 8; ++packet) {
        #pragma HLS pipeline II=1
        for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
            #pragma HLS unroll
            const auto unit = quantized_attention_unit<Policy::token_rows>(task, wave, cu);
            if (unit.active) {
                const auto word = Policy::read_output(streams, cu);
                const unsigned int row = word.range(payload + 23, payload + 16).to_uint();
                const unsigned int base = word.range(payload + 39, payload + 24).to_uint();
                const unsigned int column = pv ? base : base - tile * 128;
                const unsigned int head = unit.head_begin +
                    (task.phase == QUANTIZED_LAYER_PHASE_DECODE ? row : 0);
                if (row < unit.rows && column < 128) {
                    mm_input_block_t packed = 0;
                    for (unsigned int lane = 0; lane < 16; ++lane) {
                        #pragma HLS unroll
                        fm_t value = 0;
                        if (column + lane < (pv ? HEAD_DIM : tile_length)) {
                            value = Policy::dequantize(word, lane,
                                activation_scales[head].scale, weight_scales[unit.kv_head].scale);
                            if (!pv) value = fm_t(value * fm_t(ATTENTION_SCALE));
                        }
                        set_mm_input_block_lane(packed, lane, value);
                    }
                    result.block[cu][row][column / 16] = packed;
                }
            }
        }
    }
}

template <typename Policy>
void exchange_quantized_attention_wave(
    typename Policy::streams_t& streams, const quantized_layer_task_t& task,
    unsigned int tile, unsigned int wave, bool pv, unsigned int tile_length,
    const quantized_attention_activation_panel_t<Policy>& activation,
    const typename Policy::weight_tile_t weights[NUM_KEY_VALUE_HEADS],
    const quantized_symmetric_scale_t activation_scales[NUM_ATTENTION_HEADS],
    const quantized_symmetric_scale_t weight_scales[NUM_KEY_VALUE_HEADS],
    quantized_attention_result_panel_t<Policy>& result) {
    #pragma HLS inline off
    #pragma HLS dataflow
    emit_quantized_attention_wave<Policy>(streams, task, tile, wave, pv,
                                          tile_length, activation, weights);
#ifndef __SYNTHESIS__
    // Ordinary controller C tests seed responses. Numerical feedback tests
    // specialize this hook to run the real MAC engines between emit/drain;
    // RTL fixtures connect those engines concurrently through finite FIFOs.
    Policy::service_csim(streams, task, wave);
#endif
    drain_quantized_attention_wave<Policy>(streams, task, tile, wave, pv,
        tile_length, activation_scales, weight_scales, result);
}

template <typename Policy>
void update_quantized_attention_wave(
    typename Policy::state_t& state, typename Policy::probability_t& probability,
    attention_prob_t old_scale[Policy::token_rows][NUM_ATTENTION_HEADS],
    const quantized_attention_result_panel_t<Policy>& scores, const quantized_layer_task_t& task,
    unsigned int wave, unsigned int tile_begin, unsigned int length) {
    #pragma HLS inline off
    // Same two reductions and causal mask as the resident Fix16 online
    // softmax. Update just this wave before issuing its PV task.
    const unsigned int tile_columns = length < 128 ? length : 128;
    for (unsigned int row = 0; row < task.query_tokens; ++row) {
        const unsigned int query_position = task.position + row;
        const unsigned int causal_columns = tile_begin > query_position ? 0 :
            (query_position - tile_begin < tile_columns ?
                query_position - tile_begin + 1 : tile_columns);
        for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
            if (quantized_attention_head_in_wave<Policy::token_rows>(task, wave, head)) {
                fm_t maximum = fm_t(-128);
                for (unsigned int col = 0; col < causal_columns; ++col) {
                    #pragma HLS pipeline II=1
                    #pragma HLS loop_tripcount min=0 max=128
                    const fm_t value = quantized_attention_result_value<Policy>(scores, task, row, head, col);
                    if (value > maximum) maximum = value;
                }
                const bool previous = state.sum[row][head] != fm_accum_t(0);
                if (previous && state.maximum[row][head] > maximum)
                    maximum = state.maximum[row][head];
                const attention_prob_t scale = previous ?
                    Policy::exp(state.maximum[row][head] - maximum) : attention_prob_t(0);
                fm_accum_t sum = 0;
                // All PV consumers use tile_columns: probability_scale,
                // activation-panel preparation and the emitted k_count.
                // Overwrite causal padding inside that span with zero on
                // every call. The unused capacity tail stays unobserved,
                // including when a later tile is shorter than its predecessor.
                for (unsigned int col = 0; col < tile_columns; ++col) {
                    #pragma HLS pipeline II=1
                    #pragma HLS loop_tripcount min=0 max=128
                    const attention_prob_t value = col < causal_columns ?
                        Policy::exp(quantized_attention_result_value<Policy>(scores, task, row, head, col) - maximum) :
                        attention_prob_t(0);
                    probability.value[row][head][col] = value;
                    sum += fm_accum_t(value);
                }
                old_scale[row][head] = scale;
                state.maximum[row][head] = maximum;
                state.sum[row][head] = state.sum[row][head] * fm_accum_t(scale) + sum;
            }
        }
    }
}

template <typename Policy>
void merge_quantized_attention_wave(
    typename Policy::state_t& state, const quantized_attention_result_panel_t<Policy>& pv,
    const attention_prob_t old_scale[Policy::token_rows][NUM_ATTENTION_HEADS],
    const quantized_layer_task_t& task, unsigned int wave) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=state.context cyclic factor=16 dim=3
    for (unsigned int row = 0; row < task.query_tokens; ++row) {
        for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
            if (quantized_attention_head_in_wave<Policy::token_rows>(task, wave, head)) {
                for (unsigned int block = 0; block < (HEAD_DIM + 15) / 16; ++block) {
                    #pragma HLS pipeline II=1
                    const mm_input_block_t word = quantized_attention_result_word<Policy>(pv, task, row, head, block);
                    for (unsigned int lane = 0; lane < 16; ++lane) {
                        #pragma HLS unroll
                        const unsigned int element = block * 16 + lane;
                        if (element < HEAD_DIM)
                            state.context[row][head][element] =
                                state.context[row][head][element] * fm_accum_t(old_scale[row][head]) +
                                fm_accum_t(unpack_mm_input_block_lane(word, lane));
                    }
                }
            }
        }
    }
}

template <typename Policy>
void initialize_quantized_attention_state(typename Policy::state_t& state) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=state.context cyclic factor=16 dim=3
    for (unsigned int row = 0; row < Policy::token_rows; ++row) {
        for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
            state.maximum[row][head] = fm_t(-128);
            state.sum[row][head] = 0;
            for (unsigned int block = 0; block < (HEAD_DIM + 15) / 16; ++block) {
                #pragma HLS pipeline II=1
                for (unsigned int lane = 0; lane < 16; ++lane) {
                    #pragma HLS unroll
                    if (block * 16 + lane < HEAD_DIM)
                        state.context[row][head][block * 16 + lane] = 0;
                }
            }
        }
    }
}

template <typename Policy>
void finalize_quantized_attention_state(typename Policy::hidden_t& destination,
    const typename Policy::state_t& state, unsigned int rows) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=state.context cyclic factor=16 dim=3
    static_assert(HEAD_DIM % 16 == 0, "resident heads must align to vector words");
    for (unsigned int row = 0; row < Policy::token_rows; ++row) {
        for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
            const fm_t inverse = row < rows ? Policy::reciprocal(state.sum[row][head]) : fm_t(0);
            for (unsigned int block = 0; block < HEAD_DIM / 16; ++block) {
                #pragma HLS pipeline II=1
                mm_input_block_t word = 0;
                for (unsigned int lane = 0; lane < 16; ++lane) {
                    #pragma HLS unroll
                    const fm_t value = row < rows ?
                        fm_t(state.context[row][head][block * 16 + lane] * fm_accum_t(inverse)) : fm_t(0);
                    set_mm_input_block_lane(word, lane, value);
                }
                destination.block[row][head * (HEAD_DIM / 16) + block] = word;
            }
        }
    }
}

#include "quantized_attention_nonlinear.hpp"
#include "quantized_attention_active_state.hpp"
#if QUANTIZED_ATTENTION_WAVE_PIPELINE
#include "quantized_attention_wave_pipeline.hpp"
#endif

#if QUANTIZED_ATTENTION_TILE_PREFETCH

// The consumer holds the current tile in local BRAM while this FIFO holds
// the next tile. Transfer one packed row per beat (512 bits for W4, 1024 for
// W8), never an aggregate containing a complete tile. HLS 2022.2 cannot
// aggregate a stream item wider than 4096 bits. There are no shared arrays
// or reverse credit channels between these two processes.
template <typename Policy>
struct quantized_attention_prefetch_transport {
    static constexpr unsigned int port_bits = Policy::weight_word_t::width;
    static constexpr unsigned int beat_bits = port_bits * Policy::weight_ports;
    // Per KV head: K scale, HEAD_DIM packed K rows, V scale, <=128 V rows.
    static constexpr unsigned int tile_words =
        NUM_KEY_VALUE_HEADS * (HEAD_DIM + 128 + 2);
    using word_t = ap_uint<beat_bits>;
    static_assert(beat_bits <= 4096, "packed attention beat exceeds HLS limit");
    static_assert(beat_bits >= quant_scale_t::width, "scale must fit a header beat");
};

template <typename Policy>
void write_quantized_attention_packed_head(
    const typename Policy::weight_tile_t& weights, quant_scale_t scale,
    unsigned int rows,
    hls::stream<typename quantized_attention_prefetch_transport<Policy>::word_t>& words) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=weights.word complete dim=2
    using transport = quantized_attention_prefetch_transport<Policy>;
    typename transport::word_t header = 0;
    header.range(quant_scale_t::width - 1, 0) = scale.range(quant_scale_t::width - 1, 0);
    words.write(header);
    for (unsigned int row = 0; row < rows; ++row) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=1 max=128
        typename transport::word_t beat = 0;
        for (unsigned int port = 0; port < Policy::weight_ports; ++port) {
            #pragma HLS unroll
            beat.range((port + 1) * transport::port_bits - 1,
                       port * transport::port_bits) = weights.word[row][port];
        }
        words.write(beat);
    }
}

template <typename Policy>
void read_quantized_attention_packed_head(
    typename Policy::weight_tile_t& weights, quantized_symmetric_scale_t& scale,
    unsigned int rows,
    hls::stream<typename quantized_attention_prefetch_transport<Policy>::word_t>& words) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=weights.word complete dim=2
    using transport = quantized_attention_prefetch_transport<Policy>;
    const typename transport::word_t header = words.read();
    scale.scale.range(quant_scale_t::width - 1, 0) = header.range(quant_scale_t::width - 1, 0);
    // Packing is finished in the producer; only the dequantization scale is
    // used by the consumer. Do not transmit the unused inverse scale.
    scale.inverse_scale = 0;
    for (unsigned int row = 0; row < rows; ++row) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=1 max=128
        const typename transport::word_t beat = words.read();
        for (unsigned int port = 0; port < Policy::weight_ports; ++port) {
            #pragma HLS unroll
            weights.word[row][port] = beat.range((port + 1) * transport::port_bits - 1,
                                                 port * transport::port_bits);
        }
    }
}

template <typename Policy>
void prefetch_quantized_attention_packed_tiles(
    const quantized_layer_task_t& task,
    const mm_input_block_t* key_cache, const mm_input_block_t* value_cache,
    hls::stream<typename quantized_attention_prefetch_transport<Policy>::word_t>& packed_tiles) {
    #pragma HLS inline off
    typename Policy::kv_tile_t kv_tile;
    #pragma HLS bind_storage variable=kv_tile.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=kv_tile.block complete dim=2
    typename Policy::weight_tile_t packed_head;
    #pragma HLS bind_storage variable=packed_head.word type=ram_2p impl=bram
    #pragma HLS array_partition variable=packed_head.word complete dim=2
    const unsigned int context = task.kv_context_length + task.query_tokens;
    const unsigned int tile_count = quantized_ceildiv(context, 128);
    for (unsigned int tile = 0; tile < tile_count; ++tile) {
        #pragma HLS loop_tripcount min=1 max=16 avg=4
        const unsigned int tile_begin = tile * 128;
        const unsigned int length = context - tile_begin < 128 ?
            context - tile_begin : 128;
        for (unsigned int kv = 0; kv < NUM_KEY_VALUE_HEADS; ++kv) {
            #pragma HLS loop_tripcount min=1 max=8
            Policy::load_kv(kv_tile, key_cache, task.layer, kv,
                            tile_begin, length);
            const auto key_scale = Policy::kv_scale(kv_tile, length);
            Policy::pack_key(packed_head, kv_tile, key_scale.inverse_scale, length);
            write_quantized_attention_packed_head<Policy>(packed_head,
                key_scale.scale, HEAD_DIM, packed_tiles);
            Policy::load_kv(kv_tile, value_cache, task.layer, kv,
                            tile_begin, length);
            const auto value_scale = Policy::kv_scale(kv_tile, length);
            Policy::pack_value(packed_head, kv_tile, length, value_scale.inverse_scale);
            write_quantized_attention_packed_head<Policy>(packed_head,
                value_scale.scale, length, packed_tiles);
        }
    }
}

template <typename Policy>
void consume_quantized_attention_packed_tiles(
    const quantized_layer_task_t& task,
    const typename Policy::hidden_t& query,
    typename Policy::hidden_t& destination,
    typename Policy::streams_t& streams,
    hls::stream<typename quantized_attention_prefetch_transport<Policy>::word_t>& packed_tiles) {
    #pragma HLS inline off
    static_assert(HEAD_DIM <= 128, "attention panel must cover one head");
    constexpr unsigned int max_waves = (NUM_ATTENTION_HEADS + 3) / 4;
    quantized_attention_activation_panel_t<Policy> queries[max_waves];
    quantized_attention_activation_panel_t<Policy> pv_activation;
    quantized_attention_result_panel_t<Policy> result;
    typename Policy::probability_t probability;
    typename Policy::state_t state;
    quantized_symmetric_scale_t query_scale[NUM_ATTENTION_HEADS];
    quantized_symmetric_scale_t probability_scale[NUM_ATTENTION_HEADS];
    attention_prob_t old_scale[Policy::token_rows][NUM_ATTENTION_HEADS];
    typename Policy::weight_tile_t key_weight[NUM_KEY_VALUE_HEADS];
    typename Policy::weight_tile_t value_weight[NUM_KEY_VALUE_HEADS];
    quantized_symmetric_scale_t key_scale[NUM_KEY_VALUE_HEADS];
    quantized_symmetric_scale_t value_scale[NUM_KEY_VALUE_HEADS];
    #pragma HLS array_partition variable=key_weight complete dim=1
    #pragma HLS array_partition variable=value_weight complete dim=1
    #pragma HLS bind_storage variable=key_weight type=ram_2p impl=bram
    #pragma HLS bind_storage variable=value_weight type=ram_2p impl=bram
    #pragma HLS array_partition variable=key_scale complete
    #pragma HLS array_partition variable=value_scale complete
    #pragma HLS bind_storage variable=queries type=ram_2p impl=bram
    #pragma HLS bind_storage variable=pv_activation.word type=ram_2p impl=bram
    #pragma HLS array_partition variable=pv_activation.word complete dim=1
    #pragma HLS bind_storage variable=result.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=result.block complete dim=1
    #pragma HLS bind_storage variable=probability.value type=ram_2p impl=bram
#if QUANTIZED_ATTENTION_HEAD_LANES > 1
    #pragma HLS array_partition variable=probability.value cyclic factor=QUANTIZED_ATTENTION_HEAD_LANES dim=2
#endif
#if QUANTIZED_ATTENTION_PACK_LANES > 1
    #pragma HLS array_partition variable=probability.value cyclic factor=QUANTIZED_ATTENTION_PACK_LANES dim=3
#endif
    #pragma HLS bind_storage variable=state.context type=ram_2p impl=bram
    #pragma HLS array_partition variable=state.context cyclic factor=16 dim=3
    #pragma HLS array_partition variable=query_scale complete
    #pragma HLS array_partition variable=probability_scale complete
#if QUANTIZED_ATTENTION_ACTIVE_STATE
    initialize_quantized_attention_active_state<Policy>(state, task.query_tokens);
#else
    initialize_quantized_attention_state<Policy>(state);
#endif
    const unsigned int waves = quantized_attention_waves<Policy::token_rows>(task);
    for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head)
        query_scale[head] = Policy::query_scale(query, head, task.query_tokens);
    for (unsigned int wave = 0; wave < waves; ++wave)
        prepare_quantized_attention_activation_panel<Policy>(queries[wave],
            task, wave, false, HEAD_DIM, query, probability, query_scale);

    const unsigned int context = task.kv_context_length + task.query_tokens;
    const unsigned int tile_count = quantized_ceildiv(context, 128);
    for (unsigned int tile = 0; tile < tile_count; ++tile) {
        #pragma HLS loop_tripcount min=1 max=16 avg=4
        const unsigned int tile_begin = tile * 128;
        const unsigned int length = context - tile_begin < 128 ?
            context - tile_begin : 128;
        for (unsigned int kv = 0; kv < NUM_KEY_VALUE_HEADS; ++kv) {
            read_quantized_attention_packed_head<Policy>(key_weight[kv],
                key_scale[kv], HEAD_DIM, packed_tiles);
            read_quantized_attention_packed_head<Policy>(value_weight[kv],
                value_scale[kv], length, packed_tiles);
        }
#if QUANTIZED_ATTENTION_WAVE_PIPELINE
        run_quantized_attention_wave_pipeline<Policy>(streams, task, tile, length,
            waves, queries, key_weight, value_weight, query_scale, key_scale,
            value_scale, state, probability, old_scale, query);
#else
        for (unsigned int wave = 0; wave < waves; ++wave) {
            exchange_quantized_attention_wave<Policy>(streams, task, tile, wave,
                false, length, queries[wave], key_weight, query_scale,
                key_scale, result);
            update_and_scale_quantized_attention_wave<Policy>(state, probability, old_scale,
                result, task, wave, tile_begin, length, probability_scale);
            prepare_quantized_attention_activation_panel<Policy>(pv_activation,
                task, wave, true, length, query, probability, probability_scale);
            exchange_quantized_attention_wave<Policy>(streams, task, tile, wave,
                true, length, pv_activation, value_weight, probability_scale,
                value_scale, result);
            merge_quantized_attention_wave<Policy>(state, result, old_scale, task, wave);
        }
#endif
    }
#if QUANTIZED_ATTENTION_ACTIVE_STATE
    finalize_quantized_attention_active_state<Policy>(destination, state, task.query_tokens);
#else
    finalize_quantized_attention_state<Policy>(destination, state, task.query_tokens);
#endif
}

template <typename Policy>
void run_quantized_attention_aligned_tile_prefetch(
    const quantized_layer_task_t& task, const typename Policy::hidden_t& query,
    const mm_input_block_t* key_cache, const mm_input_block_t* value_cache,
    typename Policy::hidden_t& destination, typename Policy::streams_t& streams) {
    #pragma HLS inline off
    using transport = quantized_attention_prefetch_transport<Policy>;
    constexpr unsigned int prefetch_depth = transport::tile_words;
    hls::stream<typename transport::word_t> packed_tiles;
    #pragma HLS stream variable=packed_tiles depth=prefetch_depth
    #pragma HLS bind_storage variable=packed_tiles type=fifo impl=bram
    #pragma HLS dataflow
    prefetch_quantized_attention_packed_tiles<Policy>(task, key_cache, value_cache,
        packed_tiles);
    consume_quantized_attention_packed_tiles<Policy>(task, query, destination,
        streams, packed_tiles);
}

#endif

template <typename Policy>
void run_quantized_attention_aligned(
    const quantized_layer_task_t& task, const typename Policy::hidden_t& query,
    const mm_input_block_t* key_cache, const mm_input_block_t* value_cache,
    typename Policy::hidden_t& destination, typename Policy::streams_t& streams) {
#if QUANTIZED_ATTENTION_TILE_PREFETCH
    // CSim executes this same serialization/packing path with unbounded
    // software streams. RTL CoSim checks the finite FIFO and backpressure.
    run_quantized_attention_aligned_tile_prefetch<Policy>(
        task, query, key_cache, value_cache, destination, streams);
#else
    #pragma HLS inline
    static_assert(HEAD_DIM <= 128, "attention panel must cover one head");
    constexpr unsigned int max_waves = (NUM_ATTENTION_HEADS + 3) / 4;
    quantized_attention_activation_panel_t<Policy> queries[max_waves];
    quantized_attention_activation_panel_t<Policy> pv_activation;
    typename Policy::weight_tile_t key_weight[NUM_KEY_VALUE_HEADS];
    typename Policy::weight_tile_t value_weight[NUM_KEY_VALUE_HEADS];
    typename Policy::kv_tile_t kv_tile;
    quantized_attention_result_panel_t<Policy> result;
    typename Policy::probability_t probability;
    typename Policy::state_t state;
    quantized_symmetric_scale_t query_scale[NUM_ATTENTION_HEADS];
    quantized_symmetric_scale_t probability_scale[NUM_ATTENTION_HEADS];
    quantized_symmetric_scale_t key_scale[NUM_KEY_VALUE_HEADS];
    quantized_symmetric_scale_t value_scale[NUM_KEY_VALUE_HEADS];
    attention_prob_t old_scale[Policy::token_rows][NUM_ATTENTION_HEADS];
    #pragma HLS bind_storage variable=queries type=ram_2p impl=bram
    #pragma HLS bind_storage variable=pv_activation.word type=ram_2p impl=bram
    #pragma HLS array_partition variable=pv_activation.word complete dim=1
    #pragma HLS array_partition variable=key_weight complete dim=1
    #pragma HLS array_partition variable=value_weight complete dim=1
    #pragma HLS bind_storage variable=key_weight type=ram_2p impl=bram
    #pragma HLS bind_storage variable=value_weight type=ram_2p impl=bram
    #pragma HLS bind_storage variable=kv_tile.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=kv_tile.block complete dim=2
    #pragma HLS bind_storage variable=result.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=result.block complete dim=1
    #pragma HLS bind_storage variable=probability.value type=ram_2p impl=bram
    #pragma HLS bind_storage variable=state.context type=ram_2p impl=bram
    #pragma HLS array_partition variable=state.context cyclic factor=16 dim=3
    #pragma HLS array_partition variable=query_scale complete
    #pragma HLS array_partition variable=probability_scale complete
    #pragma HLS array_partition variable=key_scale complete
    #pragma HLS array_partition variable=value_scale complete
    initialize_quantized_attention_state<Policy>(state);
    const unsigned int waves = quantized_attention_waves<Policy::token_rows>(task);
    // Query block is cached once and reused by all K/V tiles.
    for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head)
        query_scale[head] = Policy::query_scale(query, head, task.query_tokens);
    for (unsigned int wave = 0; wave < waves; ++wave)
        prepare_quantized_attention_activation_panel<Policy>(queries[wave],
            task, wave, false, HEAD_DIM, query, probability, query_scale);
    const unsigned int context = task.kv_context_length + task.query_tokens;
    for (unsigned int tile = 0; tile < quantized_ceildiv(context, 128); ++tile) {
        const unsigned int length = context - tile * 128 < 128 ? context - tile * 128 : 128;
        // Each HBM K/V tile serves all GQA head waves before eviction.
        for (unsigned int kv = 0; kv < NUM_KEY_VALUE_HEADS; ++kv) {
            Policy::load_kv(kv_tile, key_cache, task.layer, kv, tile * 128, length);
            key_scale[kv] = Policy::kv_scale(kv_tile, length);
            Policy::pack_key(key_weight[kv], kv_tile, key_scale[kv].inverse_scale, length);
            Policy::load_kv(kv_tile, value_cache, task.layer, kv, tile * 128, length);
            value_scale[kv] = Policy::kv_scale(kv_tile, length);
            Policy::pack_value(value_weight[kv], kv_tile, length, value_scale[kv].inverse_scale);
        }
        for (unsigned int wave = 0; wave < waves; ++wave) {
            exchange_quantized_attention_wave<Policy>(streams, task, tile, wave,
                false, length, queries[wave], key_weight, query_scale, key_scale, result);
            update_quantized_attention_wave<Policy>(state, probability, old_scale,
                result, task, wave, tile * 128, length);
            for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
                if (quantized_attention_head_in_wave<Policy::token_rows>(task, wave, head))
                    probability_scale[head] = Policy::probability_scale(
                        probability, head, task.query_tokens, length);
            }
            prepare_quantized_attention_activation_panel<Policy>(pv_activation,
                task, wave, true, length, query, probability, probability_scale);
            exchange_quantized_attention_wave<Policy>(streams, task, tile, wave,
                true, length, pv_activation, value_weight, probability_scale, value_scale, result);
            merge_quantized_attention_wave<Policy>(state, result, old_scale, task, wave);
        }
    }
    finalize_quantized_attention_state<Policy>(destination, state, task.query_tokens);
#endif
}

#endif
