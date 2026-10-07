#ifndef LLM_FPGA_QUANTIZED_ATTENTION_WAVE_PIPELINE_HPP
#define LLM_FPGA_QUANTIZED_ATTENTION_WAVE_PIPELINE_HPP
#ifndef __SYNTHESIS__
#include <cassert>
#endif

// Only the matrix stage owns CU task/input/output streams. Nonlinear state
// remains private to the other stage. One complete score frame allows QK(n+1)
// to finish while nonlinear(n) prepares PV, without a shared-array PIPO cycle.
template <typename Policy>
struct quantized_attention_wave_transport {
    // Keep row/block tags so the 64-output W4 alternative may use a different
    // packet order without changing the consumer's logical result layout.
    struct result_t {
        ap_uint<256 * QUANTIZED_LAYER_COMPUTE_CUS> data;
        ap_uint<16 * QUANTIZED_LAYER_COMPUTE_CUS> tags;
    };
    using activation_t = ap_uint<Policy::activation_word_t::width * QUANTIZED_LAYER_COMPUTE_CUS>;
    static constexpr unsigned int frame_words = Policy::token_rows * 8;
};

inline bool quantized_attention_wave_step_is_pv(unsigned int step, unsigned int waves) {
    #pragma HLS inline
    return step == 2 * waves - 1 || (step > 0 && (step & 1) == 0);
}
inline unsigned int quantized_attention_wave_step_index(unsigned int step, unsigned int waves) {
    #pragma HLS inline
    return step == 0 ? 0 : step == 2 * waves - 1 ? waves - 1 :
        (step & 1) ? (step + 1) / 2 : step / 2 - 1;
}

template <typename Policy>
void receive_quantized_attention_result_frame(
    hls::stream<typename quantized_attention_wave_transport<Policy>::result_t>& pipe,
    quantized_attention_result_panel_t<Policy>& result) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=result.block complete dim=1
    for (unsigned int packet = 0; packet < Policy::token_rows * 8; ++packet) {
        #pragma HLS pipeline II=1
        const auto frame = pipe.read();
        for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
            #pragma HLS unroll
            const unsigned int row = frame.tags.range(16 * cu + 7, 16 * cu).to_uint();
            const unsigned int block = frame.tags.range(16 * cu + 15, 16 * cu + 8).to_uint();
#ifndef __SYNTHESIS__
            assert(row < Policy::token_rows && block < 8);
#endif
            if (row < Policy::token_rows && block < 8)
                result.block[cu][row][block] = frame.data.range(256 * cu + 255, 256 * cu);
        }
    }
}

template <typename Policy>
void drain_quantized_attention_result_frame(
    typename Policy::streams_t& streams, const quantized_layer_task_t& task,
    unsigned int tile, unsigned int wave, bool pv, unsigned int length,
    const quantized_symmetric_scale_t activation_scales[NUM_ATTENTION_HEADS],
    const quantized_symmetric_scale_t weight_scales[NUM_KEY_VALUE_HEADS],
    hls::stream<typename quantized_attention_wave_transport<Policy>::result_t>& pipe) {
    #pragma HLS inline off
    constexpr unsigned int payload = Policy::output_word_t::width == 576 ? 512 : 384;
    for (unsigned int packet = 0; packet < Policy::token_rows * 8; ++packet) {
        #pragma HLS pipeline II=1
        typename quantized_attention_wave_transport<Policy>::result_t frame{};
        frame.data = 0;
        frame.tags = 0;
        for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
            #pragma HLS unroll
            const auto unit = quantized_attention_unit<Policy::token_rows>(task, wave, cu);
            frame.tags.range(16 * cu + 7, 16 * cu) = packet / 8;
            frame.tags.range(16 * cu + 15, 16 * cu + 8) = packet % 8;
            if (unit.active) {
                const auto word = Policy::read_output(streams, cu);
                const unsigned int row = word.range(payload + 23, payload + 16).to_uint();
                const unsigned int base = word.range(payload + 39, payload + 24).to_uint();
                const unsigned int column = pv ? base : base - tile * 128;
                const unsigned int head = unit.head_begin +
                    (task.phase == QUANTIZED_LAYER_PHASE_DECODE ? row : 0);
                mm_input_block_t packed = 0;
                if (row < unit.rows && column < 128) {
                    for (unsigned int lane = 0; lane < 16; ++lane) {
                        #pragma HLS unroll
                        fm_t value = 0;
                        if (column + lane < (pv ? HEAD_DIM : length)) {
                            value = Policy::dequantize(word, lane, activation_scales[head].scale,
                                weight_scales[unit.kv_head].scale);
                            if (!pv) value = fm_t(value * fm_t(ATTENTION_SCALE));
                        }
                        set_mm_input_block_lane(packed, lane, value);
                    }
                }
                frame.data.range(256 * cu + 255, 256 * cu) = packed;
                frame.tags.range(16 * cu + 7, 16 * cu) = row;
                frame.tags.range(16 * cu + 15, 16 * cu + 8) = column / 16;
            }
        }
        pipe.write(frame);
    }
}

template <typename Policy>
void emit_quantized_attention_pv_from_pipe(
    typename Policy::streams_t& streams, const quantized_layer_task_t& task,
    unsigned int tile, unsigned int wave, unsigned int length,
    const typename Policy::weight_tile_t weights[NUM_KEY_VALUE_HEADS],
    hls::stream<typename quantized_attention_wave_transport<Policy>::activation_t>& activation) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=weights complete dim=1
    for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
        #pragma HLS unroll
        const auto unit = quantized_attention_unit<Policy::token_rows>(task, wave, cu);
        if (unit.active) {
            mm_stream_quantized_task_t compute{};
            compute.k_count = length;
            compute.elem_base = 0;
            compute.block_id = quantized_projection_waves_per_block() * 4 +
                (tile * quantized_attention_waves<Policy::token_rows>(task) + wave) * 8 + 4 + cu;
            compute.activation_scale = quant_scale_t(1);
            compute.weight_scale = quant_scale_t(1);
            compute.valid_tokens = unit.rows;
            compute.request_position = task.position;
            compute.kv_context_length = task.kv_context_length;
            compute.phase = static_cast<unsigned int>(task.phase);
            compute.compute_mode = MM_STREAM_QUANTIZED_MODE_ATTENTION_PV;
            Policy::write_task(streams, cu, pack_mm_stream_quantized_task(compute));
        }
    }
    for (unsigned int k = 0; k < length; ++k) {
        #pragma HLS pipeline II=1
        const auto packed = activation.read();
        typename Policy::weight_word_t words[NUM_KEY_VALUE_HEADS][Policy::weight_ports];
        #pragma HLS array_partition variable=words complete dim=0
        for (unsigned int kv = 0; kv < NUM_KEY_VALUE_HEADS; ++kv) {
            #pragma HLS unroll
            for (unsigned int port = 0; port < Policy::weight_ports; ++port) {
                #pragma HLS unroll
                words[kv][port] = weights[kv].word[k][port];
            }
        }
        for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
            #pragma HLS unroll
            const auto unit = quantized_attention_unit<Policy::token_rows>(task, wave, cu);
            if (unit.active) {
                constexpr unsigned int width = Policy::activation_word_t::width;
                Policy::write_activation(streams, cu, packed.range((cu + 1) * width - 1, cu * width));
                for (unsigned int port = 0; port < Policy::weight_ports; ++port) {
                    #pragma HLS unroll
                    Policy::write_weight(streams, cu, port, words[unit.kv_head][port]);
                }
            }
        }
    }
}

template <typename Policy>
void exchange_quantized_attention_qk_frame(
    typename Policy::streams_t& streams, const quantized_layer_task_t& task,
    unsigned int tile, unsigned int wave, unsigned int length,
    const quantized_attention_activation_panel_t<Policy>& query,
    const typename Policy::weight_tile_t weights[NUM_KEY_VALUE_HEADS],
    const quantized_symmetric_scale_t query_scale[NUM_ATTENTION_HEADS],
    const quantized_symmetric_scale_t key_scale[NUM_KEY_VALUE_HEADS],
    hls::stream<typename quantized_attention_wave_transport<Policy>::result_t>& scores) {
    #pragma HLS inline off
    #pragma HLS dataflow
    emit_quantized_attention_wave<Policy>(streams, task, tile, wave, false, length, query, weights);
#ifndef __SYNTHESIS__
    Policy::service_csim(streams, task, wave);
#endif
    drain_quantized_attention_result_frame<Policy>(streams, task, tile, wave, false,
        length, query_scale, key_scale, scores);
}

template <typename Policy>
void exchange_quantized_attention_pv_frame(
    typename Policy::streams_t& streams, const quantized_layer_task_t& task,
    unsigned int tile, unsigned int wave, unsigned int length,
    const typename Policy::weight_tile_t weights[NUM_KEY_VALUE_HEADS],
    const quantized_symmetric_scale_t probability_scale[NUM_ATTENTION_HEADS],
    const quantized_symmetric_scale_t value_scale[NUM_KEY_VALUE_HEADS],
    hls::stream<typename quantized_attention_wave_transport<Policy>::activation_t>& activation,
    hls::stream<typename quantized_attention_wave_transport<Policy>::result_t>& results) {
    #pragma HLS inline off
    #pragma HLS dataflow
    emit_quantized_attention_pv_from_pipe<Policy>(streams, task, tile, wave, length, weights, activation);
#ifndef __SYNTHESIS__
    Policy::service_csim(streams, task, wave);
#endif
    drain_quantized_attention_result_frame<Policy>(streams, task, tile, wave, true,
        length, probability_scale, value_scale, results);
}

template <typename Policy>
void prepare_quantized_attention_pv_frame(
    typename Policy::state_t& state, typename Policy::probability_t& probability,
    attention_prob_t old_scale[Policy::token_rows][NUM_ATTENTION_HEADS],
    const typename Policy::hidden_t& query, const quantized_layer_task_t& task,
    unsigned int wave, unsigned int tile_begin, unsigned int length,
    quantized_attention_result_panel_t<Policy>& result,
    hls::stream<typename quantized_attention_wave_transport<Policy>::result_t>& scores,
    hls::stream<quant_scale_t>& scales,
    hls::stream<typename quantized_attention_wave_transport<Policy>::activation_t>& activation) {
    #pragma HLS inline off
    quantized_attention_activation_panel_t<Policy> panel;
    quantized_symmetric_scale_t probability_scale[NUM_ATTENTION_HEADS];
    #pragma HLS array_partition variable=result.block complete dim=1
    #pragma HLS bind_storage variable=result.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=panel.word complete dim=1
    #pragma HLS bind_storage variable=panel.word type=ram_2p impl=bram
    #pragma HLS array_partition variable=probability_scale complete
    receive_quantized_attention_result_frame<Policy>(scores, result);
    update_and_scale_quantized_attention_wave<Policy>(state, probability, old_scale,
        result, task, wave, tile_begin, length, probability_scale);
    prepare_quantized_attention_activation_panel<Policy>(panel, task, wave, true,
        length, query, probability, probability_scale);
    for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
        #pragma HLS pipeline II=1
        scales.write(quantized_attention_head_in_wave<Policy::token_rows>(task, wave, head) ?
            probability_scale[head].scale : quant_scale_t(1));
    }
    for (unsigned int k = 0; k < length; ++k) {
        #pragma HLS pipeline II=1
        typename quantized_attention_wave_transport<Policy>::activation_t packed = 0;
        for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
            #pragma HLS unroll
            constexpr unsigned int width = Policy::activation_word_t::width;
            const auto unit = quantized_attention_unit<Policy::token_rows>(task, wave, cu);
            if (unit.active) packed.range((cu + 1) * width - 1, cu * width) = panel.word[cu][k];
        }
        activation.write(packed);
    }
}

template <typename Policy>
void finish_quantized_attention_pv_frame(
    typename Policy::state_t& state,
    attention_prob_t old_scale[Policy::token_rows][NUM_ATTENTION_HEADS],
    const quantized_layer_task_t& task, unsigned int wave,
    quantized_attention_result_panel_t<Policy>& result,
    hls::stream<typename quantized_attention_wave_transport<Policy>::result_t>& results) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=result.block complete dim=1
    #pragma HLS bind_storage variable=result.block type=ram_2p impl=bram
    receive_quantized_attention_result_frame<Policy>(results, result);
    merge_quantized_attention_wave<Policy>(state, result, old_scale, task, wave);
}

template <typename Policy>
void matrix_quantized_attention_wave_pipeline(
    typename Policy::streams_t& streams, const quantized_layer_task_t& task,
    unsigned int tile, unsigned int length, unsigned int waves,
    const quantized_attention_activation_panel_t<Policy> queries[(NUM_ATTENTION_HEADS + 3) / 4],
    const typename Policy::weight_tile_t key[NUM_KEY_VALUE_HEADS],
    const typename Policy::weight_tile_t value[NUM_KEY_VALUE_HEADS],
    const quantized_symmetric_scale_t query_scale[NUM_ATTENTION_HEADS],
    const quantized_symmetric_scale_t key_scale[NUM_KEY_VALUE_HEADS],
    const quantized_symmetric_scale_t value_scale[NUM_KEY_VALUE_HEADS],
    hls::stream<typename quantized_attention_wave_transport<Policy>::result_t>& scores,
    hls::stream<typename quantized_attention_wave_transport<Policy>::result_t>& results,
    hls::stream<quant_scale_t>& scales,
    hls::stream<typename quantized_attention_wave_transport<Policy>::activation_t>& activation) {
    #pragma HLS inline off
    for (unsigned int step = 0; step < 2 * waves; ++step) {
        const unsigned int wave = quantized_attention_wave_step_index(step, waves);
        if (!quantized_attention_wave_step_is_pv(step, waves)) {
            exchange_quantized_attention_qk_frame<Policy>(streams, task, tile, wave,
                length, queries[wave], key, query_scale, key_scale, scores);
        } else {
            quantized_symmetric_scale_t probability_scale[NUM_ATTENTION_HEADS];
            #pragma HLS array_partition variable=probability_scale complete
            for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
                #pragma HLS pipeline II=1
                probability_scale[head].scale = scales.read();
                probability_scale[head].inverse_scale = quant_inverse_scale_t(1);
            }
            exchange_quantized_attention_pv_frame<Policy>(streams, task, tile, wave,
                length, value, probability_scale, value_scale, activation, results);
        }
    }
}

template <typename Policy>
void nonlinear_quantized_attention_wave_pipeline(
    typename Policy::state_t& state, typename Policy::probability_t& probability,
    attention_prob_t old_scale[Policy::token_rows][NUM_ATTENTION_HEADS],
    const typename Policy::hidden_t& query, const quantized_layer_task_t& task,
    unsigned int tile_begin, unsigned int length, unsigned int waves,
    hls::stream<typename quantized_attention_wave_transport<Policy>::result_t>& scores,
    hls::stream<typename quantized_attention_wave_transport<Policy>::result_t>& results,
    hls::stream<quant_scale_t>& scales,
    hls::stream<typename quantized_attention_wave_transport<Policy>::activation_t>& activation) {
    #pragma HLS inline off
    quantized_attention_result_panel_t<Policy> result;
    #pragma HLS array_partition variable=result.block complete dim=1
    #pragma HLS bind_storage variable=result.block type=ram_2p impl=bram
    for (unsigned int wave = 0; wave < waves; ++wave) {
        prepare_quantized_attention_pv_frame<Policy>(state, probability, old_scale,
            query, task, wave, tile_begin, length, result, scores, scales, activation);
        finish_quantized_attention_pv_frame<Policy>(state, old_scale, task, wave, result, results);
    }
}

template <typename Policy>
void run_quantized_attention_wave_pipeline(
    typename Policy::streams_t& streams, const quantized_layer_task_t& task,
    unsigned int tile, unsigned int length, unsigned int waves,
    const quantized_attention_activation_panel_t<Policy> queries[(NUM_ATTENTION_HEADS + 3) / 4],
    const typename Policy::weight_tile_t key[NUM_KEY_VALUE_HEADS],
    const typename Policy::weight_tile_t value[NUM_KEY_VALUE_HEADS],
    const quantized_symmetric_scale_t query_scale[NUM_ATTENTION_HEADS],
    const quantized_symmetric_scale_t key_scale[NUM_KEY_VALUE_HEADS],
    const quantized_symmetric_scale_t value_scale[NUM_KEY_VALUE_HEADS],
    typename Policy::state_t& state, typename Policy::probability_t& probability,
    attention_prob_t old_scale[Policy::token_rows][NUM_ATTENTION_HEADS],
    const typename Policy::hidden_t& query) {
    #pragma HLS inline off
    using transport = quantized_attention_wave_transport<Policy>;
    hls::stream<typename transport::result_t> scores, results;
    #pragma HLS aggregate variable=scores compact=bit
    #pragma HLS aggregate variable=results compact=bit
    hls::stream<quant_scale_t> scales;
    hls::stream<typename transport::activation_t> activation;
    constexpr unsigned int score_depth = transport::frame_words;
    #pragma HLS stream variable=scores depth=score_depth
    #pragma HLS stream variable=results depth=2
    #pragma HLS stream variable=scales depth=NUM_ATTENTION_HEADS
    #pragma HLS stream variable=activation depth=16
    #pragma HLS bind_storage variable=scores type=fifo impl=bram
#ifdef __SYNTHESIS__
    #pragma HLS dataflow
    matrix_quantized_attention_wave_pipeline<Policy>(streams, task, tile, length,
        waves, queries, key, value, query_scale, key_scale, value_scale,
        scores, results, scales, activation);
    nonlinear_quantized_attention_wave_pipeline<Policy>(state, probability,
        old_scale, query, task, tile * 128, length, waves, scores, results, scales, activation);
#else
    // Execute the identical packet order with unbounded C streams. Finite
    // backpressure and concurrent stage progress are checked by RTL CoSim.
    quantized_attention_result_panel_t<Policy> result;
    for (unsigned int step = 0; step < 2 * waves; ++step) {
        const unsigned int wave = quantized_attention_wave_step_index(step, waves);
        if (!quantized_attention_wave_step_is_pv(step, waves)) {
            exchange_quantized_attention_qk_frame<Policy>(streams, task, tile, wave,
                length, queries[wave], key, query_scale, key_scale, scores);
        } else {
            prepare_quantized_attention_pv_frame<Policy>(state, probability, old_scale,
                query, task, wave, tile * 128, length, result, scores, scales, activation);
            quantized_symmetric_scale_t ps[NUM_ATTENTION_HEADS];
            for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
                ps[head].scale = scales.read();
                ps[head].inverse_scale = quant_inverse_scale_t(1);
            }
            exchange_quantized_attention_pv_frame<Policy>(streams, task, tile, wave,
                length, value, ps, value_scale, activation, results);
            finish_quantized_attention_pv_frame<Policy>(state, old_scale, task, wave, result, results);
        }
    }
#endif
}

#endif
