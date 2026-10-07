#ifndef LLM_FPGA_QUANTIZED_BATCH_PROJECTION_HPP
#define LLM_FPGA_QUANTIZED_BATCH_PROJECTION_HPP

// Q4.2-BD/Q8.2-BD: independent Decode rows share linear-projection weights.
// Scaling remains per sequence, so batching never changes its quantization.
#include "mm_stream_8x128_int4x4_block.hpp"
#include "quantized_layer_scalar_pipeline.hpp"
#ifdef QUANTIZED_ALIGNMENT_W8
#include "quantized_w8_projection_engine.hpp"
using qbd_policy_t = quantized_w8_projection_policy;
using qbd_hidden_t = quantized_w8_hidden_buffer_t;
using qbd_kv_t = quantized_w8_kv_buffer_t;
using qbd_wide_t = quantized_w8_wide_buffer_t;
constexpr unsigned int QBD_BITS = 8;
constexpr unsigned int QBD_WEIGHT_DEPTH = QUANTIZED_W8_WEIGHT_MEMORY_DEPTH;
#else
#include "quantized_w4_projection_engine.hpp"
using qbd_policy_t = quantized_w4_projection_policy;
using qbd_hidden_t = quantized_w4_hidden_buffer_t;
using qbd_kv_t = quantized_w4_kv_buffer_t;
using qbd_wide_t = quantized_w4_wide_buffer_t;
constexpr unsigned int QBD_BITS = 4;
constexpr unsigned int QBD_WEIGHT_DEPTH = QUANTIZED_W4_WEIGHT_MEMORY_DEPTH;
#endif
using qbd_weight_t = qbd_policy_t::weight_word_t;
constexpr unsigned int QBD_ROWS = qbd_policy_t::token_rows;
constexpr unsigned int QBD_HIDDEN_BLOCKS = (HIDDEN_SIZE + 15) / 16;
constexpr unsigned int QBD_KV_BLOCKS = (KV_CHANNELS + 15) / 16;
constexpr unsigned int QBD_WIDE_BLOCKS = (INTERMEDIATE_SIZE + 15) / 16;

inline void quantized_batch_projection_scales(
    const qbd_hidden_t& hidden, const qbd_wide_t& wide, bool source_is_wide,
    unsigned int rows, unsigned int elements,
    quant_scale_t scales[QBD_ROWS], quant_inverse_scale_t inverses[QBD_ROWS]) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=hidden.block complete dim=1
    #pragma HLS array_partition variable=wide.block complete dim=1
    #pragma HLS array_partition variable=scales complete
    #pragma HLS array_partition variable=inverses complete
    for (unsigned int row = 0; row < QBD_ROWS; ++row) {
        fm_t maximum = 0;
        if (row < rows) {
            for (unsigned int block = 0; block < (elements + 15) / 16; ++block) {
                #pragma HLS pipeline II=1
                #pragma HLS loop_tripcount min=1 max=688
                const auto word = source_is_wide ? wide.block[row][block] : hidden.block[row][block];
                const unsigned int remaining = elements - block * 16;
                const fm_t magnitude = quantized_word_max_abs(word, remaining < 16 ? remaining : 16);
                if (magnitude > maximum) maximum = magnitude;
            }
        }
#ifdef QUANTIZED_ALIGNMENT_W8
        const auto scale = make_quantized_symmetric_scale(maximum);
#else
        const auto scale = make_quantized_symmetric_scale_w4(maximum);
#endif
        scales[row] = scale.scale;
        inverses[row] = scale.inverse_scale;
    }
}

inline void drive_quantized_batch_projection(
    qbd_policy_t::streams_t& streams,
    hls::stream<qbd_weight_t> weights[4][qbd_policy_t::weight_ports],
    const qbd_hidden_t& hidden, const qbd_wide_t& wide, bool source_is_wide,
    const quantized_layer_task_t& task, quantized_projection_t kind,
    const quant_inverse_scale_t inverses[QBD_ROWS]) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=weights complete dim=0
    #pragma HLS array_partition variable=hidden.block complete dim=1
    #pragma HLS array_partition variable=wide.block complete dim=1
    #pragma HLS array_partition variable=inverses complete
    const auto plan = get_quantized_projection_plan(kind);
    for (unsigned int wave = 0; wave < plan.wave_count; ++wave) {
        #pragma HLS loop_tripcount min=1 max=22
        const unsigned int mask = quantized_projection_active_cu_mask(plan, wave);
        for (unsigned int cu = 0; cu < 4; ++cu) {
            #pragma HLS unroll
            if (mask & (1u << cu)) {
                const auto dispatch = make_quantized_projection_dispatch(kind, wave, cu, false, task.layer);
                // Compute emits raw accumulators. Controller applies each row's scale.
                qbd_policy_t::write_task(streams, cu, pack_mm_stream_quantized_task(
                    make_quantized_projection_task(task, dispatch, quant_scale_t(1), quant_scale_t(1))));
            }
        }
        for (unsigned int k = 0; k < plan.input_dim; ++k) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=16 max=11008
            qbd_policy_t::activation_word_t activation = 0;
            for (unsigned int row = 0; row < QBD_ROWS; ++row) {
                #pragma HLS unroll
                fm_t value = 0;
                if (row < task.query_tokens) {
                    const mm_input_block_t word = source_is_wide ? wide.block[row][k / 16] : hidden.block[row][k / 16];
                    value = unpack_mm_input_block_lane(word, k % 16);
                }
#ifdef QUANTIZED_ALIGNMENT_W8
                const ap_int<8> integer = quantize_symmetric_int8(value, inverses[row]);
#else
                const ap_int<4> integer = quantize_symmetric_int4(value, inverses[row]);
#endif
                activation.range((row + 1) * QBD_BITS - 1, row * QBD_BITS) = integer;
            }
            for (unsigned int cu = 0; cu < 4; ++cu) {
                #pragma HLS unroll
                if (mask & (1u << cu)) {
                    qbd_policy_t::write_activation(streams, cu, activation);
                    for (unsigned int port = 0; port < qbd_policy_t::weight_ports; ++port) {
                        #pragma HLS unroll
                        qbd_policy_t::write_weight(streams, cu, port, weights[cu][port].read());
                    }
                }
            }
        }
    }
}

inline void commit_quantized_batch_projection(
    qbd_wide_t& destination,
    hls::stream<qbd_policy_t::output_word_t> results[4],
    quantized_projection_t kind, unsigned int rows,
    const quant_scale_t scales[QBD_ROWS], quant_scale_t weight_scale) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=results complete
    #pragma HLS array_partition variable=destination.block complete dim=1
    #pragma HLS array_partition variable=scales complete
    const auto plan = get_quantized_projection_plan(kind);
    for (unsigned int wave = 0; wave < plan.wave_count; ++wave) {
        #pragma HLS loop_tripcount min=1 max=22
        const auto mask = quantized_projection_active_cu_mask(plan, wave);
        for (unsigned int packet = 0; packet < qbd_policy_t::packets_per_wave; ++packet) {
            #pragma HLS pipeline II=1
            for (unsigned int cu = 0; cu < 4; ++cu) {
                #pragma HLS unroll
                if (mask & (1u << cu)) {
                    const auto word = results[cu].read();
#ifdef QUANTIZED_ALIGNMENT_W8
                    const unsigned int row = word.range(535, 528).to_uint();
#else
                    const unsigned int row = word.range(407, 400).to_uint();
#endif
                    if (row < rows && row < QBD_ROWS)
                        qbd_policy_t::store_output(destination, word, rows, plan.output_dim, scales[row], weight_scale);
                }
            }
        }
    }
}

inline void quantized_batch_projection_dataflow(
    const quantized_layer_task_t& task, quantized_projection_t kind,
    const qbd_hidden_t& hidden, const qbd_wide_t& wide, bool source_is_wide,
    qbd_wide_t& destination, const quant_scale_t scales[QBD_ROWS],
    const quant_inverse_scale_t inverses[QBD_ROWS], quant_scale_t weight_scale,
    qbd_policy_t::streams_t& streams, const qbd_policy_t::memories_t& memories) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=scales complete
    #pragma HLS array_partition variable=inverses complete
    hls::stream<qbd_weight_t> weights[4][qbd_policy_t::weight_ports];
    hls::stream<qbd_policy_t::output_word_t> results[4];
    #pragma HLS array_partition variable=weights complete dim=0
    #pragma HLS array_partition variable=results complete
    #pragma HLS stream variable=weights depth=QUANTIZED_WEIGHT_FIFO_DEPTH
    #pragma HLS stream variable=results depth=QUANTIZED_RESULT_FIFO_DEPTH
    #pragma HLS bind_storage variable=weights type=fifo impl=bram
    #pragma HLS bind_storage variable=results type=fifo impl=bram
    const auto plan = get_quantized_projection_plan(kind);
    #pragma HLS dataflow
    load_quantized_projection_weight_range<qbd_policy_t>(weights, memories, kind, task.layer, 0, plan.wave_count);
    drive_quantized_batch_projection(streams, weights, hidden, wide, source_is_wide, task, kind, inverses);
    collect_quantized_projection_wave_range<qbd_policy_t>(streams, results, kind, 0, plan.wave_count);
    commit_quantized_batch_projection(destination, results, kind, task.query_tokens, scales, weight_scale);
}

inline void run_quantized_batch_projection(
    const quantized_layer_task_t& task, quantized_projection_t kind,
    const qbd_hidden_t& hidden, const qbd_wide_t& wide, bool source_is_wide,
    qbd_wide_t& destination, quant_scale_t weight_scale,
    qbd_policy_t::streams_t& streams, const qbd_policy_t::memories_t& memories) {
    #pragma HLS inline off
    quant_scale_t scales[QBD_ROWS];
    quant_inverse_scale_t inverses[QBD_ROWS];
    #pragma HLS array_partition variable=scales complete
    #pragma HLS array_partition variable=inverses complete
    const auto plan = get_quantized_projection_plan(kind);
    quantized_batch_projection_scales(hidden, wide, source_is_wide,
        task.query_tokens, plan.input_dim, scales, inverses);
    quantized_batch_projection_dataflow(task, kind, hidden, wide, source_is_wide,
        destination, scales, inverses, weight_scale, streams, memories);
}

#endif
