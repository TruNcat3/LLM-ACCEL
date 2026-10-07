#ifndef LLM_FPGA_QUANTIZED_DECODE_ROWS_PROJECTION_HPP
#define LLM_FPGA_QUANTIZED_DECODE_ROWS_PROJECTION_HPP

// Candidate B1 projection schedule. It retains the existing per-CU HBM
// stripes and groups successive 128-column stripes of the SAME CU. The
// adapter scatters output packets back into that global column order.
#include "quantized_batch_projection.hpp"

constexpr unsigned int QBD_DR_STRIPES = QBD_ROWS;
constexpr unsigned int QBD_DR_TASK_BIT = 126;

inline unsigned int quantized_dr_stripes_for_cu(
    const quantized_projection_plan_t& plan,
    unsigned int first_wave, unsigned int cu) {
    #pragma HLS inline
    unsigned int count = 0;
    for (unsigned int stripe = 0; stripe < QBD_DR_STRIPES; ++stripe) {
        #pragma HLS unroll
        count += (quantized_projection_active_cu_mask(plan, first_wave + stripe) &
                  (1u << cu)) != 0;
    }
    return count;
}

inline unsigned int quantized_dr_projection_tasks_for_cu(unsigned int cu) {
    #pragma HLS inline off
    if (cu >= QUANTIZED_LAYER_COMPUTE_CUS) return 0;
    unsigned int count = 0;
    for (unsigned int kind = 0; kind < QUANTIZED_PROJECTION_COUNT; ++kind) {
        const auto plan = get_quantized_projection_plan(
            static_cast<quantized_projection_t>(kind));
        for (unsigned int wave = 0; wave < plan.wave_count;
             wave += QBD_DR_STRIPES) {
            #pragma HLS loop_tripcount min=1 max=6
            count += quantized_dr_stripes_for_cu(plan, wave, cu) != 0;
        }
    }
    return count;
}

inline void load_quantized_dr_weights(
    hls::stream<qbd_weight_t> weights[4][qbd_policy_t::weight_ports],
    const qbd_policy_t::memories_t& memories,
    quantized_projection_t kind, unsigned int layer) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=weights complete dim=0
    const auto plan = get_quantized_projection_plan(kind);
    for (unsigned int first = 0; first < plan.wave_count; first += QBD_DR_STRIPES) {
        #pragma HLS loop_tripcount min=1 max=6
        const unsigned int stripe_count = plan.wave_count - first < QBD_DR_STRIPES ?
            plan.wave_count - first : QBD_DR_STRIPES;
        if (stripe_count == 1) {
            // No transpose is needed for a single stripe. Keep the existing
            // contiguous K loop so HLS infers burst reads, including K/V and
            // a one-stripe tail. The generic K-major interleave below is not
            // recognized as a burst even when its runtime stripe count is 1.
            load_quantized_projection_weight_range<qbd_policy_t>(
                weights, memories, kind, layer, first, first + 1);
            continue;
        }
        unsigned int k = 0;
        unsigned int stripe = 0;
        for (unsigned int beat = 0; beat < plan.input_dim * stripe_count; ++beat) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=512 max=88064
            const unsigned int old_wave = first + stripe;
            const unsigned int valid = quantized_projection_active_cu_mask(plan, old_wave);
            const std::size_t offset =
                static_cast<std::size_t>(layer) * quantized_layer_weight_words_per_shard() +
                plan.shard_word_offset + static_cast<std::size_t>(old_wave) * plan.input_dim + k;
            for (unsigned int cu = 0; cu < 4; ++cu) {
                #pragma HLS unroll
                for (unsigned int port = 0; port < qbd_policy_t::weight_ports; ++port) {
                    #pragma HLS unroll
                    // Inactive stripes cause neither an HBM read nor an
                    // external stream transfer. Padding happens inside CU.
                    if (valid & (1u << cu)) weights[cu][port].write(
                        qbd_policy_t::read_weight(memories, cu, port, offset));
                }
            }
            if (stripe + 1 == stripe_count) { stripe = 0; ++k; }
            else ++stripe;
        }
    }
}

inline void drive_quantized_dr_projection(
    qbd_policy_t::streams_t& streams,
    hls::stream<qbd_weight_t> weights[4][qbd_policy_t::weight_ports],
    const qbd_hidden_t& hidden, const qbd_wide_t& wide, bool source_is_wide,
    const quantized_layer_task_t& task, quantized_projection_t kind,
    quant_inverse_scale_t inverse) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=weights complete dim=0
    #pragma HLS array_partition variable=hidden.block complete dim=1
    #pragma HLS array_partition variable=wide.block complete dim=1
    const auto plan = get_quantized_projection_plan(kind);
    for (unsigned int first = 0; first < plan.wave_count; first += QBD_DR_STRIPES) {
        #pragma HLS loop_tripcount min=1 max=6
        const unsigned int active = quantized_projection_active_cu_mask(plan, first);
        for (unsigned int cu = 0; cu < 4; ++cu) {
            #pragma HLS unroll
            if (active & (1u << cu)) {
                const auto dispatch = make_quantized_projection_dispatch(kind, first, cu, false, task.layer);
                auto matrix = make_quantized_projection_task(task, dispatch, quant_scale_t(1), quant_scale_t(1));
                // Explicit DR metadata: valid_tokens is a stripe count ONLY
                // when bit 126 is set. Batch/P/attention retain row semantics.
                matrix.valid_tokens = quantized_dr_stripes_for_cu(plan, first, cu);
                auto packed = pack_mm_stream_quantized_task(matrix);
                packed[QBD_DR_TASK_BIT] = 1;
                qbd_policy_t::write_task(streams, cu, packed);
            }
        }
        const unsigned int stripe_count = plan.wave_count - first < QBD_DR_STRIPES ?
            plan.wave_count - first : QBD_DR_STRIPES;
        unsigned int k = 0;
        unsigned int stripe = 0;
        for (unsigned int beat = 0; beat < plan.input_dim * stripe_count; ++beat) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=512 max=88064
            if (stripe == 0) {
                const mm_input_block_t word = source_is_wide ? wide.block[0][k / 16] : hidden.block[0][k / 16];
                const fm_t value = unpack_mm_input_block_lane(word, k % 16);
                qbd_policy_t::activation_word_t activation = 0;
#ifdef QUANTIZED_ALIGNMENT_W8
                activation.range(7, 0) = quantize_symmetric_int8(value, inverse);
#else
                activation.range(3, 0) = quantize_symmetric_int4(value, inverse);
#endif
                for (unsigned int cu = 0; cu < 4; ++cu) {
                    #pragma HLS unroll
                    if (active & (1u << cu)) qbd_policy_t::write_activation(streams, cu, activation);
                }
            }
            const unsigned int valid = quantized_projection_active_cu_mask(plan, first + stripe);
            for (unsigned int cu = 0; cu < 4; ++cu) {
                #pragma HLS unroll
                if (valid & (1u << cu)) {
                    for (unsigned int port = 0; port < qbd_policy_t::weight_ports; ++port) {
                        #pragma HLS unroll
                        qbd_policy_t::write_weight(streams, cu, port, weights[cu][port].read());
                    }
                }
            }
            if (stripe + 1 == stripe_count) { stripe = 0; ++k; }
            else ++stripe;
        }
    }
}

inline void collect_quantized_dr_projection(
    qbd_policy_t::streams_t& streams,
    hls::stream<qbd_policy_t::output_word_t> results[4],
    quantized_projection_t kind) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=results complete
    const auto plan = get_quantized_projection_plan(kind);
    for (unsigned int first = 0; first < plan.wave_count; first += QBD_DR_STRIPES) {
        #pragma HLS loop_tripcount min=1 max=6
        const unsigned int active = quantized_projection_active_cu_mask(plan, first);
        for (unsigned int packet = 0; packet < qbd_policy_t::packets_per_wave; ++packet) {
            #pragma HLS pipeline II=1
            for (unsigned int cu = 0; cu < 4; ++cu) {
                #pragma HLS unroll
                if (active & (1u << cu)) results[cu].write(qbd_policy_t::read_output(streams, cu));
            }
        }
    }
}

inline void commit_quantized_dr_projection(
    qbd_wide_t& destination, hls::stream<qbd_policy_t::output_word_t> results[4],
    quantized_projection_t kind, quant_scale_t activation_scale, quant_scale_t weight_scale) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=results complete
    #pragma HLS array_partition variable=destination.block complete dim=1
    const auto plan = get_quantized_projection_plan(kind);
    for (unsigned int first = 0; first < plan.wave_count; first += QBD_DR_STRIPES) {
        #pragma HLS loop_tripcount min=1 max=6
        const unsigned int active = quantized_projection_active_cu_mask(plan, first);
        for (unsigned int packet = 0; packet < qbd_policy_t::packets_per_wave; ++packet) {
            #pragma HLS pipeline II=1
            for (unsigned int cu = 0; cu < 4; ++cu) {
                #pragma HLS unroll
                if (active & (1u << cu)) {
                    const auto word = results[cu].read();
                    // Packet coordinates were scattered by the CU adapter;
                    // the existing block store filters by output_dim.
                    qbd_policy_t::store_output(destination, word, 1, plan.output_dim,
                                               activation_scale, weight_scale);
                }
            }
        }
    }
}

inline void quantized_dr_projection_dataflow(
    const quantized_layer_task_t& task, quantized_projection_t kind,
    const qbd_hidden_t& hidden, const qbd_wide_t& wide, bool source_is_wide,
    qbd_wide_t& destination, quant_scale_t scale, quant_inverse_scale_t inverse,
    quant_scale_t weight_scale, qbd_policy_t::streams_t& streams,
    const qbd_policy_t::memories_t& memories) {
    #pragma HLS inline off
    hls::stream<qbd_weight_t> weights[4][qbd_policy_t::weight_ports];
    hls::stream<qbd_policy_t::output_word_t> results[4];
    #pragma HLS array_partition variable=weights complete dim=0
    #pragma HLS array_partition variable=results complete
    #pragma HLS stream variable=weights depth=QUANTIZED_WEIGHT_FIFO_DEPTH
    #pragma HLS stream variable=results depth=QUANTIZED_RESULT_FIFO_DEPTH
    #pragma HLS bind_storage variable=weights type=fifo impl=bram
    #pragma HLS bind_storage variable=results type=fifo impl=bram
    #pragma HLS dataflow
    load_quantized_dr_weights(weights, memories, kind, task.layer);
    drive_quantized_dr_projection(streams, weights, hidden, wide, source_is_wide, task, kind, inverse);
    collect_quantized_dr_projection(streams, results, kind);
    commit_quantized_dr_projection(destination, results, kind, scale, weight_scale);
}

inline void run_quantized_dr_projection(
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
    quantized_batch_projection_scales(hidden, wide, source_is_wide, 1,
                                      plan.input_dim, scales, inverses);
    quantized_dr_projection_dataflow(task, kind, hidden, wide, source_is_wide,
        destination, scales[0], inverses[0], weight_scale, streams, memories);
}

#endif
