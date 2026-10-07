#ifndef LLM_FPGA_QUANTIZED_PROJECTION_DATAFLOW_HPP
#define LLM_FPGA_QUANTIZED_PROJECTION_DATAFLOW_HPP

#include "quantized_dataflow_config.hpp"
#include "quantized_layer_schedule.hpp"
#include <hls_stream.h>

// These four stages implement the same topology as
// run_cc8_projection_wave_range_overlapped. W4/W8 policies adapt packed words,
// memory ports and arithmetic; the schedule is shared.
template <typename Policy>
void load_quantized_projection_weight_range(
    hls::stream<typename Policy::weight_word_t>
        weights[QUANTIZED_LAYER_COMPUTE_CUS][Policy::weight_ports],
    const typename Policy::memories_t& memories,
    quantized_projection_t kind, unsigned int layer,
    unsigned int wave_begin, unsigned int wave_end) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=weights complete dim=0
    const quantized_projection_plan_t plan = get_quantized_projection_plan(kind);
    for (unsigned int wave = wave_begin; wave < wave_end; ++wave) {
        #pragma HLS loop_tripcount min=1 max=22
        const unsigned int mask = quantized_projection_active_cu_mask(plan, wave);
        const auto dispatch = make_quantized_projection_dispatch(
            kind, wave, 0, false, layer);
        for (unsigned int k = 0; k < plan.input_dim; ++k) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=16 max=11008
            for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
                #pragma HLS unroll
                for (unsigned int port = 0; port < Policy::weight_ports; ++port) {
                    #pragma HLS unroll
                    if (mask & (1u << cu))
                        weights[cu][port].write(Policy::read_weight(
                            memories, cu, port,
                            dispatch.shard_weight_word_offset + k));
                }
            }
        }
    }
}

template <typename Policy, typename Source>
void drive_quantized_projection_wave_range(
    typename Policy::streams_t& streams,
    hls::stream<typename Policy::weight_word_t>
        weights[QUANTIZED_LAYER_COMPUTE_CUS][Policy::weight_ports],
    const Source& source, const quantized_layer_task_t& layer_task,
    quantized_projection_t kind, bool final_physical_block,
    quant_scale_t activation_scale, quant_inverse_scale_t inverse_scale,
    quant_scale_t weight_scale, unsigned int wave_begin, unsigned int wave_end) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=weights complete dim=0
    #pragma HLS array_partition variable=source.block complete dim=1
    const quantized_projection_plan_t plan = get_quantized_projection_plan(kind);
    for (unsigned int wave = wave_begin; wave < wave_end; ++wave) {
        #pragma HLS loop_tripcount min=1 max=22
        const unsigned int mask = quantized_projection_active_cu_mask(plan, wave);
        for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
            #pragma HLS unroll
            if (mask & (1u << cu)) {
                const auto dispatch = make_quantized_projection_dispatch(
                    kind, wave, cu, final_physical_block, layer_task.layer);
                Policy::write_task(streams, cu, pack_mm_stream_quantized_task(
                    make_quantized_projection_task(layer_task, dispatch,
                                                    activation_scale, weight_scale)));
            }
        }
        for (unsigned int k = 0; k < plan.input_dim; ++k) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=16 max=11008
            const typename Policy::activation_word_t activation =
                Policy::pack_activation(source, k, layer_task.query_tokens,
                                        inverse_scale);
            for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
                #pragma HLS unroll
                if (mask & (1u << cu)) {
                    Policy::write_activation(streams, cu, activation);
                    for (unsigned int port = 0; port < Policy::weight_ports; ++port) {
                        #pragma HLS unroll
                        Policy::write_weight(streams, cu, port,
                                             weights[cu][port].read());
                    }
                }
            }
        }
    }
}

template <typename Policy>
void collect_quantized_projection_wave_range(
    typename Policy::streams_t& streams,
    hls::stream<typename Policy::output_word_t> results[QUANTIZED_LAYER_COMPUTE_CUS],
    quantized_projection_t kind, unsigned int wave_begin, unsigned int wave_end) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=results complete dim=1
    const quantized_projection_plan_t plan = get_quantized_projection_plan(kind);
    for (unsigned int wave = wave_begin; wave < wave_end; ++wave) {
        #pragma HLS loop_tripcount min=1 max=22
        const unsigned int mask = quantized_projection_active_cu_mask(plan, wave);
        for (unsigned int packet = 0; packet < Policy::packets_per_wave; ++packet) {
            #pragma HLS pipeline II=1
            for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
                #pragma HLS unroll
                if (mask & (1u << cu))
                    results[cu].write(Policy::read_output(streams, cu));
            }
        }
    }
}

template <typename Policy, typename Destination>
void commit_quantized_projection_wave_range(
    Destination& destination,
    hls::stream<typename Policy::output_word_t> results[QUANTIZED_LAYER_COMPUTE_CUS],
    quantized_projection_t kind, unsigned int valid_tokens,
    quant_scale_t activation_scale, quant_scale_t weight_scale,
    unsigned int wave_begin, unsigned int wave_end) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=results complete dim=1
    #pragma HLS array_partition variable=destination.block complete dim=1
    const quantized_projection_plan_t plan = get_quantized_projection_plan(kind);
    for (unsigned int wave = wave_begin; wave < wave_end; ++wave) {
        #pragma HLS loop_tripcount min=1 max=22
        const unsigned int mask = quantized_projection_active_cu_mask(plan, wave);
        for (unsigned int packet = 0; packet < Policy::packets_per_wave; ++packet) {
            #pragma HLS pipeline II=1
            for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
                #pragma HLS unroll
                if (mask & (1u << cu))
                    Policy::store_output(destination, results[cu].read(),
                        valid_tokens, plan.output_dim, activation_scale, weight_scale);
            }
        }
    }
}

template <typename Policy, typename Source, typename Destination>
void run_quantized_projection_wave_range_overlapped(
    const quantized_layer_task_t& layer_task,
    quantized_projection_t kind, bool final_physical_block,
    const Source& source, Destination& destination,
    quant_scale_t activation_scale, quant_inverse_scale_t inverse_scale,
    quant_scale_t weight_scale, typename Policy::streams_t& streams,
    const typename Policy::memories_t& memories,
    unsigned int wave_begin, unsigned int wave_end) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=source.block complete dim=1
    #pragma HLS array_partition variable=destination.block complete dim=1
    hls::stream<typename Policy::weight_word_t>
        weights[QUANTIZED_LAYER_COMPUTE_CUS][Policy::weight_ports];
    #pragma HLS array_partition variable=weights complete dim=0
    #pragma HLS stream variable=weights depth=QUANTIZED_WEIGHT_FIFO_DEPTH
    #pragma HLS bind_storage variable=weights type=fifo impl=bram
    hls::stream<typename Policy::output_word_t> results[QUANTIZED_LAYER_COMPUTE_CUS];
    #pragma HLS array_partition variable=results complete dim=1
    #pragma HLS stream variable=results depth=QUANTIZED_RESULT_FIFO_DEPTH
    #pragma HLS bind_storage variable=results type=fifo impl=bram
    #pragma HLS dataflow
    load_quantized_projection_weight_range<Policy>(
        weights, memories, kind, layer_task.layer, wave_begin, wave_end);
    drive_quantized_projection_wave_range<Policy>(
        streams, weights, source, layer_task, kind, final_physical_block,
        activation_scale, inverse_scale, weight_scale, wave_begin, wave_end);
    collect_quantized_projection_wave_range<Policy>(
        streams, results, kind, wave_begin, wave_end);
    commit_quantized_projection_wave_range<Policy>(
        destination, results, kind, layer_task.query_tokens,
        activation_scale, weight_scale, wave_begin, wave_end);
}

// Like the Fix16 banked driver, select the resident source inside one
// physical process. The seven projections must call this through one site;
// different feature-buffer template dimensions otherwise clone the loader,
// issue/collect/commit processes and their conversion arithmetic.
template <typename Policy, typename Hidden, typename Wide>
void drive_quantized_projection_wave_range_banked(
    typename Policy::streams_t& streams,
    hls::stream<typename Policy::weight_word_t>
        weights[QUANTIZED_LAYER_COMPUTE_CUS][Policy::weight_ports],
    const Hidden& hidden_source, const Wide& wide_source, bool source_is_wide,
    const quantized_layer_task_t& layer_task, quantized_projection_t kind,
    bool final_physical_block, quant_inverse_scale_t inverse_scale,
    quant_scale_t activation_scale, quant_scale_t weight_scale,
    unsigned int wave_begin, unsigned int wave_end) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=weights complete dim=0
    #pragma HLS array_partition variable=hidden_source.block complete dim=1
    #pragma HLS array_partition variable=wide_source.block complete dim=1
    const auto plan = get_quantized_projection_plan(kind);
    for (unsigned int wave = wave_begin; wave < wave_end; ++wave) {
        #pragma HLS loop_tripcount min=1 max=22
        const unsigned int mask = quantized_projection_active_cu_mask(plan, wave);
        for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
            #pragma HLS unroll
            if (mask & (1u << cu)) {
                const auto dispatch = make_quantized_projection_dispatch(
                    kind, wave, cu, final_physical_block, layer_task.layer);
                Policy::write_task(streams, cu, pack_mm_stream_quantized_task(
                    make_quantized_projection_task(layer_task, dispatch,
                                                    activation_scale, weight_scale)));
            }
        }
        for (unsigned int k = 0; k < plan.input_dim; ++k) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=16 max=11008
            const typename Policy::activation_word_t activation = source_is_wide ?
                Policy::pack_activation(wide_source, k, layer_task.query_tokens, inverse_scale) :
                Policy::pack_activation(hidden_source, k, layer_task.query_tokens, inverse_scale);
            for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
                #pragma HLS unroll
                if (mask & (1u << cu)) {
                    Policy::write_activation(streams, cu, activation);
                    for (unsigned int port = 0; port < Policy::weight_ports; ++port) {
                        #pragma HLS unroll
                        Policy::write_weight(streams, cu, port, weights[cu][port].read());
                    }
                }
            }
        }
    }
}

template <typename Policy, typename Hidden, typename Wide>
void run_quantized_projection_banked(
    const quantized_layer_task_t& task, quantized_projection_t kind,
    bool final_physical_block, const Hidden& hidden_source, const Wide& wide_source,
    bool source_is_wide, Wide& scratch, quant_scale_t activation_scale,
    quant_inverse_scale_t inverse_scale, quant_scale_t weight_scale,
    typename Policy::streams_t& streams, const typename Policy::memories_t& memories) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=hidden_source.block complete dim=1
    #pragma HLS array_partition variable=wide_source.block complete dim=1
    #pragma HLS array_partition variable=scratch.block complete dim=1
    const auto plan = get_quantized_projection_plan(kind);
    hls::stream<typename Policy::weight_word_t>
        weights[QUANTIZED_LAYER_COMPUTE_CUS][Policy::weight_ports];
    #pragma HLS array_partition variable=weights complete dim=0
    #pragma HLS stream variable=weights depth=QUANTIZED_WEIGHT_FIFO_DEPTH
    #pragma HLS bind_storage variable=weights type=fifo impl=bram
    hls::stream<typename Policy::output_word_t> results[QUANTIZED_LAYER_COMPUTE_CUS];
    #pragma HLS array_partition variable=results complete dim=1
    #pragma HLS stream variable=results depth=QUANTIZED_RESULT_FIFO_DEPTH
    #pragma HLS bind_storage variable=results type=fifo impl=bram
    #pragma HLS dataflow
    load_quantized_projection_weight_range<Policy>(
        weights, memories, kind, task.layer, 0, plan.wave_count);
    drive_quantized_projection_wave_range_banked<Policy>(streams, weights,
        hidden_source, wide_source, source_is_wide, task, kind, final_physical_block,
        inverse_scale, activation_scale, weight_scale, 0, plan.wave_count);
    collect_quantized_projection_wave_range<Policy>(streams, results, kind, 0, plan.wave_count);
    commit_quantized_projection_wave_range<Policy>(scratch, results, kind,
        task.query_tokens, activation_scale, weight_scale, 0, plan.wave_count);
}

template <typename Policy, typename Destination, typename Source>
void copy_quantized_projection_result(Destination& destination, const Source& scratch,
                                      unsigned int rows, unsigned int elements) {
    #pragma HLS inline
    #pragma HLS array_partition variable=destination.block complete dim=1
    #pragma HLS array_partition variable=scratch.block complete dim=1
    // Commit outside DATAFLOW so a destination that is also a later source
    // never aliases the reader in the same DATAFLOW region.
    for (unsigned int row = 0; row < rows; ++row) {
        #pragma HLS loop_tripcount min=1 max=8
        for (unsigned int block = 0; block < (elements + 15) / 16; ++block) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=1 max=688
            destination.block[row][block] = scratch.block[row][block];
        }
    }
}

#endif
