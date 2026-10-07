#ifndef LLM_FPGA_QUANTIZED_DECODE_FFN_PIPELINE_HPP
#define LLM_FPGA_QUANTIZED_DECODE_FFN_PIPELINE_HPP

#include "quantized_batch_projection.hpp"
#include "quantized_vector_engine.hpp"
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
#include "quantized_decode_rows_projection.hpp"
#endif

// Gate and Up use the same normalized input and have no dependency on one
// another. After Gate completes, its SiLU can use the existing vector workers
// while Up uses the matrix workers. One actor owns task issue; one actor owns
// the ordered output stream. The gate handoff below keeps the two dataflow
// readers from aliasing the caller's gate_product storage.
inline void load_quantized_decode_ffn_weights(
    hls::stream<qbd_weight_t> weights[4][qbd_policy_t::weight_ports],
    const qbd_policy_t::memories_t& memories,
    quantized_projection_t kind, unsigned int layer, bool decode_rows) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=weights complete dim=0
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
    if (decode_rows) {
        load_quantized_dr_weights(weights, memories, kind, layer);
        return;
    }
#endif
    const auto plan = get_quantized_projection_plan(kind);
    load_quantized_projection_weight_range<qbd_policy_t>(
        weights, memories, kind, layer, 0, plan.wave_count);
}

inline void drive_quantized_decode_ffn_projection(
    qbd_policy_t::streams_t& streams,
    hls::stream<qbd_weight_t> weights[4][qbd_policy_t::weight_ports],
    const qbd_hidden_t& hidden, const qbd_wide_t& wide, bool source_is_wide,
    const quantized_layer_task_t& task, quantized_projection_t kind,
    const quant_inverse_scale_t inverses[QBD_ROWS],
    bool decode_rows, bool overlap_silu) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=weights complete dim=0
    #pragma HLS array_partition variable=inverses complete
    // Vector first in the shared result order. Operands are supplied by the
    // independent actor below, so task issue can continue into the Up waves.
    if (overlap_silu)
        emit_quantized_vector_tasks<qbd_policy_t>(
            streams, task, MM_STREAM_QUANTIZED_MODE_SILU_ONLY, INTERMEDIATE_SIZE);
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
    if (decode_rows) {
        drive_quantized_dr_projection(streams, weights, hidden, wide,
            source_is_wide, task, kind, inverses[0]);
        return;
    }
#endif
    drive_quantized_batch_projection(streams, weights, hidden, wide,
        source_is_wide, task, kind, inverses);
}

inline void emit_quantized_decode_ffn_silu(
    qbd_policy_t::streams_t& streams, const quantized_layer_task_t& task,
    const qbd_wide_t& gate, bool enabled) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gate.block complete dim=1
    if (enabled) {
        // Both operands retain the binary vector transport contract. The
        // SILU_ONLY primitive consumes operand 1 but does not multiply by it.
        emit_quantized_vector_operands<qbd_policy_t, false>(streams, gate, gate,
            task, MM_STREAM_QUANTIZED_MODE_SILU_ONLY, INTERMEDIATE_SIZE);
    }
}

// HLS 2022.2 models two readers of one array as a shared-memory PIPO.  The
// integrated caller intentionally passes gate_product as both the projection
// wide input and the SILU gate.  That alias is legal in C, but the shared PIPO
// path can crash PingpongGen while generating RTL for the depth-2 gate bank.
// Snapshot the gate before entering DATAFLOW so the SILU actor has a distinct
// read-only handoff.  The copy is outside DATAFLOW and therefore does not add
// another reader or change the task/operand ordering inside the overlap.
inline void copy_quantized_decode_ffn_silu_gate(
    qbd_wide_t& destination, const qbd_wide_t& source, unsigned int rows) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=destination.block complete dim=1
    #pragma HLS array_partition variable=source.block complete dim=1
    for (unsigned int row = 0; row < rows; ++row) {
        #pragma HLS loop_tripcount min=1 max=8
        for (unsigned int block = 0; block < QBD_WIDE_BLOCKS; ++block) {
            #pragma HLS pipeline II=1
            destination.block[row][block] = source.block[row][block];
        }
    }
}

inline void collect_quantized_decode_ffn_projection(
    qbd_policy_t::streams_t& streams,
    hls::stream<qbd_policy_t::output_word_t> results[4],
    const quantized_layer_task_t& task, quantized_projection_t kind,
    qbd_wide_t& activated_gate, bool decode_rows, bool overlap_silu) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=results complete
    #pragma HLS array_partition variable=activated_gate.block complete dim=1
    if (overlap_silu)
        collect_quantized_vector_results<qbd_policy_t>(activated_gate, streams,
            task.query_tokens, INTERMEDIATE_SIZE, MM_STREAM_QUANTIZED_MODE_SILU_ONLY);
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
    if (decode_rows) {
        collect_quantized_dr_projection(streams, results, kind);
        return;
    }
#endif
    const auto plan = get_quantized_projection_plan(kind);
    collect_quantized_projection_wave_range<qbd_policy_t>(
        streams, results, kind, 0, plan.wave_count);
}

inline void commit_quantized_decode_ffn_projection(
    qbd_wide_t& destination,
    hls::stream<qbd_policy_t::output_word_t> results[4],
    quantized_projection_t kind, unsigned int rows,
    const quant_scale_t scales[QBD_ROWS], quant_scale_t weight_scale,
    bool decode_rows) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=results complete
    #pragma HLS array_partition variable=scales complete
    #pragma HLS array_partition variable=destination.block complete dim=1
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
    if (decode_rows) {
        commit_quantized_dr_projection(destination, results, kind,
            scales[0], weight_scale);
        return;
    }
#endif
    commit_quantized_batch_projection(destination, results, kind, rows,
        scales, weight_scale);
}

#if QUANTIZED_DECODE_SCALE_PREFETCH
using quantized_decode_ffn_scales_word_t = ap_uint<QBD_ROWS * quant_scale_t::width>;

// One process owns both sequential reads of the projection input. In the
// caller, wide and gate alias gate_product; exposing the scan as a separate
// DATAFLOW process would require three readers of that same buffer.
inline void prepare_and_drive_quantized_decode_ffn_projection(
    qbd_policy_t::streams_t& streams,
    hls::stream<qbd_weight_t> weights[4][qbd_policy_t::weight_ports],
    const qbd_hidden_t& hidden, const qbd_wide_t& wide, bool source_is_wide,
    const quantized_layer_task_t& task, quantized_projection_t kind,
    hls::stream<quantized_decode_ffn_scales_word_t>& scale_tokens,
    bool decode_rows, bool overlap_silu) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=weights complete dim=0
    #pragma HLS array_partition variable=hidden.block complete dim=1
    #pragma HLS array_partition variable=wide.block complete dim=1
    quant_scale_t scales[QBD_ROWS];
    quant_inverse_scale_t inverses[QBD_ROWS];
    #pragma HLS array_partition variable=scales complete
    #pragma HLS array_partition variable=inverses complete
    const auto plan = get_quantized_projection_plan(kind);
    quantized_batch_projection_scales(hidden, wide, source_is_wide,
        task.query_tokens, plan.input_dim, scales, inverses);
    quantized_decode_ffn_scales_word_t word = 0;
    for (unsigned int row = 0; row < QBD_ROWS; ++row) {
        #pragma HLS unroll
        word.range((row + 1) * quant_scale_t::width - 1, row * quant_scale_t::width) =
            scales[row].range(quant_scale_t::width - 1, 0);
    }
    // Publish before issuing any task or operand. An array/PIPO published at
    // process completion could block commit while the finite result FIFOs
    // backpressure this driver's remaining waves.
    scale_tokens.write(word);
    drive_quantized_decode_ffn_projection(streams, weights, hidden, wide,
        source_is_wide, task, kind, inverses, decode_rows, overlap_silu);
}

inline void commit_quantized_decode_ffn_scaled_projection(
    qbd_wide_t& destination,
    hls::stream<qbd_policy_t::output_word_t> results[4],
    quantized_projection_t kind, unsigned int rows,
    hls::stream<quantized_decode_ffn_scales_word_t>& scale_tokens,
    quant_scale_t weight_scale, bool decode_rows) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=results complete
    #pragma HLS array_partition variable=destination.block complete dim=1
    quant_scale_t scales[QBD_ROWS];
    #pragma HLS array_partition variable=scales complete
    const quantized_decode_ffn_scales_word_t word = scale_tokens.read();
    for (unsigned int row = 0; row < QBD_ROWS; ++row) {
        #pragma HLS unroll
        scales[row].range(quant_scale_t::width - 1, 0) =
            word.range((row + 1) * quant_scale_t::width - 1, row * quant_scale_t::width);
    }
    commit_quantized_decode_ffn_projection(destination, results, kind, rows,
        scales, weight_scale, decode_rows);
}
#endif

inline void quantized_decode_ffn_projection_dataflow(
    const quantized_layer_task_t& task, quantized_projection_t kind,
    const qbd_hidden_t& hidden, const qbd_wide_t& wide, bool source_is_wide,
    qbd_wide_t& destination,
#if !QUANTIZED_DECODE_SCALE_PREFETCH
    const quant_scale_t scales[QBD_ROWS],
    const quant_inverse_scale_t inverses[QBD_ROWS],
#endif
    quant_scale_t weight_scale,
    qbd_policy_t::streams_t& streams, const qbd_policy_t::memories_t& memories,
    bool decode_rows, const qbd_wide_t& gate, qbd_wide_t& activated_gate) {
    #pragma HLS inline off
#if QUANTIZED_DECODE_SCALE_PREFETCH
    hls::stream<quantized_decode_ffn_scales_word_t> scale_tokens;
    #pragma HLS stream variable=scale_tokens depth=2
    #pragma HLS bind_storage variable=scale_tokens type=fifo impl=lutram
#else
    #pragma HLS array_partition variable=scales complete
    #pragma HLS array_partition variable=inverses complete
#endif
    hls::stream<qbd_weight_t> weights[4][qbd_policy_t::weight_ports];
    hls::stream<qbd_policy_t::output_word_t> results[4];
    #pragma HLS array_partition variable=weights complete dim=0
    #pragma HLS array_partition variable=results complete
    #pragma HLS stream variable=weights depth=QUANTIZED_WEIGHT_FIFO_DEPTH
    #pragma HLS stream variable=results depth=QUANTIZED_RESULT_FIFO_DEPTH
    #pragma HLS bind_storage variable=weights type=fifo impl=bram
    #pragma HLS bind_storage variable=results type=fifo impl=bram
    const bool overlap_silu = kind == QUANTIZED_PROJECTION_UP;
    #pragma HLS dataflow
    load_quantized_decode_ffn_weights(weights, memories, kind, task.layer, decode_rows);
#if QUANTIZED_DECODE_SCALE_PREFETCH
    prepare_and_drive_quantized_decode_ffn_projection(streams, weights, hidden, wide,
        source_is_wide, task, kind, scale_tokens, decode_rows, overlap_silu);
#else
    drive_quantized_decode_ffn_projection(streams, weights, hidden, wide,
        source_is_wide, task, kind, inverses, decode_rows, overlap_silu);
#endif
    emit_quantized_decode_ffn_silu(streams, task, gate, overlap_silu);
    collect_quantized_decode_ffn_projection(streams, results, task, kind,
        activated_gate, decode_rows, overlap_silu);
#if QUANTIZED_DECODE_SCALE_PREFETCH
    commit_quantized_decode_ffn_scaled_projection(destination, results, kind,
        task.query_tokens, scale_tokens, weight_scale, decode_rows);
#else
    commit_quantized_decode_ffn_projection(destination, results, kind,
        task.query_tokens, scales, weight_scale, decode_rows);
#endif
}

// Keep all seven projection kinds at one call site: the Up overlap must not
// duplicate the physical weight loader, matrix array or result commit engine.
inline void run_quantized_decode_ffn_projection(
    const quantized_layer_task_t& task, quantized_projection_t kind,
    const qbd_hidden_t& hidden, const qbd_wide_t& wide, bool source_is_wide,
    qbd_wide_t& destination, quant_scale_t weight_scale,
    qbd_policy_t::streams_t& streams, const qbd_policy_t::memories_t& memories,
    bool decode_rows, const qbd_wide_t& gate, qbd_wide_t& activated_gate) {
    #pragma HLS inline off
#if !QUANTIZED_DECODE_SCALE_PREFETCH
    quant_scale_t scales[QBD_ROWS];
    quant_inverse_scale_t inverses[QBD_ROWS];
    #pragma HLS array_partition variable=scales complete
    #pragma HLS array_partition variable=inverses complete
    const auto plan = get_quantized_projection_plan(kind);
    quantized_batch_projection_scales(hidden, wide, source_is_wide,
        task.query_tokens, plan.input_dim, scales, inverses);
#endif
    const bool overlap_silu = kind == QUANTIZED_PROJECTION_UP;
    qbd_wide_t gate_for_silu;
    #pragma HLS bind_storage variable=gate_for_silu.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=gate_for_silu.block complete dim=1
    if (overlap_silu)
        copy_quantized_decode_ffn_silu_gate(gate_for_silu, gate,
            task.query_tokens);
    // Keep one physical call site. Non-UP tasks disable the SILU reader, so
    // they neither initialize nor consume the snapshot.
    quantized_decode_ffn_projection_dataflow(task, kind, hidden, wide,
        source_is_wide, destination,
#if !QUANTIZED_DECODE_SCALE_PREFETCH
        scales, inverses,
#endif
        weight_scale,
        streams, memories, decode_rows, gate_for_silu, activated_gate);
}

#endif
