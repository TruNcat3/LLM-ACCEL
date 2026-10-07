#ifndef LLM_FPGA_QUANTIZED_BLOCK_PIPELINE_HPP
#define LLM_FPGA_QUANTIZED_BLOCK_PIPELINE_HPP

#include "quantized_vector_engine.hpp"
#include "quantized_projection_dataflow.hpp"

#ifndef QUANTIZED_PREFILL_FFN_OVERLAP
#define QUANTIZED_PREFILL_FFN_OVERLAP 0
#endif

#if QUANTIZED_PREFILL_FFN_OVERLAP && \
    (!defined(QUANTIZED_BLOCK_PIPELINE) || !QUANTIZED_BLOCK_PIPELINE || \
     !defined(QUANTIZED_ASYNC_COMPUTE) || !QUANTIZED_ASYNC_COMPUTE)
#error "QUANTIZED_PREFILL_FFN_OVERLAP requires QUANTIZED_BLOCK_PIPELINE and QUANTIZED_ASYNC_COMPUTE"
#endif

// Prefill SiLU/Up overlap is deliberately narrower than the existing block
// lookahead.  Keep the switch inert unless both the bounded controller and
// its asynchronous compute workers are selected.
#if QUANTIZED_PREFILL_FFN_OVERLAP && \
    defined(QUANTIZED_BLOCK_PIPELINE) && QUANTIZED_BLOCK_PIPELINE && \
    defined(QUANTIZED_ASYNC_COMPUTE) && QUANTIZED_ASYNC_COMPUTE
#define LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED 1
#else
#define LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED 0
#endif

// A bounded first inter-block overlap: RMSNorm for block b+1 runs beside the
// Down projection of b. Commands and results keep the existing ordered ABI.
// Only this process writes task streams, and only the collector reads results.
// The CU must enable its independent matrix/vector workers to realize overlap.
template <typename Policy, typename Hidden, typename Wide>
void drive_quantized_projection_with_lookahead(
    typename Policy::streams_t& streams,
    hls::stream<typename Policy::weight_word_t> weights[4][Policy::weight_ports],
    const Hidden& hidden_source, const Wide& wide_source, bool source_is_wide,
    const quantized_layer_task_t& task, quantized_projection_t kind,
    bool final_physical_block, quant_inverse_scale_t inverse_scale,
    quant_scale_t activation_scale, quant_scale_t weight_scale,
    bool enable_lookahead, const quantized_layer_task_t& next_task
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
    , bool enable_silu
#endif
    ) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=weights complete dim=0
    // A single task writer is important here: vector and matrix commands
    // share each CU's task FIFO, so SiLU must precede the Up matrix waves.
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
    if (enable_silu)
        emit_quantized_vector_tasks<Policy>(streams, task,
            MM_STREAM_QUANTIZED_MODE_SILU_ONLY, INTERMEDIATE_SIZE);
#endif
    if (enable_lookahead)
        emit_quantized_vector_tasks<Policy>(streams, next_task,
            MM_STREAM_QUANTIZED_MODE_RMSNORM, HIDDEN_SIZE);
    const auto plan = get_quantized_projection_plan(kind);
    drive_quantized_projection_wave_range_banked<Policy>(streams, weights,
        hidden_source, wide_source, source_is_wide, task, kind,
        final_physical_block, inverse_scale, activation_scale, weight_scale,
        0, plan.wave_count);
}

template <typename Policy, typename Hidden, typename Wide = Hidden>
void emit_quantized_lookahead_operands(
    typename Policy::streams_t& streams, bool enabled,
    const quantized_layer_task_t& next_task,
    const Hidden& next_source, const Hidden& norm_weight
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
    , const quantized_layer_task_t& silu_task,
    const Wide& silu_source, bool enable_silu
#endif
    ) {
    #pragma HLS inline off
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
    #pragma HLS array_partition variable=silu_source.block complete dim=1
    if (enable_silu)
        emit_quantized_vector_operands<Policy, false>(streams, silu_source,
            silu_source, silu_task, MM_STREAM_QUANTIZED_MODE_SILU_ONLY,
            INTERMEDIATE_SIZE);
#endif
    if (enabled)
        emit_quantized_vector_operands<Policy, false>(streams, next_source, norm_weight,
            next_task, MM_STREAM_QUANTIZED_MODE_RMSNORM, HIDDEN_SIZE);
}

template <typename Policy, typename Hidden, typename Wide = Hidden>
void collect_quantized_projection_with_lookahead(
    typename Policy::streams_t& streams,
    hls::stream<typename Policy::output_word_t> results[4],
    quantized_projection_t kind, bool enabled,
    const quantized_layer_task_t& next_task, Hidden& next_normalized
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
    , const quantized_layer_task_t& silu_task,
    Wide& activated_gate, bool enable_silu
#endif
    ) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=results complete dim=1
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
    #pragma HLS array_partition variable=activated_gate.block complete dim=1
    if (enable_silu)
        collect_quantized_vector_results<Policy>(activated_gate, streams,
            silu_task.query_tokens, INTERMEDIATE_SIZE,
            MM_STREAM_QUANTIZED_MODE_SILU_ONLY);
#endif
    if (enabled)
        collect_quantized_vector_results<Policy>(next_normalized, streams,
            next_task.query_tokens, HIDDEN_SIZE, MM_STREAM_QUANTIZED_MODE_RMSNORM);
    const auto plan = get_quantized_projection_plan(kind);
    collect_quantized_projection_wave_range<Policy>(
        streams, results, kind, 0, plan.wave_count);
}

// Used at ONE projection call site for all seven projection kinds, preventing
// a second physical weight loader / issue / result-commit engine for Down.
template <typename Policy, typename Hidden, typename Wide>
void run_quantized_projection_with_lookahead(
    const quantized_layer_task_t& task, quantized_projection_t kind,
    bool final_physical_block, const Hidden& hidden_source, const Wide& wide_source,
    bool source_is_wide, Wide& scratch, quant_scale_t activation_scale,
    quant_inverse_scale_t inverse_scale, quant_scale_t weight_scale,
    typename Policy::streams_t& streams, const typename Policy::memories_t& memories,
    bool enable_lookahead, const quantized_layer_task_t& next_task,
    const Hidden& next_source, const Hidden& norm_weight, Hidden& next_normalized
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
    , bool enable_silu, const Wide& silu_source, Wide& activated_gate
#endif
    ) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=hidden_source.block complete dim=1
    #pragma HLS array_partition variable=wide_source.block complete dim=1
    #pragma HLS array_partition variable=scratch.block complete dim=1
    #pragma HLS array_partition variable=next_source.block complete dim=1
    #pragma HLS array_partition variable=norm_weight.block complete dim=1
    #pragma HLS array_partition variable=next_normalized.block complete dim=1
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
    #pragma HLS array_partition variable=silu_source.block complete dim=1
    #pragma HLS array_partition variable=activated_gate.block complete dim=1
#endif
    hls::stream<typename Policy::weight_word_t> weights[4][Policy::weight_ports];
    hls::stream<typename Policy::output_word_t> results[4];
    #pragma HLS array_partition variable=weights complete dim=0
    #pragma HLS array_partition variable=results complete dim=1
    #pragma HLS stream variable=weights depth=QUANTIZED_WEIGHT_FIFO_DEPTH
    #pragma HLS stream variable=results depth=QUANTIZED_RESULT_FIFO_DEPTH
    #pragma HLS bind_storage variable=weights type=fifo impl=bram
    #pragma HLS bind_storage variable=results type=fifo impl=bram
    const auto plan = get_quantized_projection_plan(kind);
    #pragma HLS dataflow
    load_quantized_projection_weight_range<Policy>(
        weights, memories, kind, task.layer, 0, plan.wave_count);
    drive_quantized_projection_with_lookahead<Policy>(streams, weights,
        hidden_source, wide_source, source_is_wide, task, kind,
        final_physical_block, inverse_scale, activation_scale, weight_scale,
        enable_lookahead, next_task
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
        , enable_silu
#endif
        );
    emit_quantized_lookahead_operands<Policy>(streams, enable_lookahead,
        next_task, next_source, norm_weight
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
        , task, silu_source, enable_silu
#endif
        );
    collect_quantized_projection_with_lookahead<Policy>(streams, results,
        kind, enable_lookahead, next_task, next_normalized
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
        , task, activated_gate, enable_silu
#endif
        );
    commit_quantized_projection_wave_range<Policy>(scratch, results, kind,
        task.query_tokens, activation_scale, weight_scale, 0, plan.wave_count);
}

#endif
