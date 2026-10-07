#ifndef LLM_FPGA_QUANTIZED_LAYER_INTEGRATED_DECODE_HPP
#define LLM_FPGA_QUANTIZED_LAYER_INTEGRATED_DECODE_HPP

#include "quantized_batch_decode_task.hpp"

// The full-layer controller keeps its legacy ABI by default.  This switch
// only changes the internal D1 dispatch; Prefill remains on the layer-block
// runtime selected by the top-level precision.
#ifndef QUANTIZED_LAYER_INTEGRATED_DECODE
#define QUANTIZED_LAYER_INTEGRATED_DECODE 0
#endif
static_assert(QUANTIZED_LAYER_INTEGRATED_DECODE == 0 ||
                  QUANTIZED_LAYER_INTEGRATED_DECODE == 1,
              "Integrated Decode must be explicitly enabled or disabled");

// Integrated Decode is a composed candidate, not a partial alias.  Reject a
// build that would silently route D1 through the legacy batch projection
// implementation or fused SiLU path while the feature is selected.
#if QUANTIZED_LAYER_INTEGRATED_DECODE
#if !defined(QUANTIZED_BLOCK_PIPELINE) || !QUANTIZED_BLOCK_PIPELINE
#error "Integrated Decode requires QUANTIZED_BLOCK_PIPELINE=1 for Prefill"
#endif
#if !defined(QUANTIZED_ASYNC_COMPUTE) || !QUANTIZED_ASYNC_COMPUTE
#error "Integrated Decode requires QUANTIZED_ASYNC_COMPUTE=1"
#endif
#if !defined(QUANTIZED_DECODE_ROWS_MERGED) || !QUANTIZED_DECODE_ROWS_MERGED
#error "Integrated Decode requires QUANTIZED_DECODE_ROWS_MERGED=1"
#endif
#if !defined(QUANTIZED_DECODE_FFN_OVERLAP) || !QUANTIZED_DECODE_FFN_OVERLAP
#error "Integrated Decode requires QUANTIZED_DECODE_FFN_OVERLAP=1"
#endif
#if !defined(QUANTIZED_DECODE_SCALE_PREFETCH) || \
    !QUANTIZED_DECODE_SCALE_PREFETCH
#error "Integrated Decode requires QUANTIZED_DECODE_SCALE_PREFETCH=1"
#endif
#endif

// The existing production top has no descriptor pointer.  Map its scalar D1
// request to one qbd slot without changing hidden/KV ownership or addresses.
// All offsets are relative to the caller's existing buffers; the runtime adds
// the layer/position component to each KV base exactly as the legacy path did.
inline quantized_batch_decode_entry_t
make_quantized_layer_integrated_decode_entry(
    unsigned int sequence_id,
    unsigned int position,
    unsigned int kv_context_length,
    unsigned int hidden_input_offset = 0,
    unsigned int hidden_output_offset = 0,
    unsigned int key_cache_offset = 0,
    unsigned int value_cache_offset = 0) {
    #pragma HLS inline
    quantized_batch_decode_entry_t entry{};
    entry.sequence_id = sequence_id;
    entry.position = position;
    entry.kv_context_length = kv_context_length;
    entry.hidden_input_offset = hidden_input_offset;
    entry.hidden_output_offset = hidden_output_offset;
    entry.key_cache_offset = key_cache_offset;
    entry.value_cache_offset = value_cache_offset;
    return entry;
}

inline quantized_batch_decode_word_t
make_quantized_layer_integrated_decode_descriptor(
    unsigned int position, unsigned int kv_context_length) {
    #pragma HLS inline
    return pack_quantized_batch_decode_entry(
        make_quantized_layer_integrated_decode_entry(
            0, position, kv_context_length));
}

// These are the capacities advertised by the original precision-specific
// AXI ports.  They let the qbd validator retain its complete-sequence safety
// checks while preserving the existing Host P66->D1 allocations and offsets.
inline unsigned int quantized_layer_integrated_decode_hidden_words() {
    #pragma HLS inline
    return QUANTIZED_BATCH_DECODE_HIDDEN_WORDS;
}

inline unsigned int quantized_layer_integrated_decode_kv_words() {
    #pragma HLS inline
    return QUANTIZED_BATCH_DECODE_FULL_KV_WORDS;
}

// Keep host/C task-count calls tied to the qdr B1 grouping used by
// quantized_batch_decode_runtime.hpp.  The full qdr implementation remains in
// quantized_decode_rows_projection.hpp; this lightweight count avoids pulling
// that dataflow header into the legacy layer schedule include graph.
inline unsigned int quantized_layer_integrated_decode_rows() {
    #pragma HLS inline
#ifdef QUANTIZED_ALIGNMENT_W8
    return 4;
#else
    return 8;
#endif
}

inline unsigned int quantized_layer_integrated_decode_projection_tasks_for_cu(
    unsigned int cu) {
    #pragma HLS inline off
    if (cu >= QUANTIZED_LAYER_COMPUTE_CUS) return 0;
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
    const unsigned int stripes = quantized_layer_integrated_decode_rows();
    unsigned int count = 0;
    for (unsigned int kind = 0; kind < QUANTIZED_PROJECTION_COUNT; ++kind) {
        const quantized_projection_plan_t plan = get_quantized_projection_plan(
            static_cast<quantized_projection_t>(kind));
        for (unsigned int first = 0; first < plan.wave_count;
             first += stripes) {
            #pragma HLS loop_tripcount min=1 max=6
            bool active = false;
            for (unsigned int stripe = 0; stripe < stripes; ++stripe) {
                #pragma HLS unroll
                if (first + stripe < plan.wave_count) {
                    active = active ||
                        ((quantized_projection_active_cu_mask(
                            plan, first + stripe) & (1u << cu)) != 0);
                }
            }
            count += active;
        }
    }
    return count;
#else
    return quantized_projection_tasks_for_cu(cu, 1);
#endif
}

#endif
