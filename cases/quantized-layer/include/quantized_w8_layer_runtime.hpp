#ifndef LLM_FPGA_QUANTIZED_W8_LAYER_RUNTIME_HPP
#define LLM_FPGA_QUANTIZED_W8_LAYER_RUNTIME_HPP

#include "quantized_w8_attention_engine.hpp"
#include "quantized_vector_engine.hpp"
#if QUANTIZED_BLOCK_PIPELINE
#include "quantized_block_pipeline.hpp"
#endif

using quantized_w8_scale_word_t = ap_uint<128>;

constexpr unsigned int QUANTIZED_W8_ROPE_BLOCKS =
    (HEAD_DIM / 2 + QUANTIZED_W8_RESIDENT_LANES_PER_WORD - 1) /
        QUANTIZED_W8_RESIDENT_LANES_PER_WORD;

inline quant_scale_t get_quantized_w8_projection_weight_scale(
    const quantized_w8_scale_word_t* scale_memory,
    unsigned int layer,
    quantized_projection_t projection) {
    #pragma HLS inline
    quant_scale_t scale;
    scale.range(quant_scale_t::width - 1, 0) =
        scale_memory[layer].range(
            (static_cast<unsigned int>(projection) + 1) * 16 - 1,
            static_cast<unsigned int>(projection) * 16);
    return scale == quant_scale_t(0) ? quant_scale_t(1) : scale;
}

inline std::size_t quantized_w8_hidden_word_index(
    unsigned int token,
    unsigned int block) {
    return static_cast<std::size_t>(token) * QUANTIZED_W8_HIDDEN_BLOCKS + block;
}

inline std::size_t quantized_w8_norm_word_index(
    unsigned int layer,
    unsigned int norm,
    unsigned int block) {
    return (static_cast<std::size_t>(layer) * 2 + norm) *
        QUANTIZED_W8_HIDDEN_BLOCKS + block;
}

inline std::size_t quantized_w8_rope_word_index(
    unsigned int position,
    unsigned int component,
    unsigned int block) {
    return (static_cast<std::size_t>(position) * 2 + component) *
        QUANTIZED_W8_ROPE_BLOCKS + block;
}

inline void load_quantized_w8_hidden_block(
    quantized_w8_hidden_buffer_t& destination,
    const mm_input_block_t* source,
    unsigned int source_token_begin,
    unsigned int valid_tokens) {
    #pragma HLS inline
    for (unsigned int token = 0;
         token < QUANTIZED_W8_RESIDENT_TOKEN_ROWS; ++token) {
        for (unsigned int block = 0; block < QUANTIZED_W8_HIDDEN_BLOCKS;
             ++block) {
            #pragma HLS pipeline II=1
            destination.block[token][block] = token < valid_tokens ?
                source[quantized_w8_hidden_word_index(
                    source_token_begin + token, block)] : mm_input_block_t(0);
        }
    }
}

inline void store_quantized_w8_hidden_block(
    mm_input_block_t* destination,
    const quantized_w8_hidden_buffer_t& source,
    unsigned int destination_token_begin,
    unsigned int valid_tokens) {
    #pragma HLS inline off
    for (unsigned int token = 0;
         token < QUANTIZED_W8_RESIDENT_TOKEN_ROWS; ++token) {
        for (unsigned int block = 0; block < QUANTIZED_W8_HIDDEN_BLOCKS;
             ++block) {
            #pragma HLS pipeline II=1
            if (token < valid_tokens) {
                destination[quantized_w8_hidden_word_index(
                    destination_token_begin + token, block)] =
                    source.block[token][block];
            }
        }
    }
}

inline void load_quantized_w8_norm_weight(
    quantized_w8_hidden_buffer_t& destination,
    const mm_input_block_t* norm_memory,
    unsigned int layer,
    unsigned int norm) {
    #pragma HLS inline off
    for (unsigned int block = 0; block < QUANTIZED_W8_HIDDEN_BLOCKS;
         ++block) {
        #pragma HLS pipeline II=1
        destination.block[0][block] = norm_memory[
            quantized_w8_norm_word_index(layer, norm, block)];
    }
}

inline void load_quantized_w8_rope_coefficients(
    fm_t cosine[HEAD_DIM / 2],
    fm_t sine[HEAD_DIM / 2],
    const mm_input_block_t* rope_memory,
    unsigned int position) {
    #pragma HLS inline off
    for (unsigned int element = 0; element < HEAD_DIM / 2; ++element) {
        #pragma HLS pipeline II=1
        const unsigned int block =
            element / QUANTIZED_W8_RESIDENT_LANES_PER_WORD;
        const unsigned int lane =
            element % QUANTIZED_W8_RESIDENT_LANES_PER_WORD;
        cosine[element] = unpack_mm_input_block_lane(
            rope_memory[quantized_w8_rope_word_index(position, 0, block)], lane);
        sine[element] = unpack_mm_input_block_lane(
            rope_memory[quantized_w8_rope_word_index(position, 1, block)], lane);
    }
}

inline void run_quantized_w8_layer_block(
    const quantized_layer_task_t& layer_task,
    const mm_input_block_t* hidden_input,
    mm_input_block_t* hidden_output,
    mm_input_block_t* key_cache,
    mm_input_block_t* value_cache,
    const mm_input_block_t* norm_memory,
    const mm_input_block_t* rope_memory,
    const quantized_w8_scale_word_t* scale_memory,
    quantized_w8_projection_streams_t& streams,
    const quantized_w8_weight_memories_t& weight_memories
#if QUANTIZED_BLOCK_PIPELINE
    , bool prepared, bool has_next, const quantized_layer_task_t& next_task,
    quantized_w8_hidden_buffer_t& next_source,
    quantized_w8_hidden_buffer_t& next_normalized
#endif
    ) {
    // Flatten the block wrapper so all HBM pointers stay top-level visible.
    #pragma HLS inline
    quantized_w8_hidden_buffer_t residual;
    quantized_w8_hidden_buffer_t hidden0;
    quantized_w8_hidden_buffer_t hidden1;
    quantized_w8_hidden_buffer_t norm_weight;
    quantized_w8_kv_buffer_t key;
    quantized_w8_kv_buffer_t value;
    quantized_w8_wide_buffer_t gate_product;
    quantized_w8_wide_buffer_t up;
    quantized_w8_wide_buffer_t projection_scratch;
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
    // Keep the vector reader on an independent Gate handoff bank.  The
    // original Gate bank remains the wide-source argument for the shared Up
    // projection call, avoiding a PingpongGen alias in HLS dataflow.
    quantized_w8_wide_buffer_t gate_for_silu;
#endif
    #pragma HLS bind_storage variable=projection_scratch.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=projection_scratch.block complete dim=1
    #pragma HLS bind_storage variable=residual.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=residual.block complete dim=1
    #pragma HLS bind_storage variable=hidden0.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=hidden0.block complete dim=1
    #pragma HLS bind_storage variable=hidden1.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=hidden1.block complete dim=1
    #pragma HLS bind_storage variable=norm_weight.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=norm_weight.block complete dim=1
    #pragma HLS bind_storage variable=key.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=key.block complete dim=1
    #pragma HLS bind_storage variable=value.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=value.block complete dim=1
    #pragma HLS bind_storage variable=gate_product.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=gate_product.block complete dim=1
    #pragma HLS bind_storage variable=up.block type=ram_2p impl=bram
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
    #pragma HLS bind_storage variable=gate_for_silu.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=gate_for_silu.block complete dim=1
#endif

#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
    const bool prefill_silu_overlap =
        layer_task.op == QUANTIZED_LAYER_OP_PREFILL;
#else
    const bool prefill_silu_overlap = false;
#endif
    #pragma HLS array_partition variable=up.block complete dim=1

    const unsigned int source_token_begin =
        quantized_layer_source_token_begin(layer_task);
#if QUANTIZED_BLOCK_PIPELINE
    if (prepared) {
        copy_quantized_vector_active_rows<quantized_w8_projection_policy>(
            residual, next_source, layer_task.query_tokens, HIDDEN_SIZE);
        copy_quantized_vector_active_rows<quantized_w8_projection_policy>(
            hidden0, next_normalized, layer_task.query_tokens, HIDDEN_SIZE);
    } else {
#endif
    load_quantized_w8_hidden_block(
        residual, hidden_input, source_token_begin, layer_task.query_tokens);
    load_quantized_w8_norm_weight(
        norm_weight, norm_memory, layer_task.layer, 0);
    run_quantized_vector_exchange<quantized_w8_projection_policy>(
        layer_task, MM_STREAM_QUANTIZED_MODE_RMSNORM, HIDDEN_SIZE,
        residual, norm_weight, hidden0, streams);
#if QUANTIZED_BLOCK_PIPELINE
    }
#endif
    quantized_symmetric_scale_t projection_scale =
        quantized_w8_buffer_scale(
            hidden0, layer_task.query_tokens, HIDDEN_SIZE);

    // One call site and fixed bank interfaces preserve one physical engine,
    // as in the resident Fix16 projection-step loop.
    for (unsigned int step = 0; step < QUANTIZED_PROJECTION_COUNT; ++step) {
        #pragma HLS loop_tripcount min=7 max=7
        const auto kind = static_cast<quantized_projection_t>(step);
        if (kind == QUANTIZED_PROJECTION_O) {
            for (unsigned int token = 0;
                 token < QUANTIZED_W8_RESIDENT_TOKEN_ROWS; ++token) {
                if (token < layer_task.query_tokens) {
                    fm_t cosine[HEAD_DIM / 2];
                    fm_t sine[HEAD_DIM / 2];
                    #pragma HLS bind_storage variable=cosine type=ram_2p impl=bram
                    #pragma HLS bind_storage variable=sine type=ram_2p impl=bram
                    load_quantized_w8_rope_coefficients(
                        cosine, sine, rope_memory, layer_task.position + token);
                    apply_quantized_w8_rope(hidden1, key, token, cosine, sine);
                }
            }
            store_quantized_w8_current_kv(
                key_cache, value_cache, key, value, layer_task.layer,
                layer_task.position, layer_task.query_tokens);
            run_quantized_w8_online_attention(
                layer_task, hidden1, key_cache, value_cache, hidden0, streams);
            projection_scale = quantized_w8_buffer_scale(
                hidden0, layer_task.query_tokens, HIDDEN_SIZE);
        } else if (kind == QUANTIZED_PROJECTION_GATE) {
            run_quantized_vector_inplace<quantized_w8_projection_policy>(
                layer_task, MM_STREAM_QUANTIZED_MODE_RESIDUAL_ADD, HIDDEN_SIZE,
                residual, hidden1, streams);
            load_quantized_w8_norm_weight(
                norm_weight, norm_memory, layer_task.layer, 1);
            run_quantized_vector_exchange<quantized_w8_projection_policy>(
                layer_task, MM_STREAM_QUANTIZED_MODE_RMSNORM, HIDDEN_SIZE,
                residual, norm_weight, hidden0, streams);
            projection_scale = quantized_w8_buffer_scale(
                hidden0, layer_task.query_tokens, HIDDEN_SIZE);
        } else if (kind == QUANTIZED_PROJECTION_DOWN) {
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
            if (prefill_silu_overlap) {
                // Up's raw matrix output remains in projection_scratch.  The
                // consumed Gate bank is reused for the MUL_ONLY result.
                run_quantized_vector_exchange<quantized_w8_projection_policy>(
                    layer_task, MM_STREAM_QUANTIZED_MODE_MUL, INTERMEDIATE_SIZE,
                    up, projection_scratch, gate_product, streams);
                projection_scale = quantized_w8_buffer_scale(
                    gate_product, layer_task.query_tokens, INTERMEDIATE_SIZE);
            } else {
#endif
            run_quantized_vector_inplace<quantized_w8_projection_policy>(
                layer_task, MM_STREAM_QUANTIZED_MODE_SILU_MUL, INTERMEDIATE_SIZE,
                gate_product, up, streams);
            projection_scale = quantized_w8_buffer_scale(
                gate_product, layer_task.query_tokens, INTERMEDIATE_SIZE);
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
            }
#endif
#if QUANTIZED_BLOCK_PIPELINE
            if (has_next) {
                // Neither source nor norm is used by the current Down MM.
                // KV/attention ordering is unchanged; only initial RMS moves.
                load_quantized_w8_hidden_block(next_source, hidden_input,
                    quantized_layer_source_token_begin(next_task), next_task.query_tokens);
                load_quantized_w8_norm_weight(
                    norm_weight, norm_memory, next_task.layer, 0);
            }
#endif
        }
        const auto plan = get_quantized_projection_plan(kind);
#if QUANTIZED_BLOCK_PIPELINE
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
        const bool enable_silu = prefill_silu_overlap &&
            kind == QUANTIZED_PROJECTION_UP;
        if (enable_silu)
            copy_quantized_vector_active_rows<quantized_w8_projection_policy>(
                gate_for_silu, gate_product, layer_task.query_tokens,
                INTERMEDIATE_SIZE);
#endif
        run_quantized_projection_with_lookahead<quantized_w8_projection_policy>(
            layer_task, kind, kind == QUANTIZED_PROJECTION_DOWN && layer_task.last_block,
            hidden0, gate_product, kind == QUANTIZED_PROJECTION_DOWN,
            projection_scratch, projection_scale.scale, projection_scale.inverse_scale,
            get_quantized_w8_projection_weight_scale(scale_memory, layer_task.layer, kind),
            streams, weight_memories, kind == QUANTIZED_PROJECTION_DOWN && has_next,
            next_task, next_source, norm_weight, next_normalized
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
            , enable_silu, gate_for_silu, up
#endif
            );
#else
        run_quantized_projection_banked<quantized_w8_projection_policy>(
            layer_task, kind, kind == QUANTIZED_PROJECTION_DOWN && layer_task.last_block,
            hidden0, gate_product, kind == QUANTIZED_PROJECTION_DOWN,
            projection_scratch, projection_scale.scale, projection_scale.inverse_scale,
            get_quantized_w8_projection_weight_scale(scale_memory, layer_task.layer, kind),
            streams, weight_memories);
#endif
        if (kind == QUANTIZED_PROJECTION_K) {
            copy_quantized_projection_result<quantized_w8_projection_policy>(
                key, projection_scratch, layer_task.query_tokens, plan.output_dim);
        } else if (kind == QUANTIZED_PROJECTION_V) {
            copy_quantized_projection_result<quantized_w8_projection_policy>(
                value, projection_scratch, layer_task.query_tokens, plan.output_dim);
        } else if (kind == QUANTIZED_PROJECTION_GATE) {
            copy_quantized_projection_result<quantized_w8_projection_policy>(
                gate_product, projection_scratch, layer_task.query_tokens, plan.output_dim);
        } else if (kind == QUANTIZED_PROJECTION_UP) {
#if LLM_FPGA_QUANTIZED_PREFILL_FFN_OVERLAP_ENABLED
            if (!prefill_silu_overlap)
#endif
            copy_quantized_projection_result<quantized_w8_projection_policy>(
                up, projection_scratch, layer_task.query_tokens, plan.output_dim);
        } else {
            copy_quantized_projection_result<quantized_w8_projection_policy>(
                hidden1, projection_scratch, layer_task.query_tokens, plan.output_dim);
        }
    }
    run_quantized_vector_inplace<quantized_w8_projection_policy>(
        layer_task, MM_STREAM_QUANTIZED_MODE_RESIDUAL_ADD, HIDDEN_SIZE,
        residual, hidden1, streams);
    store_quantized_w8_hidden_block(
        hidden_output, residual, source_token_begin, layer_task.query_tokens);
}

#if QUANTIZED_BLOCK_PIPELINE
// Preserve the single-block interface used by unit fixtures. Lookahead
// ownership belongs to the outer controller, so this overload never enables it.
inline void run_quantized_w8_layer_block(
    const quantized_layer_task_t& task,
    const mm_input_block_t* hidden_input, mm_input_block_t* hidden_output,
    mm_input_block_t* key_cache, mm_input_block_t* value_cache,
    const mm_input_block_t* norm_memory, const mm_input_block_t* rope_memory,
    const quantized_w8_scale_word_t* scale_memory,
    quantized_w8_projection_streams_t& streams,
    const quantized_w8_weight_memories_t& memories) {
    #pragma HLS inline
    quantized_w8_hidden_buffer_t unused_source, unused_normalized;
    run_quantized_w8_layer_block(task, hidden_input, hidden_output,
        key_cache, value_cache, norm_memory, rope_memory, scale_memory,
        streams, memories, false, false, task, unused_source, unused_normalized);
}
#endif

#endif
