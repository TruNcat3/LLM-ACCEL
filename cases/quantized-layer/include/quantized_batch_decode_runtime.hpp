#ifndef LLM_FPGA_QUANTIZED_BATCH_DECODE_RUNTIME_HPP
#define LLM_FPGA_QUANTIZED_BATCH_DECODE_RUNTIME_HPP

#include "quantized_batch_decode_task.hpp"
#include "quantized_batch_projection.hpp"
#ifndef QUANTIZED_DECODE_FFN_OVERLAP
#define QUANTIZED_DECODE_FFN_OVERLAP 0
#endif
static_assert(QUANTIZED_DECODE_FFN_OVERLAP == 0 || QUANTIZED_DECODE_FFN_OVERLAP == 1,
              "Decode FFN overlap must be explicitly enabled or disabled");
#ifndef QUANTIZED_DECODE_SCALE_PREFETCH
#define QUANTIZED_DECODE_SCALE_PREFETCH 0
#endif
static_assert(QUANTIZED_DECODE_SCALE_PREFETCH == 0 || QUANTIZED_DECODE_SCALE_PREFETCH == 1,
              "Decode scale prefetch must be explicitly enabled or disabled");
#if QUANTIZED_DECODE_SCALE_PREFETCH && !QUANTIZED_DECODE_FFN_OVERLAP
#error "Decode scale prefetch requires the shared FFN projection dataflow"
#endif
#if QUANTIZED_DECODE_FFN_OVERLAP
#if !defined(QUANTIZED_ASYNC_COMPUTE) || !QUANTIZED_ASYNC_COMPUTE
#error "Decode FFN overlap requires independent asynchronous matrix/vector workers"
#endif
#include "quantized_decode_ffn_pipeline.hpp"
#endif
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
#include "quantized_decode_rows_projection.hpp"
#endif

// The batch controller reuses the selected single-layer helpers for norm,
// RoPE, KV-store, and attention.  Only the projection driver is batch-aware.
#ifdef QUANTIZED_ALIGNMENT_W8
#include "quantized_w8_layer_runtime.hpp"
#else
#include "quantized_w4_layer_runtime.hpp"
#endif

inline std::size_t quantized_batch_decode_kv_base_words() {
    #pragma HLS inline
    return static_cast<std::size_t>(NUM_LAYERS) * MAX_SEQ_LEN * QBD_KV_BLOCKS;
}

inline void load_quantized_batch_decode_hidden(
    qbd_hidden_t& destination,
    const mm_input_block_t* source,
    const quantized_batch_decode_entry_t entries[8],
    unsigned int first,
    unsigned int rows) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=destination.block complete dim=1
    for (unsigned int row = 0; row < QBD_ROWS; ++row) {
        for (unsigned int block = 0; block < QBD_HIDDEN_BLOCKS; ++block) {
            #pragma HLS pipeline II=1
            if (row < rows) {
                destination.block[row][block] = source[
                    static_cast<std::size_t>(entries[first + row].hidden_input_offset) +
                    block];
            }
        }
    }
}

inline void store_quantized_batch_decode_hidden(
    mm_input_block_t* destination,
    const qbd_hidden_t& source,
    const quantized_batch_decode_entry_t entries[8],
    unsigned int first,
    unsigned int rows) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=source.block complete dim=1
    for (unsigned int row = 0; row < QBD_ROWS; ++row) {
        for (unsigned int block = 0; block < QBD_HIDDEN_BLOCKS; ++block) {
            #pragma HLS pipeline II=1
            if (row < rows) {
                destination[
                    static_cast<std::size_t>(entries[first + row].hidden_output_offset) +
                    block] = source.block[row][block];
            }
        }
    }
}

inline void copy_quantized_batch_decode_hidden_row(
    qbd_hidden_t& destination,
    unsigned int destination_row,
    const qbd_hidden_t& source,
    unsigned int source_row) {
    #pragma HLS inline
    for (unsigned int block = 0; block < QBD_HIDDEN_BLOCKS; ++block) {
        #pragma HLS pipeline II=1
        destination.block[destination_row][block] =
            source.block[source_row][block];
    }
}

inline void copy_quantized_batch_decode_kv_row(
    qbd_kv_t& destination,
    unsigned int destination_row,
    const qbd_kv_t& source,
    unsigned int source_row) {
    #pragma HLS inline
    for (unsigned int block = 0; block < QBD_KV_BLOCKS; ++block) {
        #pragma HLS pipeline II=1
        destination.block[destination_row][block] =
            source.block[source_row][block];
    }
}

inline quantized_layer_task_t make_quantized_batch_decode_internal_task(
    unsigned int layer,
    const quantized_batch_decode_entry_t& first_entry,
    unsigned int rows) {
    #pragma HLS inline
    quantized_layer_task_t task{};
    task.op = QUANTIZED_LAYER_OP_PREFILL;
    task.phase = QUANTIZED_LAYER_PHASE_INTERNAL;
    task.layer = layer;
    task.position = first_entry.position;
    task.query_tokens = rows;
    task.sequence_length = 1;
    task.kv_context_length = first_entry.kv_context_length;
    task.block_index = 0;
    task.block_count = 1;
    task.input_pair = 0;
    task.output_pair = 0;
#ifdef QUANTIZED_ALIGNMENT_W8
    task.activation_format = QUANTIZED_ACTIVATION_INT8;
    task.weight_format = QUANTIZED_WEIGHT_INT8;
#else
    task.activation_format = QUANTIZED_ACTIVATION_INT4;
    task.weight_format = QUANTIZED_WEIGHT_INT4;
#endif
    task.activation_scale_id = 0;
    task.weight_scale_id = 0;
    task.physical_block_size = QBD_ROWS;
    task.last_block = true;
    task.last_task = false;
    return task;
}

inline quantized_layer_task_t make_quantized_batch_decode_attention_task(
    unsigned int layer,
    const quantized_batch_decode_entry_t& entry) {
    #pragma HLS inline
    quantized_layer_task_t task{};
    task.op = QUANTIZED_LAYER_OP_DECODE;
    task.phase = QUANTIZED_LAYER_PHASE_DECODE;
    task.layer = layer;
    task.position = entry.position;
    task.query_tokens = 1;
    task.sequence_length = 1;
    task.kv_context_length = entry.kv_context_length;
    task.block_index = 0;
    task.block_count = 1;
    task.input_pair = 0;
    task.output_pair = 1;
#ifdef QUANTIZED_ALIGNMENT_W8
    task.activation_format = QUANTIZED_ACTIVATION_INT8;
    task.weight_format = QUANTIZED_WEIGHT_INT8;
#else
    task.activation_format = QUANTIZED_ACTIVATION_INT4;
    task.weight_format = QUANTIZED_WEIGHT_INT4;
#endif
    task.activation_scale_id = 0;
    task.weight_scale_id = 0;
    task.physical_block_size = QBD_ROWS;
    task.last_block = true;
    task.last_task = true;
    return task;
}

inline quant_scale_t get_quantized_batch_decode_weight_scale(
    const ap_uint<128>* scale_memory,
    unsigned int layer,
    quantized_projection_t projection) {
    #pragma HLS inline
#ifdef QUANTIZED_ALIGNMENT_W8
    return get_quantized_w8_projection_weight_scale(
        scale_memory, layer, projection);
#else
    return get_quantized_w4_projection_weight_scale(
        scale_memory, layer, projection);
#endif
}

inline void load_quantized_batch_decode_rope(
    fm_t cosine[HEAD_DIM / 2],
    fm_t sine[HEAD_DIM / 2],
    const mm_input_block_t* rope_memory,
    unsigned int position) {
    #pragma HLS inline off
#ifdef QUANTIZED_ALIGNMENT_W8
    load_quantized_w8_rope_coefficients(cosine, sine, rope_memory, position);
#else
    load_quantized_w4_rope_coefficients(cosine, sine, rope_memory, position);
#endif
}

inline void apply_quantized_batch_decode_rope(
    qbd_hidden_t& query,
    qbd_kv_t& key,
    unsigned int token,
    const fm_t cosine[HEAD_DIM / 2],
    const fm_t sine[HEAD_DIM / 2]) {
    #pragma HLS inline off
#ifdef QUANTIZED_ALIGNMENT_W8
    apply_quantized_w8_rope(query, key, token, cosine, sine);
#else
    apply_quantized_w4_rope(query, key, token, cosine, sine);
#endif
}

inline void store_quantized_batch_decode_current_kv(
    mm_input_block_t* key_cache,
    mm_input_block_t* value_cache,
    const qbd_kv_t& key,
    const qbd_kv_t& value,
    unsigned int layer,
    unsigned int position) {
    #pragma HLS inline off
#ifdef QUANTIZED_ALIGNMENT_W8
    store_quantized_w8_current_kv(
        key_cache, value_cache, key, value, layer, position, 1);
#else
    store_quantized_w4_current_kv(
        key_cache, value_cache, key, value, layer, position, 1);
#endif
}

inline void run_quantized_batch_decode_attention(
    const quantized_layer_task_t& task,
    const qbd_hidden_t& query,
    const mm_input_block_t* key_cache,
    const mm_input_block_t* value_cache,
    qbd_hidden_t& destination,
    qbd_policy_t::streams_t& streams) {
    #pragma HLS inline off
#ifdef QUANTIZED_ALIGNMENT_W8
    run_quantized_w8_online_attention(
        task, query, key_cache, value_cache, destination, streams);
#else
    run_quantized_w4_online_attention(
        task, query, key_cache, value_cache, destination, streams);
#endif
}

inline unsigned int quantized_batch_decode_tasks_for_cu(
    const quantized_batch_decode_word_t descriptors[8],
    unsigned int batch_count,
    unsigned int layer,
    unsigned int cu) {
    #pragma HLS inline off
    if (batch_count == 0 || batch_count > QUANTIZED_BATCH_DECODE_MAX ||
        cu >= QUANTIZED_LAYER_COMPUTE_CUS || layer >= NUM_LAYERS) {
        return 0;
    }

    const unsigned int chunks =
        (batch_count + QBD_ROWS - 1) / QBD_ROWS;
    unsigned int tasks = quantized_projection_tasks_for_cu(cu, chunks);
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
    if (batch_count == 1) tasks = quantized_dr_projection_tasks_for_cu(cu);
#endif
    for (unsigned int chunk = 0; chunk < chunks; ++chunk) {
        #pragma HLS loop_tripcount min=1 max=2
        const unsigned int first = chunk * QBD_ROWS;
        const unsigned int rows =
            batch_count - first < QBD_ROWS ? batch_count - first : QBD_ROWS;
        tasks += quantized_vector_layer_tasks_for_cu(QBD_ROWS, rows, cu);
#if QUANTIZED_DECODE_FFN_OVERLAP
        // Replace one SILU_MUL with SILU_ONLY and MUL. Both use the same
        // per-sequence or B1 split-column ownership as the original task.
        tasks += quantized_vector_task_rows(QBD_ROWS, rows, cu,
            MM_STREAM_QUANTIZED_MODE_MUL, INTERMEDIATE_SIZE) != 0;
#endif
    }
    for (unsigned int index = 0; index < batch_count; ++index) {
        #pragma HLS loop_tripcount min=1 max=8
        const quantized_batch_decode_entry_t entry =
            unpack_quantized_batch_decode_entry(descriptors[index]);
        tasks +=
#ifdef QUANTIZED_ALIGNMENT_W8
            quantized_w8_attention_tasks_for_block_per_cu(
                make_quantized_batch_decode_attention_task(layer, entry), cu);
#else
            quantized_w4_attention_tasks_for_block_per_cu(
                make_quantized_batch_decode_attention_task(layer, entry), cu);
#endif
    }
    return tasks;
}

inline unsigned int run_quantized_batch_decode(
    const quantized_batch_decode_word_t descriptors[8],
    unsigned int batch_count,
    unsigned int layer,
    const mm_input_block_t* hidden_input,
    mm_input_block_t* hidden_output,
    mm_input_block_t* key_cache,
    mm_input_block_t* value_cache,
    const mm_input_block_t* norm_memory,
    const mm_input_block_t* rope_memory,
    const ap_uint<128>* scale_memory,
    qbd_policy_t::streams_t& streams,
    const qbd_policy_t::memories_t& memories,
    unsigned int input_words,
    unsigned int output_words,
    unsigned int key_words,
    unsigned int value_words) {
    #pragma HLS inline off
    const unsigned int validation = validate_quantized_batch_decode(
        descriptors, batch_count, layer, input_words, output_words,
        key_words, value_words);
    if (validation != QUANTIZED_BATCH_DECODE_STATUS_PASS) return validation;

    quantized_batch_decode_entry_t entries[QUANTIZED_BATCH_DECODE_MAX];
    for (unsigned int index = 0; index < batch_count; ++index) {
        #pragma HLS pipeline II=1
        entries[index] = unpack_quantized_batch_decode_entry(descriptors[index]);
    }

    qbd_hidden_t residual;
    qbd_hidden_t hidden0;
    qbd_hidden_t hidden1;
    qbd_hidden_t norm_weight;
    qbd_kv_t key_batch;
    qbd_kv_t value_batch;
    qbd_wide_t gate_product;
    qbd_wide_t up;
    qbd_wide_t projection_scratch;
    qbd_hidden_t query_one;
    qbd_hidden_t context_one;
    qbd_kv_t key_one;
    qbd_kv_t value_one;
    fm_t cosine[HEAD_DIM / 2];
    fm_t sine[HEAD_DIM / 2];
    #pragma HLS bind_storage variable=residual.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=hidden0.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=hidden1.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=norm_weight.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=key_batch.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=value_batch.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=gate_product.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=up.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=projection_scratch.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=query_one.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=context_one.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=key_one.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=value_one.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=cosine type=ram_2p impl=bram
    #pragma HLS bind_storage variable=sine type=ram_2p impl=bram
    #pragma HLS array_partition variable=residual.block complete dim=1
    #pragma HLS array_partition variable=hidden0.block complete dim=1
    #pragma HLS array_partition variable=hidden1.block complete dim=1
    #pragma HLS array_partition variable=norm_weight.block complete dim=1
    #pragma HLS array_partition variable=key_batch.block complete dim=1
    #pragma HLS array_partition variable=value_batch.block complete dim=1
    #pragma HLS array_partition variable=gate_product.block complete dim=1
    #pragma HLS array_partition variable=up.block complete dim=1
    #pragma HLS array_partition variable=projection_scratch.block complete dim=1
    #pragma HLS array_partition variable=query_one.block complete dim=1
    #pragma HLS array_partition variable=context_one.block complete dim=1
    #pragma HLS array_partition variable=key_one.block complete dim=1
    #pragma HLS array_partition variable=value_one.block complete dim=1

    for (unsigned int first = 0; first < batch_count; first += QBD_ROWS) {
        #pragma HLS loop_tripcount min=1 max=2
        const unsigned int rows = batch_count - first < QBD_ROWS ?
            batch_count - first : QBD_ROWS;
        const quantized_layer_task_t task =
            make_quantized_batch_decode_internal_task(
                layer, entries[first], rows);

        load_quantized_batch_decode_hidden(
            residual, hidden_input, entries, first, rows);
        // Norm weights are a single row in the selected layer's first bank.
        for (unsigned int block = 0; block < QBD_HIDDEN_BLOCKS; ++block) {
            #pragma HLS pipeline II=1
            norm_weight.block[0][block] = norm_memory[
                (static_cast<std::size_t>(layer) * 2) * QBD_HIDDEN_BLOCKS + block];
        }
        run_quantized_vector_exchange<qbd_policy_t>(
            task, MM_STREAM_QUANTIZED_MODE_RMSNORM, HIDDEN_SIZE,
            residual, norm_weight, hidden0, streams);

        // All seven projections pass through this one batch-aware call site.
        for (unsigned int step = 0; step < QUANTIZED_PROJECTION_COUNT; ++step) {
            #pragma HLS loop_tripcount min=7 max=7
            const quantized_projection_t kind =
                static_cast<quantized_projection_t>(step);
            if (kind == QUANTIZED_PROJECTION_O) {
                // Q/K/V are complete before O.  Attention is deliberately
                // serialized per sequence while sharing this one engine.
                for (unsigned int row = 0; row < rows; ++row) {
                    #pragma HLS loop_tripcount min=1 max=8
                    copy_quantized_batch_decode_hidden_row(
                        query_one, 0, hidden1, row);
                    copy_quantized_batch_decode_kv_row(
                        key_one, 0, key_batch, row);
                    copy_quantized_batch_decode_kv_row(
                        value_one, 0, value_batch, row);

                    load_quantized_batch_decode_rope(
                        cosine, sine, rope_memory, entries[first + row].position);
                    apply_quantized_batch_decode_rope(
                        query_one, key_one, 0, cosine, sine);

                    mm_input_block_t* sequence_key_cache =
                        key_cache + entries[first + row].key_cache_offset;
                    mm_input_block_t* sequence_value_cache =
                        value_cache + entries[first + row].value_cache_offset;
                    store_quantized_batch_decode_current_kv(
                        sequence_key_cache, sequence_value_cache,
                        key_one, value_one, layer,
                        entries[first + row].position);
                    const quantized_layer_task_t attention_task =
                        make_quantized_batch_decode_attention_task(
                            layer, entries[first + row]);
                    run_quantized_batch_decode_attention(
                        attention_task, query_one, sequence_key_cache,
                        sequence_value_cache, context_one, streams);
                    copy_quantized_batch_decode_hidden_row(
                        hidden0, row, context_one, 0);
                }
            }
            if (kind == QUANTIZED_PROJECTION_GATE) {
                run_quantized_vector_inplace<qbd_policy_t>(
                    task, MM_STREAM_QUANTIZED_MODE_RESIDUAL_ADD, HIDDEN_SIZE,
                    residual, hidden1, streams);
                for (unsigned int block = 0; block < QBD_HIDDEN_BLOCKS; ++block) {
                    #pragma HLS pipeline II=1
                    norm_weight.block[0][block] = norm_memory[
                        (static_cast<std::size_t>(layer) * 2 + 1) *
                            QBD_HIDDEN_BLOCKS + block];
                }
                run_quantized_vector_exchange<qbd_policy_t>(
                    task, MM_STREAM_QUANTIZED_MODE_RMSNORM, HIDDEN_SIZE,
                    residual, norm_weight, hidden0, streams);
            } else if (kind == QUANTIZED_PROJECTION_DOWN) {
#if QUANTIZED_DECODE_FFN_OVERLAP
                // Up's matrix result remains in projection_scratch, while
                // the old 'up' buffer holds SiLU(Gate). Write into the now
                // dead raw-Gate buffer, preserving the fused path's fm_t
                // activation rounding before multiplication.
                run_quantized_vector_exchange<qbd_policy_t>(
                    task, MM_STREAM_QUANTIZED_MODE_MUL,
                    INTERMEDIATE_SIZE, up, projection_scratch, gate_product, streams);
#else
                run_quantized_vector_inplace<qbd_policy_t>(
                    task, MM_STREAM_QUANTIZED_MODE_SILU_MUL,
                    INTERMEDIATE_SIZE, gate_product, up, streams);
#endif
            }
            const bool source_is_wide = kind == QUANTIZED_PROJECTION_DOWN;
            // After Q/K/V, normalized input is dead. Reuse hidden0 for the
            // gathered attention context instead of selecting aggregate
            // references, which HLS 2022.2 cannot disaggregate (214-177).
#if QUANTIZED_DECODE_FFN_OVERLAP
            bool decode_rows = false;
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
            decode_rows = batch_count == 1;
#endif
            run_quantized_decode_ffn_projection(
                task, kind, hidden0, gate_product, source_is_wide,
                projection_scratch, get_quantized_batch_decode_weight_scale(
                    scale_memory, layer, kind), streams, memories,
                decode_rows, gate_product, up);
#else
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
            if (batch_count == 1) {
                run_quantized_dr_projection(
                    task, kind, hidden0, gate_product, source_is_wide,
                    projection_scratch,
                    get_quantized_batch_decode_weight_scale(
                        scale_memory, layer, kind), streams, memories);
            } else
#endif
            run_quantized_batch_projection(
                task, kind, hidden0, gate_product, source_is_wide,
                projection_scratch,
                get_quantized_batch_decode_weight_scale(
                    scale_memory, layer, kind), streams, memories);
#endif

            const quantized_projection_plan_t plan =
                get_quantized_projection_plan(kind);
            if (kind == QUANTIZED_PROJECTION_Q) {
                copy_quantized_projection_result<qbd_policy_t>(
                    hidden1, projection_scratch, rows, plan.output_dim);
            } else if (kind == QUANTIZED_PROJECTION_K) {
                copy_quantized_projection_result<qbd_policy_t>(
                    key_batch, projection_scratch, rows, plan.output_dim);
            } else if (kind == QUANTIZED_PROJECTION_V) {
                copy_quantized_projection_result<qbd_policy_t>(
                    value_batch, projection_scratch, rows, plan.output_dim);
            } else if (kind == QUANTIZED_PROJECTION_GATE) {
                copy_quantized_projection_result<qbd_policy_t>(
                    gate_product, projection_scratch, rows, plan.output_dim);
            } else if (kind == QUANTIZED_PROJECTION_UP) {
#if !QUANTIZED_DECODE_FFN_OVERLAP
                copy_quantized_projection_result<qbd_policy_t>(
                    up, projection_scratch, rows, plan.output_dim);
#endif
            } else if (kind == QUANTIZED_PROJECTION_O) {
                copy_quantized_projection_result<qbd_policy_t>(
                    hidden1, projection_scratch, rows, plan.output_dim);
            } else {
                copy_quantized_projection_result<qbd_policy_t>(
                    hidden1, projection_scratch, rows, plan.output_dim);
            }
        }

        run_quantized_vector_inplace<qbd_policy_t>(
            task, MM_STREAM_QUANTIZED_MODE_RESIDUAL_ADD, HIDDEN_SIZE,
            residual, hidden1, streams);
        store_quantized_batch_decode_hidden(
            hidden_output, residual, entries, first, rows);
    }
    return QUANTIZED_BATCH_DECODE_STATUS_PASS;
}

#endif
