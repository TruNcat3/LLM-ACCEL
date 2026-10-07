#include "control_cache_quantized_w4_layer.hpp"
#include "quantized_weight_axi_config.hpp"

#if QUANTIZED_LAYER_INTEGRATED_DECODE
#include "quantized_batch_decode_runtime.hpp"
#endif

void control_cache_quantized_w4_layer(
    hls::stream<mm_stream_quantized_task_word_t>& task_stream0,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream1,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream2,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream3,
    hls::stream<quantized_w4_activation_word_t>& activation_stream0,
    hls::stream<quantized_w4_activation_word_t>& activation_stream1,
    hls::stream<quantized_w4_activation_word_t>& activation_stream2,
    hls::stream<quantized_w4_activation_word_t>& activation_stream3,
    hls::stream<quantized_w4_weight_word_t>& weight_stream0a,
    hls::stream<quantized_w4_weight_word_t>& weight_stream0b,
    hls::stream<quantized_w4_weight_word_t>& weight_stream1a,
    hls::stream<quantized_w4_weight_word_t>& weight_stream1b,
    hls::stream<quantized_w4_weight_word_t>& weight_stream2a,
    hls::stream<quantized_w4_weight_word_t>& weight_stream2b,
    hls::stream<quantized_w4_weight_word_t>& weight_stream3a,
    hls::stream<quantized_w4_weight_word_t>& weight_stream3b,
    hls::stream<quantized_w4_output_word_t>& out_stream0,
    hls::stream<quantized_w4_output_word_t>& out_stream1,
    hls::stream<quantized_w4_output_word_t>& out_stream2,
    hls::stream<quantized_w4_output_word_t>& out_stream3,
    hls::stream<quantized_vector_word_t>& vector_stream0a,
    hls::stream<quantized_vector_word_t>& vector_stream0b,
    hls::stream<quantized_vector_word_t>& vector_stream1a,
    hls::stream<quantized_vector_word_t>& vector_stream1b,
    hls::stream<quantized_vector_word_t>& vector_stream2a,
    hls::stream<quantized_vector_word_t>& vector_stream2b,
    hls::stream<quantized_vector_word_t>& vector_stream3a,
    hls::stream<quantized_vector_word_t>& vector_stream3b,
    const mm_input_block_t* hidden_input,
    mm_input_block_t* hidden_output,
    mm_input_block_t* key_cache,
    mm_input_block_t* value_cache,
    const mm_input_block_t* norm_memory,
    const mm_input_block_t* rope_memory,
    const quantized_w4_scale_word_t* scale_memory,
    const quantized_w4_weight_word_t* weight_mem0a,
    const quantized_w4_weight_word_t* weight_mem0b,
    const quantized_w4_weight_word_t* weight_mem1a,
    const quantized_w4_weight_word_t* weight_mem1b,
    const quantized_w4_weight_word_t* weight_mem2a,
    const quantized_w4_weight_word_t* weight_mem2b,
    const quantized_w4_weight_word_t* weight_mem3a,
    const quantized_w4_weight_word_t* weight_mem3b,
    unsigned int layer,
    unsigned int sequence_length,
    unsigned int request_position,
    unsigned int kv_context_length,
    unsigned int request_op,
    unsigned int prefill_block_size) {
    #pragma HLS interface axis port=task_stream0
    #pragma HLS interface axis port=task_stream1
    #pragma HLS interface axis port=task_stream2
    #pragma HLS interface axis port=task_stream3
    #pragma HLS interface axis port=activation_stream0
    #pragma HLS interface axis port=activation_stream1
    #pragma HLS interface axis port=activation_stream2
    #pragma HLS interface axis port=activation_stream3
    #pragma HLS interface axis port=weight_stream0a
    #pragma HLS interface axis port=weight_stream0b
    #pragma HLS interface axis port=weight_stream1a
    #pragma HLS interface axis port=weight_stream1b
    #pragma HLS interface axis port=weight_stream2a
    #pragma HLS interface axis port=weight_stream2b
    #pragma HLS interface axis port=weight_stream3a
    #pragma HLS interface axis port=weight_stream3b
    #pragma HLS interface axis port=out_stream0
    #pragma HLS interface axis port=out_stream1
    #pragma HLS interface axis port=out_stream2
    #pragma HLS interface axis port=out_stream3
    #pragma HLS interface axis port=vector_stream0a
    #pragma HLS interface axis port=vector_stream0b
    #pragma HLS interface axis port=vector_stream1a
    #pragma HLS interface axis port=vector_stream1b
    #pragma HLS interface axis port=vector_stream2a
    #pragma HLS interface axis port=vector_stream2b
    #pragma HLS interface axis port=vector_stream3a
    #pragma HLS interface axis port=vector_stream3b

    #pragma HLS interface m_axi port=hidden_input offset=slave bundle=hidden_in depth=QUANTIZED_W4_HIDDEN_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=hidden_output offset=slave bundle=hidden_out depth=QUANTIZED_W4_HIDDEN_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=key_cache offset=slave bundle=kv_k depth=QUANTIZED_W4_KV_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=value_cache offset=slave bundle=kv_v depth=QUANTIZED_W4_KV_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=norm_memory offset=slave bundle=norm depth=QUANTIZED_W4_NORM_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=rope_memory offset=slave bundle=rope depth=QUANTIZED_W4_ROPE_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=scale_memory offset=slave bundle=scales depth=QUANTIZED_W4_SCALE_MEMORY_DEPTH max_widen_bitwidth=128
    #pragma HLS interface m_axi port=weight_mem0a offset=slave bundle=weight0a depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256 num_read_outstanding=QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING max_read_burst_length=QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST
    #pragma HLS interface m_axi port=weight_mem0b offset=slave bundle=weight0b depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256 num_read_outstanding=QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING max_read_burst_length=QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST
    #pragma HLS interface m_axi port=weight_mem1a offset=slave bundle=weight1a depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256 num_read_outstanding=QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING max_read_burst_length=QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST
    #pragma HLS interface m_axi port=weight_mem1b offset=slave bundle=weight1b depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256 num_read_outstanding=QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING max_read_burst_length=QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST
    #pragma HLS interface m_axi port=weight_mem2a offset=slave bundle=weight2a depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256 num_read_outstanding=QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING max_read_burst_length=QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST
    #pragma HLS interface m_axi port=weight_mem2b offset=slave bundle=weight2b depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256 num_read_outstanding=QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING max_read_burst_length=QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST
    #pragma HLS interface m_axi port=weight_mem3a offset=slave bundle=weight3a depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256 num_read_outstanding=QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING max_read_burst_length=QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST
    #pragma HLS interface m_axi port=weight_mem3b offset=slave bundle=weight3b depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256 num_read_outstanding=QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING max_read_burst_length=QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST

    #pragma HLS interface s_axilite port=hidden_input bundle=control
    #pragma HLS interface s_axilite port=hidden_output bundle=control
    #pragma HLS interface s_axilite port=key_cache bundle=control
    #pragma HLS interface s_axilite port=value_cache bundle=control
    #pragma HLS interface s_axilite port=norm_memory bundle=control
    #pragma HLS interface s_axilite port=rope_memory bundle=control
    #pragma HLS interface s_axilite port=scale_memory bundle=control
    #pragma HLS interface s_axilite port=weight_mem0a bundle=control
    #pragma HLS interface s_axilite port=weight_mem0b bundle=control
    #pragma HLS interface s_axilite port=weight_mem1a bundle=control
    #pragma HLS interface s_axilite port=weight_mem1b bundle=control
    #pragma HLS interface s_axilite port=weight_mem2a bundle=control
    #pragma HLS interface s_axilite port=weight_mem2b bundle=control
    #pragma HLS interface s_axilite port=weight_mem3a bundle=control
    #pragma HLS interface s_axilite port=weight_mem3b bundle=control
    #pragma HLS interface s_axilite port=layer bundle=control
    #pragma HLS interface s_axilite port=sequence_length bundle=control
    #pragma HLS interface s_axilite port=request_position bundle=control
    #pragma HLS interface s_axilite port=kv_context_length bundle=control
    #pragma HLS interface s_axilite port=request_op bundle=control
    #pragma HLS interface s_axilite port=prefill_block_size bundle=control
    #pragma HLS interface s_axilite port=return bundle=control

#ifdef __SYNTHESIS__
    quantized_w4_projection_streams_t streams{
        &task_stream0, &task_stream1, &task_stream2, &task_stream3,
        &activation_stream0, &activation_stream1, &activation_stream2,
        &activation_stream3,
        &weight_stream0a, &weight_stream0b,
        &weight_stream1a, &weight_stream1b,
        &weight_stream2a, &weight_stream2b,
        &weight_stream3a, &weight_stream3b,
        &out_stream0, &out_stream1, &out_stream2, &out_stream3,
        {&vector_stream0a, &vector_stream0b, &vector_stream1a, &vector_stream1b,
         &vector_stream2a, &vector_stream2b, &vector_stream3a, &vector_stream3b}};
#else
    quantized_w4_projection_streams_t streams{};
    streams.task[0] = &task_stream0;
    streams.task[1] = &task_stream1;
    streams.task[2] = &task_stream2;
    streams.task[3] = &task_stream3;
    streams.activation[0] = &activation_stream0;
    streams.activation[1] = &activation_stream1;
    streams.activation[2] = &activation_stream2;
    streams.activation[3] = &activation_stream3;
    streams.output[0] = &out_stream0;
    streams.output[1] = &out_stream1;
    streams.output[2] = &out_stream2;
    streams.output[3] = &out_stream3;
    streams.vector = {&vector_stream0a, &vector_stream0b,
        &vector_stream1a, &vector_stream1b, &vector_stream2a, &vector_stream2b,
        &vector_stream3a, &vector_stream3b};
    streams.weight[0][0] = &weight_stream0a;
    streams.weight[0][1] = &weight_stream0b;
    streams.weight[1][0] = &weight_stream1a;
    streams.weight[1][1] = &weight_stream1b;
    streams.weight[2][0] = &weight_stream2a;
    streams.weight[2][1] = &weight_stream2b;
    streams.weight[3][0] = &weight_stream3a;
    streams.weight[3][1] = &weight_stream3b;
#endif

#ifdef __SYNTHESIS__
    quantized_w4_weight_memories_t memories{
        weight_mem0a, weight_mem0b,
        weight_mem1a, weight_mem1b,
        weight_mem2a, weight_mem2b,
        weight_mem3a, weight_mem3b};
#else
    quantized_w4_weight_memories_t memories{};
    memories.weight[0][0] = weight_mem0a;
    memories.weight[0][1] = weight_mem0b;
    memories.weight[1][0] = weight_mem1a;
    memories.weight[1][1] = weight_mem1b;
    memories.weight[2][0] = weight_mem2a;
    memories.weight[2][1] = weight_mem2b;
    memories.weight[3][0] = weight_mem3a;
    memories.weight[3][1] = weight_mem3b;
#endif

#if QUANTIZED_LAYER_INTEGRATED_DECODE
    if (request_op == QUANTIZED_LAYER_OP_DECODE) {
        // Preserve the legacy top-level ABI while translating its scalar D1
        // request into the existing qbd single-entry runtime.  The zero bases
        // are intentional: qbd adds layer/position to the same KV buffers
        // used by run_quantized_w4_layer_block, and hidden token 0 is the
        // established Host P66->D1 handoff.
        quantized_batch_decode_word_t descriptors[
            QUANTIZED_BATCH_DECODE_MAX] = {};
        descriptors[0] = make_quantized_layer_integrated_decode_descriptor(
            request_position, kv_context_length);
        (void)run_quantized_batch_decode(
            descriptors, 1, layer, hidden_input, hidden_output,
            key_cache, value_cache, norm_memory, rope_memory, scale_memory,
            streams, memories,
            quantized_layer_integrated_decode_hidden_words(),
            quantized_layer_integrated_decode_hidden_words(),
            quantized_layer_integrated_decode_kv_words(),
            quantized_layer_integrated_decode_kv_words());
        return;
    }
#endif

    const unsigned int safe_sequence = sequence_length == 0 ? 1 :
        (sequence_length > MAX_SEQ_LEN ? MAX_SEQ_LEN : sequence_length);
    const bool decode = request_op == QUANTIZED_LAYER_OP_DECODE;
    const unsigned int block_size = prefill_block_size == 0 ||
            prefill_block_size > QUANTIZED_W4_RESIDENT_TOKEN_ROWS ?
        QUANTIZED_LAYER_W4_TOKEN_BLOCK : prefill_block_size;
    const unsigned int block_count = decode ? 1 : quantized_layer_block_count(
        safe_sequence, block_size);
#if QUANTIZED_BLOCK_PIPELINE
    // One lookahead slot; the current block retains its original working banks.
    // No second FFN-wide scratch or KV cache is allocated.
    quantized_w4_hidden_buffer_t next_source, next_normalized;
    #pragma HLS array_partition variable=next_source.block complete dim=1
    #pragma HLS array_partition variable=next_normalized.block complete dim=1
    #pragma HLS bind_storage variable=next_source.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=next_normalized.block type=ram_2p impl=bram
#endif
    for (unsigned int block = 0; block < block_count; ++block) {
        #pragma HLS loop_tripcount min=1 max=512 avg=16
        quantized_layer_task_t layer_task = decode ?
            make_quantized_decode_task(
                layer, request_position, kv_context_length, 0, 1,
                QUANTIZED_ACTIVATION_INT4, QUANTIZED_WEIGHT_INT4, 0, 0,
                block_size) :
            make_quantized_prefill_block_task(
                layer, request_position, safe_sequence, block, 0, 1,
                QUANTIZED_ACTIVATION_INT4, QUANTIZED_WEIGHT_INT4, 0, 0,
                MAX_SEQ_LEN, block_size);
        if (!decode) {
            layer_task.kv_context_length =
                request_position + block * block_size;
        }
#if QUANTIZED_BLOCK_PIPELINE
        const bool has_next = !decode && block + 1 < block_count;
        quantized_layer_task_t next_task = layer_task;
        if (has_next) {
            next_task = make_quantized_prefill_block_task(
                layer, request_position, safe_sequence, block + 1, 0, 1,
                QUANTIZED_ACTIVATION_INT4, QUANTIZED_WEIGHT_INT4, 0, 0,
                MAX_SEQ_LEN, block_size);
            next_task.kv_context_length = request_position + (block + 1) * block_size;
        }
#endif
        run_quantized_w4_layer_block(
            layer_task, hidden_input, hidden_output, key_cache, value_cache,
            norm_memory, rope_memory, scale_memory, streams, memories
#if QUANTIZED_BLOCK_PIPELINE
            , !decode && block != 0, has_next, next_task, next_source, next_normalized
#endif
            );
    }
}
