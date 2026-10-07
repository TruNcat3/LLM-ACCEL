#include "quantized_layer_integration_closed_loop.hpp"

#include "compute_core_quantized_w4_unified.hpp"
#include "quantized_async_config.hpp"
#include "quantized_w4_attention_schedule.hpp"
#include "quantized_w4_layer_runtime.hpp"

namespace {

inline unsigned int integration_tasks_for_cu(
    unsigned int sequence_length,
    unsigned int request_position,
    unsigned int request_op,
    unsigned int block_size,
    unsigned int cu) {
    return quantized_w4_layer_tasks_for_cu(
        sequence_length, request_position, cu,
        request_op == QUANTIZED_LAYER_OP_DECODE, block_size);
}

#ifdef __SYNTHESIS__
static void quantized_layer_integration_dataflow(
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
    unsigned int prefill_block_size,
    unsigned int task_count0,
    unsigned int task_count1,
    unsigned int task_count2,
    unsigned int task_count3) {
    #pragma HLS inline off

    hls::stream<mm_stream_quantized_task_word_t> task_stream[4];
    hls::stream<quantized_w4_activation_word_t> activation_stream[4];
    hls::stream<quantized_w4_weight_word_t> weight_stream[4][2];
    hls::stream<quantized_w4_output_word_t> output_stream[4];
    hls::stream<quantized_vector_word_t> vector_stream[4][2];
    #pragma HLS array_partition variable=task_stream complete
    #pragma HLS array_partition variable=activation_stream complete
    #pragma HLS array_partition variable=weight_stream complete dim=0
    #pragma HLS array_partition variable=output_stream complete
    #pragma HLS array_partition variable=vector_stream complete dim=0
    #pragma HLS stream variable=task_stream depth=QUANTIZED_ASYNC_TASK_FIFO_DEPTH
    #pragma HLS stream variable=activation_stream depth=16
    #pragma HLS stream variable=weight_stream depth=QUANTIZED_WEIGHT_FIFO_DEPTH
    #pragma HLS stream variable=output_stream depth=QUANTIZED_RESULT_FIFO_DEPTH
    #pragma HLS stream variable=vector_stream depth=16
    #pragma HLS bind_storage variable=weight_stream type=fifo impl=bram
    #pragma HLS bind_storage variable=output_stream type=fifo impl=bram

    #pragma HLS dataflow
    control_cache_quantized_w4_layer(
        task_stream[0], task_stream[1], task_stream[2], task_stream[3],
        activation_stream[0], activation_stream[1],
        activation_stream[2], activation_stream[3],
        weight_stream[0][0], weight_stream[0][1],
        weight_stream[1][0], weight_stream[1][1],
        weight_stream[2][0], weight_stream[2][1],
        weight_stream[3][0], weight_stream[3][1],
        output_stream[0], output_stream[1],
        output_stream[2], output_stream[3],
        vector_stream[0][0], vector_stream[0][1],
        vector_stream[1][0], vector_stream[1][1],
        vector_stream[2][0], vector_stream[2][1],
        vector_stream[3][0], vector_stream[3][1],
        hidden_input, hidden_output, key_cache, value_cache, norm_memory,
        rope_memory, scale_memory,
        weight_mem0a, weight_mem0b, weight_mem1a, weight_mem1b,
        weight_mem2a, weight_mem2b, weight_mem3a, weight_mem3b,
        layer, sequence_length, request_position, kv_context_length,
        request_op, prefill_block_size);

    compute_core_quantized_w4_unified_nk(
        output_stream[0], task_stream[0], activation_stream[0],
        weight_stream[0][0], weight_stream[0][1],
        vector_stream[0][0], vector_stream[0][1], task_count0);
    compute_core_quantized_w4_unified_nk(
        output_stream[1], task_stream[1], activation_stream[1],
        weight_stream[1][0], weight_stream[1][1],
        vector_stream[1][0], vector_stream[1][1], task_count1);
    compute_core_quantized_w4_unified_nk(
        output_stream[2], task_stream[2], activation_stream[2],
        weight_stream[2][0], weight_stream[2][1],
        vector_stream[2][0], vector_stream[2][1], task_count2);
    compute_core_quantized_w4_unified_nk(
        output_stream[3], task_stream[3], activation_stream[3],
        weight_stream[3][0], weight_stream[3][1],
        vector_stream[3][0], vector_stream[3][1], task_count3);
}
#endif

} // namespace

unsigned int quantized_layer_integration_closed_loop(
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
    #pragma HLS interface m_axi port=hidden_input offset=slave bundle=hidden_in depth=QUANTIZED_W4_HIDDEN_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=hidden_output offset=slave bundle=hidden_out depth=QUANTIZED_W4_HIDDEN_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=key_cache offset=slave bundle=kv_k depth=QUANTIZED_W4_KV_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=value_cache offset=slave bundle=kv_v depth=QUANTIZED_W4_KV_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=norm_memory offset=slave bundle=norm depth=QUANTIZED_W4_NORM_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=rope_memory offset=slave bundle=rope depth=QUANTIZED_W4_ROPE_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=scale_memory offset=slave bundle=scales depth=QUANTIZED_W4_SCALE_MEMORY_DEPTH max_widen_bitwidth=128
    #pragma HLS interface m_axi port=weight_mem0a offset=slave bundle=weight0a depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=weight_mem0b offset=slave bundle=weight0b depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=weight_mem1a offset=slave bundle=weight1a depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=weight_mem1b offset=slave bundle=weight1b depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=weight_mem2a offset=slave bundle=weight2a depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=weight_mem2b offset=slave bundle=weight2b depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=weight_mem3a offset=slave bundle=weight3a depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256
    #pragma HLS interface m_axi port=weight_mem3b offset=slave bundle=weight3b depth=QUANTIZED_W4_WEIGHT_MEMORY_DEPTH max_widen_bitwidth=256
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

    const unsigned int task_count0 = integration_tasks_for_cu(
        sequence_length, request_position, request_op, prefill_block_size, 0);
    const unsigned int task_count1 = integration_tasks_for_cu(
        sequence_length, request_position, request_op, prefill_block_size, 1);
    const unsigned int task_count2 = integration_tasks_for_cu(
        sequence_length, request_position, request_op, prefill_block_size, 2);
    const unsigned int task_count3 = integration_tasks_for_cu(
        sequence_length, request_position, request_op, prefill_block_size, 3);

#ifdef __SYNTHESIS__
    quantized_layer_integration_dataflow(
        hidden_input, hidden_output, key_cache, value_cache, norm_memory,
        rope_memory, scale_memory,
        weight_mem0a, weight_mem0b, weight_mem1a, weight_mem1b,
        weight_mem2a, weight_mem2b, weight_mem3a, weight_mem3b,
        layer, sequence_length, request_position, kv_context_length,
        request_op, prefill_block_size,
        task_count0, task_count1, task_count2, task_count3);
#else
    hls::stream<mm_stream_quantized_task_word_t> task_stream[4];
    hls::stream<quantized_w4_activation_word_t> activation_stream[4];
    hls::stream<quantized_w4_weight_word_t> weight_stream[4][2];
    hls::stream<quantized_w4_output_word_t> output_stream[4];
    hls::stream<quantized_vector_word_t> vector_stream[4][2];
    control_cache_quantized_w4_layer(
        task_stream[0], task_stream[1], task_stream[2], task_stream[3],
        activation_stream[0], activation_stream[1],
        activation_stream[2], activation_stream[3],
        weight_stream[0][0], weight_stream[0][1],
        weight_stream[1][0], weight_stream[1][1],
        weight_stream[2][0], weight_stream[2][1],
        weight_stream[3][0], weight_stream[3][1],
        output_stream[0], output_stream[1],
        output_stream[2], output_stream[3],
        vector_stream[0][0], vector_stream[0][1],
        vector_stream[1][0], vector_stream[1][1],
        vector_stream[2][0], vector_stream[2][1],
        vector_stream[3][0], vector_stream[3][1],
        hidden_input, hidden_output, key_cache, value_cache, norm_memory,
        rope_memory, scale_memory,
        weight_mem0a, weight_mem0b, weight_mem1a, weight_mem1b,
        weight_mem2a, weight_mem2b, weight_mem3a, weight_mem3b,
        layer, sequence_length, request_position, kv_context_length,
        request_op, prefill_block_size);
#endif
    return 0;
}
