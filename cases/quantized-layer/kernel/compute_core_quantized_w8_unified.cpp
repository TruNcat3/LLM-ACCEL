#include "compute_core_quantized_w8_unified.hpp"
#include "quantized_async_compute.hpp"
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
#include "quantized_decode_rows_adapter.hpp"
#endif

void compute_core_quantized_w8_unified_nk(
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& out_stream,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream,
    hls::stream<mm_stream_4x128_int8x8_activation_word_t>& activation_stream,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream0,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream1,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream2,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream3,
    hls::stream<quantized_vector_word_t>& vector_input0_stream,
    hls::stream<quantized_vector_word_t>& vector_input1_stream,
    unsigned int task_count) {
    #pragma HLS interface axis port=out_stream
    #pragma HLS interface axis port=task_stream
    #pragma HLS interface axis port=activation_stream
    #pragma HLS interface axis port=weight_stream0
    #pragma HLS interface axis port=weight_stream1
    #pragma HLS interface axis port=weight_stream2
    #pragma HLS interface axis port=weight_stream3
    #pragma HLS interface axis port=vector_input0_stream
    #pragma HLS interface axis port=vector_input1_stream
    #pragma HLS interface s_axilite port=task_count bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
#if LLM_FPGA_QUANTIZED_ASYNC_COMPUTE_ENABLED && \
    defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
    // Match W4: expose all asynchronous workers at the enclosing DATAFLOW
    // boundary while preserving the standalone kernel's stream interfaces.
    #pragma HLS inline
    #pragma HLS dataflow
#else
    #pragma HLS inline off
#endif

#if LLM_FPGA_QUANTIZED_ASYNC_COMPUTE_ENABLED
    run_quantized_async_w8(
        out_stream, task_stream, activation_stream, weight_stream0,
        weight_stream1, weight_stream2, weight_stream3, vector_input0_stream,
        vector_input1_stream, task_count);
#else
#if !defined(QUANTIZED_DECODE_ROWS_MERGED) || !QUANTIZED_DECODE_ROWS_MERGED
    hls::stream<mm_stream_quantized_task_word_t> matrix_task;
    #pragma HLS stream variable=matrix_task depth=2
#endif
    for (unsigned int index = 0; index < task_count; ++index) {
        #pragma HLS loop_tripcount min=1 max=4096 avg=64
        const auto packed_task = task_stream.read();
        const auto task = unpack_mm_stream_quantized_task(packed_task);
        if (quantized_task_is_vector(task)) {
            run_quantized_vector_task(out_stream, vector_input0_stream,
                                      vector_input1_stream, task);
        } else {
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
            hls::stream<mm_stream_quantized_task_word_t> merged_task;
            #pragma HLS stream variable=merged_task depth=1
            merged_task.write(packed_task);
            run_quantized_decode_rows_adapter_w8(
                out_stream, merged_task, activation_stream, weight_stream0,
                weight_stream1, weight_stream2, weight_stream3, 1);
#else
            matrix_task.write(packed_task);
            compute_mm_stream_4x128_int8x8_block_nk(out_stream, matrix_task,
                activation_stream, weight_stream0, weight_stream1, weight_stream2, weight_stream3, 1);
#endif
        }
    }
#endif
}
