#include "compute_core_quantized_w4_unified.hpp"
#include "quantized_async_compute.hpp"
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
#include "quantized_decode_rows_adapter.hpp"
#endif

void compute_core_quantized_w4_unified_nk(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activation_stream,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream0,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream1,
    hls::stream<quantized_vector_word_t>& vector_input0_stream,
    hls::stream<quantized_vector_word_t>& vector_input1_stream,
    unsigned int task_count) {
    #pragma HLS interface axis port=out_stream
    #pragma HLS interface axis port=task_stream
    #pragma HLS interface axis port=activation_stream
    #pragma HLS interface axis port=weight_stream0
    #pragma HLS interface axis port=weight_stream1
    #pragma HLS interface axis port=vector_input0_stream
    #pragma HLS interface axis port=vector_input1_stream
    #pragma HLS interface s_axilite port=task_count bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
#if LLM_FPGA_QUANTIZED_ASYNC_COMPUTE_ENABLED
    // Keep the standalone CU and the closed-loop layer on the same DATAFLOW
    // boundary. This is required for the pipeline matrix path as well as the
    // merged path: an outlined async wrapper turns external FIFO waits into a
    // single ap_ctrl process and can close a task-stream dependency cycle.
    #pragma HLS inline
    #pragma HLS dataflow
#else
    #pragma HLS inline off
#endif

#if LLM_FPGA_QUANTIZED_ASYNC_COMPUTE_ENABLED
    run_quantized_async_w4(
        out_stream, task_stream, activation_stream, weight_stream0,
        weight_stream1, vector_input0_stream, vector_input1_stream, task_count);
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
            run_quantized_decode_rows_adapter_w4(
                out_stream, merged_task, activation_stream, weight_stream0,
                weight_stream1, 1);
#else
            matrix_task.write(packed_task);
            compute_mm_stream_8x128_int4x4_block_nk(out_stream, matrix_task,
                activation_stream, weight_stream0, weight_stream1, 1);
#endif
        }
    }
#endif
}
