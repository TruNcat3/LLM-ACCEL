#ifndef LLM_FPGA_QUANTIZED_ASYNC_COMPUTE_HPP
#define LLM_FPGA_QUANTIZED_ASYNC_COMPUTE_HPP

#include "quantized_async_config.hpp"

#include "mm_stream_8x128_int4x4_block.hpp"
#include "mm_stream_4x128_int8x8_block.hpp"
#include "quantized_compute_vector.hpp"
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
#include "quantized_decode_rows_adapter.hpp"
#endif

// These are internal, finite-FIFO records.  They never cross the kernel ABI.
// Keeping the task payload in the original 128-bit word means the dispatcher
// cannot accidentally change scaling, block, or last-stream metadata.
struct quantized_async_task_token_t {
    mm_stream_quantized_task_word_t task;
    bool stop;
};

struct quantized_async_result_desc_t {
    bool vector;
    unsigned int packet_count;
};

inline quantized_async_task_token_t quantized_async_stop_token() {
    #pragma HLS inline
    quantized_async_task_token_t token;
    token.task = 0;
    token.stop = true;
    return token;
}

inline unsigned int quantized_async_vector_packet_count(
    const mm_stream_quantized_task_t& task) {
    #pragma HLS inline
    return task.valid_tokens * ceildiv(task.k_count, CU_VEC_LANES);
}

// The current matrix kernels emit one complete token/output tile for each
// task.  Their repeat mechanism is outside the 128-bit quantized task ABI:
// each repeated wave is issued as another task by the controller.  Keeping
// the formulas tied to the matrix constants makes the descriptor remain
// correct for both 64- and 128-output W4 wave configurations.
inline unsigned int quantized_async_w4_matrix_packet_count() {
    #pragma HLS inline
    return MM_STREAM_8X128_INT4X4_TOKENS *
           MM_STREAM_8X128_INT4X4_OUTPUT_GROUPS;
}

inline unsigned int quantized_async_w8_matrix_packet_count() {
    #pragma HLS inline
    return MM_STREAM_4X128_INT8X8_TOKENS *
           MM_STREAM_4X128_INT8X8_OUTPUT_GROUPS;
}

#if defined(LLM_FPGA_COMPUTE_CORE_QUANTIZED_W4_UNIFIED_HPP) || \
    defined(QUANTIZED_ASYNC_COMPUTE_W4)
static void quantized_async_w4_dispatch(
    hls::stream<mm_stream_quantized_task_word_t>& input,
    hls::stream<quantized_async_task_token_t>& matrix_tasks,
    hls::stream<quantized_async_task_token_t>& vector_tasks,
    hls::stream<quantized_async_result_desc_t>& order,
    unsigned int task_count) {
    #pragma HLS inline off
    for (unsigned int index = 0; index < task_count; ++index) {
        // The controller sends a task, then its operands, before the next
        // task. Auto-pipelining this loop makes an empty next-task read stall
        // the previous task's worker/order writes, closing a backpressure
        // cycle through the finite operand FIFOs. Complete each dispatch
        // before reading the next task; matrix/vector workers remain parallel.
        #pragma HLS pipeline off
        #pragma HLS loop_tripcount min=1 max=4096 avg=64
        const mm_stream_quantized_task_word_t packed = input.read();
        const mm_stream_quantized_task_t task =
            unpack_mm_stream_quantized_task(packed);
        quantized_async_task_token_t token;
        token.task = packed;
        token.stop = false;
        quantized_async_result_desc_t descriptor;
        descriptor.vector = quantized_task_is_vector(task);
        descriptor.packet_count = descriptor.vector ?
            quantized_async_vector_packet_count(task) :
            quantized_async_w4_matrix_packet_count();
        if (descriptor.vector)
            vector_tasks.write(token);
        else
            matrix_tasks.write(token);
        order.write(descriptor);
    }
    matrix_tasks.write(quantized_async_stop_token());
    vector_tasks.write(quantized_async_stop_token());
}
#endif

#if defined(LLM_FPGA_COMPUTE_CORE_QUANTIZED_W8_UNIFIED_HPP) || \
    defined(QUANTIZED_ASYNC_COMPUTE_W8)
static void quantized_async_w8_dispatch(
    hls::stream<mm_stream_quantized_task_word_t>& input,
    hls::stream<quantized_async_task_token_t>& matrix_tasks,
    hls::stream<quantized_async_task_token_t>& vector_tasks,
    hls::stream<quantized_async_result_desc_t>& order,
    unsigned int task_count) {
    #pragma HLS inline off
    for (unsigned int index = 0; index < task_count; ++index) {
        // Preserve task-before-operands progress under finite-FIFO feedback,
        // for the same reason as the W4 dispatcher above.
        #pragma HLS pipeline off
        #pragma HLS loop_tripcount min=1 max=4096 avg=64
        const mm_stream_quantized_task_word_t packed = input.read();
        const mm_stream_quantized_task_t task =
            unpack_mm_stream_quantized_task(packed);
        quantized_async_task_token_t token;
        token.task = packed;
        token.stop = false;
        quantized_async_result_desc_t descriptor;
        descriptor.vector = quantized_task_is_vector(task);
        descriptor.packet_count = descriptor.vector ?
            quantized_async_vector_packet_count(task) :
            quantized_async_w8_matrix_packet_count();
        if (descriptor.vector)
            vector_tasks.write(token);
        else
            matrix_tasks.write(token);
        order.write(descriptor);
    }
    matrix_tasks.write(quantized_async_stop_token());
    vector_tasks.write(quantized_async_stop_token());
}
#endif

#if defined(LLM_FPGA_COMPUTE_CORE_QUANTIZED_W4_UNIFIED_HPP) || \
    defined(QUANTIZED_ASYNC_COMPUTE_W4)
static void quantized_async_w4_matrix_worker(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& results,
    hls::stream<quantized_async_task_token_t>& tasks,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activation,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight0,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight1) {
    #pragma HLS inline off
    while (true) {
        const quantized_async_task_token_t token = tasks.read();
        if (token.stop)
            return;
        hls::stream<mm_stream_quantized_task_word_t> one_task;
        #pragma HLS stream variable=one_task depth=1
        one_task.write(token.task);
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
        // All non-vector matrix modes share the one merged qdr datapath. The
        // adapter preserves the P mapping and scatters only bit-126 DR tasks.
        run_quantized_decode_rows_adapter_w4(
            results, one_task, activation, weight0, weight1, 1);
#else
        // Exactly one physical matrix engine is instantiated in this worker.
        compute_mm_stream_8x128_int4x4_block_nk(
            results, one_task, activation, weight0, weight1, 1);
#endif
    }
}
#endif

#if defined(LLM_FPGA_COMPUTE_CORE_QUANTIZED_W8_UNIFIED_HPP) || \
    defined(QUANTIZED_ASYNC_COMPUTE_W8)
static void quantized_async_w8_matrix_worker(
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& results,
    hls::stream<quantized_async_task_token_t>& tasks,
    hls::stream<mm_stream_4x128_int8x8_activation_word_t>& activation,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight0,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight1,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight2,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight3) {
    #pragma HLS inline off
    while (true) {
        const quantized_async_task_token_t token = tasks.read();
        if (token.stop)
            return;
        hls::stream<mm_stream_quantized_task_word_t> one_task;
        #pragma HLS stream variable=one_task depth=1
        one_task.write(token.task);
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
        // Keep W8 on the same merged matrix path; vector tasks are still
        // dispatched to the unchanged vector worker below.
        run_quantized_decode_rows_adapter_w8(
            results, one_task, activation, weight0, weight1, weight2, weight3,
            1);
#else
        // Exactly one physical matrix engine is instantiated in this worker.
        compute_mm_stream_4x128_int8x8_block_nk(
            results, one_task, activation, weight0, weight1, weight2, weight3,
            1);
#endif
    }
}
#endif

#if defined(LLM_FPGA_COMPUTE_CORE_QUANTIZED_W4_UNIFIED_HPP) || \
    defined(QUANTIZED_ASYNC_COMPUTE_W4)
static void quantized_async_w4_vector_worker(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& results,
    hls::stream<quantized_async_task_token_t>& tasks,
    hls::stream<quantized_vector_word_t>& input0,
    hls::stream<quantized_vector_word_t>& input1) {
    #pragma HLS inline off
    while (true) {
        const quantized_async_task_token_t token = tasks.read();
        if (token.stop)
            return;
        const mm_stream_quantized_task_t task =
            unpack_mm_stream_quantized_task(token.task);
        // Exactly one physical vector engine is instantiated in this worker.
        run_quantized_vector_task(results, input0, input1, task);
    }
}
#endif

#if defined(LLM_FPGA_COMPUTE_CORE_QUANTIZED_W8_UNIFIED_HPP) || \
    defined(QUANTIZED_ASYNC_COMPUTE_W8)
static void quantized_async_w8_vector_worker(
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& results,
    hls::stream<quantized_async_task_token_t>& tasks,
    hls::stream<quantized_vector_word_t>& input0,
    hls::stream<quantized_vector_word_t>& input1) {
    #pragma HLS inline off
    while (true) {
        const quantized_async_task_token_t token = tasks.read();
        if (token.stop)
            return;
        const mm_stream_quantized_task_t task =
            unpack_mm_stream_quantized_task(token.task);
        // Exactly one physical vector engine is instantiated in this worker.
        run_quantized_vector_task(results, input0, input1, task);
    }
}
#endif

#if defined(LLM_FPGA_COMPUTE_CORE_QUANTIZED_W4_UNIFIED_HPP) || \
    defined(QUANTIZED_ASYNC_COMPUTE_W4)
static void quantized_async_w4_merge(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& output,
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& matrix_results,
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& vector_results,
    hls::stream<quantized_async_result_desc_t>& order,
    unsigned int task_count) {
    #pragma HLS inline off
    for (unsigned int index = 0; index < task_count; ++index) {
        #pragma HLS loop_tripcount min=1 max=4096 avg=64
        const quantized_async_result_desc_t descriptor = order.read();
        for (unsigned int packet = 0;
             packet < descriptor.packet_count; ++packet) {
            #pragma HLS pipeline II=1
            if (descriptor.vector)
                output.write(vector_results.read());
            else
                output.write(matrix_results.read());
        }
    }
}
#endif

#if defined(LLM_FPGA_COMPUTE_CORE_QUANTIZED_W8_UNIFIED_HPP) || \
    defined(QUANTIZED_ASYNC_COMPUTE_W8)
static void quantized_async_w8_merge(
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& output,
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& matrix_results,
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& vector_results,
    hls::stream<quantized_async_result_desc_t>& order,
    unsigned int task_count) {
    #pragma HLS inline off
    for (unsigned int index = 0; index < task_count; ++index) {
        #pragma HLS loop_tripcount min=1 max=4096 avg=64
        const quantized_async_result_desc_t descriptor = order.read();
        for (unsigned int packet = 0;
             packet < descriptor.packet_count; ++packet) {
            #pragma HLS pipeline II=1
            if (descriptor.vector)
                output.write(vector_results.read());
            else
                output.write(matrix_results.read());
        }
    }
}
#endif

#if defined(LLM_FPGA_COMPUTE_CORE_QUANTIZED_W4_UNIFIED_HPP) || \
    defined(QUANTIZED_ASYNC_COMPUTE_W4)
static void run_quantized_async_w4(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& output,
    hls::stream<mm_stream_quantized_task_word_t>& tasks,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activation,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight0,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight1,
    hls::stream<quantized_vector_word_t>& vector0,
    hls::stream<quantized_vector_word_t>& vector1,
    unsigned int task_count) {
    // Keep the concurrent actors visible in the enclosing DATAFLOW region for
    // both the merged and pipeline matrix paths. A sequential CU wrapper
    // around this region collapses their external waits into one dependency
    // node in closed-loop RTL co-simulation.
    #pragma HLS inline
    hls::stream<quantized_async_task_token_t> matrix_tasks, vector_tasks;
    hls::stream<quantized_async_result_desc_t> order;
    hls::stream<mm_stream_8x128_int4x4_output_word_t> matrix_results;
    hls::stream<mm_stream_8x128_int4x4_output_word_t> vector_results;
    #pragma HLS stream variable=matrix_tasks depth=QUANTIZED_ASYNC_TASK_FIFO_DEPTH
    #pragma HLS stream variable=vector_tasks depth=QUANTIZED_ASYNC_TASK_FIFO_DEPTH
    #pragma HLS stream variable=order depth=QUANTIZED_ASYNC_ORDER_FIFO_DEPTH
    #pragma HLS stream variable=matrix_results depth=QUANTIZED_ASYNC_MATRIX_RESULT_FIFO_DEPTH
    #pragma HLS stream variable=vector_results depth=QUANTIZED_ASYNC_VECTOR_RESULT_FIFO_DEPTH
    #pragma HLS bind_storage variable=matrix_results type=fifo impl=bram
    #pragma HLS bind_storage variable=vector_results type=fifo impl=bram
    #pragma HLS dataflow
    quantized_async_w4_dispatch(tasks, matrix_tasks, vector_tasks, order,
                                task_count);
    quantized_async_w4_matrix_worker(matrix_results, matrix_tasks, activation,
                                      weight0, weight1);
    quantized_async_w4_vector_worker(vector_results, vector_tasks, vector0,
                                     vector1);
    quantized_async_w4_merge(output, matrix_results, vector_results, order,
                             task_count);
}

#endif

#if defined(LLM_FPGA_COMPUTE_CORE_QUANTIZED_W8_UNIFIED_HPP) || \
    defined(QUANTIZED_ASYNC_COMPUTE_W8)
static void run_quantized_async_w8(
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& output,
    hls::stream<mm_stream_quantized_task_word_t>& tasks,
    hls::stream<mm_stream_4x128_int8x8_activation_word_t>& activation,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight0,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight1,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight2,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight3,
    hls::stream<quantized_vector_word_t>& vector0,
    hls::stream<quantized_vector_word_t>& vector1,
    unsigned int task_count) {
#if defined(QUANTIZED_DECODE_ROWS_MERGED) && QUANTIZED_DECODE_ROWS_MERGED
    #pragma HLS inline
#endif
    hls::stream<quantized_async_task_token_t> matrix_tasks, vector_tasks;
    hls::stream<quantized_async_result_desc_t> order;
    hls::stream<mm_stream_4x128_int8x8_output_word_t> matrix_results;
    hls::stream<mm_stream_4x128_int8x8_output_word_t> vector_results;
    #pragma HLS stream variable=matrix_tasks depth=QUANTIZED_ASYNC_TASK_FIFO_DEPTH
    #pragma HLS stream variable=vector_tasks depth=QUANTIZED_ASYNC_TASK_FIFO_DEPTH
    #pragma HLS stream variable=order depth=QUANTIZED_ASYNC_ORDER_FIFO_DEPTH
    #pragma HLS stream variable=matrix_results depth=QUANTIZED_ASYNC_MATRIX_RESULT_FIFO_DEPTH
    #pragma HLS stream variable=vector_results depth=QUANTIZED_ASYNC_VECTOR_RESULT_FIFO_DEPTH
    #pragma HLS bind_storage variable=matrix_results type=fifo impl=bram
    #pragma HLS bind_storage variable=vector_results type=fifo impl=bram
    #pragma HLS dataflow
    quantized_async_w8_dispatch(tasks, matrix_tasks, vector_tasks, order,
                                task_count);
    quantized_async_w8_matrix_worker(matrix_results, matrix_tasks, activation,
                                      weight0, weight1, weight2, weight3);
    quantized_async_w8_vector_worker(vector_results, vector_tasks, vector0,
                                     vector1);
    quantized_async_w8_merge(output, matrix_results, vector_results, order,
                             task_count);
}

#endif

#endif  // LLM_FPGA_QUANTIZED_ASYNC_COMPUTE_HPP
