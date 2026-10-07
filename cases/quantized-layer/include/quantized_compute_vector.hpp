#ifndef LLM_FPGA_QUANTIZED_COMPUTE_VECTOR_HPP
#define LLM_FPGA_QUANTIZED_COMPUTE_VECTOR_HPP

#include "compute_core_vector.hpp"
#include "quantized_vector_packets.hpp"
#include "mm_stream_quantized_nk.hpp"

#ifndef QUANTIZED_DECODE_FFN_OVERLAP
#define QUANTIZED_DECODE_FFN_OVERLAP 0
#endif
#ifndef QUANTIZED_PREFILL_FFN_OVERLAP
#define QUANTIZED_PREFILL_FFN_OVERLAP 0
#endif

// Use the established packed Fix16 vector ABI. Matrix precision only changes
// the matrix word formats; vector operands/results remain in resident Fix16.

inline bool quantized_task_is_vector(const mm_stream_quantized_task_t& task) {
    #pragma HLS inline
    return task.compute_mode == MM_STREAM_QUANTIZED_MODE_RMSNORM ||
           task.compute_mode == MM_STREAM_QUANTIZED_MODE_SILU_MUL ||
           task.compute_mode == MM_STREAM_QUANTIZED_MODE_RESIDUAL_ADD
#if QUANTIZED_DECODE_FFN_OVERLAP || QUANTIZED_PREFILL_FFN_OVERLAP
           || task.compute_mode == MM_STREAM_QUANTIZED_MODE_SILU_ONLY ||
           task.compute_mode == MM_STREAM_QUANTIZED_MODE_MUL
#endif
           ;
}

inline cu8_task_t quantized_vector_task(const mm_stream_quantized_task_t& task) {
    #pragma HLS inline
    cu8_task_t vector{};
    vector.mode = task.compute_mode == MM_STREAM_QUANTIZED_MODE_RMSNORM ?
        CU8_MODE_RMSNORM :
        (task.compute_mode == MM_STREAM_QUANTIZED_MODE_SILU_MUL
#if QUANTIZED_DECODE_FFN_OVERLAP || QUANTIZED_PREFILL_FFN_OVERLAP
         || task.compute_mode == MM_STREAM_QUANTIZED_MODE_SILU_ONLY ||
            task.compute_mode == MM_STREAM_QUANTIZED_MODE_MUL
#endif
        ) ? CU8_MODE_SILU_MUL : CU8_MODE_RESIDUAL_ADD;
    vector.result_policy = CU8_RESULT_EMIT;
    vector.elem_count = task.k_count;
    vector.token_count = task.valid_tokens;
    vector.packet_count = task.valid_tokens * ceildiv(task.k_count, CU_VEC_LANES);
    vector.last_task = task.last_stream;
    return vector;
}

static void unpack_quantized_vector_inputs(
    hls::stream<cu_vec16_packet_t>& input0,
    hls::stream<cu_vec16_packet_t>& input1,
    hls::stream<quantized_vector_word_t>& packed0,
    hls::stream<quantized_vector_word_t>& packed1,
    const cu8_task_t& task) {
    #pragma HLS inline off
    if (task.mode == CU8_MODE_RMSNORM) {
        for (unsigned int packet = 0;
             packet < ceildiv(task.elem_count, CU_VEC_LANES); ++packet) {
            #pragma HLS pipeline II=1
            input1.write(unpack_cu8_nk_vector(packed1.read()));
        }
    }
    for (unsigned int packet = 0; packet < task.packet_count; ++packet) {
        #pragma HLS pipeline II=1
        input0.write(unpack_cu8_nk_vector(packed0.read()));
        if (task.mode != CU8_MODE_RMSNORM)
            input1.write(unpack_cu8_nk_vector(packed1.read()));
    }
}

template <typename OutputWord>
void pack_quantized_vector_outputs(
    hls::stream<OutputWord>& output,
    hls::stream<cu_vec16_packet_t>& vector_output,
    unsigned int packets) {
    #pragma HLS inline off
    static_assert(OutputWord::width >= CU8_NK_VECTOR_BITS,
                  "The matrix output stream must hold a vector packet");
    for (unsigned int packet = 0; packet < packets; ++packet) {
        #pragma HLS pipeline II=1
        output.write(OutputWord(pack_cu8_nk_vector(vector_output.read())));
    }
}

static void run_quantized_shared_vector_primitive(
    hls::stream<cu_vec16_packet_t>& output,
    hls::stream<cu_vec16_packet_t>& input0,
    hls::stream<cu_vec16_packet_t>& input1, const cu8_task_t& task,
    mm_stream_quantized_compute_mode_t source_mode) {
    #pragma HLS inline off
    // Select only the three reference primitives used by a quantized layer.
    // Online softmax belongs to the controller, so do not instantiate the
    // generic CU softmax/bypass modes in every matrix CU.
    if (task.mode == CU8_MODE_RMSNORM)
        run_cu8_rmsnorm_task(output, input0, input1, task);
    else if (task.mode == CU8_MODE_SILU_MUL) {
#if QUANTIZED_DECODE_FFN_OVERLAP || QUANTIZED_PREFILL_FFN_OVERLAP
        const bool apply_silu =
            source_mode == MM_STREAM_QUANTIZED_MODE_SILU_MUL ||
            source_mode == MM_STREAM_QUANTIZED_MODE_SILU_ONLY;
        const bool apply_mul =
            source_mode == MM_STREAM_QUANTIZED_MODE_SILU_MUL ||
            source_mode == MM_STREAM_QUANTIZED_MODE_MUL;
        stream_silu_mul_select(
            output, input0, input1, task.packet_count, apply_silu, apply_mul);
#else
        (void)source_mode;
        stream_silu_mul(output, input0, input1, task.packet_count);
#endif
    }
    else
        stream_add_residual(output, input0, input1, task.packet_count);
}

template <typename OutputWord>
void run_quantized_vector_task(
    hls::stream<OutputWord>& output,
    hls::stream<quantized_vector_word_t>& packed0,
    hls::stream<quantized_vector_word_t>& packed1,
    const mm_stream_quantized_task_t& task) {
    #pragma HLS inline off
    hls::stream<cu_vec16_packet_t> input0, input1, results;
    #pragma HLS stream variable=input0 depth=16
    #pragma HLS stream variable=input1 depth=16
    #pragma HLS stream variable=results depth=16
    const cu8_task_t vector_task = quantized_vector_task(task);
    #pragma HLS dataflow
    unpack_quantized_vector_inputs(input0, input1, packed0, packed1, vector_task);
    run_quantized_shared_vector_primitive(
        results, input0, input1, vector_task, task.compute_mode);
    pack_quantized_vector_outputs(output, results, vector_task.packet_count);
}

#endif
