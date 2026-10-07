#ifndef LLM_FPGA_COMPUTE_CORE_VECTOR_HPP
#define LLM_FPGA_COMPUTE_CORE_VECTOR_HPP

#include "compute_core_8x64_unified.hpp"
#include "compute_rmsnorm_tree.hpp"

// Shared verbatim vector task execution for Fix16 and quantized compute CUs.
// Keep the arithmetic and operand ordering in one implementation.
#if CU_RMS_LANES_CONFIG == 1
static void run_cu8_rmsnorm_task(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& input_stream,
    hls::stream<cu_vec16_packet_t>& weight_stream,
    const cu8_task_t& task
) {
    #pragma HLS inline off

    wt_norm_t weights[MAX_LINEAR_OUT_DIM];
    #pragma HLS bind_storage variable=weights type=ram_2p impl=bram
    unsigned int weight_packets = ceildiv(task.elem_count, CU_VEC_LANES);

    for (unsigned int packet = 0;
         packet < MAX_LINEAR_OUT_BLOCKS;
         packet++) {
        if (packet < weight_packets) {
            cu_vec16_packet_t weight_packet = weight_stream.read();
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS pipeline II=1
                unsigned int elem = weight_packet.elem_base + lane;
                if (elem < task.elem_count && weight_packet.valid_mask[lane]) {
                    weights[elem] = wt_norm_t(weight_packet.data[lane]);
                }
            }
        }
    }

    for (unsigned int token = 0; token < MM_STREAM_8X64_TOKENS; token++) {
        if (token < task.token_count) {
            stream_rmsnorm(
                out_stream,
                input_stream,
                weights,
                task.elem_count
            );
        }
    }
}
#else
static void run_cu8_rmsnorm_task(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& input_stream,
    hls::stream<cu_vec16_packet_t>& weight_stream,
    const cu8_task_t& task
) {
    run_cu8_rmsnorm_tree_task<CU_RMS_LANES_CONFIG>(
        out_stream, input_stream, weight_stream, task);
}
#endif

static bool cu8_mode_uses_vector_input1(cu8_mode_t mode) {
    #pragma HLS inline
    return mode == CU8_MODE_SILU_MUL ||
        mode == CU8_MODE_RMSNORM ||
        mode == CU8_MODE_RESIDUAL_ADD;
}

static void run_cu8_vector_bypass_task(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& input0_stream,
    hls::stream<cu_vec16_packet_t>& input1_stream,
    const cu8_task_t& task
) {
    #pragma HLS inline off

    if (task.mode == CU8_MODE_RMSNORM) {
        unsigned int weight_packets = ceildiv(task.elem_count, CU_VEC_LANES);
        for (unsigned int packet = 0;
             packet < MAX_LINEAR_OUT_BLOCKS;
             packet++) {
            #pragma HLS pipeline II=1
            if (packet < weight_packets) {
                (void)input1_stream.read();
            }
        }
    }

    bool consume_input1 = cu8_mode_uses_vector_input1(task.mode) &&
        task.mode != CU8_MODE_RMSNORM;
    for (unsigned int packet = 0;
         packet < CU_STREAM_MAX_PACKETS;
         packet++) {
        #pragma HLS pipeline II=1
        if (packet < task.packet_count) {
            cu_vec16_packet_t forwarded = input0_stream.read();
            if (consume_input1) {
                (void)input1_stream.read();
            }
            out_stream.write(forwarded);
        }
    }
}

static void run_cu8_vector_task(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& input0_stream,
    hls::stream<cu_vec16_packet_t>& input1_stream,
    const cu8_task_t& task
) {
    #pragma HLS inline off

    if (task.result_policy == CU8_RESULT_BYPASS) {
        run_cu8_vector_bypass_task(
            out_stream,
            input0_stream,
            input1_stream,
            task
        );
        return;
    }

    switch (task.mode) {
    case CU8_MODE_SILU:
        stream_silu(
            out_stream,
            input0_stream,
            task.packet_count
        );
        break;
    case CU8_MODE_SILU_MUL:
        stream_silu_mul(
            out_stream,
            input0_stream,
            input1_stream,
            task.packet_count
        );
        break;
    case CU8_MODE_RMSNORM:
        run_cu8_rmsnorm_task(
            out_stream,
            input0_stream,
            input1_stream,
            task
        );
        break;
    case CU8_MODE_RESIDUAL_ADD:
        stream_add_residual(
            out_stream,
            input0_stream,
            input1_stream,
            task.packet_count
        );
        break;
    case CU8_MODE_SOFTMAX:
        stream_softmax_rows(
            out_stream,
            input0_stream,
            task.token_count,
            task.elem_count
        );
        break;
    default:
        break;
    }
}

#endif
