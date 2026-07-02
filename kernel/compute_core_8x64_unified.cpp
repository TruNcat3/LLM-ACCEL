#include "compute_core_8x64_unified.hpp"

static bool cu8_mode_uses_mm(cu8_mode_t mode) {
    #pragma HLS inline
    return mode == CU8_MODE_MM ||
        mode == CU8_MODE_MM_SILU ||
        mode == CU8_MODE_MM_SCALE;
}

static void convert_cu8_mm_results(
    hls::stream<cu_vec16_packet_t>& converted_stream,
    hls::stream<cu_accum16_packet_t>& accum_stream,
    const cu8_task_t& task
) {
    #pragma HLS inline off

    for (unsigned int packet = 0;
         packet < MM_STREAM_8X64_PACKETS_PER_BLOCK;
         packet++) {
        #pragma HLS pipeline II=1
        cu_accum16_packet_t in = accum_stream.read();
        cu_vec16_packet_t out;
        out.valid_mask = in.valid_mask;
        out.token_lane = in.token_lane;
        out.elem_base = in.elem_base;
        out.block_id = in.block_id;
        out.last_block = in.last_block;
        out.last_stream = in.last_stream;
        for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
            #pragma HLS unroll
            fm_accum_t value = in.data[lane];
            if (task.mode == CU8_MODE_MM_SCALE) {
                value *= fm_accum_t(task.output_scale);
            }
            out.data[lane] = fm_t(value);
        }
        converted_stream.write(out);
    }
}

static void forward_cu8_mm_results(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& converted_stream,
    const cu8_task_t& task
) {
    #pragma HLS inline off

    if (task.mode == CU8_MODE_MM_SILU) {
        stream_silu(
            out_stream,
            converted_stream,
            MM_STREAM_8X64_PACKETS_PER_BLOCK
        );
    } else {
        for (unsigned int packet = 0;
             packet < MM_STREAM_8X64_PACKETS_PER_BLOCK;
             packet++) {
            #pragma HLS pipeline II=1
            out_stream.write(converted_stream.read());
        }
    }
}

static void run_cu8_mm_once(
    hls::stream<cu_vec16_packet_t>& out_stream,
    const cu8_task_t& task,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3
) {
    #pragma HLS inline off

    hls::stream<cu_accum16_packet_t> accum_stream;
    hls::stream<cu_vec16_packet_t> converted_stream;
    #pragma HLS stream variable=accum_stream depth=MM_STREAM_8X64_PACKETS_PER_BLOCK
    #pragma HLS stream variable=converted_stream depth=2
    #pragma HLS dataflow

    mm_stream_8x64_task_t mm_task;
    mm_task.k_count = task.k_count;
    mm_task.elem_base = task.elem_base;
    mm_task.block_id = task.block_id;
    mm_task.last_stream = task.last_task;
    compute_mm_stream_8x64_fused_mac_engine(
        accum_stream,
        mm_task,
        activation_stream,
        weight_stream0,
        weight_stream1,
        weight_stream2,
        weight_stream3
    );
    convert_cu8_mm_results(converted_stream, accum_stream, task);
    forward_cu8_mm_results(out_stream, converted_stream, task);
}

static void run_cu8_mm_task(
    hls::stream<cu_vec16_packet_t>& out_stream,
    const cu8_task_t& task,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3
) {
    #pragma HLS inline off

    unsigned int repeat_count =
        task.repeat_count == 0 ? 1 : task.repeat_count;
    for (unsigned int repeat = 0;
         repeat < CU8_MAX_MM_REPEATS;
         repeat++) {
        if (repeat < repeat_count) {
            cu8_task_t repeated = task;
            repeated.elem_base =
                task.elem_base + repeat * task.elem_stride;
            repeated.block_id =
                task.block_id + repeat * task.block_stride;
            repeated.repeat_count = 1;
            repeated.last_task =
                task.last_task && repeat + 1 == repeat_count;
            run_cu8_mm_once(
                out_stream,
                repeated,
                activation_stream,
                weight_stream0,
                weight_stream1,
                weight_stream2,
                weight_stream3
            );
        }
    }
}

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

static void run_cu8_vector_task(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& input0_stream,
    hls::stream<cu_vec16_packet_t>& input1_stream,
    const cu8_task_t& task
) {
    #pragma HLS inline off

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

void compute_core_8x64_unified(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu8_task_t>& task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3,
    hls::stream<cu_vec16_packet_t>& vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& vector_input1_stream
) {
    #pragma HLS interface axis port=out_stream
    #pragma HLS interface axis port=task_stream
    #pragma HLS interface axis port=activation_stream
    #pragma HLS interface axis port=weight_stream0
    #pragma HLS interface axis port=weight_stream1
    #pragma HLS interface axis port=weight_stream2
    #pragma HLS interface axis port=weight_stream3
    #pragma HLS interface axis port=vector_input0_stream
    #pragma HLS interface axis port=vector_input1_stream
    #pragma HLS interface ap_ctrl_hs port=return
    #pragma HLS inline off

    bool done = false;
    for (unsigned int task_idx = 0;
         task_idx < CU8_MAX_TASKS_PER_LAUNCH;
         task_idx++) {
        #pragma HLS loop_tripcount min=1 max=CU8_MAX_TASKS_PER_LAUNCH
        if (!done) {
            cu8_task_t task = task_stream.read();
            if (task.mode != CU8_MODE_STOP) {
                if (cu8_mode_uses_mm(task.mode)) {
                    run_cu8_mm_task(
                        out_stream,
                        task,
                        activation_stream,
                        weight_stream0,
                        weight_stream1,
                        weight_stream2,
                        weight_stream3
                    );
                } else {
                    run_cu8_vector_task(
                        out_stream,
                        vector_input0_stream,
                        vector_input1_stream,
                        task
                    );
                }
            }
            done = task.last_task || task.mode == CU8_MODE_STOP;
        }
    }
}
