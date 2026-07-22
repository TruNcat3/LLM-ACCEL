#include "vitis_stream_8x64.hpp"

static bool cu8_nk_mode_uses_mm(cu8_mode_t mode) {
    #pragma HLS inline
    return mode == CU8_MODE_MM ||
        mode == CU8_MODE_MM_SILU ||
        mode == CU8_MODE_MM_SCALE;
}

static void unpack_cu8_nk_inputs(
    hls::stream<cu8_nk_task_word_t>& nk_task_stream,
    hls::stream<cu8_nk_activation_word_t>& nk_activation_stream,
    hls::stream<cu8_nk_weight_word_t>& nk_weight_stream0,
    hls::stream<cu8_nk_weight_word_t>& nk_weight_stream1,
    hls::stream<cu8_nk_weight_word_t>& nk_weight_stream2,
    hls::stream<cu8_nk_weight_word_t>& nk_weight_stream3,
    hls::stream<cu8_nk_vector_word_t>& nk_vector_input0_stream,
    hls::stream<cu8_nk_vector_word_t>& nk_vector_input1_stream,
    hls::stream<cu8_task_t>& task_stream,
    hls::stream<cu8_task_t>& result_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3,
    hls::stream<cu_vec16_packet_t>& vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& vector_input1_stream
) {
    #pragma HLS inline off

    bool done = false;
    for (unsigned int task_idx = 0;
         task_idx < CU8_MAX_TASKS_PER_LAUNCH;
         task_idx++) {
        if (!done) {
            cu8_task_t task =
                unpack_cu8_nk_task(nk_task_stream.read());
            task_stream.write(task);
            result_task_stream.write(task);

            if (cu8_nk_mode_uses_mm(task.mode)) {
                unsigned int repeats =
                    task.repeat_count == 0 ? 1 : task.repeat_count;
                for (unsigned int repeat = 0;
                     repeat < repeats;
                     repeat++) {
                    #pragma HLS loop_tripcount min=1 max=CU8_MAX_MM_REPEATS avg=1
                    for (unsigned int k = 0;
                         k < task.k_count;
                         k++) {
                        #pragma HLS pipeline II=1
                        #pragma HLS loop_tripcount min=16 max=MAX_LINEAR_IN_DIM avg=HIDDEN_SIZE
                        activation_stream.write(
                            unpack_cu8_nk_activation(
                                nk_activation_stream.read()
                            )
                        );
                        weight_stream0.write(unpack_cu8_nk_weight(
                            nk_weight_stream0.read()
                        ));
                        weight_stream1.write(unpack_cu8_nk_weight(
                            nk_weight_stream1.read()
                        ));
                        weight_stream2.write(unpack_cu8_nk_weight(
                            nk_weight_stream2.read()
                        ));
                        weight_stream3.write(unpack_cu8_nk_weight(
                            nk_weight_stream3.read()
                        ));
                    }
                }
            } else if (task.mode == CU8_MODE_RMSNORM) {
                unsigned int weight_packets =
                    ceildiv(task.elem_count, CU_VEC_LANES);
                for (unsigned int packet = 0;
                     packet < weight_packets;
                     packet++) {
                    #pragma HLS pipeline II=1
                    #pragma HLS loop_tripcount min=1 max=MAX_LINEAR_OUT_BLOCKS
                    vector_input1_stream.write(unpack_cu8_nk_vector(
                        nk_vector_input1_stream.read()
                    ));
                }
                for (unsigned int packet = 0;
                     packet < task.packet_count;
                     packet++) {
                    #pragma HLS pipeline II=1
                    #pragma HLS loop_tripcount min=1 max=CU_STREAM_MAX_PACKETS
                    vector_input0_stream.write(unpack_cu8_nk_vector(
                        nk_vector_input0_stream.read()
                    ));
                }
            } else if (
                task.mode == CU8_MODE_SILU_MUL ||
                task.mode == CU8_MODE_RESIDUAL_ADD
            ) {
                for (unsigned int packet = 0;
                     packet < task.packet_count;
                     packet++) {
                    #pragma HLS pipeline II=1
                    #pragma HLS loop_tripcount min=1 max=CU_STREAM_MAX_PACKETS
                    vector_input0_stream.write(unpack_cu8_nk_vector(
                        nk_vector_input0_stream.read()
                    ));
                    vector_input1_stream.write(unpack_cu8_nk_vector(
                        nk_vector_input1_stream.read()
                    ));
                }
            } else if (
                task.mode == CU8_MODE_SILU ||
                task.mode == CU8_MODE_SOFTMAX
            ) {
                for (unsigned int packet = 0;
                     packet < task.packet_count;
                     packet++) {
                    #pragma HLS pipeline II=1
                    #pragma HLS loop_tripcount min=1 max=CU_STREAM_MAX_PACKETS
                    vector_input0_stream.write(unpack_cu8_nk_vector(
                        nk_vector_input0_stream.read()
                    ));
                }
            }
            done = task.last_task || task.mode == CU8_MODE_STOP;
        }
    }
}

static void pack_cu8_nk_results(
    hls::stream<cu8_nk_vector_word_t>& nk_out_stream,
    hls::stream<cu8_task_t>& result_task_stream,
    hls::stream<cu_vec16_packet_t>& out_stream
) {
    #pragma HLS inline off

    bool done = false;
    for (unsigned int task_idx = 0;
         task_idx < CU8_MAX_TASKS_PER_LAUNCH;
         task_idx++) {
        if (!done) {
            cu8_task_t task = result_task_stream.read();
            unsigned int repeats = cu8_nk_mode_uses_mm(task.mode) ?
                (task.repeat_count == 0 ? 1 : task.repeat_count) :
                1;
            if (task.mode != CU8_MODE_STOP) {
                for (unsigned int repeat = 0;
                     repeat < repeats;
                     repeat++) {
                    #pragma HLS loop_tripcount min=1 max=CU8_MAX_MM_REPEATS
                    for (unsigned int packet = 0;
                         packet < task.packet_count;
                         packet++) {
                        #pragma HLS pipeline II=1
                        #pragma HLS loop_tripcount min=1 max=CU_STREAM_MAX_PACKETS
                        nk_out_stream.write(
                            pack_cu8_nk_vector(out_stream.read())
                        );
                    }
                }
            }
            done = task.last_task || task.mode == CU8_MODE_STOP;
        }
    }
}

void compute_core_8x64_unified_nk(
    hls::stream<cu8_nk_vector_word_t>& out_stream,
    hls::stream<cu8_nk_task_word_t>& task_stream,
    hls::stream<cu8_nk_activation_word_t>& activation_stream,
    hls::stream<cu8_nk_weight_word_t>& weight_stream0,
    hls::stream<cu8_nk_weight_word_t>& weight_stream1,
    hls::stream<cu8_nk_weight_word_t>& weight_stream2,
    hls::stream<cu8_nk_weight_word_t>& weight_stream3,
    hls::stream<cu8_nk_vector_word_t>& vector_input0_stream,
    hls::stream<cu8_nk_vector_word_t>& vector_input1_stream
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
    #pragma HLS dataflow

    hls::stream<cu8_task_t> typed_task_stream;
    hls::stream<cu8_task_t> result_task_stream;
    hls::stream<mm_stream_8x64_activation_packet_t> typed_activation_stream;
    hls::stream<mm_stream_8x64_weight_packet_t> typed_weight_stream0;
    hls::stream<mm_stream_8x64_weight_packet_t> typed_weight_stream1;
    hls::stream<mm_stream_8x64_weight_packet_t> typed_weight_stream2;
    hls::stream<mm_stream_8x64_weight_packet_t> typed_weight_stream3;
    hls::stream<cu_vec16_packet_t> typed_vector_input0_stream;
    hls::stream<cu_vec16_packet_t> typed_vector_input1_stream;
    hls::stream<cu_vec16_packet_t> typed_out_stream;
    #pragma HLS stream variable=typed_task_stream depth=CU8_NK_TASK_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=result_task_stream depth=CU8_NK_TASK_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_activation_stream depth=CU8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_weight_stream0 depth=CU8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_weight_stream1 depth=CU8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_weight_stream2 depth=CU8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_weight_stream3 depth=CU8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_vector_input0_stream depth=CU8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_vector_input1_stream depth=CU8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_out_stream depth=CU8_NK_DATA_STREAM_DEPTH_VALUE

    unpack_cu8_nk_inputs(
        task_stream,
        activation_stream,
        weight_stream0,
        weight_stream1,
        weight_stream2,
        weight_stream3,
        vector_input0_stream,
        vector_input1_stream,
        typed_task_stream,
        result_task_stream,
        typed_activation_stream,
        typed_weight_stream0,
        typed_weight_stream1,
        typed_weight_stream2,
        typed_weight_stream3,
        typed_vector_input0_stream,
        typed_vector_input1_stream
    );
    compute_core_8x64_unified(
        typed_out_stream,
        typed_task_stream,
        typed_activation_stream,
        typed_weight_stream0,
        typed_weight_stream1,
        typed_weight_stream2,
        typed_weight_stream3,
        typed_vector_input0_stream,
        typed_vector_input1_stream
    );
    pack_cu8_nk_results(
        out_stream,
        result_task_stream,
        typed_out_stream
    );
}
