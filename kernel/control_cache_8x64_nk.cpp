#include "vitis_stream_8x64.hpp"

static bool cc8_nk_mode_uses_mm(cu8_mode_t mode) {
    #pragma HLS inline
    return mode == CU8_MODE_MM ||
        mode == CU8_MODE_MM_SILU ||
        mode == CU8_MODE_MM_SCALE;
}

#if CC8_RESIDENT_LAYER_ONLY
static void declare_cc8_nk_idle_core1_vector_ports(
    hls::stream<cu_vec16_packet_t>& vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& vector_input1_stream,
    unsigned int operator_kind
) {
    #pragma HLS inline off
    // A one-token resident layer intentionally assigns all vector work to
    // core0.  The core1 NK packer still owns the two physical AXIS ports, so
    // the surrounding dataflow region requires a syntactic producer for its
    // typed input FIFOs.  This invalid-operator probe declares that topology
    // without injecting traffic for any supported runtime operation.
    if (operator_kind == ~0u) {
        cu_vec16_packet_t packet;
        packet.valid_mask = 0;
        packet.token_lane = 0;
        packet.elem_base = 0;
        packet.block_id = 0;
        packet.last_block = true;
        packet.last_stream = true;
        for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
            #pragma HLS unroll
            packet.data[lane] = fm_t(0);
        }
        vector_input0_stream.write(packet);
        vector_input1_stream.write(packet);
    }
}
#endif

static void pack_cc8_nk_core_inputs(
    hls::stream<cu8_task_t>& task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3,
    hls::stream<cu_vec16_packet_t>& vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& vector_input1_stream,
    hls::stream<cu8_nk_task_word_t>& nk_task_stream,
    hls::stream<cu8_nk_activation_word_t>& nk_activation_stream,
    hls::stream<cu8_nk_weight_word_t>& nk_weight_stream0,
    hls::stream<cu8_nk_weight_word_t>& nk_weight_stream1,
    hls::stream<cu8_nk_weight_word_t>& nk_weight_stream2,
    hls::stream<cu8_nk_weight_word_t>& nk_weight_stream3,
    hls::stream<cu8_nk_vector_word_t>& nk_vector_input0_stream,
    hls::stream<cu8_nk_vector_word_t>& nk_vector_input1_stream,
    hls::stream<cu8_task_t>& result_task_stream
) {
    #pragma HLS inline off

    bool done = false;
    for (unsigned int task_idx = 0;
         task_idx < CU8_MAX_TASKS_PER_LAUNCH;
         task_idx++) {
        if (!done) {
            cu8_task_t task = task_stream.read();
            nk_task_stream.write(pack_cu8_nk_task(task));
            result_task_stream.write(task);

            if (cc8_nk_mode_uses_mm(task.mode)) {
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
                        nk_activation_stream.write(
                            pack_cu8_nk_activation(
                                activation_stream.read()
                            )
                        );
                        nk_weight_stream0.write(pack_cu8_nk_weight(
                            weight_stream0.read()
                        ));
                        nk_weight_stream1.write(pack_cu8_nk_weight(
                            weight_stream1.read()
                        ));
                        nk_weight_stream2.write(pack_cu8_nk_weight(
                            weight_stream2.read()
                        ));
                        nk_weight_stream3.write(pack_cu8_nk_weight(
                            weight_stream3.read()
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
                    nk_vector_input1_stream.write(pack_cu8_nk_vector(
                        vector_input1_stream.read()
                    ));
                }
                for (unsigned int packet = 0;
                     packet < task.packet_count;
                     packet++) {
                    #pragma HLS pipeline II=1
                    #pragma HLS loop_tripcount min=1 max=CU_STREAM_MAX_PACKETS
                    nk_vector_input0_stream.write(pack_cu8_nk_vector(
                        vector_input0_stream.read()
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
                    nk_vector_input0_stream.write(pack_cu8_nk_vector(
                        vector_input0_stream.read()
                    ));
                    nk_vector_input1_stream.write(pack_cu8_nk_vector(
                        vector_input1_stream.read()
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
                    nk_vector_input0_stream.write(pack_cu8_nk_vector(
                        vector_input0_stream.read()
                    ));
                }
            }
            done = task.last_task || task.mode == CU8_MODE_STOP;
        }
    }
}

static void unpack_cc8_nk_core_results(
    hls::stream<cu8_task_t>& result_task_stream,
    hls::stream<cu8_nk_vector_word_t>& nk_result_stream,
    hls::stream<cu_vec16_packet_t>& result_stream
) {
    #pragma HLS inline off

    bool done = false;
    for (unsigned int task_idx = 0;
         task_idx < CU8_MAX_TASKS_PER_LAUNCH;
         task_idx++) {
        if (!done) {
            cu8_task_t task = result_task_stream.read();
            unsigned int repeats = cc8_nk_mode_uses_mm(task.mode) ?
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
                        result_stream.write(unpack_cu8_nk_vector(
                            nk_result_stream.read()
                        ));
                    }
                }
            }
            done = task.last_task || task.mode == CU8_MODE_STOP;
        }
    }
}

static void pack_cc8_nk_status(
    hls::stream<cc8_status_packet_t>& status_stream,
    hls::stream<cu8_nk_status_word_t>& nk_status_stream
) {
    #pragma HLS inline off
    nk_status_stream.write(pack_cu8_nk_status(status_stream.read()));
}

void control_cache_8x64_dual_core_nk(
    hls::stream<cu8_nk_task_word_t>& core0_task_stream,
    hls::stream<cu8_nk_activation_word_t>& core0_activation_stream,
    hls::stream<cu8_nk_weight_word_t>& core0_weight_stream0,
    hls::stream<cu8_nk_weight_word_t>& core0_weight_stream1,
    hls::stream<cu8_nk_weight_word_t>& core0_weight_stream2,
    hls::stream<cu8_nk_weight_word_t>& core0_weight_stream3,
    hls::stream<cu8_nk_vector_word_t>& core0_vector_input0_stream,
    hls::stream<cu8_nk_vector_word_t>& core0_vector_input1_stream,
    hls::stream<cu8_nk_vector_word_t>& core0_result_stream,
    hls::stream<cu8_nk_task_word_t>& core1_task_stream,
    hls::stream<cu8_nk_activation_word_t>& core1_activation_stream,
    hls::stream<cu8_nk_weight_word_t>& core1_weight_stream0,
    hls::stream<cu8_nk_weight_word_t>& core1_weight_stream1,
    hls::stream<cu8_nk_weight_word_t>& core1_weight_stream2,
    hls::stream<cu8_nk_weight_word_t>& core1_weight_stream3,
    hls::stream<cu8_nk_vector_word_t>& core1_vector_input0_stream,
    hls::stream<cu8_nk_vector_word_t>& core1_vector_input1_stream,
    hls::stream<cu8_nk_vector_word_t>& core1_result_stream,
    hls::stream<cu8_nk_status_word_t>& status_stream,
    fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT],
    const fm_word_t input_port0[CC8_DATA_PORT_WORDS],
    const fm_word_t input_port1[CC8_DATA_PORT_WORDS],
    const fm_word_t aux_port0[CC8_DATA_PORT_WORDS],
    const fm_word_t aux_port1[CC8_DATA_PORT_WORDS],
    unsigned int operator_kind,
    unsigned int layer_id,
    unsigned int token_count,
    unsigned int position,
    unsigned int tile_len,
    QWEN_WEIGHT_SHARD_PARAMS,
    fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS],
    fm_word_t kv_cache_v[CC8_KV_CACHE_WORDS]
) {
    #pragma HLS interface axis port=core0_task_stream
    #pragma HLS interface axis port=core0_activation_stream
    #pragma HLS interface axis port=core0_weight_stream0
    #pragma HLS interface axis port=core0_weight_stream1
    #pragma HLS interface axis port=core0_weight_stream2
    #pragma HLS interface axis port=core0_weight_stream3
    #pragma HLS interface axis port=core0_vector_input0_stream
    #pragma HLS interface axis port=core0_vector_input1_stream
    #pragma HLS interface axis port=core0_result_stream
    #pragma HLS interface axis port=core1_task_stream
    #pragma HLS interface axis port=core1_activation_stream
    #pragma HLS interface axis port=core1_weight_stream0
    #pragma HLS interface axis port=core1_weight_stream1
    #pragma HLS interface axis port=core1_weight_stream2
    #pragma HLS interface axis port=core1_weight_stream3
    #pragma HLS interface axis port=core1_vector_input0_stream
    #pragma HLS interface axis port=core1_vector_input1_stream
    #pragma HLS interface axis port=core1_result_stream
    #pragma HLS interface axis port=status_stream
    #pragma HLS interface m_axi port=output_port0 offset=slave bundle=data0 depth=CC8_FEATURE_WORDS_PER_PORT max_widen_bitwidth=512
    #pragma HLS interface m_axi port=output_port1 offset=slave bundle=data1 depth=CC8_FEATURE_WORDS_PER_PORT max_widen_bitwidth=512
    #pragma HLS interface m_axi port=input_port0 offset=slave bundle=data0 depth=CC8_DATA_PORT_WORDS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=input_port1 offset=slave bundle=data1 depth=CC8_DATA_PORT_WORDS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=aux_port0 offset=slave bundle=data2 depth=CC8_DATA_PORT_WORDS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=aux_port1 offset=slave bundle=data3 depth=CC8_DATA_PORT_WORDS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard0 offset=slave bundle=w0 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard1 offset=slave bundle=w1 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard2 offset=slave bundle=w2 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard3 offset=slave bundle=w3 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard4 offset=slave bundle=w4 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard5 offset=slave bundle=w5 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard6 offset=slave bundle=w6 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard7 offset=slave bundle=w7 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard8 offset=slave bundle=w8 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard9 offset=slave bundle=w9 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard10 offset=slave bundle=w10 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard11 offset=slave bundle=w11 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard12 offset=slave bundle=w12 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard13 offset=slave bundle=w13 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard14 offset=slave bundle=w14 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=layer_weights_shard15 offset=slave bundle=w15 depth=WEIGHT_SHARD_BLOCKS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=kv_cache_k offset=slave bundle=kvk depth=CC8_KV_CACHE_WORDS max_widen_bitwidth=512
    #pragma HLS interface m_axi port=kv_cache_v offset=slave bundle=kvv depth=CC8_KV_CACHE_WORDS max_widen_bitwidth=512
    #pragma HLS interface s_axilite port=output_port0 bundle=control
    #pragma HLS interface s_axilite port=output_port1 bundle=control
    #pragma HLS interface s_axilite port=input_port0 bundle=control
    #pragma HLS interface s_axilite port=input_port1 bundle=control
    #pragma HLS interface s_axilite port=aux_port0 bundle=control
    #pragma HLS interface s_axilite port=aux_port1 bundle=control
    #pragma HLS interface s_axilite port=operator_kind bundle=control
    #pragma HLS interface s_axilite port=layer_id bundle=control
    #pragma HLS interface s_axilite port=token_count bundle=control
    #pragma HLS interface s_axilite port=position bundle=control
    #pragma HLS interface s_axilite port=tile_len bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard0 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard1 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard2 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard3 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard4 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard5 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard6 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard7 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard8 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard9 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard10 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard11 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard12 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard13 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard14 bundle=control
    #pragma HLS interface s_axilite port=layer_weights_shard15 bundle=control
    #pragma HLS interface s_axilite port=kv_cache_k bundle=control
    #pragma HLS interface s_axilite port=kv_cache_v bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
    #pragma HLS interface ap_ctrl_hs port=return
    #pragma HLS dataflow

    hls::stream<cu8_task_t> typed_core0_task_stream;
    hls::stream<mm_stream_8x64_activation_packet_t> typed_core0_activation_stream;
    hls::stream<mm_stream_8x64_weight_packet_t> typed_core0_weight_stream0;
    hls::stream<mm_stream_8x64_weight_packet_t> typed_core0_weight_stream1;
    hls::stream<mm_stream_8x64_weight_packet_t> typed_core0_weight_stream2;
    hls::stream<mm_stream_8x64_weight_packet_t> typed_core0_weight_stream3;
    hls::stream<cu_vec16_packet_t> typed_core0_vector_input0_stream;
    hls::stream<cu_vec16_packet_t> typed_core0_vector_input1_stream;
    hls::stream<cu_vec16_packet_t> typed_core0_result_stream;
    hls::stream<cu8_task_t> typed_core1_task_stream;
    hls::stream<mm_stream_8x64_activation_packet_t> typed_core1_activation_stream;
    hls::stream<mm_stream_8x64_weight_packet_t> typed_core1_weight_stream0;
    hls::stream<mm_stream_8x64_weight_packet_t> typed_core1_weight_stream1;
    hls::stream<mm_stream_8x64_weight_packet_t> typed_core1_weight_stream2;
    hls::stream<mm_stream_8x64_weight_packet_t> typed_core1_weight_stream3;
    hls::stream<cu_vec16_packet_t> typed_core1_vector_input0_stream;
    hls::stream<cu_vec16_packet_t> typed_core1_vector_input1_stream;
    hls::stream<cu_vec16_packet_t> typed_core1_result_stream;
    hls::stream<cc8_status_packet_t> typed_status_stream;
    hls::stream<cu8_task_t> core0_result_task_stream;
    hls::stream<cu8_task_t> core1_result_task_stream;
    #pragma HLS stream variable=typed_core0_task_stream depth=CC8_NK_TASK_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core1_task_stream depth=CC8_NK_TASK_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core0_result_task_stream depth=CC8_NK_TASK_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core1_result_task_stream depth=CC8_NK_TASK_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core0_activation_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core1_activation_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core0_weight_stream0 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core0_weight_stream1 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core0_weight_stream2 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core0_weight_stream3 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core1_weight_stream0 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core1_weight_stream1 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core1_weight_stream2 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core1_weight_stream3 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core0_vector_input0_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core0_vector_input1_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core1_vector_input0_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core1_vector_input1_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core0_result_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_core1_result_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=typed_status_stream depth=CC8_NK_STATUS_STREAM_DEPTH_VALUE

    control_cache_8x64_dual_core(
        typed_core0_task_stream,
        typed_core0_activation_stream,
        typed_core0_weight_stream0,
        typed_core0_weight_stream1,
        typed_core0_weight_stream2,
        typed_core0_weight_stream3,
        typed_core0_vector_input0_stream,
        typed_core0_vector_input1_stream,
        typed_core0_result_stream,
        typed_core1_task_stream,
        typed_core1_activation_stream,
        typed_core1_weight_stream0,
        typed_core1_weight_stream1,
        typed_core1_weight_stream2,
        typed_core1_weight_stream3,
        typed_core1_vector_input0_stream,
        typed_core1_vector_input1_stream,
        typed_core1_result_stream,
        typed_status_stream,
        output_port0,
        output_port1,
        input_port0,
        input_port1,
        aux_port0,
        aux_port1,
        operator_kind,
        layer_id,
        token_count,
        position,
        tile_len,
        QWEN_WEIGHT_SHARD_ARGS,
        kv_cache_k,
        kv_cache_v
    );
#if CC8_RESIDENT_LAYER_ONLY
    declare_cc8_nk_idle_core1_vector_ports(
        typed_core1_vector_input0_stream,
        typed_core1_vector_input1_stream,
        operator_kind
    );
#endif
    pack_cc8_nk_core_inputs(
        typed_core0_task_stream,
        typed_core0_activation_stream,
        typed_core0_weight_stream0,
        typed_core0_weight_stream1,
        typed_core0_weight_stream2,
        typed_core0_weight_stream3,
        typed_core0_vector_input0_stream,
        typed_core0_vector_input1_stream,
        core0_task_stream,
        core0_activation_stream,
        core0_weight_stream0,
        core0_weight_stream1,
        core0_weight_stream2,
        core0_weight_stream3,
        core0_vector_input0_stream,
        core0_vector_input1_stream,
        core0_result_task_stream
    );
    unpack_cc8_nk_core_results(
        core0_result_task_stream,
        core0_result_stream,
        typed_core0_result_stream
    );
    pack_cc8_nk_core_inputs(
        typed_core1_task_stream,
        typed_core1_activation_stream,
        typed_core1_weight_stream0,
        typed_core1_weight_stream1,
        typed_core1_weight_stream2,
        typed_core1_weight_stream3,
        typed_core1_vector_input0_stream,
        typed_core1_vector_input1_stream,
        core1_task_stream,
        core1_activation_stream,
        core1_weight_stream0,
        core1_weight_stream1,
        core1_weight_stream2,
        core1_weight_stream3,
        core1_vector_input0_stream,
        core1_vector_input1_stream,
        core1_result_task_stream
    );
    unpack_cc8_nk_core_results(
        core1_result_task_stream,
        core1_result_stream,
        typed_core1_result_stream
    );
    pack_cc8_nk_status(typed_status_stream, status_stream);
}
