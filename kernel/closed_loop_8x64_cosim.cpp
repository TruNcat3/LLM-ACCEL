#include "closed_loop_8x64_cosim.hpp"

#ifndef __SYNTHESIS__
static void seed_cc8_closed_loop_result_block(
    hls::stream<cu_vec16_packet_t>& result_stream,
    fm_t value,
    unsigned int elem_base,
    unsigned int block_id,
    bool last_stream
) {
    for (unsigned int token = 0;
         token < MM_STREAM_8X64_TOKENS;
         token++) {
        for (unsigned int group = 0;
             group < MM_STREAM_8X64_WEIGHT_GROUPS;
             group++) {
            cu_vec16_packet_t packet;
            packet.valid_mask = 0xffff;
            packet.token_lane = token;
            packet.elem_base = elem_base + group * CU_VEC_LANES;
            packet.block_id = block_id;
            packet.last_block =
                token + 1 == MM_STREAM_8X64_TOKENS &&
                group + 1 == MM_STREAM_8X64_WEIGHT_GROUPS;
            packet.last_stream = last_stream && packet.last_block;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                const unsigned int elem = packet.elem_base + lane;
                packet.data[lane] =
                    token < GQA_GROUP_SIZE && elem < HEAD_DIM ?
                    value : fm_t(0);
            }
            result_stream.write(packet);
        }
    }
}

static void seed_cc8_closed_loop_c_model(
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream
) {
    // HLS cosim first executes the C top sequentially to generate test vectors.
    // Seed that C-only pass with the exact zero-QK/current-V responses used by
    // the deterministic smoke case. The synthesized RTL has the real feedback
    // ring and receives every result from the two compute processes.
    seed_cc8_closed_loop_result_block(
        core0_result_stream,
        fm_t(0),
        0,
        0,
        false
    );
    seed_cc8_closed_loop_result_block(
        core1_result_stream,
        fm_t(0),
        0,
        0,
        false
    );
    seed_cc8_closed_loop_result_block(
        core0_result_stream,
        fm_t(1),
        0,
        0,
        true
    );
    seed_cc8_closed_loop_result_block(
        core1_result_stream,
        fm_t(2),
        0,
        0,
        true
    );
}

static void seed_cc8_prefill_block_c_model(
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int query_begin,
    unsigned int token_count
) {
    const unsigned int context_len = query_begin + token_count;
    for (unsigned int tile_begin = 0;
         tile_begin < context_len;
         tile_begin += CC8_ATTN_TILE) {
        for (unsigned int head = 0; head < GQA_GROUP_SIZE; head++) {
            seed_cc8_closed_loop_result_block(
                core0_result_stream, fm_t(0), 0, head, false
            );
            seed_cc8_closed_loop_result_block(
                core1_result_stream, fm_t(0), 0, head, false
            );
            for (unsigned int wave = 0;
                 wave < CC8_ATTN_PV_WAVES;
                 wave++) {
                const bool last =
                    tile_begin + CC8_ATTN_TILE >= context_len &&
                    head + 1 == GQA_GROUP_SIZE &&
                    wave + 1 == CC8_ATTN_PV_WAVES;
                seed_cc8_closed_loop_result_block(
                    core0_result_stream, fm_t(0),
                    wave * MM_STREAM_8X64_OUTPUTS, wave, last
                );
                seed_cc8_closed_loop_result_block(
                    core1_result_stream, fm_t(0),
                    wave * MM_STREAM_8X64_OUTPUTS, wave, last
                );
            }
        }
    }
}

static bool cc8_closed_loop_uses_mm(cc8_operator_t op) {
    switch (op) {
    case CC8_OP_Q_PROJECTION:
    case CC8_OP_K_PROJECTION:
    case CC8_OP_V_PROJECTION:
    case CC8_OP_O_PROJECTION:
    case CC8_OP_FFN_GATE:
    case CC8_OP_FFN_UP:
    case CC8_OP_FFN_DOWN:
        return true;
    default:
        return false;
    }
}

static void seed_cc8_closed_loop_mm_c_model(
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int first_wave,
    unsigned int wave_count,
    unsigned int core_mask
) {
    for (unsigned int local_wave = 0;
         local_wave < wave_count;
         local_wave++) {
        unsigned int output_wave = first_wave + local_wave;
        bool last_stream = local_wave + 1 == wave_count;
        if ((core_mask & 1u) != 0u) {
            seed_cc8_closed_loop_result_block(
                core0_result_stream,
                fm_t(0),
                output_wave * CC8_OUTPUTS_PER_WAVE,
                output_wave,
                last_stream
            );
        }
        if ((core_mask & 2u) != 0u) {
            seed_cc8_closed_loop_result_block(
                core1_result_stream,
                fm_t(0),
                output_wave * CC8_OUTPUTS_PER_WAVE +
                    MM_STREAM_8X64_OUTPUTS,
                output_wave,
                last_stream
            );
        }
    }
}

static void seed_cc8_closed_loop_vector_results(
    hls::stream<cu_vec16_packet_t>& result_stream,
    unsigned int token_begin,
    unsigned int token_count,
    unsigned int elem_count,
    unsigned int operator_kind
) {
    unsigned int block_count = ceildiv(elem_count, CU_VEC_LANES);
    for (unsigned int local_token = 0;
         local_token < token_count;
         local_token++) {
        for (unsigned int block = 0; block < block_count; block++) {
            cu_vec16_packet_t packet;
            packet.valid_mask = 0;
            packet.token_lane = token_begin + local_token;
            packet.elem_base = block * CU_VEC_LANES;
            packet.block_id = 0;
            packet.last_block =
                local_token + 1 == token_count &&
                block + 1 == block_count;
            packet.last_stream = packet.last_block;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                unsigned int elem = packet.elem_base + lane;
                bool valid = elem < elem_count;
                fm_t value = fm_t(0);
                if (operator_kind == unsigned(CC8_OP_RESIDUAL_ADD) && valid) {
                    fm_t lhs =
                        fm_t((packet.token_lane + 1) * 16 + (elem & 15)) /
                        fm_t(16);
                    fm_t rhs =
                        fm_t((packet.token_lane + 2) * 8 + (elem & 7)) /
                        fm_t(8);
                    value = lhs + rhs;
                }
                packet.data[lane] = value;
                packet.valid_mask[lane] = valid;
            }
            result_stream.write(packet);
        }
    }
}

static void seed_cc8_closed_loop_dual_vector_results(
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int token_count,
    unsigned int elem_count,
    unsigned int operator_kind
) {
    const unsigned int core0_tokens =
        token_count < CC8_TOKENS_PER_DATA_PORT ?
        token_count : CC8_TOKENS_PER_DATA_PORT;
    const unsigned int core1_tokens =
        token_count > CC8_TOKENS_PER_DATA_PORT ?
        token_count - CC8_TOKENS_PER_DATA_PORT : 0;
    seed_cc8_closed_loop_vector_results(
        core0_result_stream,
        0,
        core0_tokens,
        elem_count,
        operator_kind
    );
    seed_cc8_closed_loop_vector_results(
        core1_result_stream,
        CC8_TOKENS_PER_DATA_PORT,
        core1_tokens,
        elem_count,
        operator_kind
    );
}

static void seed_cc8_closed_loop_zero_mm_wave(
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int output_wave,
    bool attention_layout
) {
    unsigned int core0_base = attention_layout ?
        output_wave * MM_STREAM_8X64_OUTPUTS :
        output_wave * CC8_OUTPUTS_PER_WAVE;
    unsigned int core1_base = attention_layout ?
        core0_base : core0_base + MM_STREAM_8X64_OUTPUTS;
    seed_cc8_closed_loop_result_block(
        core0_result_stream,
        fm_t(0),
        core0_base,
        output_wave,
        false
    );
    seed_cc8_closed_loop_result_block(
        core1_result_stream,
        fm_t(0),
        core1_base,
        output_wave,
        false
    );
}

static unsigned int cc8_resident_layer_mm_wave_slots(
    unsigned int position,
    unsigned int token_count
) {
    unsigned int attention_output_waves =
        ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS);
    unsigned int slots =
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(KV_CHANNELS, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(KV_CHANNELS, CC8_OUTPUTS_PER_WAVE);
    for (unsigned int token = 0; token < CC8_GBUF_TOKEN_ROWS; token++) {
        if (token < token_count) {
            slots += ceildiv(position + token + 1, CC8_ATTN_TILE) *
                (1 + attention_output_waves);
        }
    }
    return slots +
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE);
}

static unsigned int cc8_attention_sublayer_mm_wave_slots(
    unsigned int position,
    unsigned int token_count
) {
    const unsigned int attention_output_waves =
        ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS);
    unsigned int slots =
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE) +
        2 * ceildiv(KV_CHANNELS, CC8_OUTPUTS_PER_WAVE);
    for (unsigned int token = 0; token < CC8_GBUF_TOKEN_ROWS; token++) {
        if (token < token_count) {
            slots += ceildiv(position + token + 1, CC8_ATTN_TILE) *
                (1 + attention_output_waves);
        }
    }
    return slots + ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE);
}

static unsigned int cc8_ffn_sublayer_mm_wave_slots() {
    return
        2 * ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE);
}

static void seed_cc8_closed_loop_attention_sublayer_c_model(
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int position,
    unsigned int token_count
) {
    seed_cc8_closed_loop_dual_vector_results(
        core0_result_stream,
        core1_result_stream,
        token_count,
        HIDDEN_SIZE,
        unsigned(CC8_OP_RMSNORM)
    );

    const mm_projection_kind_t qkv[3] = {
        MM_PROJECTION_Q,
        MM_PROJECTION_K,
        MM_PROJECTION_V
    };
    for (unsigned int projection_idx = 0;
         projection_idx < 3;
         projection_idx++) {
        mm_projection_spec_t projection;
        get_mm_projection_spec(qkv[projection_idx], projection);
        const unsigned int waves =
            ceildiv(projection.out_dim, CC8_OUTPUTS_PER_WAVE);
        for (unsigned int wave = 0; wave < waves; wave++) {
            seed_cc8_closed_loop_zero_mm_wave(
                core0_result_stream,
                core1_result_stream,
                wave,
                false
            );
        }
    }

    const unsigned int attention_output_waves =
        ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS);
    for (unsigned int token = 0; token < CC8_GBUF_TOKEN_ROWS; token++) {
        if (token < token_count) {
            const unsigned int attention_tiles =
                ceildiv(position + token + 1, CC8_ATTN_TILE);
            for (unsigned int tile = 0; tile < attention_tiles; tile++) {
                seed_cc8_closed_loop_zero_mm_wave(
                    core0_result_stream,
                    core1_result_stream,
                    0,
                    true
                );
                for (unsigned int wave = 0;
                     wave < attention_output_waves;
                     wave++) {
                    seed_cc8_closed_loop_zero_mm_wave(
                        core0_result_stream,
                        core1_result_stream,
                        wave,
                        true
                    );
                }
            }
        }
    }

    const unsigned int hidden_waves =
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE);
    for (unsigned int wave = 0; wave < hidden_waves; wave++) {
        seed_cc8_closed_loop_zero_mm_wave(
            core0_result_stream,
            core1_result_stream,
            wave,
            false
        );
    }
    seed_cc8_closed_loop_dual_vector_results(
        core0_result_stream,
        core1_result_stream,
        token_count,
        HIDDEN_SIZE,
        unsigned(CC8_OP_RMSNORM)
    );
}

static void seed_cc8_closed_loop_ffn_sublayer_c_model(
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int token_count
) {
    const unsigned int hidden_waves =
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE);
    seed_cc8_closed_loop_dual_vector_results(
        core0_result_stream,
        core1_result_stream,
        token_count,
        HIDDEN_SIZE,
        unsigned(CC8_OP_RMSNORM)
    );

    const unsigned int intermediate_waves =
        ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE);
    for (unsigned int projection = 0; projection < 2; projection++) {
        for (unsigned int wave = 0; wave < intermediate_waves; wave++) {
            seed_cc8_closed_loop_zero_mm_wave(
                core0_result_stream,
                core1_result_stream,
                wave,
                false
            );
        }
    }
    seed_cc8_closed_loop_dual_vector_results(
        core0_result_stream,
        core1_result_stream,
        token_count,
        INTERMEDIATE_SIZE,
        unsigned(CC8_OP_RMSNORM)
    );
    for (unsigned int wave = 0; wave < hidden_waves; wave++) {
        seed_cc8_closed_loop_zero_mm_wave(
            core0_result_stream,
            core1_result_stream,
            wave,
            false
        );
    }
    seed_cc8_closed_loop_dual_vector_results(
        core0_result_stream,
        core1_result_stream,
        token_count,
        HIDDEN_SIZE,
        unsigned(CC8_OP_RMSNORM)
    );
}

static void seed_cc8_closed_loop_resident_layer_c_model(
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int position,
    unsigned int token_count
) {
    seed_cc8_closed_loop_attention_sublayer_c_model(
        core0_result_stream,
        core1_result_stream,
        position,
        token_count
    );
    seed_cc8_closed_loop_ffn_sublayer_c_model(
        core0_result_stream,
        core1_result_stream,
        token_count
    );
}

static void drain_cc8_closed_loop_c_model_results(
    hls::stream<cu_vec16_packet_t>& result_stream,
    unsigned int packet_count
) {
    for (unsigned int packet = 0; packet < packet_count; packet++) {
        result_stream.read();
    }
}
#endif

static void cc8_closed_loop_compute_core0(
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
    #pragma HLS inline off
    compute_core_8x64_unified(
        out_stream,
        task_stream,
        activation_stream,
        weight_stream0,
        weight_stream1,
        weight_stream2,
        weight_stream3,
        vector_input0_stream,
        vector_input1_stream
    );
}

static void cc8_closed_loop_compute_core1(
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
    #pragma HLS inline off
    compute_core_8x64_unified(
        out_stream,
        task_stream,
        activation_stream,
        weight_stream0,
        weight_stream1,
        weight_stream2,
        weight_stream3,
        vector_input0_stream,
        vector_input1_stream
    );
}

#if (CC8_RESIDENT_LAYER_ONLY && !CC8_RESIDENT_DUAL_VECTOR_PORTS) || \
    defined(CC8_PREFILL_BLOCK_SYNTH_ONLY)
static void declare_cc8_closed_loop_idle_vector_ports(
    hls::stream<cu_vec16_packet_t>& vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& vector_input1_stream,
    unsigned int operator_kind
) {
    #pragma HLS inline off
    // A specialized controller may intentionally leave vector AXIS ports
    // idle, but a closed-loop HLS dataflow top still requires every internal
    // stream to have a syntactic producer.  This unreachable probe declares
    // that topology without injecting traffic into any valid operation.
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

void cc8_closed_loop_inner_cosim(
    fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t status_output[1],
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
    #pragma HLS interface m_axi port=output_port0 offset=slave bundle=data0 depth=CC8_FEATURE_WORDS_PER_PORT max_widen_bitwidth=512
    #pragma HLS interface m_axi port=output_port1 offset=slave bundle=data1 depth=CC8_FEATURE_WORDS_PER_PORT max_widen_bitwidth=512
    #pragma HLS interface m_axi port=status_output offset=slave bundle=status depth=1 max_widen_bitwidth=512
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
    #pragma HLS interface s_axilite port=status_output bundle=control
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
    #pragma HLS dataflow

    hls::stream<cu8_task_t> core0_task_stream;
    hls::stream<mm_stream_8x64_activation_packet_t> core0_activation_stream;
    hls::stream<mm_stream_8x64_weight_packet_t> core0_weight_stream0;
    hls::stream<mm_stream_8x64_weight_packet_t> core0_weight_stream1;
    hls::stream<mm_stream_8x64_weight_packet_t> core0_weight_stream2;
    hls::stream<mm_stream_8x64_weight_packet_t> core0_weight_stream3;
    hls::stream<cu_vec16_packet_t> core0_vector_input0_stream;
    hls::stream<cu_vec16_packet_t> core0_vector_input1_stream;
    hls::stream<cu_vec16_packet_t> core0_result_stream;
    hls::stream<cu8_task_t> core1_task_stream;
    hls::stream<mm_stream_8x64_activation_packet_t> core1_activation_stream;
    hls::stream<mm_stream_8x64_weight_packet_t> core1_weight_stream0;
    hls::stream<mm_stream_8x64_weight_packet_t> core1_weight_stream1;
    hls::stream<mm_stream_8x64_weight_packet_t> core1_weight_stream2;
    hls::stream<mm_stream_8x64_weight_packet_t> core1_weight_stream3;
    hls::stream<cu_vec16_packet_t> core1_vector_input0_stream;
    hls::stream<cu_vec16_packet_t> core1_vector_input1_stream;
    hls::stream<cu_vec16_packet_t> core1_result_stream;
    hls::stream<cc8_status_packet_t> status_stream;

    #pragma HLS stream variable=core0_task_stream depth=CC8_NK_TASK_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core1_task_stream depth=CC8_NK_TASK_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core0_activation_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core0_weight_stream0 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core0_weight_stream1 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core0_weight_stream2 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core0_weight_stream3 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core0_vector_input0_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core0_vector_input1_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core0_result_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core1_activation_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core1_weight_stream0 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core1_weight_stream1 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core1_weight_stream2 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core1_weight_stream3 depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core1_vector_input0_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core1_vector_input1_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=core1_result_stream depth=CC8_NK_DATA_STREAM_DEPTH_VALUE
    #pragma HLS stream variable=status_stream depth=CC8_NK_STATUS_STREAM_DEPTH_VALUE

#ifndef __SYNTHESIS__
    unsigned int c_model_core0_result_packets =
        2 * MM_STREAM_8X64_PACKETS_PER_BLOCK;
    unsigned int c_model_core1_result_packets =
        2 * MM_STREAM_8X64_PACKETS_PER_BLOCK;
    bool vector_case =
        operator_kind == unsigned(CC8_OP_RESIDUAL_ADD) ||
        operator_kind == unsigned(CC8_OP_SILU_MUL) ||
        operator_kind == unsigned(CC8_OP_RMSNORM) ||
        operator_kind == unsigned(CC8_OP_FINAL_NORM);
    bool profiled_mm_case =
        tile_len != 0 &&
        cc8_closed_loop_uses_mm(cc8_operator_t(operator_kind));
    if (operator_kind == unsigned(CC8_OP_ATTN_PREFILL_BLOCK)) {
        const unsigned int tile_count =
            ceildiv(position + token_count, CC8_ATTN_TILE);
        const unsigned int tasks_per_core =
            tile_count * GQA_GROUP_SIZE * (1 + CC8_ATTN_PV_WAVES);
        c_model_core0_result_packets =
            tasks_per_core * MM_STREAM_8X64_PACKETS_PER_BLOCK;
        c_model_core1_result_packets = c_model_core0_result_packets;
        seed_cc8_prefill_block_c_model(
            core0_result_stream,
            core1_result_stream,
            position,
            token_count
        );
    } else if (operator_kind == unsigned(CC8_OP_DECODER_LAYER)) {
        unsigned int wave_slots =
            cc8_resident_layer_mm_wave_slots(position, token_count);
        const unsigned int core0_tokens =
            token_count < CC8_TOKENS_PER_DATA_PORT ?
            token_count : CC8_TOKENS_PER_DATA_PORT;
        const unsigned int core1_tokens = token_count - core0_tokens;
        const unsigned int vector_packets_per_token =
            4 * ceildiv(HIDDEN_SIZE, CU_VEC_LANES) +
            ceildiv(INTERMEDIATE_SIZE, CU_VEC_LANES);
        c_model_core0_result_packets =
            wave_slots * MM_STREAM_8X64_PACKETS_PER_BLOCK +
            core0_tokens * vector_packets_per_token;
        c_model_core1_result_packets =
            wave_slots * MM_STREAM_8X64_PACKETS_PER_BLOCK +
            core1_tokens * vector_packets_per_token;
        seed_cc8_closed_loop_resident_layer_c_model(
            core0_result_stream,
            core1_result_stream,
            position,
            token_count
        );
    } else if (operator_kind == unsigned(CC8_OP_ATTENTION_SUBLAYER)) {
        const unsigned int wave_slots =
            cc8_attention_sublayer_mm_wave_slots(position, token_count);
        const unsigned int core0_tokens =
            token_count < CC8_TOKENS_PER_DATA_PORT ?
            token_count : CC8_TOKENS_PER_DATA_PORT;
        const unsigned int core1_tokens = token_count - core0_tokens;
        const unsigned int vector_packets_per_token =
            2 * ceildiv(HIDDEN_SIZE, CU_VEC_LANES);
        c_model_core0_result_packets =
            wave_slots * MM_STREAM_8X64_PACKETS_PER_BLOCK +
            core0_tokens * vector_packets_per_token;
        c_model_core1_result_packets =
            wave_slots * MM_STREAM_8X64_PACKETS_PER_BLOCK +
            core1_tokens * vector_packets_per_token;
        seed_cc8_closed_loop_attention_sublayer_c_model(
            core0_result_stream,
            core1_result_stream,
            position,
            token_count
        );
    } else if (operator_kind == unsigned(CC8_OP_FFN_SUBLAYER)) {
        const unsigned int wave_slots =
            cc8_ffn_sublayer_mm_wave_slots();
        const unsigned int core0_tokens =
            token_count < CC8_TOKENS_PER_DATA_PORT ?
            token_count : CC8_TOKENS_PER_DATA_PORT;
        const unsigned int core1_tokens = token_count - core0_tokens;
        const unsigned int vector_packets_per_token =
            2 * ceildiv(HIDDEN_SIZE, CU_VEC_LANES) +
            ceildiv(INTERMEDIATE_SIZE, CU_VEC_LANES);
        c_model_core0_result_packets =
            wave_slots * MM_STREAM_8X64_PACKETS_PER_BLOCK +
            core0_tokens * vector_packets_per_token;
        c_model_core1_result_packets =
            wave_slots * MM_STREAM_8X64_PACKETS_PER_BLOCK +
            core1_tokens * vector_packets_per_token;
        seed_cc8_closed_loop_ffn_sublayer_c_model(
            core0_result_stream,
            core1_result_stream,
            token_count
        );
    } else if (vector_case) {
        unsigned int core0_tokens =
            token_count < CC8_TOKENS_PER_DATA_PORT ?
            token_count : CC8_TOKENS_PER_DATA_PORT;
        unsigned int core1_tokens =
            token_count > CC8_TOKENS_PER_DATA_PORT ?
            token_count - CC8_TOKENS_PER_DATA_PORT : 0;
        unsigned int elem_count =
            operator_kind == unsigned(CC8_OP_SILU_MUL) ?
            INTERMEDIATE_SIZE : HIDDEN_SIZE;
        unsigned int blocks_per_token = ceildiv(elem_count, CU_VEC_LANES);
        c_model_core0_result_packets = core0_tokens * blocks_per_token;
        c_model_core1_result_packets = core1_tokens * blocks_per_token;
        seed_cc8_closed_loop_vector_results(
            core0_result_stream,
            0,
            core0_tokens,
            elem_count,
            operator_kind
        );
        seed_cc8_closed_loop_vector_results(
            core1_result_stream,
            CC8_TOKENS_PER_DATA_PORT,
            core1_tokens,
            elem_count,
            operator_kind
        );
    } else if (profiled_mm_case) {
        cc8_operator_spec_t spec;
        (void)cc8_get_operator_spec(
            cc8_operator_t(operator_kind),
            spec
        );
        unsigned int output_waves =
            ceildiv(spec.out_dim, CC8_OUTPUTS_PER_WAVE);
        unsigned int first_wave =
            position < output_waves ? position : output_waves;
        unsigned int requested_waves = tile_len & 0xffffu;
        if (requested_waves == 0u) {
            requested_waves = output_waves - first_wave;
        }
        unsigned int remaining_waves = output_waves - first_wave;
        unsigned int active_waves =
            requested_waves < remaining_waves ?
            requested_waves : remaining_waves;
        unsigned int profile_flags = token_count >> 16;
        unsigned int core_mask = (profile_flags >> 5) & 0x3u;
        if (core_mask == 0u) {
            core_mask = 0x3u;
        }
        c_model_core0_result_packets =
            (core_mask & 1u) != 0u ?
            active_waves * MM_STREAM_8X64_PACKETS_PER_BLOCK : 0;
        c_model_core1_result_packets =
            (core_mask & 2u) != 0u ?
            active_waves * MM_STREAM_8X64_PACKETS_PER_BLOCK : 0;
        seed_cc8_closed_loop_mm_c_model(
            core0_result_stream,
            core1_result_stream,
            first_wave,
            active_waves,
            core_mask
        );
    } else {
        seed_cc8_closed_loop_c_model(
            core0_result_stream,
            core1_result_stream
        );
    }
#endif

    control_cache_8x64_dual_core(
        core0_task_stream,
        core0_activation_stream,
        core0_weight_stream0,
        core0_weight_stream1,
        core0_weight_stream2,
        core0_weight_stream3,
        core0_vector_input0_stream,
        core0_vector_input1_stream,
        core0_result_stream,
        core1_task_stream,
        core1_activation_stream,
        core1_weight_stream0,
        core1_weight_stream1,
        core1_weight_stream2,
        core1_weight_stream3,
        core1_vector_input0_stream,
        core1_vector_input1_stream,
        core1_result_stream,
        status_stream,
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
#if CC8_RESIDENT_LAYER_ONLY && !CC8_RESIDENT_DUAL_VECTOR_PORTS
    declare_cc8_closed_loop_idle_vector_ports(
        core1_vector_input0_stream,
        core1_vector_input1_stream,
        operator_kind
    );
#endif
#ifdef CC8_PREFILL_BLOCK_SYNTH_ONLY
    // Block-prefill schedules only MM/QK/PV work, so all four compute-core
    // vector-input FIFOs are intentionally idle.  As in the production NK
    // wrapper, retain an unreachable producer process so that the closed-loop
    // HLS dataflow graph has exactly one syntactic producer and one consumer
    // for every internal stream without injecting packets for a valid op.
    declare_cc8_closed_loop_idle_vector_ports(
        core0_vector_input0_stream,
        core0_vector_input1_stream,
        operator_kind
    );
    declare_cc8_closed_loop_idle_vector_ports(
        core1_vector_input0_stream,
        core1_vector_input1_stream,
        operator_kind
    );
#endif
    cc8_closed_loop_compute_core0(
        core0_result_stream,
        core0_task_stream,
        core0_activation_stream,
        core0_weight_stream0,
        core0_weight_stream1,
        core0_weight_stream2,
        core0_weight_stream3,
        core0_vector_input0_stream,
        core0_vector_input1_stream
    );
    cc8_closed_loop_compute_core1(
        core1_result_stream,
        core1_task_stream,
        core1_activation_stream,
        core1_weight_stream0,
        core1_weight_stream1,
        core1_weight_stream2,
        core1_weight_stream3,
        core1_vector_input0_stream,
        core1_vector_input1_stream
    );
#ifndef __SYNTHESIS__
    drain_cc8_closed_loop_c_model_results(
        core0_result_stream,
        c_model_core0_result_packets
    );
    drain_cc8_closed_loop_c_model_results(
        core1_result_stream,
        c_model_core1_result_packets
    );
#endif
    cc8_status_sink(status_stream, status_output);
}
