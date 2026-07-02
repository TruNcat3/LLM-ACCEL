#include "control_cache_8x64.hpp"

struct cc8_weight_tile_t {
    wt_linear_t value[CU_VEC_LANES][MM_PE_IN];
};

bool cc8_get_operator_spec(
    cc8_operator_t op,
    cc8_operator_spec_t& spec
) {
    #pragma HLS inline

    spec.uses_mm = false;
    spec.uses_vector = false;
    spec.projection = MM_PROJECTION_Q;
    spec.compute_mode = CU8_MODE_STOP;
    spec.in_dim = 0;
    spec.out_dim = 0;
    spec.vector_inputs = 0;

    switch (op) {
    case CC8_OP_Q_PROJECTION:
        spec.uses_mm = true;
        spec.compute_mode = CU8_MODE_MM;
        spec.projection = MM_PROJECTION_Q;
        break;
    case CC8_OP_K_PROJECTION:
        spec.uses_mm = true;
        spec.compute_mode = CU8_MODE_MM;
        spec.projection = MM_PROJECTION_K;
        break;
    case CC8_OP_V_PROJECTION:
        spec.uses_mm = true;
        spec.compute_mode = CU8_MODE_MM;
        spec.projection = MM_PROJECTION_V;
        break;
    case CC8_OP_O_PROJECTION:
        spec.uses_mm = true;
        spec.compute_mode = CU8_MODE_MM;
        spec.projection = MM_PROJECTION_ATTN_O;
        break;
    case CC8_OP_FFN_GATE:
        spec.uses_mm = true;
        spec.compute_mode = CU8_MODE_MM;
        spec.projection = MM_PROJECTION_FFN_GATE;
        break;
    case CC8_OP_FFN_UP:
        spec.uses_mm = true;
        spec.compute_mode = CU8_MODE_MM;
        spec.projection = MM_PROJECTION_FFN_UP;
        break;
    case CC8_OP_FFN_DOWN:
        spec.uses_mm = true;
        spec.compute_mode = CU8_MODE_MM;
        spec.projection = MM_PROJECTION_FFN_DOWN;
        break;
    case CC8_OP_RMSNORM:
        spec.uses_vector = true;
        spec.compute_mode = CU8_MODE_RMSNORM;
        spec.in_dim = HIDDEN_SIZE;
        spec.out_dim = HIDDEN_SIZE;
        spec.vector_inputs = 2;
        return true;
    case CC8_OP_SILU_MUL:
        spec.uses_vector = true;
        spec.compute_mode = CU8_MODE_SILU_MUL;
        spec.in_dim = INTERMEDIATE_SIZE;
        spec.out_dim = INTERMEDIATE_SIZE;
        spec.vector_inputs = 2;
        return true;
    case CC8_OP_RESIDUAL_ADD:
        spec.uses_vector = true;
        spec.compute_mode = CU8_MODE_RESIDUAL_ADD;
        spec.in_dim = HIDDEN_SIZE;
        spec.out_dim = HIDDEN_SIZE;
        spec.vector_inputs = 2;
        return true;
    case CC8_OP_ATTN_QK:
        spec.compute_mode = CU8_MODE_MM_SCALE;
        spec.in_dim = HEAD_DIM;
        spec.out_dim = CC8_ATTN_TILE;
        return true;
    case CC8_OP_ATTN_PV:
        spec.compute_mode = CU8_MODE_MM;
        spec.in_dim = CC8_ATTN_TILE;
        spec.out_dim = HEAD_DIM;
        return true;
    case CC8_OP_SOFTMAX:
        spec.compute_mode = CU8_MODE_SOFTMAX;
        spec.in_dim = CC8_ATTN_TILE;
        spec.out_dim = CC8_ATTN_TILE;
        spec.uses_vector = true;
        return true;
    case CC8_OP_NOP:
        return true;
    default:
        return false;
    }

    mm_projection_spec_t projection_spec;
    get_mm_projection_spec(spec.projection, projection_spec);
    spec.in_dim = projection_spec.in_dim;
    spec.out_dim = projection_spec.out_dim;
    return true;
}

static fm_word_t read_cc8_feature_word(
    unsigned int port,
    unsigned int word_idx,
    const fm_word_t port0[CC8_DATA_PORT_WORDS],
    const fm_word_t port1[CC8_DATA_PORT_WORDS]
) {
    #pragma HLS inline
    return port == 0 ? port0[word_idx] : port1[word_idx];
}

static void load_cc8_feature_gbuf(
    cc8_global_buffer_t& gbuf,
    const fm_word_t port0[CC8_DATA_PORT_WORDS],
    const fm_word_t port1[CC8_DATA_PORT_WORDS],
    unsigned int token_slots,
    unsigned int valid_tokens,
    unsigned int elem_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf.block complete dim=1

    unsigned int words_per_token = ceildiv(elem_count, FM_BLOCK_SIZE);
    for (unsigned int token = 0; token < LINEAR_TOKEN_TILE_ACTIVE; token++) {
        unsigned int port = token / CC8_TOKENS_PER_DATA_PORT;
        unsigned int token_in_port =
            token - port * CC8_TOKENS_PER_DATA_PORT;
        for (unsigned int word_idx = 0;
             word_idx < CC8_FEATURE_WORDS_PER_TOKEN;
             word_idx++) {
            #pragma HLS pipeline II=1
            if (token < token_slots && word_idx < words_per_token) {
                fm_word_t word = 0;
                if (token < valid_tokens) {
                    unsigned int port_word =
                        token_in_port * CC8_FEATURE_WORDS_PER_TOKEN +
                        word_idx;
                    word = read_cc8_feature_word(
                        port,
                        port_word,
                        port0,
                        port1
                    );
                }

                unsigned int block0 = 2 * word_idx;
                unsigned int block1 = block0 + 1;
                if (block0 < CC8_GBUF_BLOCKS) {
                    gbuf.block[token][block0] = word.range(255, 0);
                }
                if (block1 < CC8_GBUF_BLOCKS) {
                    gbuf.block[token][block1] = word.range(511, 256);
                }
            }
        }
    }
}

static void clear_cc8_gbuf(
    cc8_global_buffer_t& gbuf,
    unsigned int elem_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf.block complete dim=1

    unsigned int block_count = ceildiv(elem_count, MM_PE_IN);
    for (unsigned int token = 0; token < LINEAR_TOKEN_TILE_ACTIVE; token++) {
        for (unsigned int block = 0; block < CC8_GBUF_BLOCKS; block++) {
            #pragma HLS pipeline II=1
            if (block < block_count) {
                gbuf.block[token][block] = 0;
            }
        }
    }
}

template <unsigned int GROUP>
static void read_cc8_weight_tile_words(
    wt_block_t packed[MM_TILE_WEIGHT_BLOCKS],
    weight_addr_t local_block,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline
    #pragma HLS array_partition variable=packed complete dim=1

    if (GROUP == 0) {
        packed[0] = layer_weights_shard0[local_block];
        packed[1] = layer_weights_shard1[local_block];
        packed[2] = layer_weights_shard2[local_block];
        packed[3] = layer_weights_shard3[local_block];
        packed[4] = layer_weights_shard4[local_block];
        packed[5] = layer_weights_shard5[local_block];
        packed[6] = layer_weights_shard6[local_block];
        packed[7] = layer_weights_shard7[local_block];
    } else {
        packed[0] = layer_weights_shard8[local_block];
        packed[1] = layer_weights_shard9[local_block];
        packed[2] = layer_weights_shard10[local_block];
        packed[3] = layer_weights_shard11[local_block];
        packed[4] = layer_weights_shard12[local_block];
        packed[5] = layer_weights_shard13[local_block];
        packed[6] = layer_weights_shard14[local_block];
        packed[7] = layer_weights_shard15[local_block];
    }
}

static void clear_cc8_weight_tile(
    cc8_weight_tile_t& tile
) {
    #pragma HLS inline
    #pragma HLS array_partition variable=tile.value complete dim=0

    for (unsigned int out_lane = 0; out_lane < CU_VEC_LANES; out_lane++) {
        #pragma HLS unroll
        for (unsigned int k_lane = 0; k_lane < MM_PE_IN; k_lane++) {
            #pragma HLS unroll
            tile.value[out_lane][k_lane] = wt_linear_t(0);
        }
    }
}

template <unsigned int GROUP>
static void load_cc8_weight_tile(
    cc8_weight_tile_t& tile,
    weight_addr_t local_block,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline
    #pragma HLS array_partition variable=tile.value complete dim=0

    wt_block_t packed[MM_TILE_WEIGHT_BLOCKS];
    #pragma HLS array_partition variable=packed complete dim=1
    read_cc8_weight_tile_words<GROUP>(
        packed,
        local_block,
        QWEN_WEIGHT_SHARD_ARGS
    );

    for (unsigned int out_lane = 0; out_lane < CU_VEC_LANES; out_lane++) {
        #pragma HLS unroll
        for (unsigned int k_lane = 0; k_lane < MM_PE_IN; k_lane++) {
            #pragma HLS unroll
            unsigned int lane_in_tile =
                out_lane * MM_PE_IN + k_lane;
            unsigned int word = lane_in_tile / WT_BLOCK_SIZE;
            unsigned int lane = lane_in_tile - word * WT_BLOCK_SIZE;
            tile.value[out_lane][k_lane] =
                unpack_wt_block_lane(packed[word], lane);
        }
    }
}

static void load_cc8_weight_panels(
    hls::stream<cc8_weight_tile_t>
        tile_stream[CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS],
    unsigned int output_wave,
    unsigned int in_dim,
    unsigned int out_dim,
    weight_addr_t weight_base,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=tile_stream complete dim=0

    unsigned int in_tile_count = ceildiv(in_dim, MM_PE_IN);
    weight_addr_t matrix_tile_base =
        weight_base / weight_addr_t(MM_TILE_WEIGHT_ELEMS);

    for (unsigned int in_tile = 0;
         in_tile < MAX_LINEAR_IN_BLOCKS;
        in_tile++) {
        if (in_tile < in_tile_count) {
            for (unsigned int tile_slot = 0;
                 tile_slot < CC8_OUTPUT_TILES_PER_WAVE;
                 tile_slot++) {
                #pragma HLS pipeline II=1
                cc8_weight_tile_t tile;
                #pragma HLS array_partition variable=tile.value complete dim=0
                unsigned int out_tile =
                    output_wave * CC8_OUTPUT_TILES_PER_WAVE +
                    tile_slot;
                unsigned int out_base = out_tile * MM_PE_OUT;

                if (out_base < out_dim) {
                    weight_addr_t global_tile =
                        matrix_tile_base +
                        weight_addr_t(out_tile) *
                            weight_addr_t(in_tile_count) +
                        weight_addr_t(in_tile);
                    weight_addr_t local_block = global_tile >> 1;
                    if ((global_tile & weight_addr_t(1)) == 0) {
                        load_cc8_weight_tile<0>(
                            tile,
                            local_block,
                            QWEN_WEIGHT_SHARD_ARGS
                        );
                    } else {
                        load_cc8_weight_tile<1>(
                            tile,
                            local_block,
                            QWEN_WEIGHT_SHARD_ARGS
                        );
                    }
                } else {
                    clear_cc8_weight_tile(tile);
                }

                switch (tile_slot) {
                case 0:
                    tile_stream[0][0].write(tile);
                    break;
                case 1:
                    tile_stream[0][1].write(tile);
                    break;
                case 2:
                    tile_stream[0][2].write(tile);
                    break;
                case 3:
                    tile_stream[0][3].write(tile);
                    break;
                case 4:
                    tile_stream[1][0].write(tile);
                    break;
                case 5:
                    tile_stream[1][1].write(tile);
                    break;
                case 6:
                    tile_stream[1][2].write(tile);
                    break;
                default:
                    tile_stream[1][3].write(tile);
                    break;
                }
            }
        }
    }
}

static void emit_cc8_mm_wave_inputs(
    hls::stream<mm_stream_8x64_activation_packet_t>& core0_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream3,
    hls::stream<mm_stream_8x64_activation_packet_t>& core1_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream3,
    hls::stream<cc8_weight_tile_t>
        tile_stream[CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS],
    const cc8_global_buffer_t& source,
    unsigned int token_count,
    unsigned int in_dim
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=source.block complete dim=1
    #pragma HLS array_partition variable=tile_stream complete dim=0

    cc8_weight_tile_t tiles
        [CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS];
    #pragma HLS array_partition variable=tiles complete dim=0

    for (unsigned int k = 0; k < MAX_LINEAR_IN_DIM; k++) {
        #pragma HLS pipeline II=1
        if (k < in_dim) {
            unsigned int in_tile = k / MM_PE_IN;
            unsigned int k_lane = k - in_tile * MM_PE_IN;
            if (k_lane == 0) {
                tiles[0][0] = tile_stream[0][0].read();
                tiles[0][1] = tile_stream[0][1].read();
                tiles[0][2] = tile_stream[0][2].read();
                tiles[0][3] = tile_stream[0][3].read();
                tiles[1][0] = tile_stream[1][0].read();
                tiles[1][1] = tile_stream[1][1].read();
                tiles[1][2] = tile_stream[1][2].read();
                tiles[1][3] = tile_stream[1][3].read();
            }

            mm_stream_8x64_activation_packet_t activation;
            #pragma HLS array_partition variable=activation.data complete dim=1
            for (unsigned int token = 0;
                 token < MM_STREAM_8X64_TOKENS;
                 token++) {
                #pragma HLS unroll
                activation.data[token] =
                    token < token_count ?
                    unpack_mm_input_block_lane(
                        source.block[token][in_tile],
                        k_lane
                    ) :
                    fm_t(0);
            }

            mm_stream_8x64_weight_packet_t c0w0;
            mm_stream_8x64_weight_packet_t c0w1;
            mm_stream_8x64_weight_packet_t c0w2;
            mm_stream_8x64_weight_packet_t c0w3;
            mm_stream_8x64_weight_packet_t c1w0;
            mm_stream_8x64_weight_packet_t c1w1;
            mm_stream_8x64_weight_packet_t c1w2;
            mm_stream_8x64_weight_packet_t c1w3;
            #pragma HLS array_partition variable=c0w0.data complete dim=1
            #pragma HLS array_partition variable=c0w1.data complete dim=1
            #pragma HLS array_partition variable=c0w2.data complete dim=1
            #pragma HLS array_partition variable=c0w3.data complete dim=1
            #pragma HLS array_partition variable=c1w0.data complete dim=1
            #pragma HLS array_partition variable=c1w1.data complete dim=1
            #pragma HLS array_partition variable=c1w2.data complete dim=1
            #pragma HLS array_partition variable=c1w3.data complete dim=1

            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS unroll
                c0w0.data[lane] = tiles[0][0].value[lane][k_lane];
                c0w1.data[lane] = tiles[0][1].value[lane][k_lane];
                c0w2.data[lane] = tiles[0][2].value[lane][k_lane];
                c0w3.data[lane] = tiles[0][3].value[lane][k_lane];
                c1w0.data[lane] = tiles[1][0].value[lane][k_lane];
                c1w1.data[lane] = tiles[1][1].value[lane][k_lane];
                c1w2.data[lane] = tiles[1][2].value[lane][k_lane];
                c1w3.data[lane] = tiles[1][3].value[lane][k_lane];
            }

            core0_activation_stream.write(activation);
            core1_activation_stream.write(activation);
            core0_weight_stream0.write(c0w0);
            core0_weight_stream1.write(c0w1);
            core0_weight_stream2.write(c0w2);
            core0_weight_stream3.write(c0w3);
            core1_weight_stream0.write(c1w0);
            core1_weight_stream1.write(c1w1);
            core1_weight_stream2.write(c1w2);
            core1_weight_stream3.write(c1w3);
        }
    }
}

static void store_cc8_result_packet(
    cc8_global_buffer_t& destination,
    const cu_vec16_packet_t& packet,
    unsigned int token_count,
    unsigned int out_dim
) {
    #pragma HLS inline
    #pragma HLS array_partition variable=destination.block complete dim=1

    if (packet.token_lane < token_count) {
        mm_input_block_t packed = 0;
        for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
            #pragma HLS unroll
            unsigned int elem = packet.elem_base + lane;
            fm_t value =
                elem < out_dim && packet.valid_mask[lane] ?
                packet.data[lane] :
                fm_t(0);
            set_mm_input_block_lane(packed, lane, value);
        }
        unsigned int block = packet.elem_base / MM_PE_IN;
        if (block < CC8_GBUF_BLOCKS) {
            destination.block[packet.token_lane][block] = packed;
        }
    }
}

static void collect_cc8_mm_wave_results(
    cc8_global_buffer_t& destination,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int token_count,
    unsigned int out_dim
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=destination.block complete dim=1

    for (unsigned int packet_idx = 0;
         packet_idx < MM_STREAM_8X64_PACKETS_PER_BLOCK;
         packet_idx++) {
        #pragma HLS pipeline II=1
        cu_vec16_packet_t packet0 = core0_result_stream.read();
        cu_vec16_packet_t packet1 = core1_result_stream.read();
        store_cc8_result_packet(
            destination,
            packet0,
            token_count,
            out_dim
        );
        store_cc8_result_packet(
            destination,
            packet1,
            token_count,
            out_dim
        );
    }
}

static void run_cc8_mm_wave(
    hls::stream<mm_stream_8x64_activation_packet_t>& core0_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core1_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    const cc8_global_buffer_t& source,
    cc8_global_buffer_t& destination,
    unsigned int output_wave,
    unsigned int token_count,
    unsigned int in_dim,
    unsigned int out_dim,
    weight_addr_t weight_base,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline off

    hls::stream<cc8_weight_tile_t>
        tile_stream[CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS];
    #pragma HLS array_partition variable=tile_stream complete dim=0
    #pragma HLS stream variable=tile_stream depth=2
    #pragma HLS dataflow

    load_cc8_weight_panels(
        tile_stream,
        output_wave,
        in_dim,
        out_dim,
        weight_base,
        QWEN_WEIGHT_SHARD_ARGS
    );
    emit_cc8_mm_wave_inputs(
        core0_activation_stream,
        core0_weight_stream0,
        core0_weight_stream1,
        core0_weight_stream2,
        core0_weight_stream3,
        core1_activation_stream,
        core1_weight_stream0,
        core1_weight_stream1,
        core1_weight_stream2,
        core1_weight_stream3,
        tile_stream,
        source,
        token_count,
        in_dim
    );
    collect_cc8_mm_wave_results(
        destination,
        core0_result_stream,
        core1_result_stream,
        token_count,
        out_dim
    );
}

static void write_cc8_output_word(
    fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT],
    unsigned int token,
    unsigned int word_idx,
    fm_word_t word
) {
    #pragma HLS inline

    unsigned int port = token / CC8_TOKENS_PER_DATA_PORT;
    unsigned int token_in_port =
        token - port * CC8_TOKENS_PER_DATA_PORT;
    unsigned int port_word =
        token_in_port * CC8_FEATURE_WORDS_PER_TOKEN + word_idx;
    if (port == 0) {
        output_port0[port_word] = word;
    } else {
        output_port1[port_word] = word;
    }
}

static void store_cc8_gbuf_to_hbm(
    fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT],
    const cc8_global_buffer_t& gbuf,
    unsigned int elem_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf.block complete dim=1

    unsigned int words_per_token = ceildiv(elem_count, FM_BLOCK_SIZE);
    unsigned int blocks_per_token = ceildiv(elem_count, MM_PE_IN);
    for (unsigned int token = 0; token < LINEAR_TOKEN_TILE_ACTIVE; token++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_FEATURE_WORDS_PER_TOKEN;
             word_idx++) {
            #pragma HLS pipeline II=1
            if (word_idx < words_per_token) {
                unsigned int block0 = 2 * word_idx;
                unsigned int block1 = block0 + 1;
                fm_word_t word = 0;
                if (block0 < blocks_per_token) {
                    word.range(255, 0) = gbuf.block[token][block0];
                }
                if (block1 < blocks_per_token) {
                    word.range(511, 256) = gbuf.block[token][block1];
                }
                write_cc8_output_word(
                    output_port0,
                    output_port1,
                    token,
                    word_idx,
                    word
                );
            }
        }
    }
}

static cu_vec16_packet_t build_cc8_vec_packet(
    const cc8_global_buffer_t& gbuf,
    unsigned int token,
    unsigned int valid_tokens,
    unsigned int elem_count,
    unsigned int block,
    unsigned int block_id,
    bool last_block,
    bool last_stream
) {
    #pragma HLS inline
    #pragma HLS array_partition variable=gbuf.block complete dim=1

    cu_vec16_packet_t packet;
    packet.valid_mask = 0;
    packet.token_lane = token;
    packet.elem_base = block * CU_VEC_LANES;
    packet.block_id = block_id;
    packet.last_block = last_block;
    packet.last_stream = last_stream && last_block;
    mm_input_block_t packed = gbuf.block[token][block];
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        #pragma HLS unroll
        unsigned int elem = packet.elem_base + lane;
        bool valid = token < valid_tokens && elem < elem_count;
        packet.data[lane] = valid ?
            unpack_mm_input_block_lane(packed, lane) :
            fm_t(0);
        packet.valid_mask[lane] = valid;
    }
    return packet;
}

static void emit_cc8_vec_range_from_gbuf(
    hls::stream<cu_vec16_packet_t>& stream,
    const cc8_global_buffer_t& gbuf,
    unsigned int token_begin,
    unsigned int token_slots,
    unsigned int valid_tokens,
    unsigned int elem_count,
    unsigned int block_id,
    bool last_stream
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf.block complete dim=1

    unsigned int block_count = ceildiv(elem_count, CU_VEC_LANES);
    for (unsigned int token = 0; token < LINEAR_TOKEN_TILE_ACTIVE; token++) {
        for (unsigned int block = 0; block < MAX_LINEAR_IN_BLOCKS; block++) {
            #pragma HLS pipeline II=1
            bool token_active =
                token >= token_begin &&
                token < token_begin + token_slots;
            if (token_active && block < block_count) {
                bool is_last =
                    token + 1 == token_begin + token_slots &&
                    block + 1 == block_count;
                stream.write(build_cc8_vec_packet(
                    gbuf,
                    token,
                    valid_tokens,
                    elem_count,
                    block,
                    block_id,
                    is_last,
                    last_stream
                ));
            }
        }
    }
}

static void emit_cc8_vec_pair_from_gbuf(
    hls::stream<cu_vec16_packet_t>& stream0,
    hls::stream<cu_vec16_packet_t>& stream1,
    const cc8_global_buffer_t& gbuf0,
    const cc8_global_buffer_t& gbuf1,
    unsigned int token_begin,
    unsigned int token_slots,
    unsigned int valid_tokens,
    unsigned int elem_count,
    unsigned int block_id,
    bool last_stream
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf0.block complete dim=1
    #pragma HLS array_partition variable=gbuf1.block complete dim=1

    unsigned int block_count = ceildiv(elem_count, CU_VEC_LANES);
    for (unsigned int token = 0; token < LINEAR_TOKEN_TILE_ACTIVE; token++) {
        for (unsigned int block = 0; block < MAX_LINEAR_IN_BLOCKS; block++) {
            #pragma HLS pipeline II=1
            bool token_active =
                token >= token_begin &&
                token < token_begin + token_slots;
            if (token_active && block < block_count) {
                bool is_last =
                    token + 1 == token_begin + token_slots &&
                    block + 1 == block_count;
                stream0.write(build_cc8_vec_packet(
                    gbuf0,
                    token,
                    valid_tokens,
                    elem_count,
                    block,
                    block_id,
                    is_last,
                    last_stream
                ));
                stream1.write(build_cc8_vec_packet(
                    gbuf1,
                    token,
                    valid_tokens,
                    elem_count,
                    block,
                    block_id,
                    is_last,
                    last_stream
                ));
            }
        }
    }
}

static void collect_cc8_vector_results(
    fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT],
    hls::stream<cu_vec16_packet_t>& result_stream,
    unsigned int packet_count
) {
    #pragma HLS inline off

    fm_word_t pending_word = 0;
    unsigned int pending_token = 0;
    unsigned int pending_word_idx = 0;
    bool pending_valid = false;

    for (unsigned int packet_idx = 0;
         packet_idx < CU_STREAM_MAX_PACKETS;
         packet_idx++) {
        if (packet_idx < packet_count) {
            cu_vec16_packet_t packet = result_stream.read();
            unsigned int word_idx = packet.elem_base / FM_BLOCK_SIZE;
            unsigned int half =
                (packet.elem_base % FM_BLOCK_SIZE) / CU_VEC_LANES;
            bool same_word =
                pending_valid &&
                pending_token == packet.token_lane &&
                pending_word_idx == word_idx;
            if (pending_valid && !same_word) {
                write_cc8_output_word(
                    output_port0,
                    output_port1,
                    pending_token,
                    pending_word_idx,
                    pending_word
                );
                pending_valid = false;
            }
            if (!pending_valid) {
                pending_word = 0;
                pending_token = packet.token_lane;
                pending_word_idx = word_idx;
                pending_valid = true;
            }

            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS pipeline II=1
                unsigned int word_lane = half * CU_VEC_LANES + lane;
                fm_t value = packet.valid_mask[lane] ?
                    packet.data[lane] :
                    fm_t(0);
                set_fm_word_lane(pending_word, word_lane, value);
            }

            if (half == 1 || packet.last_stream) {
                write_cc8_output_word(
                    output_port0,
                    output_port1,
                    pending_token,
                    pending_word_idx,
                    pending_word
                );
                pending_valid = false;
            }
        }
    }

    if (pending_valid) {
        write_cc8_output_word(
            output_port0,
            output_port1,
            pending_token,
            pending_word_idx,
            pending_word
        );
    }
}

static cu8_task_t build_cc8_compute_task(
    cu8_mode_t mode,
    unsigned int k_count,
    unsigned int token_count,
    unsigned int elem_count,
    unsigned int packet_count,
    unsigned int elem_base,
    unsigned int block_id,
    bool last_task
) {
    #pragma HLS inline

    cu8_task_t task;
    task.mode = mode;
    task.k_count = k_count;
    task.token_count = token_count;
    task.elem_count = elem_count;
    task.packet_count = packet_count;
    task.elem_base = elem_base;
    task.block_id = block_id;
    task.repeat_count = 1;
    task.elem_stride = 0;
    task.block_stride = 0;
    task.output_scale = fm_t(1);
    task.last_task = last_task;
    return task;
}

static cu8_task_t build_cc8_stop_task() {
    #pragma HLS inline
    return build_cc8_compute_task(
        CU8_MODE_STOP,
        0,
        0,
        0,
        0,
        0,
        0,
        true
    );
}

static void load_cc8_flat_rows(
    cc8_attention_buffer_t& gbuf,
    const fm_word_t words[CC8_DATA_PORT_WORDS],
    unsigned int row_count,
    unsigned int elem_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf.block complete dim=1

    unsigned int words_per_row = ceildiv(elem_count, FM_BLOCK_SIZE);
    for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_ATTN_BUFFER_WORDS;
             word_idx++) {
            #pragma HLS pipeline II=1
            if (word_idx < words_per_row) {
                fm_word_t word =
                    row < row_count ?
                    words[row * words_per_row + word_idx] :
                    fm_word_t(0);
                unsigned int block0 = 2 * word_idx;
                unsigned int block1 = block0 + 1;
                if (block0 < CC8_ATTN_BUFFER_BLOCKS) {
                    gbuf.block[row][block0] = word.range(255, 0);
                }
                if (block1 < CC8_ATTN_BUFFER_BLOCKS) {
                    gbuf.block[row][block1] = word.range(511, 256);
                }
            }
        }
    }
}

static void load_cc8_attention_panel(
    cc8_attention_panel_t& panel,
    const fm_word_t words[CC8_DATA_PORT_WORDS],
    unsigned int tile_len
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=panel.word complete dim=1

    for (unsigned int pos = 0; pos < CC8_ATTN_TILE; pos++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_HEAD_WORDS;
             word_idx++) {
            #pragma HLS pipeline II=1
            panel.word[pos][word_idx] =
                pos < tile_len ?
                words[pos * CC8_HEAD_WORDS + word_idx] :
                fm_word_t(0);
        }
    }
}

static fm_t read_cc8_gbuf_value(
    const cc8_attention_buffer_t& gbuf,
    unsigned int row,
    unsigned int elem
) {
    #pragma HLS inline
    unsigned int block = elem / MM_PE_IN;
    unsigned int lane = elem % MM_PE_IN;
    return unpack_mm_input_block_lane(gbuf.block[row][block], lane);
}

static fm_t read_cc8_attention_panel_value(
    const cc8_attention_panel_t& panel,
    unsigned int pos,
    unsigned int elem
) {
    #pragma HLS inline
    unsigned int word = elem / FM_BLOCK_SIZE;
    unsigned int lane = elem % FM_BLOCK_SIZE;
    return unpack_fm_word_lane(panel.word[pos][word], lane);
}

static void emit_cc8_attention_qk_inputs(
    hls::stream<mm_stream_8x64_activation_packet_t>& core0_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream3,
    hls::stream<mm_stream_8x64_activation_packet_t>& core1_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream3,
    const cc8_attention_buffer_t& q0,
    const cc8_attention_buffer_t& q1,
    const cc8_attention_panel_t& k0,
    const cc8_attention_panel_t& k1,
    unsigned int tile_len
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=q0.block complete dim=1
    #pragma HLS array_partition variable=q1.block complete dim=1
    #pragma HLS array_partition variable=k0.word complete dim=1
    #pragma HLS array_partition variable=k1.word complete dim=1

    for (unsigned int k = 0; k < HEAD_DIM; k++) {
        #pragma HLS pipeline II=1
        mm_stream_8x64_activation_packet_t activation0;
        mm_stream_8x64_activation_packet_t activation1;
        mm_stream_8x64_weight_packet_t c0w0;
        mm_stream_8x64_weight_packet_t c0w1;
        mm_stream_8x64_weight_packet_t c0w2;
        mm_stream_8x64_weight_packet_t c0w3;
        mm_stream_8x64_weight_packet_t c1w0;
        mm_stream_8x64_weight_packet_t c1w1;
        mm_stream_8x64_weight_packet_t c1w2;
        mm_stream_8x64_weight_packet_t c1w3;
        #pragma HLS array_partition variable=activation0.data complete dim=1
        #pragma HLS array_partition variable=activation1.data complete dim=1
        #pragma HLS array_partition variable=c0w0.data complete dim=1
        #pragma HLS array_partition variable=c0w1.data complete dim=1
        #pragma HLS array_partition variable=c0w2.data complete dim=1
        #pragma HLS array_partition variable=c0w3.data complete dim=1
        #pragma HLS array_partition variable=c1w0.data complete dim=1
        #pragma HLS array_partition variable=c1w1.data complete dim=1
        #pragma HLS array_partition variable=c1w2.data complete dim=1
        #pragma HLS array_partition variable=c1w3.data complete dim=1

        for (unsigned int row = 0;
             row < MM_STREAM_8X64_TOKENS;
             row++) {
            #pragma HLS unroll
            activation0.data[row] =
                row < GQA_GROUP_SIZE ?
                read_cc8_gbuf_value(q0, row, k) :
                fm_t(0);
            activation1.data[row] =
                row < GQA_GROUP_SIZE ?
                read_cc8_gbuf_value(q1, row, k) :
                fm_t(0);
        }

        for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
            #pragma HLS unroll
            unsigned int pos0 = lane;
            unsigned int pos1 = CU_VEC_LANES + lane;
            unsigned int pos2 = 2 * CU_VEC_LANES + lane;
            unsigned int pos3 = 3 * CU_VEC_LANES + lane;
            c0w0.data[lane] = pos0 < tile_len ?
                wt_linear_t(read_cc8_attention_panel_value(k0, pos0, k)) :
                wt_linear_t(0);
            c0w1.data[lane] = pos1 < tile_len ?
                wt_linear_t(read_cc8_attention_panel_value(k0, pos1, k)) :
                wt_linear_t(0);
            c0w2.data[lane] = pos2 < tile_len ?
                wt_linear_t(read_cc8_attention_panel_value(k0, pos2, k)) :
                wt_linear_t(0);
            c0w3.data[lane] = pos3 < tile_len ?
                wt_linear_t(read_cc8_attention_panel_value(k0, pos3, k)) :
                wt_linear_t(0);
            c1w0.data[lane] = pos0 < tile_len ?
                wt_linear_t(read_cc8_attention_panel_value(k1, pos0, k)) :
                wt_linear_t(0);
            c1w1.data[lane] = pos1 < tile_len ?
                wt_linear_t(read_cc8_attention_panel_value(k1, pos1, k)) :
                wt_linear_t(0);
            c1w2.data[lane] = pos2 < tile_len ?
                wt_linear_t(read_cc8_attention_panel_value(k1, pos2, k)) :
                wt_linear_t(0);
            c1w3.data[lane] = pos3 < tile_len ?
                wt_linear_t(read_cc8_attention_panel_value(k1, pos3, k)) :
                wt_linear_t(0);
        }

        core0_activation_stream.write(activation0);
        core1_activation_stream.write(activation1);
        core0_weight_stream0.write(c0w0);
        core0_weight_stream1.write(c0w1);
        core0_weight_stream2.write(c0w2);
        core0_weight_stream3.write(c0w3);
        core1_weight_stream0.write(c1w0);
        core1_weight_stream1.write(c1w1);
        core1_weight_stream2.write(c1w2);
        core1_weight_stream3.write(c1w3);
    }
}

static void emit_cc8_attention_pv_inputs(
    hls::stream<mm_stream_8x64_activation_packet_t>& core0_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream3,
    hls::stream<mm_stream_8x64_activation_packet_t>& core1_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream3,
    const cc8_attention_buffer_t& p0,
    const cc8_attention_buffer_t& p1,
    const cc8_attention_panel_t& v0,
    const cc8_attention_panel_t& v1,
    unsigned int output_wave,
    unsigned int tile_len
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=p0.block complete dim=1
    #pragma HLS array_partition variable=p1.block complete dim=1
    #pragma HLS array_partition variable=v0.word complete dim=1
    #pragma HLS array_partition variable=v1.word complete dim=1

    for (unsigned int pos = 0; pos < CC8_ATTN_TILE; pos++) {
        #pragma HLS pipeline II=1
        if (pos < tile_len) {
            mm_stream_8x64_activation_packet_t activation0;
            mm_stream_8x64_activation_packet_t activation1;
            mm_stream_8x64_weight_packet_t weights0[4];
            mm_stream_8x64_weight_packet_t weights1[4];
            #pragma HLS array_partition variable=activation0.data complete dim=1
            #pragma HLS array_partition variable=activation1.data complete dim=1
            #pragma HLS array_partition variable=weights0 complete dim=0
            #pragma HLS array_partition variable=weights1 complete dim=0

            for (unsigned int row = 0;
                 row < MM_STREAM_8X64_TOKENS;
                 row++) {
                #pragma HLS unroll
                activation0.data[row] =
                    row < GQA_GROUP_SIZE ?
                    read_cc8_gbuf_value(p0, row, pos) :
                    fm_t(0);
                activation1.data[row] =
                    row < GQA_GROUP_SIZE ?
                    read_cc8_gbuf_value(p1, row, pos) :
                    fm_t(0);
            }
            for (unsigned int group = 0; group < 4; group++) {
                #pragma HLS unroll
                for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                    #pragma HLS unroll
                    unsigned int out =
                        output_wave * MM_STREAM_8X64_OUTPUTS +
                        group * CU_VEC_LANES +
                        lane;
                    weights0[group].data[lane] =
                        out < HEAD_DIM ?
                        wt_linear_t(
                            read_cc8_attention_panel_value(v0, pos, out)
                        ) :
                        wt_linear_t(0);
                    weights1[group].data[lane] =
                        out < HEAD_DIM ?
                        wt_linear_t(
                            read_cc8_attention_panel_value(v1, pos, out)
                        ) :
                        wt_linear_t(0);
                }
            }

            core0_activation_stream.write(activation0);
            core1_activation_stream.write(activation1);
            core0_weight_stream0.write(weights0[0]);
            core0_weight_stream1.write(weights0[1]);
            core0_weight_stream2.write(weights0[2]);
            core0_weight_stream3.write(weights0[3]);
            core1_weight_stream0.write(weights1[0]);
            core1_weight_stream1.write(weights1[1]);
            core1_weight_stream2.write(weights1[2]);
            core1_weight_stream3.write(weights1[3]);
        }
    }
}

static void collect_cc8_separate_mm_results(
    cc8_attention_buffer_t& destination0,
    cc8_attention_buffer_t& destination1,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int row_count,
    unsigned int elem_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=destination0.block complete dim=1
    #pragma HLS array_partition variable=destination1.block complete dim=1

    for (unsigned int packet = 0;
         packet < MM_STREAM_8X64_PACKETS_PER_BLOCK;
         packet++) {
        #pragma HLS pipeline II=1
        cu_vec16_packet_t packet0 = core0_result_stream.read();
        cu_vec16_packet_t packet1 = core1_result_stream.read();
        if (packet0.token_lane < row_count) {
            mm_input_block_t packed0 = 0;
            mm_input_block_t packed1 = 0;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS unroll
                unsigned int elem0 = packet0.elem_base + lane;
                unsigned int elem1 = packet1.elem_base + lane;
                set_mm_input_block_lane(
                    packed0,
                    lane,
                    elem0 < elem_count && packet0.valid_mask[lane] ?
                    packet0.data[lane] :
                    fm_t(0)
                );
                set_mm_input_block_lane(
                    packed1,
                    lane,
                    elem1 < elem_count && packet1.valid_mask[lane] ?
                    packet1.data[lane] :
                    fm_t(0)
                );
            }
            destination0.block[packet0.token_lane]
                [packet0.elem_base / MM_PE_IN] = packed0;
            destination1.block[packet1.token_lane]
                [packet1.elem_base / MM_PE_IN] = packed1;
        }
    }
}

static void store_cc8_flat_rows(
    fm_word_t output[CC8_FEATURE_WORDS_PER_PORT],
    const cc8_attention_buffer_t& gbuf,
    unsigned int row_count,
    unsigned int elem_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf.block complete dim=1

    unsigned int words_per_row = ceildiv(elem_count, FM_BLOCK_SIZE);
    for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_ATTN_BUFFER_WORDS;
             word_idx++) {
            #pragma HLS pipeline II=1
            if (row < row_count && word_idx < words_per_row) {
                fm_word_t word = 0;
                unsigned int block0 = 2 * word_idx;
                unsigned int block1 = block0 + 1;
                word.range(255, 0) = gbuf.block[row][block0];
                if (block1 < ceildiv(elem_count, MM_PE_IN)) {
                    word.range(511, 256) = gbuf.block[row][block1];
                }
                output[row * words_per_row + word_idx] = word;
            }
        }
    }
}

static void emit_cc8_attention_vectors(
    hls::stream<cu_vec16_packet_t>& stream,
    const cc8_attention_buffer_t& gbuf,
    unsigned int row_count,
    unsigned int elem_count,
    bool last_stream
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf.block complete dim=1

    unsigned int block_count = ceildiv(elem_count, CU_VEC_LANES);
    for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
        for (unsigned int block = 0;
             block < CC8_ATTN_BUFFER_BLOCKS;
             block++) {
            #pragma HLS pipeline II=1
            if (row < row_count && block < block_count) {
                cu_vec16_packet_t packet;
                packet.valid_mask = 0;
                packet.token_lane = row;
                packet.elem_base = block * CU_VEC_LANES;
                packet.block_id = 0;
                packet.last_block =
                    row + 1 == row_count &&
                    block + 1 == block_count;
                packet.last_stream = last_stream && packet.last_block;
                mm_input_block_t packed = gbuf.block[row][block];
                for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                    #pragma HLS unroll
                    unsigned int elem = packet.elem_base + lane;
                    bool valid = elem < elem_count;
                    packet.data[lane] = valid ?
                        unpack_mm_input_block_lane(packed, lane) :
                        fm_t(0);
                    packet.valid_mask[lane] = valid;
                }
                stream.write(packet);
            }
        }
    }
}

static void collect_cc8_attention_vectors(
    cc8_attention_buffer_t& destination,
    hls::stream<cu_vec16_packet_t>& result_stream,
    unsigned int row_count,
    unsigned int elem_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=destination.block complete dim=1

    unsigned int packet_count =
        row_count * ceildiv(elem_count, CU_VEC_LANES);
    for (unsigned int packet_idx = 0;
         packet_idx < MM_STREAM_8X64_PACKETS_PER_BLOCK;
         packet_idx++) {
        #pragma HLS pipeline II=1
        if (packet_idx < packet_count) {
            cu_vec16_packet_t packet = result_stream.read();
            mm_input_block_t packed = 0;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS unroll
                unsigned int elem = packet.elem_base + lane;
                set_mm_input_block_lane(
                    packed,
                    lane,
                    elem < elem_count && packet.valid_mask[lane] ?
                    packet.data[lane] :
                    fm_t(0)
                );
            }
            destination.block[packet.token_lane]
                [packet.elem_base / MM_PE_IN] = packed;
        }
    }
}

void control_cache_8x64_dual_core(
    hls::stream<cu8_task_t>& core0_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core0_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core0_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core0_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu8_task_t>& core1_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core1_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core1_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core1_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    hls::stream<cc8_status_packet_t>& status_stream,
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
    QWEN_WEIGHT_SHARD_PARAMS
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
    #pragma HLS interface s_axilite port=return bundle=control
    #pragma HLS inline off

    cc8_status_packet_t status;
    status.op = CC8_OP_NOP;
    status.status = CC8_STATUS_OK;
    status.token_count = token_count;
    status.output_waves = 0;
    status.dispatched_mm_tasks = 0;
    status.dispatched_vector_tasks = 0;
    status.completed_output_packets = 0;
    status.last_task = true;

    if (token_count > LINEAR_TOKEN_TILE_ACTIVE) {
        status.status = CC8_STATUS_BAD_TOKEN_COUNT;
        core0_task_stream.write(build_cc8_stop_task());
        core1_task_stream.write(build_cc8_stop_task());
        status_stream.write(status);
        return;
    }

    cc8_operator_t op = cc8_operator_t(operator_kind);
    status.op = op;
    cc8_operator_spec_t spec;
    if (!cc8_get_operator_spec(op, spec)) {
        status.status = CC8_STATUS_BAD_OPERATOR;
        core0_task_stream.write(build_cc8_stop_task());
        core1_task_stream.write(build_cc8_stop_task());
        status_stream.write(status);
        return;
    }

    if (op == CC8_OP_NOP) {
        core0_task_stream.write(build_cc8_stop_task());
        core1_task_stream.write(build_cc8_stop_task());
        status_stream.write(status);
        return;
    }

    cc8_global_buffer_t gbuf0;
    cc8_global_buffer_t gbuf1;
    #pragma HLS bind_storage variable=gbuf0.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=gbuf1.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=gbuf0.block complete dim=1
    #pragma HLS array_partition variable=gbuf1.block complete dim=1

    if (op == CC8_OP_ATTN_QK ||
        op == CC8_OP_ATTN_PV ||
        op == CC8_OP_SOFTMAX) {
        if (tile_len == 0 || tile_len > CC8_ATTN_TILE) {
            status.status = CC8_STATUS_BAD_TILE_LENGTH;
            core0_task_stream.write(build_cc8_stop_task());
            core1_task_stream.write(build_cc8_stop_task());
            status_stream.write(status);
            return;
        }

        cc8_attention_buffer_t source0;
        cc8_attention_buffer_t source1;
        cc8_attention_buffer_t destination0;
        cc8_attention_buffer_t destination1;
        #pragma HLS bind_storage variable=source0.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=source1.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=destination0.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=destination1.block type=ram_2p impl=bram
        #pragma HLS array_partition variable=source0.block complete dim=1
        #pragma HLS array_partition variable=source1.block complete dim=1
        #pragma HLS array_partition variable=destination0.block complete dim=1
        #pragma HLS array_partition variable=destination1.block complete dim=1

        unsigned int source_elems =
            op == CC8_OP_ATTN_QK ? HEAD_DIM : CC8_ATTN_TILE;
        load_cc8_flat_rows(
            source0,
            input_port0,
            GQA_GROUP_SIZE,
            source_elems
        );
        load_cc8_flat_rows(
            source1,
            input_port1,
            GQA_GROUP_SIZE,
            source_elems
        );

        if (op == CC8_OP_SOFTMAX) {
            unsigned int packets =
                GQA_GROUP_SIZE *
                ceildiv(tile_len, CU_VEC_LANES);
            cu8_task_t softmax_task = build_cc8_compute_task(
                CU8_MODE_SOFTMAX,
                0,
                GQA_GROUP_SIZE,
                tile_len,
                packets,
                0,
                0,
                true
            );
            core0_task_stream.write(softmax_task);
            core1_task_stream.write(softmax_task);
            emit_cc8_attention_vectors(
                core0_vector_input0_stream,
                source0,
                GQA_GROUP_SIZE,
                tile_len,
                true
            );
            emit_cc8_attention_vectors(
                core1_vector_input0_stream,
                source1,
                GQA_GROUP_SIZE,
                tile_len,
                true
            );
            collect_cc8_attention_vectors(
                destination0,
                core0_result_stream,
                GQA_GROUP_SIZE,
                tile_len
            );
            collect_cc8_attention_vectors(
                destination1,
                core1_result_stream,
                GQA_GROUP_SIZE,
                tile_len
            );
            store_cc8_flat_rows(
                output_port0,
                destination0,
                GQA_GROUP_SIZE,
                tile_len
            );
            store_cc8_flat_rows(
                output_port1,
                destination1,
                GQA_GROUP_SIZE,
                tile_len
            );
            status.dispatched_vector_tasks = CC8_MM_CORE_COUNT;
            status.completed_output_packets =
                CC8_MM_CORE_COUNT * packets;
            status_stream.write(status);
            return;
        }

        cc8_attention_panel_t panel0;
        cc8_attention_panel_t panel1;
        #pragma HLS bind_storage variable=panel0.word type=ram_2p impl=bram
        #pragma HLS bind_storage variable=panel1.word type=ram_2p impl=bram
        #pragma HLS array_partition variable=panel0.word complete dim=1
        #pragma HLS array_partition variable=panel1.word complete dim=1
        load_cc8_attention_panel(panel0, aux_port0, tile_len);
        load_cc8_attention_panel(panel1, aux_port1, tile_len);

        if (op == CC8_OP_ATTN_QK) {
            cu8_task_t qk_task = build_cc8_compute_task(
                CU8_MODE_MM_SCALE,
                HEAD_DIM,
                GQA_GROUP_SIZE,
                CC8_ATTN_TILE,
                MM_STREAM_8X64_PACKETS_PER_BLOCK,
                0,
                0,
                true
            );
            qk_task.output_scale = fm_t(ATTENTION_SCALE);
            core0_task_stream.write(qk_task);
            core1_task_stream.write(qk_task);
            emit_cc8_attention_qk_inputs(
                core0_activation_stream,
                core0_weight_stream0,
                core0_weight_stream1,
                core0_weight_stream2,
                core0_weight_stream3,
                core1_activation_stream,
                core1_weight_stream0,
                core1_weight_stream1,
                core1_weight_stream2,
                core1_weight_stream3,
                source0,
                source1,
                panel0,
                panel1,
                tile_len
            );
            collect_cc8_separate_mm_results(
                destination0,
                destination1,
                core0_result_stream,
                core1_result_stream,
                GQA_GROUP_SIZE,
                CC8_ATTN_TILE
            );
            store_cc8_flat_rows(
                output_port0,
                destination0,
                GQA_GROUP_SIZE,
                CC8_ATTN_TILE
            );
            store_cc8_flat_rows(
                output_port1,
                destination1,
                GQA_GROUP_SIZE,
                CC8_ATTN_TILE
            );
            status.output_waves = 1;
            status.dispatched_mm_tasks = CC8_MM_CORE_COUNT;
            status.completed_output_packets =
                CC8_MM_CORE_COUNT *
                MM_STREAM_8X64_PACKETS_PER_BLOCK;
            status_stream.write(status);
            return;
        }

        unsigned int output_waves =
            ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS);
        for (unsigned int output_wave = 0;
             output_wave < ceildiv(
                 CC8_ATTN_BUFFER_ELEMS,
                 MM_STREAM_8X64_OUTPUTS
             );
             output_wave++) {
            if (output_wave < output_waves) {
                bool last_wave = output_wave + 1 == output_waves;
                cu8_task_t pv_task = build_cc8_compute_task(
                    CU8_MODE_MM,
                    tile_len,
                    GQA_GROUP_SIZE,
                    MM_STREAM_8X64_OUTPUTS,
                    MM_STREAM_8X64_PACKETS_PER_BLOCK,
                    output_wave * MM_STREAM_8X64_OUTPUTS,
                    output_wave,
                    last_wave
                );
                core0_task_stream.write(pv_task);
                core1_task_stream.write(pv_task);
                emit_cc8_attention_pv_inputs(
                    core0_activation_stream,
                    core0_weight_stream0,
                    core0_weight_stream1,
                    core0_weight_stream2,
                    core0_weight_stream3,
                    core1_activation_stream,
                    core1_weight_stream0,
                    core1_weight_stream1,
                    core1_weight_stream2,
                    core1_weight_stream3,
                    source0,
                    source1,
                    panel0,
                    panel1,
                    output_wave,
                    tile_len
                );
                collect_cc8_separate_mm_results(
                    destination0,
                    destination1,
                    core0_result_stream,
                    core1_result_stream,
                    GQA_GROUP_SIZE,
                    HEAD_DIM
                );
            }
        }
        store_cc8_flat_rows(
            output_port0,
            destination0,
            GQA_GROUP_SIZE,
            HEAD_DIM
        );
        store_cc8_flat_rows(
            output_port1,
            destination1,
            GQA_GROUP_SIZE,
            HEAD_DIM
        );
        status.output_waves = output_waves;
        status.dispatched_mm_tasks =
            output_waves * CC8_MM_CORE_COUNT;
        status.completed_output_packets =
            output_waves *
            CC8_MM_CORE_COUNT *
            MM_STREAM_8X64_PACKETS_PER_BLOCK;
        status_stream.write(status);
        return;
    }

    if (spec.uses_mm) {
        load_cc8_feature_gbuf(
            gbuf0,
            input_port0,
            input_port1,
            LINEAR_TOKEN_TILE_ACTIVE,
            token_count,
            spec.in_dim
        );
        clear_cc8_gbuf(gbuf1, spec.out_dim);

        mm_projection_spec_t projection_spec;
        get_mm_projection_spec(spec.projection, projection_spec);
        weight_addr_t layer_base =
            weight_addr_t(layer_id) * weight_addr_t(LAYER_WEIGHT_SIZE);
        weight_addr_t weight_base =
            layer_base + projection_spec.weight_base;
        unsigned int output_waves =
            ceildiv(spec.out_dim, CC8_OUTPUTS_PER_WAVE);
        status.output_waves = output_waves;
        status.dispatched_mm_tasks = output_waves * CC8_MM_CORE_COUNT;

        for (unsigned int output_wave = 0;
             output_wave < ceildiv(MAX_LINEAR_OUT_DIM, CC8_OUTPUTS_PER_WAVE);
             output_wave++) {
            if (output_wave < output_waves) {
                bool last_wave = output_wave + 1 == output_waves;
                cu8_task_t task0 = build_cc8_compute_task(
                    CU8_MODE_MM,
                    spec.in_dim,
                    token_count,
                    MM_STREAM_8X64_OUTPUTS,
                    MM_STREAM_8X64_PACKETS_PER_BLOCK,
                    output_wave * CC8_OUTPUTS_PER_WAVE,
                    output_wave,
                    last_wave
                );
                cu8_task_t task1 = task0;
                task1.elem_base += MM_STREAM_8X64_OUTPUTS;
                core0_task_stream.write(task0);
                core1_task_stream.write(task1);

                run_cc8_mm_wave(
                    core0_activation_stream,
                    core0_weight_stream0,
                    core0_weight_stream1,
                    core0_weight_stream2,
                    core0_weight_stream3,
                    core0_result_stream,
                    core1_activation_stream,
                    core1_weight_stream0,
                    core1_weight_stream1,
                    core1_weight_stream2,
                    core1_weight_stream3,
                    core1_result_stream,
                    gbuf0,
                    gbuf1,
                    output_wave,
                    token_count,
                    spec.in_dim,
                    spec.out_dim,
                    weight_base,
                    QWEN_WEIGHT_SHARD_ARGS
                );
            }
        }
        store_cc8_gbuf_to_hbm(
            output_port0,
            output_port1,
            gbuf1,
            spec.out_dim
        );
        status.completed_output_packets =
            output_waves *
            CC8_MM_CORE_COUNT *
            MM_STREAM_8X64_PACKETS_PER_BLOCK;
        status_stream.write(status);
        return;
    }

    load_cc8_feature_gbuf(
        gbuf0,
        input_port0,
        input_port1,
        LINEAR_TOKEN_TILE_ACTIVE,
        token_count,
        spec.in_dim
    );
    unsigned int rhs_token_slots =
        op == CC8_OP_RMSNORM ?
        1 :
        LINEAR_TOKEN_TILE_ACTIVE;
    unsigned int rhs_valid_tokens =
        op == CC8_OP_RMSNORM ?
        1 :
        token_count;
    load_cc8_feature_gbuf(
        gbuf1,
        aux_port0,
        aux_port1,
        rhs_token_slots,
        rhs_valid_tokens,
        spec.in_dim
    );

    unsigned int packets_per_core =
        CC8_TOKENS_PER_DATA_PORT *
        ceildiv(spec.out_dim, CU_VEC_LANES);
    cu8_task_t compute_task = build_cc8_compute_task(
        spec.compute_mode,
        0,
        CC8_TOKENS_PER_DATA_PORT,
        spec.in_dim,
        packets_per_core,
        0,
        0,
        true
    );
    core0_task_stream.write(compute_task);
    core1_task_stream.write(compute_task);

    if (op == CC8_OP_RMSNORM) {
        // RMSNorm consumes the shared weight row before any activation row.
        emit_cc8_vec_range_from_gbuf(
            core0_vector_input1_stream,
            gbuf1,
            0,
            rhs_token_slots,
            rhs_valid_tokens,
            spec.in_dim,
            0,
            true
        );
        emit_cc8_vec_range_from_gbuf(
            core1_vector_input1_stream,
            gbuf1,
            0,
            rhs_token_slots,
            rhs_valid_tokens,
            spec.in_dim,
            0,
            true
        );
        emit_cc8_vec_range_from_gbuf(
            core0_vector_input0_stream,
            gbuf0,
            0,
            CC8_TOKENS_PER_DATA_PORT,
            token_count,
            spec.in_dim,
            0,
            true
        );
        emit_cc8_vec_range_from_gbuf(
            core1_vector_input0_stream,
            gbuf0,
            CC8_TOKENS_PER_DATA_PORT,
            CC8_TOKENS_PER_DATA_PORT,
            token_count,
            spec.in_dim,
            0,
            true
        );
    } else {
        // Binary vector modes send lhs/rhs as pairs so finite AXIS FIFOs
        // cannot deadlock while the consumer waits for the second operand.
        emit_cc8_vec_pair_from_gbuf(
            core0_vector_input0_stream,
            core0_vector_input1_stream,
            gbuf0,
            gbuf1,
            0,
            CC8_TOKENS_PER_DATA_PORT,
            token_count,
            spec.in_dim,
            0,
            true
        );
        emit_cc8_vec_pair_from_gbuf(
            core1_vector_input0_stream,
            core1_vector_input1_stream,
            gbuf0,
            gbuf1,
            CC8_TOKENS_PER_DATA_PORT,
            CC8_TOKENS_PER_DATA_PORT,
            token_count,
            spec.in_dim,
            0,
            true
        );
    }

    collect_cc8_vector_results(
        output_port0,
        output_port1,
        core0_result_stream,
        packets_per_core
    );
    collect_cc8_vector_results(
        output_port0,
        output_port1,
        core1_result_stream,
        packets_per_core
    );
    status.dispatched_vector_tasks = CC8_MM_CORE_COUNT;
    status.completed_output_packets =
        CC8_MM_CORE_COUNT * packets_per_core;
    status_stream.write(status);
}
