#include "control_cache_8x64.hpp"
#include <hls_math.h>

struct cc8_weight_tile_t {
    wt_linear_t value[CU_VEC_LANES][MM_PE_IN];
};

static unsigned int cc8_core_mask_count(unsigned int core_mask) {
    #pragma HLS inline
    unsigned int count = 0;
    if ((core_mask & 1u) != 0u) {
        count++;
    }
    if ((core_mask & 2u) != 0u) {
        count++;
    }
    return count;
}

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
    case CC8_OP_ATTN_FLASH:
    case CC8_OP_DECODE_SMOKE:
    case CC8_OP_ATTN_PREFILL_BLOCK:
        spec.compute_mode = CU8_MODE_MM_SCALE;
        spec.in_dim = HEAD_DIM;
        spec.out_dim = HEAD_DIM;
        return true;
    case CC8_OP_DECODER_LAYER:
    case CC8_OP_ATTENTION_SUBLAYER:
    case CC8_OP_FFN_SUBLAYER:
    case CC8_OP_FINAL_NORM:
        // The resident layer path uses a fixed descriptor sequence rather
        // than dispatching one of the individual operator specifications.
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

template <typename BufferT>
static void load_cc8_feature_gbuf(
    BufferT& gbuf,
    const fm_word_t port0[CC8_DATA_PORT_WORDS],
    const fm_word_t port1[CC8_DATA_PORT_WORDS],
    unsigned int token_slots,
    unsigned int valid_tokens,
    unsigned int elem_count,
    unsigned int word_offset = 0
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf.block complete dim=1

    unsigned int words_per_token = ceildiv(elem_count, FM_BLOCK_SIZE);
    for (unsigned int token = 0; token < CC8_GBUF_TOKEN_ROWS; token++) {
        unsigned int port = token / CC8_TOKENS_PER_DATA_PORT;
        unsigned int token_in_port =
            token - port * CC8_TOKENS_PER_DATA_PORT;
        for (unsigned int word_idx = 0;
             word_idx < ceildiv(BufferT::kBlockCount, 2u);
             word_idx++) {
            #pragma HLS pipeline II=1
            if (token < token_slots && word_idx < words_per_token) {
                fm_word_t word = 0;
                if (token < valid_tokens) {
                    unsigned int port_word =
                        word_offset +
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
                if (block0 < BufferT::kBlockCount) {
                    gbuf.block[token][block0] = word.range(255, 0);
                }
                if (block1 < BufferT::kBlockCount) {
                    gbuf.block[token][block1] = word.range(511, 256);
                }
            }
        }
    }
}

template <typename BufferT>
static void clear_cc8_gbuf(
    BufferT& gbuf,
    unsigned int elem_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf.block complete dim=1

    unsigned int block_count = ceildiv(elem_count, MM_PE_IN);
    for (unsigned int token = 0; token < CC8_GBUF_TOKEN_ROWS; token++) {
        for (unsigned int block = 0;
             block < BufferT::kBlockCount;
             block++) {
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

static void load_cc8_weight_panels_legacy(
    hls::stream<mm_stream_8x64_weight_packet_t>
        tile_stream[CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS],
    unsigned int output_wave,
    unsigned int active_in_dim,
    unsigned int matrix_in_dim,
    unsigned int out_dim,
    weight_addr_t weight_base,
    unsigned int core_mask,
    bool zero_weight_stream,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=tile_stream complete dim=0

    unsigned int active_in_tile_count = ceildiv(active_in_dim, MM_PE_IN);
    unsigned int matrix_in_tile_count = ceildiv(matrix_in_dim, MM_PE_IN);
    weight_addr_t matrix_tile_base =
        weight_base / weight_addr_t(MM_TILE_WEIGHT_ELEMS);

    for (unsigned int in_tile = 0;
         in_tile < MAX_LINEAR_IN_BLOCKS;
        in_tile++) {
        if (in_tile < active_in_tile_count) {
            cc8_weight_tile_t tiles
                [CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS];
            #pragma HLS array_partition variable=tiles complete dim=0

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

                if (zero_weight_stream || out_base >= out_dim) {
                    clear_cc8_weight_tile(tile);
                } else {
                    weight_addr_t global_tile =
                        matrix_tile_base +
                        weight_addr_t(out_tile) *
                            weight_addr_t(matrix_in_tile_count) +
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
                }

                switch (tile_slot) {
                case 0:
                    tiles[0][0] = tile;
                    break;
                case 1:
                    tiles[0][1] = tile;
                    break;
                case 2:
                    tiles[0][2] = tile;
                    break;
                case 3:
                    tiles[0][3] = tile;
                    break;
                case 4:
                    tiles[1][0] = tile;
                    break;
                case 5:
                    tiles[1][1] = tile;
                    break;
                case 6:
                    tiles[1][2] = tile;
                    break;
                default:
                    tiles[1][3] = tile;
                    break;
                }
            }

            for (unsigned int k_lane = 0;
                 k_lane < MM_PE_IN;
                 k_lane++) {
                #pragma HLS pipeline II=1
                mm_stream_8x64_weight_packet_t packets
                    [CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS];
                #pragma HLS array_partition variable=packets complete dim=0

                for (unsigned int core = 0;
                     core < CC8_MM_CORE_COUNT;
                     core++) {
                    #pragma HLS unroll
                    for (unsigned int group = 0;
                         group < MM_STREAM_8X64_WEIGHT_GROUPS;
                         group++) {
                        #pragma HLS unroll
                        for (unsigned int lane = 0;
                             lane < CU_VEC_LANES;
                             lane++) {
                            #pragma HLS unroll
                            packets[core][group].data[lane] =
                                tiles[core][group].value[lane][k_lane];
                        }
                    }
                }

                if ((core_mask & 1u) != 0u) {
                    tile_stream[0][0].write(packets[0][0]);
                    tile_stream[0][1].write(packets[0][1]);
                    tile_stream[0][2].write(packets[0][2]);
                    tile_stream[0][3].write(packets[0][3]);
                }
                if ((core_mask & 2u) != 0u) {
                    tile_stream[1][0].write(packets[1][0]);
                    tile_stream[1][1].write(packets[1][1]);
                    tile_stream[1][2].write(packets[1][2]);
                    tile_stream[1][3].write(packets[1][3]);
                }
            }
        }
    }
}

static void load_cc8_packed_weight_blocks(
    hls::stream<wt_block_t>
        packed_stream[CC8_OUTPUT_TILES_PER_WAVE],
    unsigned int output_wave,
    unsigned int active_in_tile_count,
    unsigned int matrix_in_tile_count,
    unsigned int out_dim,
    weight_addr_t matrix_tile_base,
    bool zero_weight_stream,
    const wt_block_t* group0_shard,
    const wt_block_t* group1_shard
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=packed_stream complete dim=1

    unsigned int packed_tile_count =
        active_in_tile_count * CC8_OUTPUT_TILES_PER_WAVE;
    for (unsigned int flat = 0;
         flat < packed_tile_count;
         flat++) {
        #pragma HLS pipeline II=CC8_WEIGHT_TILE_LOAD_II_VALUE
        #pragma HLS loop_tripcount min=CC8_OUTPUT_TILES_PER_WAVE max=CC8_MAX_PACKED_WEIGHT_TILES
        unsigned int in_tile = flat / CC8_OUTPUT_TILES_PER_WAVE;
        unsigned int tile_slot =
            flat - in_tile * CC8_OUTPUT_TILES_PER_WAVE;
        unsigned int out_tile =
            output_wave * CC8_OUTPUT_TILES_PER_WAVE + tile_slot;
        unsigned int out_base = out_tile * MM_PE_OUT;

        wt_block_t packed = 0;
        if (zero_weight_stream || out_base >= out_dim) {
            packed = 0;
        } else {
            weight_addr_t global_tile =
                matrix_tile_base +
                weight_addr_t(out_tile) *
                    weight_addr_t(matrix_in_tile_count) +
                weight_addr_t(in_tile);
            weight_addr_t local_block = global_tile >> 1;
            if ((global_tile & weight_addr_t(1)) == 0) {
                packed = group0_shard[local_block];
            } else {
                packed = group1_shard[local_block];
            }
        }

        switch (tile_slot) {
        case 0:
            packed_stream[0].write(packed);
            break;
        case 1:
            packed_stream[1].write(packed);
            break;
        case 2:
            packed_stream[2].write(packed);
            break;
        case 3:
            packed_stream[3].write(packed);
            break;
        case 4:
            packed_stream[4].write(packed);
            break;
        case 5:
            packed_stream[5].write(packed);
            break;
        case 6:
            packed_stream[6].write(packed);
            break;
        default:
            packed_stream[7].write(packed);
            break;
        }
    }
}

static void load_cc8_packed_weight_block_range(
    hls::stream<wt_block_t>
        packed_stream[CC8_OUTPUT_TILES_PER_WAVE],
    unsigned int wave_begin,
    unsigned int wave_end,
    unsigned int active_in_tile_count,
    unsigned int matrix_in_tile_count,
    unsigned int out_dim,
    weight_addr_t matrix_tile_base,
    bool zero_weight_stream,
    const wt_block_t* group0_shard,
    const wt_block_t* group1_shard
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=packed_stream complete dim=1

    for (unsigned int output_wave = wave_begin;
         output_wave < wave_end;
         output_wave++) {
        #pragma HLS loop_tripcount min=1 max=CC8_MAX_OUTPUT_WAVES
        load_cc8_packed_weight_blocks(
            packed_stream,
            output_wave,
            active_in_tile_count,
            matrix_in_tile_count,
            out_dim,
            matrix_tile_base,
            zero_weight_stream,
            group0_shard,
            group1_shard
        );
    }
}

static void emit_cc8_packed_weight_slot(
    hls::stream<wt_block_t>& packed_stream0,
    hls::stream<wt_block_t>& packed_stream1,
    hls::stream<wt_block_t>& packed_stream2,
    hls::stream<wt_block_t>& packed_stream3,
    hls::stream<wt_block_t>& packed_stream4,
    hls::stream<wt_block_t>& packed_stream5,
    hls::stream<wt_block_t>& packed_stream6,
    hls::stream<wt_block_t>& packed_stream7,
    hls::stream<bool>& enable_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream,
    unsigned int active_in_tile_count
) {
    #pragma HLS inline off

    bool enabled = enable_stream.read();

    constexpr unsigned int kPackedRowBits =
        MM_PE_IN * wt_linear_t::width;
    ap_uint<kPackedRowBits> row_shift[CU_VEC_LANES];
    #pragma HLS array_partition variable=row_shift complete dim=1
    unsigned int packet_count = active_in_tile_count * MM_PE_IN;
    for (unsigned int packet_index = 0;
         packet_index < packet_count;
         packet_index++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=MM_PE_IN max=MAX_LINEAR_IN_DIM
        unsigned int k_lane = packet_index % MM_PE_IN;
        if (k_lane == 0) {
            wt_block_t packed[MM_TILE_WEIGHT_BLOCKS];
            #pragma HLS array_partition variable=packed complete dim=1
            packed[0] = packed_stream0.read();
            packed[1] = packed_stream1.read();
            packed[2] = packed_stream2.read();
            packed[3] = packed_stream3.read();
            packed[4] = packed_stream4.read();
            packed[5] = packed_stream5.read();
            packed[6] = packed_stream6.read();
            packed[7] = packed_stream7.read();
            for (unsigned int block = 0;
                 block < MM_TILE_WEIGHT_BLOCKS;
                 block++) {
                #pragma HLS unroll
                row_shift[2 * block] =
                    packed[block].range(kPackedRowBits - 1, 0);
                row_shift[2 * block + 1] =
                    packed[block].range(2 * kPackedRowBits - 1,
                                        kPackedRowBits);
            }
        }

        mm_stream_8x64_weight_packet_t packet;
        #pragma HLS array_partition variable=packet.data complete dim=1
        for (unsigned int lane = 0;
             lane < CU_VEC_LANES;
             lane++) {
            #pragma HLS unroll
            wt_linear_t value;
            value.range(wt_linear_t::width - 1, 0) =
                row_shift[lane].range(wt_linear_t::width - 1, 0);
            packet.data[lane] = value;
            row_shift[lane] >>= wt_linear_t::width;
        }
        if (enabled) {
            weight_stream.write(packet);
        }
    }
}

static void emit_cc8_packed_weight_slot_range(
    hls::stream<wt_block_t>& packed_stream0,
    hls::stream<wt_block_t>& packed_stream1,
    hls::stream<wt_block_t>& packed_stream2,
    hls::stream<wt_block_t>& packed_stream3,
    hls::stream<wt_block_t>& packed_stream4,
    hls::stream<wt_block_t>& packed_stream5,
    hls::stream<wt_block_t>& packed_stream6,
    hls::stream<wt_block_t>& packed_stream7,
    hls::stream<bool>& enable_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream,
    unsigned int wave_count,
    unsigned int active_in_tile_count
) {
    #pragma HLS inline off

    for (unsigned int wave = 0; wave < wave_count; wave++) {
        #pragma HLS loop_tripcount min=1 max=CC8_MAX_OUTPUT_WAVES
        emit_cc8_packed_weight_slot(
            packed_stream0,
            packed_stream1,
            packed_stream2,
            packed_stream3,
            packed_stream4,
            packed_stream5,
            packed_stream6,
            packed_stream7,
            enable_stream,
            weight_stream,
            active_in_tile_count
        );
    }
}

static void distribute_cc8_weight_enables(
    hls::stream<bool> enable_stream[CC8_OUTPUT_TILES_PER_WAVE],
    unsigned int core_mask
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=enable_stream complete dim=1

    bool core0_enabled = (core_mask & 1u) != 0u;
    bool core1_enabled = (core_mask & 2u) != 0u;
    enable_stream[0].write(core0_enabled);
    enable_stream[1].write(core0_enabled);
    enable_stream[2].write(core0_enabled);
    enable_stream[3].write(core0_enabled);
    enable_stream[4].write(core1_enabled);
    enable_stream[5].write(core1_enabled);
    enable_stream[6].write(core1_enabled);
    enable_stream[7].write(core1_enabled);
}

static void distribute_cc8_weight_enable_range(
    hls::stream<bool> enable_stream[CC8_OUTPUT_TILES_PER_WAVE],
    unsigned int core_mask,
    unsigned int wave_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=enable_stream complete dim=1

    for (unsigned int wave = 0; wave < wave_count; wave++) {
        #pragma HLS loop_tripcount min=1 max=CC8_MAX_OUTPUT_WAVES
        distribute_cc8_weight_enables(enable_stream, core_mask);
    }
}

static void load_cc8_weight_panels_streaming(
    hls::stream<mm_stream_8x64_weight_packet_t>
        tile_stream[CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS],
    unsigned int wave_begin,
    unsigned int wave_end,
    unsigned int active_in_dim,
    unsigned int matrix_in_dim,
    unsigned int out_dim,
    weight_addr_t weight_base,
    unsigned int core_mask,
    bool zero_weight_stream,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=tile_stream complete dim=0

    hls::stream<wt_block_t> packed_stream
        [MM_TILE_WEIGHT_BLOCKS][CC8_OUTPUT_TILES_PER_WAVE];
    hls::stream<bool> enable_stream[CC8_OUTPUT_TILES_PER_WAVE];
    #pragma HLS array_partition variable=packed_stream complete dim=0
    #pragma HLS array_partition variable=enable_stream complete dim=1
    #pragma HLS stream variable=packed_stream depth=CC8_WEIGHT_TILE_FIFO_DEPTH_VALUE
    #pragma HLS stream variable=enable_stream depth=1
    #pragma HLS bind_storage variable=packed_stream type=fifo impl=lutram
    #pragma HLS dataflow

    unsigned int active_in_tile_count = ceildiv(active_in_dim, MM_PE_IN);
    unsigned int matrix_in_tile_count = ceildiv(matrix_in_dim, MM_PE_IN);
    unsigned int wave_count = wave_end > wave_begin ?
        wave_end - wave_begin : 0;
    weight_addr_t matrix_tile_base =
        weight_base / weight_addr_t(MM_TILE_WEIGHT_ELEMS);

    distribute_cc8_weight_enable_range(
        enable_stream, core_mask, wave_count
    );
    load_cc8_packed_weight_block_range(
        packed_stream[0], wave_begin, wave_end,
        active_in_tile_count, matrix_in_tile_count, out_dim,
        matrix_tile_base, zero_weight_stream,
        layer_weights_shard0, layer_weights_shard8);
    load_cc8_packed_weight_block_range(
        packed_stream[1], wave_begin, wave_end,
        active_in_tile_count, matrix_in_tile_count, out_dim,
        matrix_tile_base, zero_weight_stream,
        layer_weights_shard1, layer_weights_shard9);
    load_cc8_packed_weight_block_range(
        packed_stream[2], wave_begin, wave_end,
        active_in_tile_count, matrix_in_tile_count, out_dim,
        matrix_tile_base, zero_weight_stream,
        layer_weights_shard2, layer_weights_shard10);
    load_cc8_packed_weight_block_range(
        packed_stream[3], wave_begin, wave_end,
        active_in_tile_count, matrix_in_tile_count, out_dim,
        matrix_tile_base, zero_weight_stream,
        layer_weights_shard3, layer_weights_shard11);
    load_cc8_packed_weight_block_range(
        packed_stream[4], wave_begin, wave_end,
        active_in_tile_count, matrix_in_tile_count, out_dim,
        matrix_tile_base, zero_weight_stream,
        layer_weights_shard4, layer_weights_shard12);
    load_cc8_packed_weight_block_range(
        packed_stream[5], wave_begin, wave_end,
        active_in_tile_count, matrix_in_tile_count, out_dim,
        matrix_tile_base, zero_weight_stream,
        layer_weights_shard5, layer_weights_shard13);
    load_cc8_packed_weight_block_range(
        packed_stream[6], wave_begin, wave_end,
        active_in_tile_count, matrix_in_tile_count, out_dim,
        matrix_tile_base, zero_weight_stream,
        layer_weights_shard6, layer_weights_shard14);
    load_cc8_packed_weight_block_range(
        packed_stream[7], wave_begin, wave_end,
        active_in_tile_count, matrix_in_tile_count, out_dim,
        matrix_tile_base, zero_weight_stream,
        layer_weights_shard7, layer_weights_shard15);
    emit_cc8_packed_weight_slot_range(
        packed_stream[0][0], packed_stream[1][0],
        packed_stream[2][0], packed_stream[3][0],
        packed_stream[4][0], packed_stream[5][0],
        packed_stream[6][0], packed_stream[7][0],
        enable_stream[0],
        tile_stream[0][0], wave_count, active_in_tile_count
    );
    emit_cc8_packed_weight_slot_range(
        packed_stream[0][1], packed_stream[1][1],
        packed_stream[2][1], packed_stream[3][1],
        packed_stream[4][1], packed_stream[5][1],
        packed_stream[6][1], packed_stream[7][1],
        enable_stream[1],
        tile_stream[0][1], wave_count, active_in_tile_count
    );
    emit_cc8_packed_weight_slot_range(
        packed_stream[0][2], packed_stream[1][2],
        packed_stream[2][2], packed_stream[3][2],
        packed_stream[4][2], packed_stream[5][2],
        packed_stream[6][2], packed_stream[7][2],
        enable_stream[2],
        tile_stream[0][2], wave_count, active_in_tile_count
    );
    emit_cc8_packed_weight_slot_range(
        packed_stream[0][3], packed_stream[1][3],
        packed_stream[2][3], packed_stream[3][3],
        packed_stream[4][3], packed_stream[5][3],
        packed_stream[6][3], packed_stream[7][3],
        enable_stream[3],
        tile_stream[0][3], wave_count, active_in_tile_count
    );
    emit_cc8_packed_weight_slot_range(
        packed_stream[0][4], packed_stream[1][4],
        packed_stream[2][4], packed_stream[3][4],
        packed_stream[4][4], packed_stream[5][4],
        packed_stream[6][4], packed_stream[7][4],
        enable_stream[4],
        tile_stream[1][0], wave_count, active_in_tile_count
    );
    emit_cc8_packed_weight_slot_range(
        packed_stream[0][5], packed_stream[1][5],
        packed_stream[2][5], packed_stream[3][5],
        packed_stream[4][5], packed_stream[5][5],
        packed_stream[6][5], packed_stream[7][5],
        enable_stream[5],
        tile_stream[1][1], wave_count, active_in_tile_count
    );
    emit_cc8_packed_weight_slot_range(
        packed_stream[0][6], packed_stream[1][6],
        packed_stream[2][6], packed_stream[3][6],
        packed_stream[4][6], packed_stream[5][6],
        packed_stream[6][6], packed_stream[7][6],
        enable_stream[6],
        tile_stream[1][2], wave_count, active_in_tile_count
    );
    emit_cc8_packed_weight_slot_range(
        packed_stream[0][7], packed_stream[1][7],
        packed_stream[2][7], packed_stream[3][7],
        packed_stream[4][7], packed_stream[5][7],
        packed_stream[6][7], packed_stream[7][7],
        enable_stream[7],
        tile_stream[1][3], wave_count, active_in_tile_count
    );
}

static void load_cc8_weight_panels(
    hls::stream<mm_stream_8x64_weight_packet_t>
        tile_stream[CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS],
    unsigned int output_wave,
    unsigned int active_in_dim,
    unsigned int matrix_in_dim,
    unsigned int out_dim,
    weight_addr_t weight_base,
    unsigned int core_mask,
    bool zero_weight_stream,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline off
#if CC8_ENABLE_WEIGHT_TILE_PIPELINE
    load_cc8_weight_panels_streaming(
        tile_stream,
        output_wave,
        output_wave + 1,
        active_in_dim,
        matrix_in_dim,
        out_dim,
        weight_base,
        core_mask,
        zero_weight_stream,
        QWEN_WEIGHT_SHARD_ARGS
    );
#else
    load_cc8_weight_panels_legacy(
        tile_stream,
        output_wave,
        active_in_dim,
        matrix_in_dim,
        out_dim,
        weight_base,
        core_mask,
        zero_weight_stream,
        QWEN_WEIGHT_SHARD_ARGS
    );
#endif
}

static void load_cc8_weight_panel_range(
    hls::stream<mm_stream_8x64_weight_packet_t>
        tile_stream[CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS],
    unsigned int wave_begin,
    unsigned int wave_end,
    unsigned int active_in_dim,
    unsigned int matrix_in_dim,
    unsigned int out_dim,
    weight_addr_t weight_base,
    unsigned int core_mask,
    bool zero_weight_stream,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=tile_stream complete dim=0
#if CC8_ENABLE_WEIGHT_TILE_PIPELINE
    load_cc8_weight_panels_streaming(
        tile_stream,
        wave_begin,
        wave_end,
        active_in_dim,
        matrix_in_dim,
        out_dim,
        weight_base,
        core_mask,
        zero_weight_stream,
        QWEN_WEIGHT_SHARD_ARGS
    );
#else
    for (unsigned int output_wave = wave_begin;
         output_wave < wave_end;
         output_wave++) {
        #pragma HLS loop_tripcount min=1 max=CC8_MAX_OUTPUT_WAVES
        load_cc8_weight_panels_legacy(
            tile_stream,
            output_wave,
            active_in_dim,
            matrix_in_dim,
            out_dim,
            weight_base,
            core_mask,
            zero_weight_stream,
            QWEN_WEIGHT_SHARD_ARGS
        );
    }
#endif
}

template <typename SourceBufferT>
static void emit_cc8_mm_wave_inputs_flat(
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
    hls::stream<mm_stream_8x64_weight_packet_t>
        tile_stream[CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS],
    const SourceBufferT& source,
    unsigned int token_count,
    unsigned int in_dim,
    unsigned int core_mask
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=tile_stream complete dim=0

    for (unsigned int k = 0; k < in_dim; k++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=16 max=MAX_LINEAR_IN_DIM avg=HIDDEN_SIZE
        unsigned int in_tile = k / MM_PE_IN;
        unsigned int k_lane = k - in_tile * MM_PE_IN;

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

        if ((core_mask & 1u) != 0u) {
            mm_stream_8x64_weight_packet_t c0w0 =
                tile_stream[0][0].read();
            mm_stream_8x64_weight_packet_t c0w1 =
                tile_stream[0][1].read();
            mm_stream_8x64_weight_packet_t c0w2 =
                tile_stream[0][2].read();
            mm_stream_8x64_weight_packet_t c0w3 =
                tile_stream[0][3].read();
            core0_activation_stream.write(activation);
            core0_weight_stream0.write(c0w0);
            core0_weight_stream1.write(c0w1);
            core0_weight_stream2.write(c0w2);
            core0_weight_stream3.write(c0w3);
        }
        if ((core_mask & 2u) != 0u) {
            mm_stream_8x64_weight_packet_t c1w0 =
                tile_stream[1][0].read();
            mm_stream_8x64_weight_packet_t c1w1 =
                tile_stream[1][1].read();
            mm_stream_8x64_weight_packet_t c1w2 =
                tile_stream[1][2].read();
            mm_stream_8x64_weight_packet_t c1w3 =
                tile_stream[1][3].read();
            core1_activation_stream.write(activation);
            core1_weight_stream0.write(c1w0);
            core1_weight_stream1.write(c1w1);
            core1_weight_stream2.write(c1w2);
            core1_weight_stream3.write(c1w3);
        }
    }
}

static void store_cc8_result_packet_flat(
    mm_input_block_t* __restrict destination,
    unsigned int destination_stride,
    const cu_vec16_packet_t& packet,
    unsigned int token_count,
    unsigned int out_dim
) {
    #pragma HLS inline

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
        if (block < destination_stride) {
            destination[
                packet.token_lane * destination_stride + block
            ] = packed;
        }
    }
}

template <typename DestinationBufferT>
static void store_cc8_result_packet(
    DestinationBufferT& destination,
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
        if (block < DestinationBufferT::kBlockCount) {
            destination.block[packet.token_lane][block] = packed;
        }
    }
}

template <typename DestinationBufferT>
static void collect_cc8_mm_wave_results_flat(
    DestinationBufferT& destination,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int token_count,
    unsigned int out_dim,
    unsigned int core_mask
) {
    #pragma HLS inline off

    for (unsigned int packet_idx = 0;
         packet_idx < MM_STREAM_8X64_PACKETS_PER_BLOCK;
         packet_idx++) {
        #pragma HLS pipeline II=1
        if ((core_mask & 1u) != 0u) {
            cu_vec16_packet_t packet0 = core0_result_stream.read();
            store_cc8_result_packet(
                destination,
                packet0,
                token_count,
                out_dim
            );
        }
        if ((core_mask & 2u) != 0u) {
            cu_vec16_packet_t packet1 = core1_result_stream.read();
            store_cc8_result_packet(
                destination,
                packet1,
                token_count,
                out_dim
            );
        }
    }
}

template <typename SourceBufferT, typename DestinationBufferT>
static void run_cc8_mm_wave_flat(
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
    const SourceBufferT& source,
    DestinationBufferT& destination,
    unsigned int output_wave,
    unsigned int token_count,
    unsigned int active_in_dim,
    unsigned int matrix_in_dim,
    unsigned int out_dim,
    weight_addr_t weight_base,
    unsigned int core_mask,
    bool zero_weight_stream,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline off

    hls::stream<mm_stream_8x64_weight_packet_t>
        tile_stream[CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS];
    #pragma HLS array_partition variable=tile_stream complete dim=0
    #pragma HLS stream variable=tile_stream depth=16
    #pragma HLS bind_storage variable=tile_stream type=fifo impl=bram
    #pragma HLS dataflow

    load_cc8_weight_panels(
        tile_stream,
        output_wave,
        active_in_dim,
        matrix_in_dim,
        out_dim,
        weight_base,
        core_mask,
        zero_weight_stream,
        QWEN_WEIGHT_SHARD_ARGS
    );
    emit_cc8_mm_wave_inputs_flat(
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
        active_in_dim,
        core_mask
    );
    collect_cc8_mm_wave_results_flat(
        destination,
        core0_result_stream,
        core1_result_stream,
        token_count,
        out_dim,
        core_mask
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

template <typename BufferT>
static void store_cc8_gbuf_to_hbm(
    fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT],
    const BufferT& gbuf,
    unsigned int elem_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf.block complete dim=1

    unsigned int words_per_token = ceildiv(elem_count, FM_BLOCK_SIZE);
    unsigned int blocks_per_token = ceildiv(elem_count, MM_PE_IN);
    for (unsigned int token = 0; token < CC8_GBUF_TOKEN_ROWS; token++) {
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

template <typename BufferT>
static cu_vec16_packet_t build_cc8_vec_packet(
    const BufferT& gbuf,
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

template <typename BufferT>
static void emit_cc8_vec_dual_range_from_gbuf(
    hls::stream<cu_vec16_packet_t>& stream0,
    hls::stream<cu_vec16_packet_t>& stream1,
    const BufferT& gbuf,
    unsigned int token_begin0,
    unsigned int token_slots0,
    unsigned int token_begin1,
    unsigned int token_slots1,
    unsigned int valid_tokens,
    unsigned int elem_count,
    unsigned int block_id,
    bool last_stream
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf.block complete dim=1

    unsigned int block_count = ceildiv(elem_count, CU_VEC_LANES);
    unsigned int token_slots = token_slots0 > token_slots1 ?
        token_slots0 : token_slots1;
    for (unsigned int local_token = 0;
         local_token < token_slots;
         local_token++) {
        #pragma HLS loop_tripcount min=0 max=CC8_TOKENS_PER_DATA_PORT
        for (unsigned int block = 0; block < block_count; block++) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=1 max=MAX_LINEAR_IN_BLOCKS
            if (local_token < token_slots0) {
                bool is_last0 =
                    local_token + 1 == token_slots0 &&
                    block + 1 == block_count;
                stream0.write(build_cc8_vec_packet(
                    gbuf,
                    token_begin0 + local_token,
                    valid_tokens,
                    elem_count,
                    block,
                    block_id,
                    is_last0,
                    last_stream
                ));
            }
            if (local_token < token_slots1) {
                bool is_last1 =
                    local_token + 1 == token_slots1 &&
                    block + 1 == block_count;
                stream1.write(build_cc8_vec_packet(
                    gbuf,
                    token_begin1 + local_token,
                    valid_tokens,
                    elem_count,
                    block,
                    block_id,
                    is_last1,
                    last_stream
                ));
            }
        }
    }
}

template <typename Buffer0T, typename Buffer1T>
static void emit_cc8_vec_pair_dual_from_gbuf(
    hls::stream<cu_vec16_packet_t>& core0_stream0,
    hls::stream<cu_vec16_packet_t>& core0_stream1,
    hls::stream<cu_vec16_packet_t>& core1_stream0,
    hls::stream<cu_vec16_packet_t>& core1_stream1,
    const Buffer0T& gbuf0,
    const Buffer1T& gbuf1,
    unsigned int core0_token_slots,
    unsigned int core1_token_slots,
    unsigned int valid_tokens,
    unsigned int elem_count,
    unsigned int block_id,
    bool last_stream
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf0.block complete dim=1
    #pragma HLS array_partition variable=gbuf1.block complete dim=1

    unsigned int block_count = ceildiv(elem_count, CU_VEC_LANES);
    unsigned int token_slots = core0_token_slots > core1_token_slots ?
        core0_token_slots : core1_token_slots;
    for (unsigned int local_token = 0;
         local_token < token_slots;
         local_token++) {
        #pragma HLS loop_tripcount min=0 max=CC8_TOKENS_PER_DATA_PORT
        for (unsigned int block = 0; block < block_count; block++) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=1 max=MAX_LINEAR_IN_BLOCKS
            if (local_token < core0_token_slots) {
                bool is_last0 =
                    local_token + 1 == core0_token_slots &&
                    block + 1 == block_count;
                unsigned int token0 = local_token;
                core0_stream0.write(build_cc8_vec_packet(
                    gbuf0,
                    token0,
                    valid_tokens,
                    elem_count,
                    block,
                    block_id,
                    is_last0,
                    last_stream
                ));
                core0_stream1.write(build_cc8_vec_packet(
                    gbuf1,
                    token0,
                    valid_tokens,
                    elem_count,
                    block,
                    block_id,
                    is_last0,
                    last_stream
                ));
            }
            if (local_token < core1_token_slots) {
                bool is_last1 =
                    local_token + 1 == core1_token_slots &&
                    block + 1 == block_count;
                unsigned int token1 =
                    CC8_TOKENS_PER_DATA_PORT + local_token;
                core1_stream0.write(build_cc8_vec_packet(
                    gbuf0,
                    token1,
                    valid_tokens,
                    elem_count,
                    block,
                    block_id,
                    is_last1,
                    last_stream
                ));
                core1_stream1.write(build_cc8_vec_packet(
                    gbuf1,
                    token1,
                    valid_tokens,
                    elem_count,
                    block,
                    block_id,
                    is_last1,
                    last_stream
                ));
            }
        }
    }
}

template <typename Buffer0T, typename Buffer1T>
static void emit_cc8_generic_vector_inputs(
    cc8_operator_t op,
    hls::stream<cu_vec16_packet_t>& core0_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core0_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core1_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core1_vector_input1_stream,
    const Buffer0T& gbuf0,
    const Buffer1T& gbuf1,
    unsigned int token_count,
    unsigned int core0_token_count,
    unsigned int core1_token_count,
    unsigned int rhs_token_slots,
    unsigned int rhs_valid_tokens,
    unsigned int elem_count
) {
    #pragma HLS inline off

    if (op == CC8_OP_RMSNORM) {
        // RMSNorm consumes the shared weight row before any activation row.
        emit_cc8_vec_dual_range_from_gbuf(
            core0_vector_input1_stream,
            core1_vector_input1_stream,
            gbuf1,
            0,
            core0_token_count != 0 ? rhs_token_slots : 0,
            0,
            core1_token_count != 0 ? rhs_token_slots : 0,
            rhs_valid_tokens,
            elem_count,
            0,
            true
        );
        emit_cc8_vec_dual_range_from_gbuf(
            core0_vector_input0_stream,
            core1_vector_input0_stream,
            gbuf0,
            0,
            core0_token_count,
            CC8_TOKENS_PER_DATA_PORT,
            core1_token_count,
            token_count,
            elem_count,
            0,
            true
        );
    } else {
        // Binary vector modes send lhs/rhs as pairs so finite AXIS FIFOs
        // cannot deadlock while the consumer waits for the second operand.
        emit_cc8_vec_pair_dual_from_gbuf(
            core0_vector_input0_stream,
            core0_vector_input1_stream,
            core1_vector_input0_stream,
            core1_vector_input1_stream,
            gbuf0,
            gbuf1,
            core0_token_count,
            core1_token_count,
            token_count,
            elem_count,
            0,
            true
        );
    }
}

static void write_cc8_vector_output_word(
    fm_word_t output_port[CC8_FEATURE_WORDS_PER_PORT],
    unsigned int token,
    unsigned int token_begin,
    unsigned int word_idx,
    fm_word_t word
) {
    #pragma HLS inline

    if (token >= token_begin &&
        token < token_begin + CC8_TOKENS_PER_DATA_PORT) {
        unsigned int local_token = token - token_begin;
        unsigned int port_word =
            local_token * CC8_FEATURE_WORDS_PER_TOKEN + word_idx;
        output_port[port_word] = word;
    }
}

static_assert(
    MM_INPUT_BLOCK_BIT_WIDTH == CU_VEC_LANES * fm_t::width,
    "one vector result packet must fill one packed half word"
);
static_assert(
    AXI_XFER_BIT_WIDTH == 2 * MM_INPUT_BLOCK_BIT_WIDTH,
    "two packed vector result packets must fill one AXI word"
);

static mm_input_block_t pack_cc8_vector_result_half(
    const cu_vec16_packet_t& packet
) {
    #pragma HLS inline
    #pragma HLS array_partition variable=packet.data complete dim=1

    mm_input_block_t packed = 0;
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        #pragma HLS unroll
        fm_t value = packet.valid_mask[lane] ?
            packet.data[lane] : fm_t(0);
        set_mm_input_block_lane(packed, lane, value);
    }
    return packed;
}

static void collect_cc8_vector_results(
    fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT],
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int core0_packet_count,
    unsigned int core1_packet_count
) {
    #pragma HLS inline off

    fm_word_t pending_word0 = 0;
    fm_word_t pending_word1 = 0;
    unsigned int pending_token0 = 0;
    unsigned int pending_token1 = 0;
    unsigned int pending_word_idx0 = 0;
    unsigned int pending_word_idx1 = 0;
    bool pending_valid0 = false;
    bool pending_valid1 = false;
    unsigned int packet_count =
        core0_packet_count > core1_packet_count ?
        core0_packet_count : core1_packet_count;

    // The compute protocol emits half 0 before half 1 for each token/word.
    // Keep that ordering invariant when adding new vector result producers.
    for (unsigned int packet_idx = 0; packet_idx < packet_count; packet_idx++) {
        #pragma HLS loop_tripcount min=0 max=CU_STREAM_MAX_PACKETS
        bool read0 = packet_idx < core0_packet_count;
        bool read1 = packet_idx < core1_packet_count;
        cu_vec16_packet_t packet0;
        cu_vec16_packet_t packet1;
        #pragma HLS array_partition variable=packet0.data complete dim=1
        #pragma HLS array_partition variable=packet1.data complete dim=1
        unsigned int word_idx0 = 0;
        unsigned int word_idx1 = 0;
        unsigned int half0 = 0;
        unsigned int half1 = 0;

        if (read0) {
            packet0 = core0_result_stream.read();
            word_idx0 = packet0.elem_base / FM_BLOCK_SIZE;
            half0 = (packet0.elem_base % FM_BLOCK_SIZE) / CU_VEC_LANES;
            bool same_word0 =
                pending_valid0 &&
                pending_token0 == packet0.token_lane &&
                pending_word_idx0 == word_idx0;
            if (pending_valid0 && !same_word0) {
                write_cc8_vector_output_word(
                    output_port0,
                    pending_token0,
                    0,
                    pending_word_idx0,
                    pending_word0
                );
                pending_valid0 = false;
            }
            if (!pending_valid0) {
                pending_word0 = 0;
                pending_token0 = packet0.token_lane;
                pending_word_idx0 = word_idx0;
                pending_valid0 = true;
            }
            mm_input_block_t packed0 =
                pack_cc8_vector_result_half(packet0);
            if (half0 == 0) {
                pending_word0.range(MM_INPUT_BLOCK_BIT_WIDTH - 1, 0) =
                    packed0;
            } else {
                pending_word0.range(
                    2 * MM_INPUT_BLOCK_BIT_WIDTH - 1,
                    MM_INPUT_BLOCK_BIT_WIDTH
                ) = packed0;
            }
        }

        if (read1) {
            packet1 = core1_result_stream.read();
            word_idx1 = packet1.elem_base / FM_BLOCK_SIZE;
            half1 = (packet1.elem_base % FM_BLOCK_SIZE) / CU_VEC_LANES;
            bool same_word1 =
                pending_valid1 &&
                pending_token1 == packet1.token_lane &&
                pending_word_idx1 == word_idx1;
            if (pending_valid1 && !same_word1) {
                write_cc8_vector_output_word(
                    output_port1,
                    pending_token1,
                    CC8_TOKENS_PER_DATA_PORT,
                    pending_word_idx1,
                    pending_word1
                );
                pending_valid1 = false;
            }
            if (!pending_valid1) {
                pending_word1 = 0;
                pending_token1 = packet1.token_lane;
                pending_word_idx1 = word_idx1;
                pending_valid1 = true;
            }
            mm_input_block_t packed1 =
                pack_cc8_vector_result_half(packet1);
            if (half1 == 0) {
                pending_word1.range(MM_INPUT_BLOCK_BIT_WIDTH - 1, 0) =
                    packed1;
            } else {
                pending_word1.range(
                    2 * MM_INPUT_BLOCK_BIT_WIDTH - 1,
                    MM_INPUT_BLOCK_BIT_WIDTH
                ) = packed1;
            }
        }

        if (read0 && (half0 == 1 || packet0.last_stream)) {
            write_cc8_vector_output_word(
                output_port0,
                pending_token0,
                0,
                pending_word_idx0,
                pending_word0
            );
            pending_valid0 = false;
        }
        if (read1 && (half1 == 1 || packet1.last_stream)) {
            write_cc8_vector_output_word(
                output_port1,
                pending_token1,
                CC8_TOKENS_PER_DATA_PORT,
                pending_word_idx1,
                pending_word1
            );
            pending_valid1 = false;
        }
    }

    if (pending_valid0) {
        write_cc8_vector_output_word(
            output_port0,
            pending_token0,
            0,
            pending_word_idx0,
            pending_word0
        );
    }
    if (pending_valid1) {
        write_cc8_vector_output_word(
            output_port1,
            pending_token1,
            CC8_TOKENS_PER_DATA_PORT,
            pending_word_idx1,
            pending_word1
        );
    }
}

template <typename DestinationBufferT>
static void collect_cc8_vector_results_to_gbuf(
    DestinationBufferT& destination,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int token_count,
    unsigned int elem_count,
    unsigned int core0_packet_count,
    unsigned int core1_packet_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=destination.block complete dim=1

    unsigned int packet_count =
        core0_packet_count > core1_packet_count ?
        core0_packet_count : core1_packet_count;
    for (unsigned int packet_idx = 0;
         packet_idx < packet_count;
         packet_idx++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=0 max=CU_STREAM_MAX_PACKETS
        if (packet_idx < core0_packet_count) {
            cu_vec16_packet_t packet = core0_result_stream.read();
            store_cc8_result_packet(
                destination,
                packet,
                token_count,
                elem_count
            );
        }
        if (packet_idx < core1_packet_count) {
            cu_vec16_packet_t packet = core1_result_stream.read();
            store_cc8_result_packet(
                destination,
                packet,
                token_count,
                elem_count
            );
        }
    }
}

template <typename Buffer0T, typename Buffer1T>
static void run_cc8_generic_vector_exchange(
    fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT],
    hls::stream<cu_vec16_packet_t>& core0_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core0_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core1_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    const Buffer0T& gbuf0,
    const Buffer1T& gbuf1,
    cc8_operator_t op,
    unsigned int token_count,
    unsigned int core0_token_count,
    unsigned int core1_token_count,
    unsigned int rhs_token_slots,
    unsigned int rhs_valid_tokens,
    unsigned int elem_count,
    unsigned int core0_packet_count,
    unsigned int core1_packet_count
) {
    #pragma HLS inline off
    #pragma HLS dataflow

    emit_cc8_generic_vector_inputs(
        op,
        core0_vector_input0_stream,
        core0_vector_input1_stream,
        core1_vector_input0_stream,
        core1_vector_input1_stream,
        gbuf0,
        gbuf1,
        token_count,
        core0_token_count,
        core1_token_count,
        rhs_token_slots,
        rhs_valid_tokens,
        elem_count
    );
    collect_cc8_vector_results(
        output_port0,
        output_port1,
        core0_result_stream,
        core1_result_stream,
        core0_packet_count,
        core1_packet_count
    );
}

template <typename DestinationBufferT, typename Buffer0T, typename Buffer1T>
static void run_cc8_generic_vector_gbuf_exchange(
    DestinationBufferT& destination,
    hls::stream<cu_vec16_packet_t>& core0_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core0_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core1_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    const Buffer0T& source0,
    const Buffer1T& source1,
    cc8_operator_t op,
    unsigned int token_count,
    unsigned int core0_token_count,
    unsigned int core1_token_count,
    unsigned int rhs_token_slots,
    unsigned int rhs_valid_tokens,
    unsigned int elem_count,
    unsigned int core0_packet_count,
    unsigned int core1_packet_count
) {
    #pragma HLS inline off
    #pragma HLS dataflow

    emit_cc8_generic_vector_inputs(
        op,
        core0_vector_input0_stream,
        core0_vector_input1_stream,
        core1_vector_input0_stream,
        core1_vector_input1_stream,
        source0,
        source1,
        token_count,
        core0_token_count,
        core1_token_count,
        rhs_token_slots,
        rhs_valid_tokens,
        elem_count
    );
    collect_cc8_vector_results_to_gbuf(
        destination,
        core0_result_stream,
        core1_result_stream,
        token_count,
        elem_count,
        core0_packet_count,
        core1_packet_count
    );
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
    task.result_policy = CU8_RESULT_RELEASE;
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

template <typename Buffer0T, typename Buffer1T, typename DestinationBufferT>
static unsigned int run_cc8_vector_gbuf_task(
    hls::stream<cu8_task_t>& core0_task_stream,
    hls::stream<cu_vec16_packet_t>& core0_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core0_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu8_task_t>& core1_task_stream,
    hls::stream<cu_vec16_packet_t>& core1_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core1_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    const Buffer0T& source0,
    const Buffer1T& source1,
    DestinationBufferT& destination,
    cc8_operator_t op,
    cu8_mode_t mode,
    unsigned int token_count,
    unsigned int elem_count,
    unsigned int rhs_token_slots,
    unsigned int rhs_valid_tokens,
    bool end_compute_session
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=source0.block complete dim=1
    #pragma HLS array_partition variable=source1.block complete dim=1
    #pragma HLS array_partition variable=destination.block complete dim=1

    unsigned int core0_token_count =
        token_count < CC8_TOKENS_PER_DATA_PORT ?
        token_count : CC8_TOKENS_PER_DATA_PORT;
    unsigned int core1_token_count =
        token_count > CC8_TOKENS_PER_DATA_PORT ?
        token_count - CC8_TOKENS_PER_DATA_PORT : 0;
    unsigned int blocks_per_token = ceildiv(elem_count, CU_VEC_LANES);
    unsigned int core0_packet_count =
        core0_token_count * blocks_per_token;
    unsigned int core1_packet_count =
        core1_token_count * blocks_per_token;

    if (core0_token_count != 0) {
        core0_task_stream.write(build_cc8_compute_task(
            mode,
            0,
            core0_token_count,
            elem_count,
            core0_packet_count,
            0,
            0,
            end_compute_session
        ));
    } else if (end_compute_session) {
        core0_task_stream.write(build_cc8_stop_task());
    }
    if (core1_token_count != 0) {
        core1_task_stream.write(build_cc8_compute_task(
            mode,
            0,
            core1_token_count,
            elem_count,
            core1_packet_count,
            0,
            0,
            end_compute_session
        ));
    } else if (end_compute_session) {
        core1_task_stream.write(build_cc8_stop_task());
    }

    run_cc8_generic_vector_gbuf_exchange(
        destination,
        core0_vector_input0_stream,
        core0_vector_input1_stream,
        core0_result_stream,
        core1_vector_input0_stream,
        core1_vector_input1_stream,
        core1_result_stream,
        source0,
        source1,
        op,
        token_count,
        core0_token_count,
        core1_token_count,
        rhs_token_slots,
        rhs_valid_tokens,
        elem_count,
        core0_packet_count,
        core1_packet_count
    );
    return (core0_token_count != 0 ? 1u : 0u) +
        (core1_token_count != 0 ? 1u : 0u);
}

// A binary vector result cannot be written back to its left-hand GBUF from
// the same DATAFLOW region that is still reading that GBUF.  HLS treats that
// as a read-before-write channel, and a sequential send/receive window is not
// sufficient either: a deep vector pipeline may need more input packets than
// a shallow window contains before it can produce its first result.  Drain
// the complete result stream into a BRAM scratch buffer while the complete
// input stream is emitted, then commit the scratch buffer after DATAFLOW has
// finished.  This keeps the inter-kernel FIFOs shallow without making
// correctness depend on the vector operator's pipeline depth.
template <typename InOutBufferT, typename RhsBufferT>
static unsigned int run_cc8_vector_gbuf_task_inplace_binary(
    hls::stream<cu8_task_t>& core0_task_stream,
    hls::stream<cu_vec16_packet_t>& core0_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core0_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu8_task_t>& core1_task_stream,
    hls::stream<cu_vec16_packet_t>& core1_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core1_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    InOutBufferT& inout,
    const RhsBufferT& rhs,
    cu8_mode_t mode,
    unsigned int token_count,
    unsigned int elem_count,
    bool end_compute_session
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=inout.block complete dim=1
    #pragma HLS array_partition variable=rhs.block complete dim=1

    InOutBufferT result_scratch;
    #pragma HLS bind_storage variable=result_scratch.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=result_scratch.block complete dim=1

    unsigned int core0_token_count =
        token_count < CC8_TOKENS_PER_DATA_PORT ?
        token_count : CC8_TOKENS_PER_DATA_PORT;
    unsigned int core1_token_count =
        token_count > CC8_TOKENS_PER_DATA_PORT ?
        token_count - CC8_TOKENS_PER_DATA_PORT : 0;
    unsigned int block_count = ceildiv(elem_count, CU_VEC_LANES);
    unsigned int core0_packet_count = core0_token_count * block_count;
    unsigned int core1_packet_count = core1_token_count * block_count;

    if (core0_token_count != 0) {
        core0_task_stream.write(build_cc8_compute_task(
            mode,
            0,
            core0_token_count,
            elem_count,
            core0_packet_count,
            0,
            0,
            end_compute_session
        ));
    } else if (end_compute_session) {
        core0_task_stream.write(build_cc8_stop_task());
    }
    if (core1_token_count != 0) {
        core1_task_stream.write(build_cc8_compute_task(
            mode,
            0,
            core1_token_count,
            elem_count,
            core1_packet_count,
            0,
            0,
            end_compute_session
        ));
    } else if (end_compute_session) {
        core1_task_stream.write(build_cc8_stop_task());
    }

    cc8_operator_t op = mode == CU8_MODE_SILU_MUL ?
        CC8_OP_SILU_MUL : CC8_OP_RESIDUAL_ADD;
    run_cc8_generic_vector_gbuf_exchange(
        result_scratch,
        core0_vector_input0_stream,
        core0_vector_input1_stream,
        core0_result_stream,
        core1_vector_input0_stream,
        core1_vector_input1_stream,
        core1_result_stream,
        inout,
        rhs,
        op,
        token_count,
        core0_token_count,
        core1_token_count,
        token_count,
        token_count,
        elem_count,
        core0_packet_count,
        core1_packet_count
    );

    for (unsigned int token = 0;
         token < CC8_GBUF_TOKEN_ROWS;
         token++) {
        #pragma HLS loop_tripcount min=1 max=CC8_GBUF_TOKEN_ROWS
        for (unsigned int block = 0;
             block < InOutBufferT::kBlockCount;
             block++) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=1 max=MAX_LINEAR_IN_BLOCKS
            if (token < token_count && block < block_count) {
                inout.block[token][block] =
                    result_scratch.block[token][block];
            }
        }
    }

    return (core0_token_count != 0 ? 1u : 0u) +
        (core1_token_count != 0 ? 1u : 0u);
}

template <typename SourceBufferT, typename DestinationBufferT>
static void run_cc8_projection_waves_flat(
    hls::stream<cu8_task_t>& core0_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core0_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu8_task_t>& core1_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core1_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    const SourceBufferT& source,
    DestinationBufferT& destination,
    unsigned int wave_begin,
    unsigned int wave_end,
    unsigned int token_count,
    unsigned int active_in_dim,
    unsigned int matrix_in_dim,
    unsigned int out_dim,
    weight_addr_t weight_base,
    unsigned int core_mask,
    bool zero_weight_stream,
    bool end_compute_session,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline off

    unsigned int wave_count = wave_end > wave_begin ?
        wave_end - wave_begin : 0;

#if CC8_ENABLE_MM_WAVE_REPEAT
    cu8_task_t repeated_task0 = build_cc8_compute_task(
        CU8_MODE_MM,
        active_in_dim,
        token_count,
        MM_STREAM_8X64_OUTPUTS,
        MM_STREAM_8X64_PACKETS_PER_BLOCK,
        wave_begin * CC8_OUTPUTS_PER_WAVE,
        wave_begin,
        end_compute_session
    );
    repeated_task0.repeat_count = wave_count;
    repeated_task0.elem_stride = CC8_OUTPUTS_PER_WAVE;
    repeated_task0.block_stride = 1;
    cu8_task_t repeated_task1 = repeated_task0;
    repeated_task1.elem_base += MM_STREAM_8X64_OUTPUTS;
    if ((core_mask & 1u) != 0u) {
        core0_task_stream.write(repeated_task0);
    } else if (end_compute_session) {
        core0_task_stream.write(build_cc8_stop_task());
    }
    if ((core_mask & 2u) != 0u) {
        core1_task_stream.write(repeated_task1);
    } else if (end_compute_session) {
        core1_task_stream.write(build_cc8_stop_task());
    }
#endif

    for (unsigned int output_wave = wave_begin;
         output_wave < wave_end;
         output_wave++) {
        #pragma HLS loop_tripcount min=1 max=CC8_MAX_OUTPUT_WAVES
#if !CC8_ENABLE_MM_WAVE_REPEAT
        bool last_wave = output_wave + 1 == wave_end;
        bool last_task = end_compute_session && last_wave;
        cu8_task_t task0 = build_cc8_compute_task(
            CU8_MODE_MM,
            active_in_dim,
            token_count,
            MM_STREAM_8X64_OUTPUTS,
            MM_STREAM_8X64_PACKETS_PER_BLOCK,
            output_wave * CC8_OUTPUTS_PER_WAVE,
            output_wave,
            last_task
        );
        cu8_task_t task1 = task0;
        task1.elem_base += MM_STREAM_8X64_OUTPUTS;
        if ((core_mask & 1u) != 0u) {
            core0_task_stream.write(task0);
        } else if (end_compute_session && last_wave) {
            core0_task_stream.write(build_cc8_stop_task());
        }
        if ((core_mask & 2u) != 0u) {
            core1_task_stream.write(task1);
        } else if (end_compute_session && last_wave) {
            core1_task_stream.write(build_cc8_stop_task());
        }
#endif

        run_cc8_mm_wave_flat(
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
            source,
            destination,
            output_wave,
            token_count,
            active_in_dim,
            matrix_in_dim,
            out_dim,
            weight_base,
            core_mask,
            zero_weight_stream,
            QWEN_WEIGHT_SHARD_ARGS
        );
    }
}

// All resident projections use the same hardware scheduler and weight
// loader.  Buffer capacity is an address stride, not a C++ template
// parameter of the heavy datapath; otherwise HLS creates one complete copy
// for hidden->hidden, hidden->FFN and FFN->hidden projections.
template <typename SourceBufferT, typename DestinationBufferT>
static void run_cc8_projection_waves(
    hls::stream<cu8_task_t>& core0_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core0_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu8_task_t>& core1_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core1_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    const SourceBufferT& source,
    DestinationBufferT& destination,
    unsigned int wave_begin,
    unsigned int wave_end,
    unsigned int token_count,
    unsigned int active_in_dim,
    unsigned int matrix_in_dim,
    unsigned int out_dim,
    weight_addr_t weight_base,
    unsigned int core_mask,
    bool zero_weight_stream,
    bool end_compute_session,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline
    #pragma HLS array_partition variable=source.block complete dim=1
    #pragma HLS array_partition variable=destination.block complete dim=1

    run_cc8_projection_waves_flat(
        core0_task_stream,
        core0_activation_stream,
        core0_weight_stream0,
        core0_weight_stream1,
        core0_weight_stream2,
        core0_weight_stream3,
        core0_result_stream,
        core1_task_stream,
        core1_activation_stream,
        core1_weight_stream0,
        core1_weight_stream1,
        core1_weight_stream2,
        core1_weight_stream3,
        core1_result_stream,
        source,
        destination,
        wave_begin,
        wave_end,
        token_count,
        active_in_dim,
        matrix_in_dim,
        out_dim,
        weight_base,
        core_mask,
        zero_weight_stream,
        end_compute_session,
        QWEN_WEIGHT_SHARD_ARGS
    );
}

enum cc8_projection_bank_t {
    CC8_PROJECTION_HIDDEN0 = 0,
    CC8_PROJECTION_HIDDEN1 = 1,
    CC8_PROJECTION_WIDE0 = 2,
    CC8_PROJECTION_WIDE1 = 3
};

struct cc8_mm_wave_result_layout_t {
    static constexpr unsigned int kDataBits =
        CU_VEC_LANES * fm_t::width;
    static constexpr unsigned int kValidMaskOffset = kDataBits;
    static constexpr unsigned int kTokenLaneOffset =
        kValidMaskOffset + CU_VEC_LANES;
    static constexpr unsigned int kElemBaseOffset =
        kTokenLaneOffset + 32;
    static constexpr unsigned int kWordBits =
        kElemBaseOffset + 32;

    using word_t = ap_uint<kWordBits>;
};

static cc8_mm_wave_result_layout_t::word_t pack_cc8_mm_wave_result(
    const cu_vec16_packet_t& packet
) {
    #pragma HLS inline
    cc8_mm_wave_result_layout_t::word_t word = 0;
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        #pragma HLS unroll
        word.range(
            (lane + 1) * fm_t::width - 1,
            lane * fm_t::width
        ) = packet.data[lane].range(fm_t::width - 1, 0);
    }
    word.range(
        cc8_mm_wave_result_layout_t::kTokenLaneOffset - 1,
        cc8_mm_wave_result_layout_t::kValidMaskOffset
    ) = packet.valid_mask;
    word.range(
        cc8_mm_wave_result_layout_t::kElemBaseOffset - 1,
        cc8_mm_wave_result_layout_t::kTokenLaneOffset
    ) = packet.token_lane;
    word.range(
        cc8_mm_wave_result_layout_t::kWordBits - 1,
        cc8_mm_wave_result_layout_t::kElemBaseOffset
    ) = packet.elem_base;
    return word;
}

static cu_vec16_packet_t unpack_cc8_mm_wave_result(
    const cc8_mm_wave_result_layout_t::word_t& word
) {
    #pragma HLS inline
    cu_vec16_packet_t packet;
    #pragma HLS array_partition variable=packet.data complete dim=1
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        #pragma HLS unroll
        packet.data[lane].range(fm_t::width - 1, 0) = word.range(
            (lane + 1) * fm_t::width - 1,
            lane * fm_t::width
        );
    }
    packet.valid_mask = word.range(
        cc8_mm_wave_result_layout_t::kTokenLaneOffset - 1,
        cc8_mm_wave_result_layout_t::kValidMaskOffset
    );
    packet.token_lane = word.range(
        cc8_mm_wave_result_layout_t::kElemBaseOffset - 1,
        cc8_mm_wave_result_layout_t::kTokenLaneOffset
    ).to_uint();
    packet.elem_base = word.range(
        cc8_mm_wave_result_layout_t::kWordBits - 1,
        cc8_mm_wave_result_layout_t::kElemBaseOffset
    ).to_uint();
    packet.block_id = 0;
    packet.last_block = false;
    packet.last_stream = false;
    return packet;
}

static mm_input_block_t read_cc8_projection_bank(
    const cc8_hidden_buffer_t& hidden0,
    const cc8_hidden_buffer_t& hidden1,
    const cc8_global_buffer_t& wide0,
    const cc8_global_buffer_t& wide1,
    cc8_projection_bank_t bank,
    unsigned int token,
    unsigned int block
) {
    #pragma HLS inline
    #pragma HLS array_partition variable=hidden0.block complete dim=1
    #pragma HLS array_partition variable=hidden1.block complete dim=1
    #pragma HLS array_partition variable=wide0.block complete dim=1
    #pragma HLS array_partition variable=wide1.block complete dim=1

    if (bank == CC8_PROJECTION_HIDDEN0) {
        return block < cc8_hidden_buffer_t::kBlockCount ?
            hidden0.block[token][block] : mm_input_block_t(0);
    }
    if (bank == CC8_PROJECTION_HIDDEN1) {
        return block < cc8_hidden_buffer_t::kBlockCount ?
            hidden1.block[token][block] : mm_input_block_t(0);
    }
    if (bank == CC8_PROJECTION_WIDE0) {
        return wide0.block[token][block];
    }
    return wide1.block[token][block];
}

static void emit_cc8_mm_wave_inputs_banked(
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
    hls::stream<mm_stream_8x64_weight_packet_t>
        tile_stream[CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS],
    const cc8_hidden_buffer_t& hidden0,
    const cc8_hidden_buffer_t& hidden1,
    const cc8_global_buffer_t& wide0,
    const cc8_global_buffer_t& wide1,
    cc8_projection_bank_t source_bank,
    unsigned int token_count,
    unsigned int in_dim,
    unsigned int core_mask
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=tile_stream complete dim=0
    #pragma HLS array_partition variable=hidden0.block complete dim=1
    #pragma HLS array_partition variable=hidden1.block complete dim=1
    #pragma HLS array_partition variable=wide0.block complete dim=1
    #pragma HLS array_partition variable=wide1.block complete dim=1

    for (unsigned int k = 0; k < in_dim; k++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=16 max=MAX_LINEAR_IN_DIM avg=HIDDEN_SIZE
        unsigned int in_tile = k / MM_PE_IN;
        unsigned int k_lane = k - in_tile * MM_PE_IN;

        mm_stream_8x64_activation_packet_t activation;
        #pragma HLS array_partition variable=activation.data complete dim=1
        for (unsigned int token = 0;
             token < MM_STREAM_8X64_TOKENS;
             token++) {
            #pragma HLS unroll
            mm_input_block_t source_block =
                token < token_count ?
                read_cc8_projection_bank(
                    hidden0,
                    hidden1,
                    wide0,
                    wide1,
                    source_bank,
                    token,
                    in_tile
                ) :
                mm_input_block_t(0);
            activation.data[token] =
                unpack_mm_input_block_lane(source_block, k_lane);
        }

        if ((core_mask & 1u) != 0u) {
            core0_activation_stream.write(activation);
            core0_weight_stream0.write(tile_stream[0][0].read());
            core0_weight_stream1.write(tile_stream[0][1].read());
            core0_weight_stream2.write(tile_stream[0][2].read());
            core0_weight_stream3.write(tile_stream[0][3].read());
        }
        if ((core_mask & 2u) != 0u) {
            core1_activation_stream.write(activation);
            core1_weight_stream0.write(tile_stream[1][0].read());
            core1_weight_stream1.write(tile_stream[1][1].read());
            core1_weight_stream2.write(tile_stream[1][2].read());
            core1_weight_stream3.write(tile_stream[1][3].read());
        }
    }
}

static void collect_cc8_mm_wave_results_buffered(
    hls::stream<cc8_mm_wave_result_layout_t::word_t>& result0_stream,
    hls::stream<cc8_mm_wave_result_layout_t::word_t>& result1_stream,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int core_mask
) {
    #pragma HLS inline off

    for (unsigned int packet_idx = 0;
         packet_idx < MM_STREAM_8X64_PACKETS_PER_BLOCK;
         packet_idx++) {
        #pragma HLS pipeline II=1
        if ((core_mask & 1u) != 0u) {
            result0_stream.write(pack_cc8_mm_wave_result(
                core0_result_stream.read()
            ));
        }
        if ((core_mask & 2u) != 0u) {
            result1_stream.write(pack_cc8_mm_wave_result(
                core1_result_stream.read()
            ));
        }
    }
}

static void store_cc8_projection_result_packet(
    cc8_hidden_buffer_t& hidden0,
    cc8_hidden_buffer_t& hidden1,
    cc8_global_buffer_t& wide0,
    cc8_global_buffer_t& wide1,
    cc8_projection_bank_t destination_bank,
    const cu_vec16_packet_t& packet,
    unsigned int token_count,
    unsigned int out_dim
) {
    #pragma HLS inline
    if (destination_bank == CC8_PROJECTION_HIDDEN0) {
        store_cc8_result_packet(hidden0, packet, token_count, out_dim);
    } else if (destination_bank == CC8_PROJECTION_HIDDEN1) {
        store_cc8_result_packet(hidden1, packet, token_count, out_dim);
    } else if (destination_bank == CC8_PROJECTION_WIDE0) {
        store_cc8_result_packet(wide0, packet, token_count, out_dim);
    } else {
        store_cc8_result_packet(wide1, packet, token_count, out_dim);
    }
}

static void commit_cc8_mm_wave_results(
    cc8_hidden_buffer_t& hidden0,
    cc8_hidden_buffer_t& hidden1,
    cc8_global_buffer_t& wide0,
    cc8_global_buffer_t& wide1,
    hls::stream<cc8_mm_wave_result_layout_t::word_t>& result0_stream,
    hls::stream<cc8_mm_wave_result_layout_t::word_t>& result1_stream,
    cc8_projection_bank_t destination_bank,
    unsigned int token_count,
    unsigned int out_dim,
    unsigned int core_mask
) {
    #pragma HLS inline off

    for (unsigned int packet_idx = 0;
         packet_idx < MM_STREAM_8X64_PACKETS_PER_BLOCK;
        packet_idx++) {
        #pragma HLS pipeline II=1
        if ((core_mask & 1u) != 0u) {
            cu_vec16_packet_t packet0 = unpack_cc8_mm_wave_result(
                result0_stream.read()
            );
            store_cc8_projection_result_packet(
                hidden0,
                hidden1,
                wide0,
                wide1,
                destination_bank,
                packet0,
                token_count,
                out_dim
            );
        }
        if ((core_mask & 2u) != 0u) {
            cu_vec16_packet_t packet1 = unpack_cc8_mm_wave_result(
                result1_stream.read()
            );
            store_cc8_projection_result_packet(
                hidden0,
                hidden1,
                wide0,
                wide1,
                destination_bank,
                packet1,
                token_count,
                out_dim
            );
        }
    }
}

static void drive_cc8_projection_wave_range_banked(
    hls::stream<cu8_task_t>& core0_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core0_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream3,
    hls::stream<cu8_task_t>& core1_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core1_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream3,
    hls::stream<mm_stream_8x64_weight_packet_t>
        tile_stream[CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS],
    const cc8_hidden_buffer_t& hidden0,
    const cc8_hidden_buffer_t& hidden1,
    const cc8_global_buffer_t& wide0,
    const cc8_global_buffer_t& wide1,
    cc8_projection_bank_t source_bank,
    unsigned int wave_begin,
    unsigned int wave_end,
    unsigned int token_count,
    unsigned int active_in_dim,
    unsigned int core_mask,
    bool end_compute_session
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=tile_stream complete dim=0
    #pragma HLS array_partition variable=hidden0.block complete dim=1
    #pragma HLS array_partition variable=hidden1.block complete dim=1
    #pragma HLS array_partition variable=wide0.block complete dim=1
    #pragma HLS array_partition variable=wide1.block complete dim=1

    unsigned int wave_count = wave_end > wave_begin ?
        wave_end - wave_begin : 0;
#if CC8_ENABLE_MM_WAVE_REPEAT
    if (wave_count != 0) {
        cu8_task_t repeated_task0 = build_cc8_compute_task(
            CU8_MODE_MM,
            active_in_dim,
            token_count,
            MM_STREAM_8X64_OUTPUTS,
            MM_STREAM_8X64_PACKETS_PER_BLOCK,
            wave_begin * CC8_OUTPUTS_PER_WAVE,
            wave_begin,
            end_compute_session
        );
        repeated_task0.repeat_count = wave_count;
        repeated_task0.elem_stride = CC8_OUTPUTS_PER_WAVE;
        repeated_task0.block_stride = 1;
        cu8_task_t repeated_task1 = repeated_task0;
        repeated_task1.elem_base += MM_STREAM_8X64_OUTPUTS;
        if ((core_mask & 1u) != 0u) {
            core0_task_stream.write(repeated_task0);
        } else if (end_compute_session) {
            core0_task_stream.write(build_cc8_stop_task());
        }
        if ((core_mask & 2u) != 0u) {
            core1_task_stream.write(repeated_task1);
        } else if (end_compute_session) {
            core1_task_stream.write(build_cc8_stop_task());
        }
    }
#endif

    for (unsigned int output_wave = wave_begin;
         output_wave < wave_end;
         output_wave++) {
        #pragma HLS loop_tripcount min=1 max=CC8_MAX_OUTPUT_WAVES
#if !CC8_ENABLE_MM_WAVE_REPEAT
        bool last_wave = output_wave + 1 == wave_end;
        bool last_task = end_compute_session && last_wave;
        cu8_task_t task0 = build_cc8_compute_task(
            CU8_MODE_MM,
            active_in_dim,
            token_count,
            MM_STREAM_8X64_OUTPUTS,
            MM_STREAM_8X64_PACKETS_PER_BLOCK,
            output_wave * CC8_OUTPUTS_PER_WAVE,
            output_wave,
            last_task
        );
        cu8_task_t task1 = task0;
        task1.elem_base += MM_STREAM_8X64_OUTPUTS;
        if ((core_mask & 1u) != 0u) {
            core0_task_stream.write(task0);
        } else if (end_compute_session && last_wave) {
            core0_task_stream.write(build_cc8_stop_task());
        }
        if ((core_mask & 2u) != 0u) {
            core1_task_stream.write(task1);
        } else if (end_compute_session && last_wave) {
            core1_task_stream.write(build_cc8_stop_task());
        }
#endif

        emit_cc8_mm_wave_inputs_banked(
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
            hidden0,
            hidden1,
            wide0,
            wide1,
            source_bank,
            token_count,
            active_in_dim,
            core_mask
        );
    }
}

static void collect_cc8_projection_wave_range(
    hls::stream<cc8_mm_wave_result_layout_t::word_t>& result0_stream,
    hls::stream<cc8_mm_wave_result_layout_t::word_t>& result1_stream,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    unsigned int wave_begin,
    unsigned int wave_end,
    unsigned int core_mask
) {
    #pragma HLS inline off

    for (unsigned int output_wave = wave_begin;
         output_wave < wave_end;
         output_wave++) {
        #pragma HLS loop_tripcount min=1 max=CC8_MAX_OUTPUT_WAVES
        collect_cc8_mm_wave_results_buffered(
            result0_stream,
            result1_stream,
            core0_result_stream,
            core1_result_stream,
            core_mask
        );
    }
}

static void commit_cc8_projection_wave_range(
    cc8_global_buffer_t& projection_scratch,
    hls::stream<cc8_mm_wave_result_layout_t::word_t>& result0_stream,
    hls::stream<cc8_mm_wave_result_layout_t::word_t>& result1_stream,
    unsigned int wave_begin,
    unsigned int wave_end,
    unsigned int token_count,
    unsigned int out_dim,
    unsigned int core_mask
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=projection_scratch.block complete dim=1

    for (unsigned int output_wave = wave_begin;
         output_wave < wave_end;
         output_wave++) {
        #pragma HLS loop_tripcount min=1 max=CC8_MAX_OUTPUT_WAVES
        for (unsigned int packet_idx = 0;
             packet_idx < MM_STREAM_8X64_PACKETS_PER_BLOCK;
             packet_idx++) {
            #pragma HLS pipeline II=1
            if ((core_mask & 1u) != 0u) {
                cu_vec16_packet_t packet0 = unpack_cc8_mm_wave_result(
                    result0_stream.read()
                );
                store_cc8_result_packet(
                    projection_scratch,
                    packet0,
                    token_count,
                    out_dim
                );
            }
            if ((core_mask & 2u) != 0u) {
                cu_vec16_packet_t packet1 = unpack_cc8_mm_wave_result(
                    result1_stream.read()
                );
                store_cc8_result_packet(
                    projection_scratch,
                    packet1,
                    token_count,
                    out_dim
                );
            }
        }
    }
}

static void copy_cc8_projection_scratch_to_bank(
    cc8_hidden_buffer_t& hidden0,
    cc8_hidden_buffer_t& hidden1,
    cc8_global_buffer_t& wide0,
    cc8_global_buffer_t& wide1,
    const cc8_global_buffer_t& projection_scratch,
    cc8_projection_bank_t destination_bank,
    unsigned int token_count,
    unsigned int out_dim
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=hidden0.block complete dim=1
    #pragma HLS array_partition variable=hidden1.block complete dim=1
    #pragma HLS array_partition variable=wide0.block complete dim=1
    #pragma HLS array_partition variable=wide1.block complete dim=1
    #pragma HLS array_partition variable=projection_scratch.block complete dim=1

    unsigned int block_count = ceildiv(out_dim, MM_PE_IN);
    for (unsigned int token = 0;
         token < CC8_GBUF_TOKEN_ROWS;
         token++) {
        #pragma HLS loop_tripcount min=1 max=CC8_GBUF_TOKEN_ROWS
        for (unsigned int block = 0;
             block < CC8_GBUF_BLOCKS;
             block++) {
            #pragma HLS pipeline II=1
            if (token < token_count && block < block_count) {
                mm_input_block_t value =
                    projection_scratch.block[token][block];
                if (destination_bank == CC8_PROJECTION_HIDDEN0) {
                    if (block < cc8_hidden_buffer_t::kBlockCount) {
                        hidden0.block[token][block] = value;
                    }
                } else if (destination_bank == CC8_PROJECTION_HIDDEN1) {
                    if (block < cc8_hidden_buffer_t::kBlockCount) {
                        hidden1.block[token][block] = value;
                    }
                } else if (destination_bank == CC8_PROJECTION_WIDE0) {
                    wide0.block[token][block] = value;
                } else {
                    wide1.block[token][block] = value;
                }
            }
        }
    }
}

static void run_cc8_mm_wave_banked(
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
    hls::stream<cc8_mm_wave_result_layout_t::word_t>& result0_stream,
    hls::stream<cc8_mm_wave_result_layout_t::word_t>& result1_stream,
    const cc8_hidden_buffer_t& hidden0,
    const cc8_hidden_buffer_t& hidden1,
    const cc8_global_buffer_t& wide0,
    const cc8_global_buffer_t& wide1,
    cc8_projection_bank_t source_bank,
    unsigned int output_wave,
    unsigned int token_count,
    unsigned int active_in_dim,
    unsigned int matrix_in_dim,
    unsigned int out_dim,
    weight_addr_t weight_base,
    unsigned int core_mask,
    bool zero_weight_stream,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline off

    hls::stream<mm_stream_8x64_weight_packet_t>
        tile_stream[CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS];
    #pragma HLS array_partition variable=tile_stream complete dim=0
    #pragma HLS stream variable=tile_stream depth=16
    #pragma HLS bind_storage variable=tile_stream type=fifo impl=bram
    #pragma HLS dataflow

    load_cc8_weight_panels(
        tile_stream,
        output_wave,
        active_in_dim,
        matrix_in_dim,
        out_dim,
        weight_base,
        core_mask,
        zero_weight_stream,
        QWEN_WEIGHT_SHARD_ARGS
    );
    emit_cc8_mm_wave_inputs_banked(
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
        hidden0,
        hidden1,
        wide0,
        wide1,
        source_bank,
        token_count,
        active_in_dim,
        core_mask
    );
    collect_cc8_mm_wave_results_buffered(
        result0_stream,
        result1_stream,
        core0_result_stream,
        core1_result_stream,
        core_mask
    );
}

static void run_cc8_projection_wave_range_overlapped(
    hls::stream<cu8_task_t>& core0_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core0_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu8_task_t>& core1_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core1_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    const cc8_hidden_buffer_t& hidden0,
    const cc8_hidden_buffer_t& hidden1,
    const cc8_global_buffer_t& wide0,
    const cc8_global_buffer_t& wide1,
    cc8_global_buffer_t& projection_scratch,
    cc8_projection_bank_t source_bank,
    unsigned int wave_begin,
    unsigned int wave_end,
    unsigned int token_count,
    unsigned int active_in_dim,
    unsigned int matrix_in_dim,
    unsigned int out_dim,
    weight_addr_t weight_base,
    unsigned int core_mask,
    bool zero_weight_stream,
    bool end_compute_session,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=hidden0.block complete dim=1
    #pragma HLS array_partition variable=hidden1.block complete dim=1
    #pragma HLS array_partition variable=wide0.block complete dim=1
    #pragma HLS array_partition variable=wide1.block complete dim=1
    #pragma HLS array_partition variable=projection_scratch.block complete dim=1

    hls::stream<mm_stream_8x64_weight_packet_t>
        tile_stream[CC8_MM_CORE_COUNT][MM_STREAM_8X64_WEIGHT_GROUPS];
    #pragma HLS array_partition variable=tile_stream complete dim=0
    #pragma HLS stream variable=tile_stream depth=16
    #pragma HLS bind_storage variable=tile_stream type=fifo impl=bram

    hls::stream<cc8_mm_wave_result_layout_t::word_t>
        wave_result_stream[CC8_MM_CORE_COUNT];
    #pragma HLS array_partition variable=wave_result_stream complete dim=1
    #pragma HLS stream variable=wave_result_stream depth=CC8_MM_WAVE_RESULT_FIFO_DEPTH_VALUE
    #pragma HLS bind_storage variable=wave_result_stream type=fifo impl=bram
    #pragma HLS dataflow

    load_cc8_weight_panel_range(
        tile_stream,
        wave_begin,
        wave_end,
        active_in_dim,
        matrix_in_dim,
        out_dim,
        weight_base,
        core_mask,
        zero_weight_stream,
        QWEN_WEIGHT_SHARD_ARGS
    );
    drive_cc8_projection_wave_range_banked(
        core0_task_stream,
        core0_activation_stream,
        core0_weight_stream0,
        core0_weight_stream1,
        core0_weight_stream2,
        core0_weight_stream3,
        core1_task_stream,
        core1_activation_stream,
        core1_weight_stream0,
        core1_weight_stream1,
        core1_weight_stream2,
        core1_weight_stream3,
        tile_stream,
        hidden0,
        hidden1,
        wide0,
        wide1,
        source_bank,
        wave_begin,
        wave_end,
        token_count,
        active_in_dim,
        core_mask,
        end_compute_session
    );
    collect_cc8_projection_wave_range(
        wave_result_stream[0],
        wave_result_stream[1],
        core0_result_stream,
        core1_result_stream,
        wave_begin,
        wave_end,
        core_mask
    );
    commit_cc8_projection_wave_range(
        projection_scratch,
        wave_result_stream[0],
        wave_result_stream[1],
        wave_begin,
        wave_end,
        token_count,
        out_dim,
        core_mask
    );
}

static void run_cc8_projection_waves_banked(
    hls::stream<cu8_task_t>& core0_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core0_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu8_task_t>& core1_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core1_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    cc8_hidden_buffer_t& hidden0,
    cc8_hidden_buffer_t& hidden1,
    cc8_global_buffer_t& wide0,
    cc8_global_buffer_t& wide1,
    cc8_projection_bank_t source_bank,
    cc8_projection_bank_t destination_bank,
    unsigned int wave_begin,
    unsigned int wave_end,
    unsigned int token_count,
    unsigned int active_in_dim,
    unsigned int matrix_in_dim,
    unsigned int out_dim,
    weight_addr_t weight_base,
    unsigned int core_mask,
    bool zero_weight_stream,
    bool end_compute_session,
    QWEN_WEIGHT_SHARD_PARAMS
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=hidden0.block complete dim=1
    #pragma HLS array_partition variable=hidden1.block complete dim=1
    #pragma HLS array_partition variable=wide0.block complete dim=1
    #pragma HLS array_partition variable=wide1.block complete dim=1

#if CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW
    cc8_global_buffer_t projection_scratch;
    #pragma HLS bind_storage variable=projection_scratch.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=projection_scratch.block complete dim=1

    // The overlap region only reads the selected source banks and writes the
    // local projection scratch.  A batched copy after DATAFLOW avoids the
    // read/write parameter alias that Vitis HLS rejects for dynamic GBUF
    // selection, while still removing every per-wave commit barrier.
    run_cc8_projection_wave_range_overlapped(
        core0_task_stream,
        core0_activation_stream,
        core0_weight_stream0,
        core0_weight_stream1,
        core0_weight_stream2,
        core0_weight_stream3,
        core0_result_stream,
        core1_task_stream,
        core1_activation_stream,
        core1_weight_stream0,
        core1_weight_stream1,
        core1_weight_stream2,
        core1_weight_stream3,
        core1_result_stream,
        hidden0,
        hidden1,
        wide0,
        wide1,
        projection_scratch,
        source_bank,
        wave_begin,
        wave_end,
        token_count,
        active_in_dim,
        matrix_in_dim,
        out_dim,
        weight_base,
        core_mask,
        zero_weight_stream,
        end_compute_session,
        QWEN_WEIGHT_SHARD_ARGS
    );
    copy_cc8_projection_scratch_to_bank(
        hidden0,
        hidden1,
        wide0,
        wide1,
        projection_scratch,
        destination_bank,
        token_count,
        out_dim
    );
#else
    hls::stream<cc8_mm_wave_result_layout_t::word_t>
        wave_result_stream[CC8_MM_CORE_COUNT];
    #pragma HLS array_partition variable=wave_result_stream complete dim=1
    #pragma HLS stream variable=wave_result_stream depth=CC8_MM_WAVE_RESULT_FIFO_DEPTH_VALUE
    #pragma HLS bind_storage variable=wave_result_stream type=fifo impl=bram

    unsigned int wave_count = wave_end > wave_begin ?
        wave_end - wave_begin : 0;
#if CC8_ENABLE_MM_WAVE_REPEAT
    cu8_task_t repeated_task0 = build_cc8_compute_task(
        CU8_MODE_MM,
        active_in_dim,
        token_count,
        MM_STREAM_8X64_OUTPUTS,
        MM_STREAM_8X64_PACKETS_PER_BLOCK,
        wave_begin * CC8_OUTPUTS_PER_WAVE,
        wave_begin,
        end_compute_session
    );
    repeated_task0.repeat_count = wave_count;
    repeated_task0.elem_stride = CC8_OUTPUTS_PER_WAVE;
    repeated_task0.block_stride = 1;
    cu8_task_t repeated_task1 = repeated_task0;
    repeated_task1.elem_base += MM_STREAM_8X64_OUTPUTS;
    if ((core_mask & 1u) != 0u) {
        core0_task_stream.write(repeated_task0);
    } else if (end_compute_session) {
        core0_task_stream.write(build_cc8_stop_task());
    }
    if ((core_mask & 2u) != 0u) {
        core1_task_stream.write(repeated_task1);
    } else if (end_compute_session) {
        core1_task_stream.write(build_cc8_stop_task());
    }
#endif

    for (unsigned int output_wave = wave_begin;
         output_wave < wave_end;
         output_wave++) {
        #pragma HLS loop_tripcount min=1 max=CC8_MAX_OUTPUT_WAVES
#if !CC8_ENABLE_MM_WAVE_REPEAT
        bool last_wave = output_wave + 1 == wave_end;
        bool last_task = end_compute_session && last_wave;
        cu8_task_t task0 = build_cc8_compute_task(
            CU8_MODE_MM,
            active_in_dim,
            token_count,
            MM_STREAM_8X64_OUTPUTS,
            MM_STREAM_8X64_PACKETS_PER_BLOCK,
            output_wave * CC8_OUTPUTS_PER_WAVE,
            output_wave,
            last_task
        );
        cu8_task_t task1 = task0;
        task1.elem_base += MM_STREAM_8X64_OUTPUTS;
        if ((core_mask & 1u) != 0u) {
            core0_task_stream.write(task0);
        } else if (end_compute_session && last_wave) {
            core0_task_stream.write(build_cc8_stop_task());
        }
        if ((core_mask & 2u) != 0u) {
            core1_task_stream.write(task1);
        } else if (end_compute_session && last_wave) {
            core1_task_stream.write(build_cc8_stop_task());
        }
#endif

        run_cc8_mm_wave_banked(
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
            wave_result_stream[0],
            wave_result_stream[1],
            hidden0,
            hidden1,
            wide0,
            wide1,
            source_bank,
            output_wave,
            token_count,
            active_in_dim,
            matrix_in_dim,
            out_dim,
            weight_base,
            core_mask,
            zero_weight_stream,
            QWEN_WEIGHT_SHARD_ARGS
        );
        commit_cc8_mm_wave_results(
            hidden0,
            hidden1,
            wide0,
            wide1,
            wave_result_stream[0],
            wave_result_stream[1],
            destination_bank,
            token_count,
            out_dim,
            core_mask
        );
    }
#endif
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
    #pragma HLS array_partition variable=panel.word complete dim=0

    for (unsigned int pos = 0; pos < CC8_ATTN_TILE; pos++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_HEAD_WORDS;
             word_idx++) {
            #pragma HLS pipeline II=1
            fm_word_t word =
                pos < tile_len ?
                words[pos * CC8_HEAD_WORDS + word_idx] :
                fm_word_t(0);
            wt_block_t packed = 0;
            for (unsigned int lane = 0;
                 lane < FM_BLOCK_SIZE;
                 lane++) {
                #pragma HLS unroll
                unsigned int elem = word_idx * FM_BLOCK_SIZE + lane;
                if (elem < HEAD_DIM) {
                    set_wt_block_lane(
                        packed,
                        lane,
                        wt_linear_t(unpack_fm_word_lane(word, lane))
                    );
                }
            }
            panel.word[pos][word_idx] = packed;
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

static wt_linear_t read_cc8_attention_panel_value(
    const cc8_attention_panel_t& panel,
    unsigned int pos,
    unsigned int elem
) {
    #pragma HLS inline
    unsigned int word_idx = elem / WT_BLOCK_SIZE;
    unsigned int lane = elem % WT_BLOCK_SIZE;
    return unpack_wt_block_lane(panel.word[pos][word_idx], lane);
}

static wt_linear_t read_cc8_current_kv_value(
    const cc8_current_kv_words_t& current,
    unsigned int kv_head,
    unsigned int elem
) {
    #pragma HLS inline
    unsigned int word_idx = elem / FM_BLOCK_SIZE;
    unsigned int lane = elem % FM_BLOCK_SIZE;
    fm_word_t word = current.word[kv_head][word_idx];
    return wt_linear_t(unpack_fm_word_lane(word, lane));
}

static void set_cc8_attention_weight_packet_lane(
    cc8_attention_weight_packet_word_t& word,
    unsigned int lane,
    wt_linear_t value
) {
    #pragma HLS inline
    word.range(
        (lane + 1) * wt_linear_t::width - 1,
        lane * wt_linear_t::width
    ) = value.range(wt_linear_t::width - 1, 0);
}

static mm_stream_8x64_weight_packet_t
unpack_cc8_attention_weight_packet(
    const cc8_attention_weight_packet_word_t& word
) {
    #pragma HLS inline
    mm_stream_8x64_weight_packet_t packet;
    #pragma HLS array_partition variable=packet.data complete dim=1
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        #pragma HLS unroll
        packet.data[lane].range(wt_linear_t::width - 1, 0) =
            word.range(
                (lane + 1) * wt_linear_t::width - 1,
                lane * wt_linear_t::width
            );
    }
    return packet;
}

static wt_linear_t read_cc8_attention_panel_or_current(
    const cc8_attention_panel_t& panel,
    const cc8_current_kv_words_t& current,
    unsigned int kv_head,
    bool use_current,
    unsigned int current_pos,
    unsigned int pos,
    unsigned int elem
) {
    #pragma HLS inline
#if CC8_RESIDENT_LAYER_ONLY
    // The resident layer stores the current K/V row to HBM before loading
    // the attention tile.  Read the complete tile back through one uniform
    // path so current_pos/use_current are not broadcast into every QK/PV
    // lane.  The legacy diagnostic image keeps the explicit bypass below.
    return read_cc8_attention_panel_value(panel, pos, elem);
#else
    wt_linear_t value = read_cc8_attention_panel_value(panel, pos, elem);
    if (use_current && pos == current_pos) {
        value = read_cc8_current_kv_value(current, kv_head, elem);
    }
    return value;
#endif
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
    const cc8_current_kv_words_t& current_k,
    bool use_current_k,
    unsigned int current_pos
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=q0.block complete dim=1
    #pragma HLS array_partition variable=q1.block complete dim=1
    #pragma HLS array_partition variable=k0.word complete dim=0
    #pragma HLS array_partition variable=k1.word complete dim=0
    #pragma HLS array_partition variable=current_k.word complete dim=0

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
            c0w0.data[lane] =
                read_cc8_attention_panel_or_current(
                    k0, current_k, 0, use_current_k, current_pos, pos0, k);
            c0w1.data[lane] =
                read_cc8_attention_panel_or_current(
                    k0, current_k, 0, use_current_k, current_pos, pos1, k);
            c0w2.data[lane] =
                read_cc8_attention_panel_or_current(
                    k0, current_k, 0, use_current_k, current_pos, pos2, k);
            c0w3.data[lane] =
                read_cc8_attention_panel_or_current(
                    k0, current_k, 0, use_current_k, current_pos, pos3, k);
            c1w0.data[lane] =
                read_cc8_attention_panel_or_current(
                    k1, current_k, 1, use_current_k, current_pos, pos0, k);
            c1w1.data[lane] =
                read_cc8_attention_panel_or_current(
                    k1, current_k, 1, use_current_k, current_pos, pos1, k);
            c1w2.data[lane] =
                read_cc8_attention_panel_or_current(
                    k1, current_k, 1, use_current_k, current_pos, pos2, k);
            c1w3.data[lane] =
                read_cc8_attention_panel_or_current(
                    k1, current_k, 1, use_current_k, current_pos, pos3, k);
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

static void load_cc8_prefill_query_block(
    cc8_prefill_query_block_t& query,
    const fm_word_t words[CC8_DATA_PORT_WORDS],
    unsigned int token_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=query.block complete dim=2
    constexpr unsigned int kHeadBlocks = ceildiv(HEAD_DIM, MM_PE_IN);
    constexpr unsigned int kWordsPerHead = ceildiv(HEAD_DIM, FM_BLOCK_SIZE);

    for (unsigned int head = 0; head < GQA_GROUP_SIZE; head++) {
        for (unsigned int token = 0;
             token < MM_STREAM_8X64_TOKENS;
             token++) {
            for (unsigned int word_idx = 0;
                 word_idx < kWordsPerHead;
                 word_idx++) {
                #pragma HLS pipeline II=1
                const unsigned int source_word =
                    (token * GQA_GROUP_SIZE + head) * kWordsPerHead +
                    word_idx;
                const fm_word_t word = token < token_count ?
                    words[source_word] : fm_word_t(0);
                query.block[head][token][2 * word_idx] =
                    word.range(255, 0);
                if (2 * word_idx + 1 < kHeadBlocks) {
                    query.block[head][token][2 * word_idx + 1] =
                        word.range(511, 256);
                }
            }
        }
    }
}

static fm_t read_cc8_prefill_query_value(
    const cc8_prefill_query_block_t& query,
    unsigned int head,
    unsigned int token,
    unsigned int elem
) {
    #pragma HLS inline
    return unpack_mm_input_block_lane(
        query.block[head][token][elem / MM_PE_IN],
        elem % MM_PE_IN
    );
}

static void emit_cc8_prefill_qk_inputs(
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
    const cc8_prefill_query_block_t& query0,
    const cc8_prefill_query_block_t& query1,
    const cc8_attention_qk_packet_panel_t& k0,
    const cc8_attention_qk_packet_panel_t& k1,
    unsigned int query_head,
    unsigned int token_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=query0.block complete dim=2
    #pragma HLS array_partition variable=query1.block complete dim=2
    #pragma HLS array_partition variable=k0.packet complete dim=1
    #pragma HLS array_partition variable=k1.packet complete dim=1

    for (unsigned int k = 0; k < HEAD_DIM; k++) {
        #pragma HLS pipeline II=1
        mm_stream_8x64_activation_packet_t activation0;
        mm_stream_8x64_activation_packet_t activation1;
        mm_stream_8x64_weight_packet_t weights0[CC8_ATTN_PACKET_GROUPS];
        mm_stream_8x64_weight_packet_t weights1[CC8_ATTN_PACKET_GROUPS];
        #pragma HLS array_partition variable=activation0.data complete dim=1
        #pragma HLS array_partition variable=activation1.data complete dim=1
        #pragma HLS array_partition variable=weights0 complete dim=0
        #pragma HLS array_partition variable=weights1 complete dim=0

        for (unsigned int token = 0;
             token < MM_STREAM_8X64_TOKENS;
             token++) {
            #pragma HLS unroll
            activation0.data[token] = token < token_count ?
                read_cc8_prefill_query_value(query0, query_head, token, k) :
                fm_t(0);
            activation1.data[token] = token < token_count ?
                read_cc8_prefill_query_value(query1, query_head, token, k) :
                fm_t(0);
        }
        for (unsigned int group = 0;
             group < CC8_ATTN_PACKET_GROUPS;
             group++) {
            #pragma HLS unroll
            weights0[group] = unpack_cc8_attention_weight_packet(
                k0.packet[group][k]
            );
            weights1[group] = unpack_cc8_attention_weight_packet(
                k1.packet[group][k]
            );
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

static void emit_cc8_attention_qk_packet_inputs(
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
    const cc8_resident_query_buffer_t& q0,
    const cc8_resident_query_buffer_t& q1,
    const cc8_attention_qk_packet_panel_t& k0,
    const cc8_attention_qk_packet_panel_t& k1
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=q0.value complete dim=1
    #pragma HLS array_partition variable=q0.value complete dim=2
    #pragma HLS array_partition variable=q1.value complete dim=1
    #pragma HLS array_partition variable=q1.value complete dim=2
    #pragma HLS array_partition variable=k0.packet complete dim=1
    #pragma HLS array_partition variable=k1.packet complete dim=1

    for (unsigned int k = 0; k < HEAD_DIM; k++) {
        #pragma HLS pipeline II=1
        mm_stream_8x64_activation_packet_t activation0;
        mm_stream_8x64_activation_packet_t activation1;
        mm_stream_8x64_weight_packet_t weights0[CC8_ATTN_PACKET_GROUPS];
        mm_stream_8x64_weight_packet_t weights1[CC8_ATTN_PACKET_GROUPS];
        #pragma HLS array_partition variable=activation0.data complete dim=1
        #pragma HLS array_partition variable=activation1.data complete dim=1
        #pragma HLS array_partition variable=weights0 complete dim=0
        #pragma HLS array_partition variable=weights1 complete dim=0

        const unsigned int query_bank =
            k < CC8_ROPE_HALF_ELEMS ? 0 : 1;
        const unsigned int query_bank_index =
            k < CC8_ROPE_HALF_ELEMS ?
            k : k - CC8_ROPE_HALF_ELEMS;

        for (unsigned int row = 0;
             row < MM_STREAM_8X64_TOKENS;
            row++) {
            #pragma HLS unroll
            activation0.data[row] = row < GQA_GROUP_SIZE ?
                q0.value[row][query_bank][query_bank_index] : fm_t(0);
            activation1.data[row] = row < GQA_GROUP_SIZE ?
                q1.value[row][query_bank][query_bank_index] : fm_t(0);
        }
        for (unsigned int group = 0;
             group < CC8_ATTN_PACKET_GROUPS;
             group++) {
            #pragma HLS unroll
            weights0[group] = unpack_cc8_attention_weight_packet(
                k0.packet[group][k]
            );
            weights1[group] = unpack_cc8_attention_weight_packet(
                k1.packet[group][k]
            );
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
    const cc8_current_kv_words_t& current_v,
    bool use_current_v,
    unsigned int current_pos,
    unsigned int output_wave,
    unsigned int tile_len
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=p0.block complete dim=1
    #pragma HLS array_partition variable=p1.block complete dim=1
    #pragma HLS array_partition variable=v0.word complete dim=0
    #pragma HLS array_partition variable=v1.word complete dim=0
    #pragma HLS array_partition variable=current_v.word complete dim=0

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
                        read_cc8_attention_panel_or_current(
                            v0,
                            current_v,
                            0,
                            use_current_v,
                            current_pos,
                            pos,
                            out
                        ) :
                        wt_linear_t(0);
                    weights1[group].data[lane] =
                        out < HEAD_DIM ?
                        read_cc8_attention_panel_or_current(
                            v1,
                            current_v,
                            1,
                            use_current_v,
                            current_pos,
                            pos,
                            out
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

static void emit_cc8_attention_pv_packet_inputs(
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
    const cc8_resident_probability_buffer_t& p0,
    const cc8_resident_probability_buffer_t& p1,
    const cc8_attention_pv_packet_panel_t& v0,
    const cc8_attention_pv_packet_panel_t& v1,
    unsigned int output_wave,
    unsigned int tile_len
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=p0.value complete dim=1
    #pragma HLS array_partition variable=p1.value complete dim=1
    #pragma HLS array_partition variable=v0.packet complete dim=1
    #pragma HLS array_partition variable=v1.packet complete dim=1

    for (unsigned int pos = 0; pos < CC8_ATTN_TILE; pos++) {
        #pragma HLS pipeline II=1
        if (pos < tile_len) {
            mm_stream_8x64_activation_packet_t activation0;
            mm_stream_8x64_activation_packet_t activation1;
            mm_stream_8x64_weight_packet_t weights0[CC8_ATTN_PACKET_GROUPS];
            mm_stream_8x64_weight_packet_t weights1[CC8_ATTN_PACKET_GROUPS];
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
                    p0.value[row][pos] :
                    fm_t(0);
                activation1.data[row] =
                    row < GQA_GROUP_SIZE ?
                    p1.value[row][pos] :
                    fm_t(0);
            }

            const unsigned int panel_index =
                output_wave * CC8_ATTN_TILE + pos;
            for (unsigned int group = 0;
                 group < CC8_ATTN_PACKET_GROUPS;
                 group++) {
                #pragma HLS unroll
                weights0[group] = unpack_cc8_attention_weight_packet(
                    v0.packet[group][panel_index]
                );
                weights1[group] = unpack_cc8_attention_weight_packet(
                    v1.packet[group][panel_index]
                );
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

// Keep request and response in one scheduler boundary. Separate top-level
// calls can be reordered into collect-before-emit deadlock.
static void run_cc8_attention_qk_exchange(
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
    const cc8_attention_buffer_t& query0,
    const cc8_attention_buffer_t& query1,
    const cc8_attention_panel_t& key0,
    const cc8_attention_panel_t& key1,
    const cc8_current_kv_words_t& current_key,
    cc8_attention_buffer_t& destination0,
    cc8_attention_buffer_t& destination1,
    bool use_current_key,
    unsigned int current_pos,
    unsigned int row_count,
    unsigned int elem_count
) {
    #pragma HLS inline off
    #pragma HLS dataflow

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
        query0,
        query1,
        key0,
        key1,
        current_key,
        use_current_key,
        current_pos
    );
    collect_cc8_separate_mm_results(
        destination0,
        destination1,
        core0_result_stream,
        core1_result_stream,
        row_count,
        elem_count
    );
}

static void run_cc8_attention_qk_packet_exchange(
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
    const cc8_resident_query_buffer_t& query0,
    const cc8_resident_query_buffer_t& query1,
    const cc8_attention_qk_packet_panel_t& key0,
    const cc8_attention_qk_packet_panel_t& key1,
    cc8_attention_buffer_t& destination0,
    cc8_attention_buffer_t& destination1,
    unsigned int row_count,
    unsigned int elem_count
) {
    #pragma HLS inline off
    #pragma HLS dataflow

    emit_cc8_attention_qk_packet_inputs(
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
        query0,
        query1,
        key0,
        key1
    );
    collect_cc8_separate_mm_results(
        destination0,
        destination1,
        core0_result_stream,
        core1_result_stream,
        row_count,
        elem_count
    );
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

static fm_t cc8_clamp_fm(fm_t x, fm_t lo, fm_t hi) {
    #pragma HLS inline
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

static fm_t cc8_exp_xilinx(fm_t x) {
    #pragma HLS inline
    fm_t xc = cc8_clamp_fm(x, fm_t(-8), fm_t(8));
    ap_fixed<16, 8> exp_in = ap_fixed<16, 8>(xc);
    return fm_t(hls::exp(exp_in));
}

static attention_prob_t cc8_exp_attention_probability(fm_t x) {
    #pragma HLS inline
    // exp() only sees non-positive differences in online softmax.  A wider
    // temporary supplies enough fractional bits for Q2.14 without changing
    // any external stream width.
    const fm_t xc = cc8_clamp_fm(x, fm_t(-8), fm_t(0));
    // The score difference originates in Q8.8, while Q2.14 is the desired
    // probability format.  Eighteen total bits retain all 14 output
    // fractional bits; a wider exp input adds no source information and
    // substantially increases HLS elaboration/resource cost.
    const ap_fixed<18, 4> exp_in = ap_fixed<18, 4>(xc);
    return attention_prob_t(hls::exp(exp_in));
}

static fm_t cc8_recip_safe(fm_accum_t x) {
    #pragma HLS inline
    fm_accum_t safe = x < fm_accum_t(0.000244140625) ?
        fm_accum_t(0.000244140625) :
        x;
    return fm_t(fm_accum_t(1) / safe);
}

static unsigned int cc8_kv_cache_word_index(
    unsigned int layer,
    unsigned int position,
    unsigned int kv_head,
    unsigned int word_idx
) {
    #pragma HLS inline
    return
        (
            (
                layer * MAX_SEQ_LEN +
                position
            ) *
            NUM_KEY_VALUE_HEADS +
            kv_head
        ) *
        CC8_HEAD_WORDS +
        word_idx;
}

static unsigned int cc8_k_cache_transposed_word_index(
    unsigned int layer,
    unsigned int kv_head,
    unsigned int elem,
    unsigned int position_word
) {
    #pragma HLS inline
    return
        CC8_KV_CACHE_ROW_MAJOR_WORDS +
        (
            (
                layer * NUM_KEY_VALUE_HEADS +
                kv_head
            ) *
            HEAD_DIM +
            elem
        ) *
        CC8_K_CACHE_POSITION_WORDS +
        position_word;
}

static void clear_cc8_current_kv_words(
    cc8_current_kv_words_t& current
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=current.word complete dim=0

    for (unsigned int kv_head = 0;
         kv_head < NUM_KEY_VALUE_HEADS;
         kv_head++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_HEAD_WORDS;
             word_idx++) {
            #pragma HLS pipeline II=1
            current.word[kv_head][word_idx] = 0;
        }
    }
}

static void load_cc8_current_kv_words(
    cc8_current_kv_words_t& current,
    const fm_word_t source[CC8_DATA_PORT_WORDS]
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=current.word complete dim=0

    for (unsigned int kv_head = 0;
         kv_head < NUM_KEY_VALUE_HEADS;
         kv_head++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_HEAD_WORDS;
             word_idx++) {
            #pragma HLS pipeline II=1
            current.word[kv_head][word_idx] =
                source[kv_head * CC8_HEAD_WORDS + word_idx];
        }
    }
}

static void store_cc8_current_kv_to_cache(
    fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS],
    fm_word_t kv_cache_v[CC8_KV_CACHE_WORDS],
    const cc8_current_kv_words_t& k_words,
    const cc8_current_kv_words_t& v_words,
    unsigned int layer,
    unsigned int position
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=k_words.word complete dim=0
    #pragma HLS array_partition variable=v_words.word complete dim=0

    for (unsigned int kv_head = 0;
         kv_head < NUM_KEY_VALUE_HEADS;
         kv_head++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_HEAD_WORDS;
             word_idx++) {
            #pragma HLS pipeline II=1
            unsigned int cache_idx = cc8_kv_cache_word_index(
                layer,
                position,
                kv_head,
                word_idx
            );
            kv_cache_k[cache_idx] = k_words.word[kv_head][word_idx];
            kv_cache_v[cache_idx] = v_words.word[kv_head][word_idx];
        }
    }
}

static void store_cc8_current_k_to_transposed_cache(
    fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS],
    const cc8_current_kv_words_t& k_words,
    unsigned int layer,
    unsigned int position
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=k_words.word complete dim=0

    const unsigned int position_word = position / FM_BLOCK_SIZE;
    const unsigned int position_lane = position % FM_BLOCK_SIZE;
    // cache_idx is injective over (kv_head, elem) for a fixed layer and
    // position_word.  Without this assertion Vitis HLS 2022.2 conservatively
    // treats the AXI read-modify-write as a loop-carried dependence and
    // schedules the loop at II ~= the AXI response latency.
    #pragma HLS dependence variable=kv_cache_k inter false
    for (unsigned int kv_head = 0;
         kv_head < NUM_KEY_VALUE_HEADS;
         kv_head++) {
        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            #pragma HLS pipeline II=1
            const unsigned int source_word = elem / FM_BLOCK_SIZE;
            const unsigned int source_lane = elem % FM_BLOCK_SIZE;
            const fm_t value = unpack_fm_word_lane(
                k_words.word[kv_head][source_word],
                source_lane
            );
            const unsigned int cache_idx =
                cc8_k_cache_transposed_word_index(
                    layer,
                    kv_head,
                    elem,
                    position_word
                );
            fm_word_t packed_positions = kv_cache_k[cache_idx];
            set_fm_word_lane(
                packed_positions,
                position_lane,
                value
            );
            kv_cache_k[cache_idx] = packed_positions;
        }
    }
}

static fm_t read_cc8_resident_k_value(
    const cc8_resident_k_buffer_t& current_k,
    unsigned int kv_head,
    unsigned int elem
) {
    #pragma HLS inline
    #pragma HLS array_partition variable=current_k.value complete dim=1
    #pragma HLS array_partition variable=current_k.value complete dim=2

    const unsigned int bank = elem < CC8_ROPE_HALF_ELEMS ? 0 : 1;
    const unsigned int bank_index = elem < CC8_ROPE_HALF_ELEMS ?
        elem : elem - CC8_ROPE_HALF_ELEMS;
    return current_k.value[kv_head][bank][bank_index];
}

static fm_word_t pack_cc8_resident_k_word(
    const cc8_resident_k_buffer_t& current_k,
    unsigned int kv_head,
    unsigned int word_idx
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=current_k.value complete dim=1
    #pragma HLS array_partition variable=current_k.value complete dim=2

    // Shift by a compile-time lane width instead of updating a dynamic
    // 512-bit slice.  The sequential packer is off the attention stream hot
    // loop and keeps the public row-major K mirror coherent for diagnostics.
    fm_word_t packed = 0;
    for (unsigned int lane = 0; lane < FM_BLOCK_SIZE; lane++) {
        #pragma HLS pipeline II=1
        const unsigned int elem = word_idx * FM_BLOCK_SIZE + lane;
        const fm_t value = elem < HEAD_DIM ?
            read_cc8_resident_k_value(current_k, kv_head, elem) : fm_t(0);
        const ap_uint<fm_t::width> raw =
            value.range(fm_t::width - 1, 0);
        packed >>= fm_t::width;
        packed.range(
            fm_word_t::width - 1,
            fm_word_t::width - fm_t::width
        ) = raw;
    }
    return packed;
}

static void store_cc8_resident_current_k_row_major(
    fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS],
    const cc8_resident_k_buffer_t& current_k,
    unsigned int layer,
    unsigned int position
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=current_k.value complete dim=1
    #pragma HLS array_partition variable=current_k.value complete dim=2

    for (unsigned int kv_head = 0;
         kv_head < NUM_KEY_VALUE_HEADS;
         kv_head++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_HEAD_WORDS;
             word_idx++) {
            kv_cache_k[cc8_kv_cache_word_index(
                layer,
                position,
                kv_head,
                word_idx
            )] = pack_cc8_resident_k_word(
                current_k,
                kv_head,
                word_idx
            );
        }
    }
}

static void store_cc8_resident_current_v_row_major(
    fm_word_t kv_cache_v[CC8_KV_CACHE_WORDS],
    const cc8_current_kv_words_t& current_v,
    unsigned int layer,
    unsigned int position
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=current_v.word complete dim=0

    for (unsigned int kv_head = 0;
         kv_head < NUM_KEY_VALUE_HEADS;
         kv_head++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_HEAD_WORDS;
             word_idx++) {
            #pragma HLS pipeline II=1
            kv_cache_v[cc8_kv_cache_word_index(
                layer,
                position,
                kv_head,
                word_idx
            )] = current_v.word[kv_head][word_idx];
        }
    }
}

static void store_cc8_resident_current_k_transposed(
    fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS],
    const cc8_resident_k_buffer_t& current_k,
    unsigned int layer,
    unsigned int position
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=current_k.value complete dim=1
    #pragma HLS array_partition variable=current_k.value complete dim=2

    const unsigned int position_word = position / FM_BLOCK_SIZE;
    const unsigned int position_lane = position % FM_BLOCK_SIZE;
    // cache_idx is injective over (kv_head, elem) for a fixed layer and
    // position_word.  Without this assertion Vitis HLS 2022.2 conservatively
    // treats the AXI read-modify-write as a loop-carried dependence and
    // schedules the loop at II ~= the AXI response latency.
    #pragma HLS dependence variable=kv_cache_k inter false
    for (unsigned int kv_head = 0;
         kv_head < NUM_KEY_VALUE_HEADS;
         kv_head++) {
        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            #pragma HLS pipeline II=1
            const unsigned int cache_idx =
                cc8_k_cache_transposed_word_index(
                    layer,
                    kv_head,
                    elem,
                    position_word
                );
            fm_word_t packed_positions = kv_cache_k[cache_idx];
            set_fm_word_lane(
                packed_positions,
                position_lane,
                read_cc8_resident_k_value(current_k, kv_head, elem)
            );
            kv_cache_k[cache_idx] = packed_positions;
        }
    }
}

static void load_cc8_kv_attention_panel(
    cc8_attention_panel_t& panel0,
    cc8_attention_panel_t& panel1,
    const fm_word_t kv_cache[CC8_KV_CACHE_WORDS],
    unsigned int layer,
    unsigned int tile_begin,
    unsigned int tile_len
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=panel0.word complete dim=0
    #pragma HLS array_partition variable=panel1.word complete dim=0

    for (unsigned int pos = 0; pos < CC8_ATTN_TILE; pos++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_HEAD_WORDS;
             word_idx++) {
            #pragma HLS pipeline II=1
            fm_word_t word0 = 0;
            fm_word_t word1 = 0;
            if (pos < tile_len) {
                unsigned int absolute_pos = tile_begin + pos;
                word0 = kv_cache[cc8_kv_cache_word_index(
                    layer,
                    absolute_pos,
                    0,
                    word_idx
                )];
                word1 = kv_cache[cc8_kv_cache_word_index(
                    layer,
                    absolute_pos,
                    1,
                    word_idx
                )];
            }
            wt_block_t packed0 = 0;
            wt_block_t packed1 = 0;
            for (unsigned int lane = 0;
                 lane < FM_BLOCK_SIZE;
                 lane++) {
                #pragma HLS unroll
                unsigned int elem = word_idx * FM_BLOCK_SIZE + lane;
                if (elem < HEAD_DIM) {
                    set_wt_block_lane(
                        packed0,
                        lane,
                        wt_linear_t(unpack_fm_word_lane(word0, lane))
                    );
                    set_wt_block_lane(
                        packed1,
                        lane,
                        wt_linear_t(unpack_fm_word_lane(word1, lane))
                    );
                }
            }
            panel0.word[pos][word_idx] = packed0;
            panel1.word[pos][word_idx] = packed1;
        }
    }
}

static void load_cc8_k_attention_packet_panel(
    cc8_attention_qk_packet_panel_t& panel0,
    cc8_attention_qk_packet_panel_t& panel1,
    const fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS],
    unsigned int layer,
    unsigned int tile_begin,
    unsigned int tile_len
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=panel0.packet complete dim=1
    #pragma HLS array_partition variable=panel1.packet complete dim=1

    // K^T in HBM stores 32 sequence positions in one 512-bit word for a
    // fixed head/element.  One read therefore becomes two ready-to-stream
    // 16-lane packets, avoiding the 64-way dynamic lane crossbar in QK.
    for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
        for (unsigned int position_half = 0;
             position_half < CC8_ATTN_TILE / FM_BLOCK_SIZE;
             position_half++) {
            #pragma HLS pipeline II=1
            const unsigned int local_position_base =
                position_half * FM_BLOCK_SIZE;
            fm_word_t word0 = 0;
            fm_word_t word1 = 0;
            if (local_position_base < tile_len) {
                const unsigned int position_word =
                    tile_begin / FM_BLOCK_SIZE + position_half;
                word0 = kv_cache_k[cc8_k_cache_transposed_word_index(
                    layer,
                    0,
                    elem,
                    position_word
                )];
                word1 = kv_cache_k[cc8_k_cache_transposed_word_index(
                    layer,
                    1,
                    elem,
                    position_word
                )];
            }

            cc8_attention_weight_packet_word_t packed0[2];
            cc8_attention_weight_packet_word_t packed1[2];
            #pragma HLS array_partition variable=packed0 complete dim=1
            #pragma HLS array_partition variable=packed1 complete dim=1
            packed0[0] = 0;
            packed0[1] = 0;
            packed1[0] = 0;
            packed1[1] = 0;
            for (unsigned int half = 0; half < 2; half++) {
                #pragma HLS unroll
                for (unsigned int lane = 0;
                     lane < CU_VEC_LANES;
                     lane++) {
                    #pragma HLS unroll
                    const unsigned int word_lane =
                        half * CU_VEC_LANES + lane;
                    const unsigned int local_position =
                        local_position_base + word_lane;
                    const bool valid = local_position < tile_len;
                    set_cc8_attention_weight_packet_lane(
                        packed0[half],
                        lane,
                        valid ?
                            wt_linear_t(
                                unpack_fm_word_lane(word0, word_lane)
                            ) :
                            wt_linear_t(0)
                    );
                    set_cc8_attention_weight_packet_lane(
                        packed1[half],
                        lane,
                        valid ?
                            wt_linear_t(
                                unpack_fm_word_lane(word1, word_lane)
                            ) :
                            wt_linear_t(0)
                    );
                }
            }

            const unsigned int group_base = 2 * position_half;
            panel0.packet[group_base][elem] = packed0[0];
            panel0.packet[group_base + 1][elem] = packed0[1];
            panel1.packet[group_base][elem] = packed1[0];
            panel1.packet[group_base + 1][elem] = packed1[1];
        }
    }
}

static void clear_cc8_v_attention_packet_panel(
    cc8_attention_pv_packet_panel_t& panel0,
    cc8_attention_pv_packet_panel_t& panel1
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=panel0.packet complete dim=1
    #pragma HLS array_partition variable=panel1.packet complete dim=1

    for (unsigned int index = 0;
         index < CC8_ATTN_PV_PACKET_DEPTH;
         index++) {
        #pragma HLS pipeline II=1
        for (unsigned int group = 0;
             group < CC8_ATTN_PACKET_GROUPS;
             group++) {
            #pragma HLS unroll
            panel0.packet[group][index] = 0;
            panel1.packet[group][index] = 0;
        }
    }
}

static void load_cc8_v_attention_packet_panel(
    cc8_attention_pv_packet_panel_t& panel0,
    cc8_attention_pv_packet_panel_t& panel1,
    const fm_word_t kv_cache_v[CC8_KV_CACHE_WORDS],
    unsigned int layer,
    unsigned int tile_begin,
    unsigned int tile_len
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=panel0.packet complete dim=1
    #pragma HLS array_partition variable=panel1.packet complete dim=1

    clear_cc8_v_attention_packet_panel(panel0, panel1);
    constexpr unsigned int kWordsPerOutputWave =
        MM_STREAM_8X64_OUTPUTS / FM_BLOCK_SIZE;
    for (unsigned int pos = 0; pos < CC8_ATTN_TILE; pos++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_HEAD_WORDS;
             word_idx++) {
            #pragma HLS pipeline II=1
            fm_word_t word0 = 0;
            fm_word_t word1 = 0;
            if (pos < tile_len) {
                const unsigned int absolute_pos = tile_begin + pos;
                word0 = kv_cache_v[cc8_kv_cache_word_index(
                    layer,
                    absolute_pos,
                    0,
                    word_idx
                )];
                word1 = kv_cache_v[cc8_kv_cache_word_index(
                    layer,
                    absolute_pos,
                    1,
                    word_idx
                )];
            }

            cc8_attention_weight_packet_word_t packed0[2];
            cc8_attention_weight_packet_word_t packed1[2];
            #pragma HLS array_partition variable=packed0 complete dim=1
            #pragma HLS array_partition variable=packed1 complete dim=1
            packed0[0] = 0;
            packed0[1] = 0;
            packed1[0] = 0;
            packed1[1] = 0;
            for (unsigned int half = 0; half < 2; half++) {
                #pragma HLS unroll
                for (unsigned int lane = 0;
                     lane < CU_VEC_LANES;
                     lane++) {
                    #pragma HLS unroll
                    const unsigned int word_lane =
                        half * CU_VEC_LANES + lane;
                    const unsigned int elem =
                        word_idx * FM_BLOCK_SIZE + word_lane;
                    const bool valid = pos < tile_len && elem < HEAD_DIM;
                    set_cc8_attention_weight_packet_lane(
                        packed0[half],
                        lane,
                        valid ?
                            wt_linear_t(
                                unpack_fm_word_lane(word0, word_lane)
                            ) :
                            wt_linear_t(0)
                    );
                    set_cc8_attention_weight_packet_lane(
                        packed1[half],
                        lane,
                        valid ?
                            wt_linear_t(
                                unpack_fm_word_lane(word1, word_lane)
                            ) :
                            wt_linear_t(0)
                    );
                }
            }

            const unsigned int output_wave =
                word_idx / kWordsPerOutputWave;
            const unsigned int group_base =
                2 * (word_idx % kWordsPerOutputWave);
            const unsigned int panel_index =
                output_wave * CC8_ATTN_TILE + pos;
            panel0.packet[group_base][panel_index] = packed0[0];
            panel0.packet[group_base + 1][panel_index] = packed0[1];
            panel1.packet[group_base][panel_index] = packed1[0];
            panel1.packet[group_base + 1][panel_index] = packed1[1];
        }
    }
}

static void clear_cc8_attention_buffer(
    cc8_attention_buffer_t& gbuf
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=gbuf.block complete dim=1

    for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
        for (unsigned int block = 0;
             block < CC8_ATTN_BUFFER_BLOCKS;
             block++) {
            #pragma HLS pipeline II=1
            gbuf.block[row][block] = 0;
        }
    }
}

static void write_cc8_attention_buffer_value(
    cc8_attention_buffer_t& gbuf,
    unsigned int row,
    unsigned int elem,
    fm_t value
) {
    #pragma HLS inline
    unsigned int block = elem / MM_PE_IN;
    unsigned int lane = elem % MM_PE_IN;
    mm_input_block_t packed = gbuf.block[row][block];
    set_mm_input_block_lane(packed, lane, value);
    gbuf.block[row][block] = packed;
}

static void write_cc8_attention_probability(
    cc8_attention_buffer_t& gbuf,
    unsigned int row,
    unsigned int elem,
    attention_prob_t value
) {
    #pragma HLS inline
    const unsigned int block = elem / MM_PE_IN;
    const unsigned int lane = elem % MM_PE_IN;
    mm_input_block_t packed = gbuf.block[row][block];
    packed.range(
        (lane + 1) * attention_prob_t::width - 1,
        lane * attention_prob_t::width
    ) = value.range(attention_prob_t::width - 1, 0);
    gbuf.block[row][block] = packed;
}

static void init_cc8_flash_state(
    fm_t max0[MM_STREAM_8X64_TOKENS],
    fm_t max1[MM_STREAM_8X64_TOKENS],
    fm_accum_t sum0[MM_STREAM_8X64_TOKENS],
    fm_accum_t sum1[MM_STREAM_8X64_TOKENS],
    cc8_flash_accumulator_t& acc0,
    cc8_flash_accumulator_t& acc1
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=max0 complete dim=1
    #pragma HLS array_partition variable=max1 complete dim=1
    #pragma HLS array_partition variable=sum0 complete dim=1
    #pragma HLS array_partition variable=sum1 complete dim=1
    #pragma HLS array_partition variable=acc0.value complete dim=1
    #pragma HLS array_partition variable=acc0.value complete dim=3
    #pragma HLS array_partition variable=acc1.value complete dim=1
    #pragma HLS array_partition variable=acc1.value complete dim=3

    for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
        max0[row] = fm_t(-128);
        max1[row] = fm_t(-128);
        sum0[row] = fm_accum_t(0);
        sum1[row] = fm_accum_t(0);
        for (unsigned int block = 0;
             block < CC8_FLASH_ACCUM_BLOCKS;
             block++) {
            #pragma HLS pipeline II=1
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS unroll
                acc0.value[row][block][lane] = fm_accum_t(0);
                acc1.value[row][block][lane] = fm_accum_t(0);
            }
        }
    }
}

static void compute_cc8_flash_probabilities_one_core(
    cc8_attention_buffer_t& probabilities,
    const cc8_attention_buffer_t& scores,
    fm_t online_max[MM_STREAM_8X64_TOKENS],
    fm_accum_t online_sum[MM_STREAM_8X64_TOKENS],
    fm_t old_scale[MM_STREAM_8X64_TOKENS],
    unsigned int tile_len
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=probabilities.block complete dim=1
    #pragma HLS array_partition variable=scores.block complete dim=1
    #pragma HLS array_partition variable=online_max complete dim=1
    #pragma HLS array_partition variable=online_sum complete dim=1
    #pragma HLS array_partition variable=old_scale complete dim=1

    clear_cc8_attention_buffer(probabilities);
    for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
        if (row < GQA_GROUP_SIZE) {
            fm_t row_max = online_max[row];
            for (unsigned int elem = 0; elem < CC8_ATTN_TILE; elem++) {
                #pragma HLS pipeline II=1
                if (elem < tile_len) {
                    fm_t score = read_cc8_gbuf_value(scores, row, elem);
                    if (score > row_max) {
                        row_max = score;
                    }
                }
            }

            fm_t row_old_scale = cc8_exp_xilinx(online_max[row] - row_max);
            fm_accum_t tile_sum = fm_accum_t(0);
            for (unsigned int elem = 0; elem < CC8_ATTN_TILE; elem++) {
                #pragma HLS pipeline II=1
                fm_t probability = fm_t(0);
                if (elem < tile_len) {
                    fm_t score = read_cc8_gbuf_value(scores, row, elem);
                    probability = cc8_exp_xilinx(score - row_max);
                    tile_sum += fm_accum_t(probability);
                }
                write_cc8_attention_buffer_value(
                    probabilities,
                    row,
                    elem,
                    probability
                );
            }
            old_scale[row] = row_old_scale;
            online_sum[row] =
                online_sum[row] * fm_accum_t(row_old_scale) + tile_sum;
            online_max[row] = row_max;
        } else {
            old_scale[row] = fm_t(0);
        }
    }
}

static void init_cc8_prefill_flash_state(
    fm_t online_max[GQA_GROUP_SIZE][MM_STREAM_8X64_TOKENS],
    fm_accum_t online_sum[GQA_GROUP_SIZE][MM_STREAM_8X64_TOKENS],
    cc8_flash_accumulator_t accumulator[GQA_GROUP_SIZE]
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=accumulator complete dim=1

    for (unsigned int head = 0; head < GQA_GROUP_SIZE; head++) {
        for (unsigned int token = 0;
             token < MM_STREAM_8X64_TOKENS;
             token++) {
            online_max[head][token] = fm_t(-128);
            online_sum[head][token] = fm_accum_t(0);
            for (unsigned int block = 0;
                 block < CC8_FLASH_ACCUM_BLOCKS;
                 block++) {
                #pragma HLS pipeline II=1
                for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                    #pragma HLS unroll
                    accumulator[head].value[token][block][lane] =
                        fm_accum_t(0);
                }
            }
        }
    }
}

static void compute_cc8_prefill_flash_probabilities_one_core(
    cc8_attention_buffer_t& probabilities,
    const cc8_attention_buffer_t& scores,
    fm_t online_max[MM_STREAM_8X64_TOKENS],
    fm_accum_t online_sum[MM_STREAM_8X64_TOKENS],
    attention_prob_t old_scale[MM_STREAM_8X64_TOKENS],
    unsigned int query_begin,
    unsigned int token_count,
    unsigned int tile_begin,
    unsigned int tile_len
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=probabilities.block complete dim=1
    #pragma HLS array_partition variable=scores.block complete dim=1

    clear_cc8_attention_buffer(probabilities);
    for (unsigned int token = 0;
         token < MM_STREAM_8X64_TOKENS;
         token++) {
        if (token < token_count) {
            const unsigned int query_position = query_begin + token;
            fm_t row_max = online_max[token];
            for (unsigned int elem = 0; elem < CC8_ATTN_TILE; elem++) {
                #pragma HLS pipeline II=1
                const bool valid = elem < tile_len &&
                    tile_begin + elem <= query_position;
                if (valid) {
                    const fm_t score =
                        read_cc8_gbuf_value(scores, token, elem);
                    if (score > row_max) {
                        row_max = score;
                    }
                }
            }

            const attention_prob_t row_old_scale =
                cc8_exp_attention_probability(
                    online_max[token] - row_max
                );
            fm_accum_t tile_sum = fm_accum_t(0);
            for (unsigned int elem = 0; elem < CC8_ATTN_TILE; elem++) {
                #pragma HLS pipeline II=1
                const bool valid = elem < tile_len &&
                    tile_begin + elem <= query_position;
                attention_prob_t probability = attention_prob_t(0);
                if (valid) {
                    const fm_t score =
                        read_cc8_gbuf_value(scores, token, elem);
                    probability =
                        cc8_exp_attention_probability(score - row_max);
                    tile_sum += fm_accum_t(probability);
                }
                write_cc8_attention_probability(
                    probabilities,
                    token,
                    elem,
                    probability
                );
            }
            old_scale[token] = row_old_scale;
            online_sum[token] =
                online_sum[token] * fm_accum_t(row_old_scale) + tile_sum;
            online_max[token] = row_max;
        } else {
            old_scale[token] = fm_t(0);
        }
    }
}

static void store_cc8_prefill_flash_output_one_core(
    fm_word_t output[CC8_FEATURE_WORDS_PER_PORT],
    fm_accum_t online_sum[GQA_GROUP_SIZE][MM_STREAM_8X64_TOKENS],
    cc8_flash_accumulator_t accumulator[GQA_GROUP_SIZE],
    unsigned int token_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=accumulator complete dim=1
    constexpr unsigned int kWordsPerToken =
        GQA_GROUP_SIZE * CC8_HEAD_WORDS;

    for (unsigned int token = 0;
         token < MM_STREAM_8X64_TOKENS;
         token++) {
        for (unsigned int head = 0; head < GQA_GROUP_SIZE; head++) {
            const fm_t inv_sum = token < token_count ?
                cc8_recip_safe(online_sum[head][token]) : fm_t(0);
            for (unsigned int word_idx = 0;
                 word_idx < CC8_HEAD_WORDS;
                 word_idx++) {
                #pragma HLS pipeline II=1
                fm_word_t packed = 0;
                for (unsigned int lane = 0; lane < FM_BLOCK_SIZE; lane++) {
                    #pragma HLS unroll
                    const unsigned int elem = word_idx * FM_BLOCK_SIZE + lane;
                    fm_t value = fm_t(0);
                    if (token < token_count && elem < HEAD_DIM) {
                        value = fm_t(
                            accumulator[head]
                                .value[token]
                                      [elem / CU_VEC_LANES]
                                      [elem % CU_VEC_LANES] *
                            fm_accum_t(inv_sum)
                        );
                    }
                    set_fm_word_lane(packed, lane, value);
                }
                const unsigned int destination_word =
                    token * kWordsPerToken +
                    head * CC8_HEAD_WORDS + word_idx;
                output[destination_word] = packed;
            }
        }
    }
}

static void compute_cc8_resident_flash_probabilities_one_core(
    cc8_resident_probability_buffer_t& probabilities,
    const cc8_attention_buffer_t& scores,
    fm_t online_max[MM_STREAM_8X64_TOKENS],
    fm_accum_t online_sum[MM_STREAM_8X64_TOKENS],
    fm_t old_scale[MM_STREAM_8X64_TOKENS],
    unsigned int tile_len
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=probabilities.value complete dim=1
    #pragma HLS array_partition variable=scores.block complete dim=1
    #pragma HLS array_partition variable=online_max complete dim=1
    #pragma HLS array_partition variable=online_sum complete dim=1
    #pragma HLS array_partition variable=old_scale complete dim=1

    for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
        if (row < GQA_GROUP_SIZE) {
            fm_t row_max = online_max[row];
            for (unsigned int elem = 0; elem < CC8_ATTN_TILE; elem++) {
                #pragma HLS pipeline II=1
                if (elem < tile_len) {
                    const fm_t score =
                        read_cc8_gbuf_value(scores, row, elem);
                    if (score > row_max) {
                        row_max = score;
                    }
                }
            }

            const fm_t row_old_scale =
                cc8_exp_xilinx(online_max[row] - row_max);
            fm_accum_t tile_sum = fm_accum_t(0);
            for (unsigned int elem = 0; elem < CC8_ATTN_TILE; elem++) {
                #pragma HLS pipeline II=1
                fm_t probability = fm_t(0);
                if (elem < tile_len) {
                    const fm_t score =
                        read_cc8_gbuf_value(scores, row, elem);
                    probability = cc8_exp_xilinx(score - row_max);
                    tile_sum += fm_accum_t(probability);
                }
                probabilities.value[row][elem] = probability;
            }
            old_scale[row] = row_old_scale;
            online_sum[row] =
                online_sum[row] * fm_accum_t(row_old_scale) + tile_sum;
            online_max[row] = row_max;
        } else {
            old_scale[row] = fm_t(0);
        }
    }
}

static void scale_cc8_flash_accumulator_one_core(
    cc8_flash_accumulator_t& acc,
    const fm_t old_scale[MM_STREAM_8X64_TOKENS]
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=acc.value complete dim=1
    #pragma HLS array_partition variable=acc.value complete dim=3
    #pragma HLS array_partition variable=old_scale complete dim=1

    for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
        for (unsigned int block = 0;
             block < CC8_FLASH_ACCUM_BLOCKS;
             block++) {
            #pragma HLS pipeline II=1
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS unroll
                unsigned int elem = block * CU_VEC_LANES + lane;
                if (row < GQA_GROUP_SIZE && elem < HEAD_DIM) {
                    acc.value[row][block][lane] *=
                        fm_accum_t(old_scale[row]);
                }
            }
        }
    }
}

static void accumulate_cc8_flash_pv_one_core(
    cc8_flash_accumulator_t& acc,
    hls::stream<cu_vec16_packet_t>& result_stream
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=acc.value complete dim=1
    #pragma HLS array_partition variable=acc.value complete dim=3

    for (unsigned int packet_idx = 0;
         packet_idx < MM_STREAM_8X64_PACKETS_PER_BLOCK;
         packet_idx++) {
        #pragma HLS pipeline II=1
        cu_vec16_packet_t packet = result_stream.read();
        if (packet.token_lane < GQA_GROUP_SIZE) {
            unsigned int block = packet.elem_base / CU_VEC_LANES;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS unroll
                unsigned int elem = packet.elem_base + lane;
                if (block < CC8_FLASH_ACCUM_BLOCKS &&
                    elem < HEAD_DIM &&
                    packet.valid_mask[lane]) {
                    acc.value[packet.token_lane][block][lane] +=
                        fm_accum_t(packet.data[lane]);
                }
            }
        }
    }
}

static void scale_cc8_prefill_flash_accumulator_one_core(
    cc8_flash_accumulator_t& acc,
    const attention_prob_t old_scale[MM_STREAM_8X64_TOKENS],
    unsigned int token_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=acc.value complete dim=3

    for (unsigned int token = 0;
         token < MM_STREAM_8X64_TOKENS;
         token++) {
        for (unsigned int block = 0;
             block < CC8_FLASH_ACCUM_BLOCKS;
             block++) {
            #pragma HLS pipeline II=1
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS unroll
                const unsigned int elem = block * CU_VEC_LANES + lane;
                if (token < token_count && elem < HEAD_DIM) {
                    acc.value[token][block][lane] *=
                        fm_accum_t(old_scale[token]);
                }
            }
        }
    }
}

static void accumulate_cc8_prefill_flash_pv_one_core(
    cc8_flash_accumulator_t& acc,
    hls::stream<cu_vec16_packet_t>& result_stream,
    unsigned int token_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=acc.value complete dim=3

    for (unsigned int packet_idx = 0;
         packet_idx < MM_STREAM_8X64_PACKETS_PER_BLOCK;
         packet_idx++) {
        #pragma HLS pipeline II=1
        const cu_vec16_packet_t packet = result_stream.read();
        if (packet.token_lane < token_count) {
            const unsigned int block = packet.elem_base / CU_VEC_LANES;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS unroll
                const unsigned int elem = packet.elem_base + lane;
                if (block < CC8_FLASH_ACCUM_BLOCKS &&
                    elem < HEAD_DIM &&
                    packet.valid_mask[lane]) {
                    acc.value[packet.token_lane][block][lane] +=
                        fm_accum_t(packet.data[lane]);
                }
            }
        }
    }
}

static void store_cc8_flash_output_one_core(
    fm_word_t output[CC8_FEATURE_WORDS_PER_PORT],
    cc8_flash_accumulator_t& acc,
    const fm_accum_t online_sum[MM_STREAM_8X64_TOKENS]
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=acc.value complete dim=1
    #pragma HLS array_partition variable=acc.value complete dim=3
    #pragma HLS array_partition variable=online_sum complete dim=1

    cc8_attention_buffer_t destination;
    #pragma HLS bind_storage variable=destination.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=destination.block complete dim=1
    clear_cc8_attention_buffer(destination);

    for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
        if (row < GQA_GROUP_SIZE) {
            fm_t inv_sum = cc8_recip_safe(online_sum[row]);
            for (unsigned int block = 0;
                 block < CC8_FLASH_ACCUM_BLOCKS;
                 block++) {
                #pragma HLS pipeline II=1
                mm_input_block_t packed = 0;
                for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                    #pragma HLS unroll
                    unsigned int elem = block * CU_VEC_LANES + lane;
                    fm_t value = fm_t(0);
                    if (elem < HEAD_DIM) {
                        value = fm_t(
                            acc.value[row][block][lane] *
                            fm_accum_t(inv_sum)
                        );
                    }
                    set_mm_input_block_lane(packed, lane, value);
                }
                destination.block[row][block] = packed;
            }
        }
    }
    store_cc8_flat_rows(output, destination, GQA_GROUP_SIZE, HEAD_DIM);
}

template <typename SourceBufferT>
static void extract_cc8_resident_query_group_from_gbuf(
    cc8_resident_query_buffer_t& query,
    const SourceBufferT& source,
    unsigned int token,
    unsigned int query_head_begin
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=query.value complete dim=1
    #pragma HLS array_partition variable=query.value complete dim=2
    #pragma HLS array_partition variable=source.block complete dim=1

    constexpr unsigned int kHeadBlocks = ceildiv(HEAD_DIM, MM_PE_IN);
    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int i = 0; i < CC8_ROPE_HALF_ELEMS; i++) {
            #pragma HLS pipeline II=1
            const unsigned int head = query_head_begin + row;
            const unsigned int low_block =
                head * kHeadBlocks + i / MM_PE_IN;
            const unsigned int high_elem = CC8_ROPE_HALF_ELEMS + i;
            const unsigned int high_block =
                head * kHeadBlocks + high_elem / MM_PE_IN;
            query.value[row][0][i] = unpack_mm_input_block_lane(
                source.block[token][low_block],
                i % MM_PE_IN
            );
            query.value[row][1][i] = unpack_mm_input_block_lane(
                source.block[token][high_block],
                high_elem % MM_PE_IN
            );
        }
    }
}

template <typename SourceBufferT>
static void extract_cc8_prefill_query_groups_from_gbuf(
    cc8_prefill_query_block_t& query0,
    cc8_prefill_query_block_t& query1,
    const SourceBufferT& source,
    unsigned int token_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=query0.block complete dim=2
    #pragma HLS array_partition variable=query1.block complete dim=2
    #pragma HLS array_partition variable=source.block complete dim=1

    constexpr unsigned int kHeadBlocks = ceildiv(HEAD_DIM, MM_PE_IN);
    for (unsigned int token = 0;
         token < CC8_GBUF_TOKEN_ROWS;
         token++) {
        for (unsigned int head = 0; head < GQA_GROUP_SIZE; head++) {
            for (unsigned int block = 0; block < kHeadBlocks; block++) {
                #pragma HLS pipeline II=1
                query0.block[head][token][block] = token < token_count ?
                    source.block[token][head * kHeadBlocks + block] :
                    mm_input_block_t(0);
                query1.block[head][token][block] = token < token_count ?
                    source.block[token][
                        (GQA_GROUP_SIZE + head) * kHeadBlocks + block
                    ] :
                    mm_input_block_t(0);
            }
        }
    }
}

static void load_cc8_resident_query_from_prefill_block(
    cc8_resident_query_buffer_t& query0,
    cc8_resident_query_buffer_t& query1,
    const cc8_prefill_query_block_t& block0,
    const cc8_prefill_query_block_t& block1,
    unsigned int token
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=query0.value complete dim=1
    #pragma HLS array_partition variable=query0.value complete dim=2
    #pragma HLS array_partition variable=query1.value complete dim=1
    #pragma HLS array_partition variable=query1.value complete dim=2
    #pragma HLS array_partition variable=block0.block complete dim=2
    #pragma HLS array_partition variable=block1.block complete dim=2

    for (unsigned int head = 0; head < GQA_GROUP_SIZE; head++) {
        for (unsigned int i = 0; i < CC8_ROPE_HALF_ELEMS; i++) {
            #pragma HLS pipeline II=1
            query0.value[head][0][i] =
                read_cc8_prefill_query_value(block0, head, token, i);
            query0.value[head][1][i] = read_cc8_prefill_query_value(
                block0,
                head,
                token,
                CC8_ROPE_HALF_ELEMS + i
            );
            query1.value[head][0][i] =
                read_cc8_prefill_query_value(block1, head, token, i);
            query1.value[head][1][i] = read_cc8_prefill_query_value(
                block1,
                head,
                token,
                CC8_ROPE_HALF_ELEMS + i
            );
        }
    }
}

template <typename SourceBufferT>
static void extract_cc8_current_kv_from_gbuf(
    cc8_current_kv_words_t& current,
    const SourceBufferT& source,
    unsigned int token
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=current.word complete dim=0
    #pragma HLS array_partition variable=source.block complete dim=1

    constexpr unsigned int kHeadBlocks = ceildiv(HEAD_DIM, MM_PE_IN);
    for (unsigned int head = 0; head < NUM_KEY_VALUE_HEADS; head++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_HEAD_WORDS;
             word_idx++) {
            #pragma HLS pipeline II=1
            unsigned int block0 = head * kHeadBlocks + 2 * word_idx;
            unsigned int block1 = block0 + 1;
            fm_word_t word = 0;
            word.range(MM_INPUT_BLOCK_BIT_WIDTH - 1, 0) =
                source.block[token][block0];
            if (block1 < (head + 1) * kHeadBlocks) {
                word.range(
                    2 * MM_INPUT_BLOCK_BIT_WIDTH - 1,
                    MM_INPUT_BLOCK_BIT_WIDTH
                ) = source.block[token][block1];
            }
            current.word[head][word_idx] = word;
        }
    }
}

template <typename SourceBufferT>
static void extract_cc8_resident_k_from_gbuf(
    cc8_resident_k_buffer_t& current_k,
    const SourceBufferT& source,
    unsigned int token
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=current_k.value complete dim=1
    #pragma HLS array_partition variable=current_k.value complete dim=2
    #pragma HLS array_partition variable=source.block complete dim=1

    constexpr unsigned int kHeadBlocks = ceildiv(HEAD_DIM, MM_PE_IN);
    for (unsigned int head = 0; head < NUM_KEY_VALUE_HEADS; head++) {
        for (unsigned int i = 0; i < CC8_ROPE_HALF_ELEMS; i++) {
            #pragma HLS pipeline II=1
            const unsigned int low_block =
                head * kHeadBlocks + i / MM_PE_IN;
            const unsigned int high_elem = CC8_ROPE_HALF_ELEMS + i;
            const unsigned int high_block =
                head * kHeadBlocks + high_elem / MM_PE_IN;
            current_k.value[head][0][i] = unpack_mm_input_block_lane(
                source.block[token][low_block],
                i % MM_PE_IN
            );
            current_k.value[head][1][i] = unpack_mm_input_block_lane(
                source.block[token][high_block],
                high_elem % MM_PE_IN
            );
        }
    }
}

static void set_cc8_attention_buffer_value(
    cc8_attention_buffer_t& buffer,
    unsigned int row,
    unsigned int elem,
    fm_t value
) {
    #pragma HLS inline
    unsigned int block = elem / MM_PE_IN;
    unsigned int lane = elem % MM_PE_IN;
    mm_input_block_t packed = buffer.block[row][block];
    set_mm_input_block_lane(packed, lane, value);
    buffer.block[row][block] = packed;
}

static fm_t read_cc8_current_kv_fm(
    const cc8_current_kv_words_t& current,
    unsigned int head,
    unsigned int elem
) {
    #pragma HLS inline
    unsigned int word_idx = elem / FM_BLOCK_SIZE;
    unsigned int lane = elem % FM_BLOCK_SIZE;
    return unpack_fm_word_lane(current.word[head][word_idx], lane);
}

static void set_cc8_current_kv_fm(
    cc8_current_kv_words_t& current,
    unsigned int head,
    unsigned int elem,
    fm_t value
) {
    #pragma HLS inline
    unsigned int word_idx = elem / FM_BLOCK_SIZE;
    unsigned int lane = elem % FM_BLOCK_SIZE;
    fm_word_t word = current.word[head][word_idx];
    ap_uint<fm_t::width> raw = value.range(fm_t::width - 1, 0);
    word.range((lane + 1) * fm_t::width - 1, lane * fm_t::width) = raw;
    current.word[head][word_idx] = word;
}

static fm_t read_cc8_layer_aux_value(
    const fm_word_t aux[CC8_DATA_PORT_WORDS],
    unsigned int word_offset,
    unsigned int elem
) {
    #pragma HLS inline
    unsigned int word_idx = word_offset + elem / FM_BLOCK_SIZE;
    unsigned int lane = elem % FM_BLOCK_SIZE;
    return unpack_fm_word_lane(aux[word_idx], lane);
}

static void load_cc8_rope_coefficients(
    cc8_rope_coefficient_buffer_t& coefficients,
    const fm_word_t rope_table[CC8_DATA_PORT_WORDS],
    unsigned int position
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=coefficients.value complete dim=1

    // aux1 is a persistent position-major RoPE table in HBM. Read one
    // coefficient per cycle into two independent local banks; all query rows
    // and both KV heads then reuse these values without repeated AXI reads.
    const unsigned int position_word_offset =
        position * CC8_ROPE_POSITION_WORDS;
    for (unsigned int i = 0; i < CC8_ROPE_HALF_ELEMS; i++) {
        #pragma HLS pipeline II=1
        coefficients.value[0][i] = read_cc8_layer_aux_value(
            rope_table,
            position_word_offset + CC8_ROPE_COS_WORD_OFFSET,
            i
        );
    }
    for (unsigned int i = 0; i < CC8_ROPE_HALF_ELEMS; i++) {
        #pragma HLS pipeline II=1
        coefficients.value[1][i] = read_cc8_layer_aux_value(
            rope_table,
            position_word_offset + CC8_ROPE_SIN_WORD_OFFSET,
            i
        );
    }
}

static void apply_cc8_rope_from_aux(
    cc8_resident_query_buffer_t& query0,
    cc8_resident_query_buffer_t& query1,
    cc8_resident_k_buffer_t& current_k,
    const fm_word_t rope_table[CC8_DATA_PORT_WORDS],
    unsigned int position = 0
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=query0.value complete dim=1
    #pragma HLS array_partition variable=query0.value complete dim=2
    #pragma HLS array_partition variable=query1.value complete dim=1
    #pragma HLS array_partition variable=query1.value complete dim=2
    #pragma HLS array_partition variable=current_k.value complete dim=1
    #pragma HLS array_partition variable=current_k.value complete dim=2

    cc8_rope_coefficient_buffer_t coefficients;
    #pragma HLS bind_storage variable=coefficients.value type=ram_2p impl=bram
    #pragma HLS array_partition variable=coefficients.value complete dim=1
    load_cc8_rope_coefficients(coefficients, rope_table, position);

    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int i = 0; i < CC8_ROPE_HALF_ELEMS; i++) {
            #pragma HLS pipeline II=1
            const fm_t cosine = coefficients.value[0][i];
            const fm_t sine = coefficients.value[1][i];
            const fm_t q00 = query0.value[row][0][i];
            const fm_t q01 = query0.value[row][1][i];
            const fm_t q10 = query1.value[row][0][i];
            const fm_t q11 = query1.value[row][1][i];
            query0.value[row][0][i] =
                fm_t(q00 * cosine - q01 * sine);
            query0.value[row][1][i] =
                fm_t(q01 * cosine + q00 * sine);
            query1.value[row][0][i] =
                fm_t(q10 * cosine - q11 * sine);
            query1.value[row][1][i] =
                fm_t(q11 * cosine + q10 * sine);
        }
    }

    for (unsigned int head = 0; head < NUM_KEY_VALUE_HEADS; head++) {
        for (unsigned int i = 0; i < CC8_ROPE_HALF_ELEMS; i++) {
            #pragma HLS pipeline II=1
            const fm_t cosine = coefficients.value[0][i];
            const fm_t sine = coefficients.value[1][i];
            const fm_t k0 = current_k.value[head][0][i];
            const fm_t k1 = current_k.value[head][1][i];
            current_k.value[head][0][i] =
                fm_t(k0 * cosine - k1 * sine);
            current_k.value[head][1][i] =
                fm_t(k1 * cosine + k0 * sine);
        }
    }
}

template <typename DestinationBufferT>
static void store_cc8_flash_output_group_to_gbuf(
    DestinationBufferT& destination,
    unsigned int token,
    unsigned int query_head_begin,
    cc8_flash_accumulator_t& acc,
    const fm_accum_t online_sum[MM_STREAM_8X64_TOKENS]
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=destination.block complete dim=1
    #pragma HLS array_partition variable=acc.value complete dim=1
    #pragma HLS array_partition variable=acc.value complete dim=3
    #pragma HLS array_partition variable=online_sum complete dim=1

    constexpr unsigned int kHeadBlocks = ceildiv(HEAD_DIM, CU_VEC_LANES);
    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        fm_t inv_sum = cc8_recip_safe(online_sum[row]);
        for (unsigned int block = 0; block < kHeadBlocks; block++) {
            #pragma HLS pipeline II=1
            mm_input_block_t packed = 0;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS unroll
                unsigned int elem = block * CU_VEC_LANES + lane;
                fm_t value = elem < HEAD_DIM ?
                    fm_t(acc.value[row][block][lane] * fm_accum_t(inv_sum)) :
                    fm_t(0);
                set_mm_input_block_lane(packed, lane, value);
            }
            unsigned int destination_block =
                (query_head_begin + row) * kHeadBlocks + block;
            destination.block[token][destination_block] = packed;
        }
    }
}

template <typename DestinationBufferT>
static unsigned int run_cc8_resident_decode_attention(
    hls::stream<cu8_task_t>& core0_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core0_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu8_task_t>& core1_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core1_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    const cc8_resident_query_buffer_t& query0,
    const cc8_resident_query_buffer_t& query1,
    const cc8_resident_k_buffer_t& current_k,
    const cc8_current_kv_words_t& current_v_words,
    DestinationBufferT& destination,
    unsigned int destination_token,
    unsigned int layer_id,
    unsigned int position,
    fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS],
    fm_word_t kv_cache_v[CC8_KV_CACHE_WORDS]
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=query0.value complete dim=1
    #pragma HLS array_partition variable=query0.value complete dim=2
    #pragma HLS array_partition variable=query1.value complete dim=1
    #pragma HLS array_partition variable=query1.value complete dim=2
    #pragma HLS array_partition variable=current_k.value complete dim=1
    #pragma HLS array_partition variable=current_k.value complete dim=2
    #pragma HLS array_partition variable=current_v_words.word complete dim=0
    #pragma HLS array_partition variable=destination.block complete dim=1

    cc8_attention_buffer_t qk0;
    cc8_attention_buffer_t qk1;
    cc8_resident_probability_buffer_t prob0;
    cc8_resident_probability_buffer_t prob1;
    cc8_attention_qk_packet_panel_t k_panel0;
    cc8_attention_qk_packet_panel_t k_panel1;
    cc8_attention_pv_packet_panel_t v_panel0;
    cc8_attention_pv_packet_panel_t v_panel1;
    #pragma HLS bind_storage variable=qk0.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=qk1.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=prob0.value type=ram_2p impl=bram
    #pragma HLS bind_storage variable=prob1.value type=ram_2p impl=bram
    #pragma HLS array_partition variable=qk0.block complete dim=1
    #pragma HLS array_partition variable=qk1.block complete dim=1
    #pragma HLS array_partition variable=prob0.value complete dim=1
    #pragma HLS array_partition variable=prob1.value complete dim=1
    #pragma HLS bind_storage variable=k_panel0.packet type=ram_2p impl=bram
    #pragma HLS bind_storage variable=k_panel1.packet type=ram_2p impl=bram
    #pragma HLS bind_storage variable=v_panel0.packet type=ram_2p impl=bram
    #pragma HLS bind_storage variable=v_panel1.packet type=ram_2p impl=bram
    #pragma HLS array_partition variable=k_panel0.packet complete dim=1
    #pragma HLS array_partition variable=k_panel1.packet complete dim=1
    #pragma HLS array_partition variable=v_panel0.packet complete dim=1
    #pragma HLS array_partition variable=v_panel1.packet complete dim=1

    fm_t online_max0[MM_STREAM_8X64_TOKENS];
    fm_t online_max1[MM_STREAM_8X64_TOKENS];
    fm_t old_scale0[MM_STREAM_8X64_TOKENS];
    fm_t old_scale1[MM_STREAM_8X64_TOKENS];
    fm_accum_t online_sum0[MM_STREAM_8X64_TOKENS];
    fm_accum_t online_sum1[MM_STREAM_8X64_TOKENS];
    cc8_flash_accumulator_t acc0;
    cc8_flash_accumulator_t acc1;
    #pragma HLS array_partition variable=online_max0 complete dim=1
    #pragma HLS array_partition variable=online_max1 complete dim=1
    #pragma HLS array_partition variable=old_scale0 complete dim=1
    #pragma HLS array_partition variable=old_scale1 complete dim=1
    #pragma HLS array_partition variable=online_sum0 complete dim=1
    #pragma HLS array_partition variable=online_sum1 complete dim=1
    #pragma HLS array_partition variable=acc0.value complete dim=1
    #pragma HLS array_partition variable=acc0.value complete dim=3
    #pragma HLS array_partition variable=acc1.value complete dim=1
    #pragma HLS array_partition variable=acc1.value complete dim=3

    store_cc8_resident_current_k_row_major(
        kv_cache_k,
        current_k,
        layer_id,
        position
    );
    store_cc8_resident_current_v_row_major(
        kv_cache_v,
        current_v_words,
        layer_id,
        position
    );
    store_cc8_resident_current_k_transposed(
        kv_cache_k,
        current_k,
        layer_id,
        position
    );
    init_cc8_flash_state(
        online_max0,
        online_max1,
        online_sum0,
        online_sum1,
        acc0,
        acc1
    );

    unsigned int context_len = position + 1;
    unsigned int output_waves =
        ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS);
    unsigned int tile_count = ceildiv(context_len, CC8_ATTN_TILE);

    for (unsigned int tile_begin = 0;
         tile_begin < MAX_SEQ_LEN;
         tile_begin += CC8_ATTN_TILE) {
        #pragma HLS loop_tripcount min=1 max=ATTENTION_NUM_TILES
        if (tile_begin < context_len) {
            unsigned int current_tile_len = CC8_ATTN_TILE;
            if (tile_begin + current_tile_len > context_len) {
                current_tile_len = context_len - tile_begin;
            }
            // Both row-major and transposed resident-cache stores completed
            // before this loop, so
            // the resident image can load the complete tile, including the
            // current position.  This avoids a wide per-lane current-K/V
            // bypass network in both QK and PV emitters.
            unsigned int load_tile_len = current_tile_len;

            load_cc8_k_attention_packet_panel(
                k_panel0,
                k_panel1,
                kv_cache_k,
                layer_id,
                tile_begin,
                load_tile_len
            );
            load_cc8_v_attention_packet_panel(
                v_panel0,
                v_panel1,
                kv_cache_v,
                layer_id,
                tile_begin,
                load_tile_len
            );

            cu8_task_t qk_task = build_cc8_compute_task(
                CU8_MODE_MM_SCALE,
                HEAD_DIM,
                GQA_GROUP_SIZE,
                CC8_ATTN_TILE,
                MM_STREAM_8X64_PACKETS_PER_BLOCK,
                0,
                0,
                false
            );
            qk_task.output_scale = fm_t(ATTENTION_SCALE);
            core0_task_stream.write(qk_task);
            core1_task_stream.write(qk_task);
            run_cc8_attention_qk_packet_exchange(
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
                query0,
                query1,
                k_panel0,
                k_panel1,
                qk0,
                qk1,
                GQA_GROUP_SIZE,
                CC8_ATTN_TILE
            );
            compute_cc8_resident_flash_probabilities_one_core(
                prob0,
                qk0,
                online_max0,
                online_sum0,
                old_scale0,
                current_tile_len
            );
            compute_cc8_resident_flash_probabilities_one_core(
                prob1,
                qk1,
                online_max1,
                online_sum1,
                old_scale1,
                current_tile_len
            );
            scale_cc8_flash_accumulator_one_core(acc0, old_scale0);
            scale_cc8_flash_accumulator_one_core(acc1, old_scale1);

            for (unsigned int output_wave = 0;
                 output_wave < ceildiv(
                     CC8_ATTN_BUFFER_ELEMS,
                     MM_STREAM_8X64_OUTPUTS
                 );
                 output_wave++) {
                if (output_wave < output_waves) {
                    cu8_task_t pv_task = build_cc8_compute_task(
                        CU8_MODE_MM,
                        current_tile_len,
                        GQA_GROUP_SIZE,
                        MM_STREAM_8X64_OUTPUTS,
                        MM_STREAM_8X64_PACKETS_PER_BLOCK,
                        output_wave * MM_STREAM_8X64_OUTPUTS,
                        output_wave,
                        false
                    );
                    core0_task_stream.write(pv_task);
                    core1_task_stream.write(pv_task);
                    emit_cc8_attention_pv_packet_inputs(
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
                        prob0,
                        prob1,
                        v_panel0,
                        v_panel1,
                        output_wave,
                        current_tile_len
                    );
                    accumulate_cc8_flash_pv_one_core(
                        acc0,
                        core0_result_stream
                    );
                    accumulate_cc8_flash_pv_one_core(
                        acc1,
                        core1_result_stream
                    );
                }
            }
        }
    }

    store_cc8_flash_output_group_to_gbuf(
        destination,
        destination_token,
        0,
        acc0,
        online_sum0
    );
    store_cc8_flash_output_group_to_gbuf(
        destination,
        destination_token,
        GQA_GROUP_SIZE,
        acc1,
        online_sum1
    );
    return CC8_MM_CORE_COUNT * tile_count * (1 + output_waves);
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
    #pragma HLS inline off
    unsigned int profile_flags = 0;
    if (tile_len != 0) {
        profile_flags = token_count >> 16;
        token_count = token_count & 0xffffu;
    }

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

#ifdef CC8_PREFILL_BLOCK_SYNTH_ONLY
    // Resource probe for the block-prefill scheduler.  Making the operator a
    // compile-time constant lets HLS remove the legacy diagnostic schedulers,
    // so this report measures the new path instead of their union.
    cc8_operator_t op = CC8_OP_ATTN_PREFILL_BLOCK;
#else
    cc8_operator_t op = cc8_operator_t(operator_kind);
#endif
    status.op = op;
    cc8_operator_spec_t spec;
    if (!cc8_get_operator_spec(op, spec)) {
        status.status = CC8_STATUS_BAD_OPERATOR;
        core0_task_stream.write(build_cc8_stop_task());
        core1_task_stream.write(build_cc8_stop_task());
        status_stream.write(status);
        return;
    }
#ifdef CC8_PREFILL_LAYER_ONLY
    // Keep the runtime operator ABI required by the staged prefill host, but
    // reject operators whose schedulers are deliberately absent from this
    // image.  The corresponding preprocessor guards below let HLS eliminate
    // decode, resident-layer, and diagnostic-attention control networks.
    const bool prefill_layer_operator =
        op == CC8_OP_NOP ||
        op == CC8_OP_Q_PROJECTION ||
        op == CC8_OP_K_PROJECTION ||
        op == CC8_OP_V_PROJECTION ||
        op == CC8_OP_O_PROJECTION ||
        op == CC8_OP_FFN_GATE ||
        op == CC8_OP_FFN_UP ||
        op == CC8_OP_FFN_DOWN ||
        op == CC8_OP_RMSNORM ||
        op == CC8_OP_SILU_MUL ||
        op == CC8_OP_RESIDUAL_ADD ||
        op == CC8_OP_ATTN_PREFILL_BLOCK;
    if (!prefill_layer_operator) {
        status.status = CC8_STATUS_BAD_OPERATOR;
        core0_task_stream.write(build_cc8_stop_task());
        core1_task_stream.write(build_cc8_stop_task());
        status_stream.write(status);
        return;
    }
#endif

    if (op == CC8_OP_NOP) {
        core0_task_stream.write(build_cc8_stop_task());
        core1_task_stream.write(build_cc8_stop_task());
        status_stream.write(status);
        return;
    }

    if (layer_id >= NUM_LAYERS) {
        status.status = CC8_STATUS_BAD_LAYER;
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

#if !defined(CC8_PREFILL_LAYER_ONLY)
    cc8_hidden_buffer_t hidden0;
    cc8_hidden_buffer_t hidden1;
    #pragma HLS bind_storage variable=hidden0.block type=ram_2p impl=bram
    #pragma HLS bind_storage variable=hidden1.block type=ram_2p impl=bram
    #pragma HLS array_partition variable=hidden0.block complete dim=1
    #pragma HLS array_partition variable=hidden1.block complete dim=1

    if (op == CC8_OP_FINAL_NORM) {
        if (token_count == 0 || token_count > CC8_GBUF_TOKEN_ROWS) {
            status.status = CC8_STATUS_BAD_TOKEN_COUNT;
            core0_task_stream.write(build_cc8_stop_task());
            core1_task_stream.write(build_cc8_stop_task());
            status_stream.write(status);
            return;
        }
        load_cc8_feature_gbuf(
            hidden0,
            input_port0,
            input_port1,
            LINEAR_TOKEN_TILE_ACTIVE,
            token_count,
            HIDDEN_SIZE
        );
        load_cc8_feature_gbuf(
            gbuf0,
            aux_port0,
            aux_port0,
            1,
            1,
            HIDDEN_SIZE,
            cc8_final_norm_word_offset()
        );
        status.dispatched_vector_tasks = run_cc8_vector_gbuf_task(
            core0_task_stream,
            core0_vector_input0_stream,
            core0_vector_input1_stream,
            core0_result_stream,
            core1_task_stream,
            core1_vector_input0_stream,
            core1_vector_input1_stream,
            core1_result_stream,
            hidden0,
            gbuf0,
            hidden1,
            CC8_OP_RMSNORM,
            CU8_MODE_RMSNORM,
            token_count,
            HIDDEN_SIZE,
            1,
            1,
            true
        );
        status.completed_output_packets =
            token_count * ceildiv(HIDDEN_SIZE, CU_VEC_LANES);
        store_cc8_gbuf_to_hbm(
            output_port0,
            output_port1,
            hidden1,
            HIDDEN_SIZE
        );
        status_stream.write(status);
        return;
    }

    const bool resident_layer_task =
        op == CC8_OP_DECODER_LAYER ||
        op == CC8_OP_ATTENTION_SUBLAYER ||
        op == CC8_OP_FFN_SUBLAYER;
    if (resident_layer_task) {
        // The controller exposes both a compatibility full-layer task and
        // two coarse sublayer tasks.  Attention writes its residual state to
        // HBM; a following FFN task reads that device-resident state through
        // a host-side buffer rebind, without a CPU copy.  All tensors inside
        // either sublayer remain in the local hidden/wide banks.
        if (token_count == 0 || token_count > CC8_GBUF_TOKEN_ROWS) {
            status.status = CC8_STATUS_BAD_TOKEN_COUNT;
            core0_task_stream.write(build_cc8_stop_task());
            core1_task_stream.write(build_cc8_stop_task());
            status_stream.write(status);
            return;
        }
        if (position >= MAX_SEQ_LEN || position + token_count > MAX_SEQ_LEN) {
            status.status = CC8_STATUS_BAD_POSITION;
            core0_task_stream.write(build_cc8_stop_task());
            core1_task_stream.write(build_cc8_stop_task());
            status_stream.write(status);
            return;
        }

        cc8_resident_query_buffer_t query0;
        cc8_resident_query_buffer_t query1;
        cc8_prefill_query_block_t query_block0;
        cc8_prefill_query_block_t query_block1;
        cc8_resident_k_buffer_t current_k[CC8_GBUF_TOKEN_ROWS];
        cc8_current_kv_words_t current_v_words;
        #pragma HLS bind_storage variable=query0.value type=ram_2p impl=bram
        #pragma HLS bind_storage variable=query1.value type=ram_2p impl=bram
        #pragma HLS bind_storage variable=query_block0.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=query_block1.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=current_k type=ram_2p impl=bram
        #pragma HLS array_partition variable=query0.value complete dim=1
        #pragma HLS array_partition variable=query0.value complete dim=2
        #pragma HLS array_partition variable=query1.value complete dim=1
        #pragma HLS array_partition variable=query1.value complete dim=2
        #pragma HLS array_partition variable=query_block0.block complete dim=2
        #pragma HLS array_partition variable=query_block1.block complete dim=2
        #pragma HLS array_partition variable=current_v_words.word complete dim=0

        unsigned int total_mm_tasks = 0;
        unsigned int total_vector_tasks = 0;
        unsigned int total_packets = 0;
        unsigned int total_wave_slots = 0;
        weight_addr_t layer_base =
            weight_addr_t(layer_id) * weight_addr_t(LAYER_WEIGHT_SIZE);

        // Attention/full-layer starts with hidden0 -> hidden1.  Standalone
        // FFN deliberately loads the HBM sublayer boundary into hidden1 and
        // normalizes to hidden0, which recreates exactly the bank state at
        // projection step 4 of the full resident schedule.
        const bool starts_with_attention =
            op != CC8_OP_FFN_SUBLAYER;
        if (starts_with_attention) {
            load_cc8_feature_gbuf(
                hidden0,
                input_port0,
                input_port1,
                LINEAR_TOKEN_TILE_ACTIVE,
                token_count,
                HIDDEN_SIZE
            );
            load_cc8_feature_gbuf(
                gbuf0,
                aux_port0,
                aux_port0,
                1,
                1,
                HIDDEN_SIZE,
                cc8_attention_norm_word_offset(layer_id)
            );
            total_vector_tasks += run_cc8_vector_gbuf_task(
                core0_task_stream,
                core0_vector_input0_stream,
                core0_vector_input1_stream,
                core0_result_stream,
                core1_task_stream,
                core1_vector_input0_stream,
                core1_vector_input1_stream,
                core1_result_stream,
                hidden0,
                gbuf0,
                hidden1,
                CC8_OP_RMSNORM,
                CU8_MODE_RMSNORM,
                token_count,
                HIDDEN_SIZE,
                1,
                1,
                false
            );
        } else {
            load_cc8_feature_gbuf(
                hidden1,
                input_port0,
                input_port1,
                LINEAR_TOKEN_TILE_ACTIVE,
                token_count,
                HIDDEN_SIZE
            );
            load_cc8_feature_gbuf(
                gbuf1,
                aux_port0,
                aux_port0,
                1,
                1,
                HIDDEN_SIZE,
                cc8_ffn_norm_word_offset(layer_id)
            );
            total_vector_tasks += run_cc8_vector_gbuf_task(
                core0_task_stream,
                core0_vector_input0_stream,
                core0_vector_input1_stream,
                core0_result_stream,
                core1_task_stream,
                core1_vector_input0_stream,
                core1_vector_input1_stream,
                core1_result_stream,
                hidden1,
                gbuf1,
                hidden0,
                CC8_OP_RMSNORM,
                CU8_MODE_RMSNORM,
                token_count,
                HIDDEN_SIZE,
                1,
                1,
                false
            );
        }
        total_packets +=
            token_count * ceildiv(HIDDEN_SIZE, CU_VEC_LANES);

        // Stages 2-5 share one physical projection engine.  Keeping the
        // call at one syntactic site prevents HLS from cloning the complete
        // 16-shard weight loader for hidden and FFN buffer port shapes.
        const unsigned int projection_begin =
            op == CC8_OP_FFN_SUBLAYER ? 4 : 0;
        const unsigned int projection_end =
            op == CC8_OP_ATTENTION_SUBLAYER ? 4 : 7;
        for (unsigned int projection_step = projection_begin;
             projection_step < projection_end;
             projection_step++) {
            #pragma HLS loop_tripcount min=3 max=7

            mm_projection_kind_t projection =
                projection_step == 0 ? MM_PROJECTION_Q :
                projection_step == 1 ? MM_PROJECTION_K :
                projection_step == 2 ? MM_PROJECTION_V :
                projection_step == 3 ? MM_PROJECTION_ATTN_O :
                projection_step == 4 ? MM_PROJECTION_FFN_GATE :
                projection_step == 5 ? MM_PROJECTION_FFN_UP :
                                       MM_PROJECTION_FFN_DOWN;
            mm_projection_spec_t projection_spec;
            get_mm_projection_spec(projection, projection_spec);

            cc8_projection_bank_t source_bank =
                CC8_PROJECTION_HIDDEN1;
            cc8_projection_bank_t destination_bank =
                CC8_PROJECTION_WIDE0;

            if (projection_step <= 2) {
                // Q/K/V consume the first normalized hidden state and reuse
                // gbuf0 after each result is compacted into resident caches.
                // V is the exception: all token rows remain in gbuf0 until
                // the causal attention walk consumes and overwrites them.
                clear_cc8_gbuf(gbuf0, projection_spec.out_dim);
            } else if (projection_step == 3) {
                // One coarse prefill task holds all Q rows and K rows on chip.
                // V remains in gbuf0.  For each query token, append K/V to the
                // controller-owned cache, execute causal online attention,
                // and overwrite only that token's V row with its context.
                // The compute array still uses all eight rows: here rows map
                // to the eight GQA heads, while projections/FFN map them to
                // consecutive query tokens.
                unsigned int attention_tasks = 0;
                for (unsigned int token = 0;
                     token < CC8_GBUF_TOKEN_ROWS;
                     token++) {
                    #pragma HLS loop_tripcount min=1 max=LINEAR_TOKEN_TILE_ACTIVE
                    if (token < token_count) {
                        load_cc8_resident_query_from_prefill_block(
                            query0,
                            query1,
                            query_block0,
                            query_block1,
                            token
                        );
                        extract_cc8_current_kv_from_gbuf(
                            current_v_words,
                            gbuf0,
                            token
                        );
                        apply_cc8_rope_from_aux(
                            query0,
                            query1,
                            current_k[token],
                            aux_port1,
                            position + token
                        );
                        attention_tasks +=
                            run_cc8_resident_decode_attention(
                        core0_task_stream,
                        core0_activation_stream,
                        core0_weight_stream0,
                        core0_weight_stream1,
                        core0_weight_stream2,
                        core0_weight_stream3,
                        core0_result_stream,
                        core1_task_stream,
                        core1_activation_stream,
                        core1_weight_stream0,
                        core1_weight_stream1,
                        core1_weight_stream2,
                        core1_weight_stream3,
                        core1_result_stream,
                        query0,
                        query1,
                        current_k[token],
                        current_v_words,
                        gbuf0,
                        token,
                        layer_id,
                        position + token,
                        kv_cache_k,
                        kv_cache_v
                    );
                    }
                }
                total_mm_tasks += attention_tasks;
                total_packets +=
                    attention_tasks * MM_STREAM_8X64_PACKETS_PER_BLOCK;
                total_wave_slots +=
                    attention_tasks / CC8_MM_CORE_COUNT;

                clear_cc8_gbuf(gbuf1, projection_spec.out_dim);
                source_bank = CC8_PROJECTION_WIDE0;
                destination_bank = CC8_PROJECTION_WIDE1;
            } else if (projection_step == 4) {
                // Gate starts the FFN-width ping-pong pair.
                clear_cc8_gbuf(gbuf0, projection_spec.out_dim);
                source_bank = CC8_PROJECTION_HIDDEN0;
                destination_bank = CC8_PROJECTION_WIDE0;
            } else if (projection_step == 5) {
                // Up preserves Gate in gbuf0 and fills the second wide bank.
                clear_cc8_gbuf(gbuf1, projection_spec.out_dim);
                source_bank = CC8_PROJECTION_HIDDEN0;
                destination_bank = CC8_PROJECTION_WIDE1;
            } else {
                // Down consumes the in-place SiLU*Up product from gbuf0.
                clear_cc8_gbuf(hidden0, projection_spec.out_dim);
                source_bank = CC8_PROJECTION_WIDE0;
                destination_bank = CC8_PROJECTION_HIDDEN0;
            }

            unsigned int output_waves =
                ceildiv(projection_spec.out_dim, CC8_OUTPUTS_PER_WAVE);
            run_cc8_projection_waves_banked(
                core0_task_stream,
                core0_activation_stream,
                core0_weight_stream0,
                core0_weight_stream1,
                core0_weight_stream2,
                core0_weight_stream3,
                core0_result_stream,
                core1_task_stream,
                core1_activation_stream,
                core1_weight_stream0,
                core1_weight_stream1,
                core1_weight_stream2,
                core1_weight_stream3,
                core1_result_stream,
                hidden0,
                hidden1,
                gbuf0,
                gbuf1,
                source_bank,
                destination_bank,
                0,
                output_waves,
                token_count,
                projection_spec.in_dim,
                projection_spec.in_dim,
                projection_spec.out_dim,
                layer_base + projection_spec.weight_base,
                0x3u,
                false,
                false,
                QWEN_WEIGHT_SHARD_ARGS
            );
            total_wave_slots += output_waves;
            total_mm_tasks += CC8_MM_CORE_COUNT * output_waves;
            total_packets +=
                CC8_MM_CORE_COUNT * output_waves *
                MM_STREAM_8X64_PACKETS_PER_BLOCK;

            if (projection_step == 0) {
                extract_cc8_prefill_query_groups_from_gbuf(
                    query_block0,
                    query_block1,
                    gbuf0,
                    token_count
                );
            } else if (projection_step == 1) {
                for (unsigned int token = 0;
                     token < CC8_GBUF_TOKEN_ROWS;
                     token++) {
                    #pragma HLS loop_tripcount min=1 max=LINEAR_TOKEN_TILE_ACTIVE
                    if (token < token_count) {
                        extract_cc8_resident_k_from_gbuf(
                            current_k[token],
                            gbuf0,
                            token
                        );
                    }
                }
            } else if (projection_step == 2) {
                // Preserve every V row in gbuf0.  Stage 3 consumes each row
                // immediately before replacing it with the attention output.
            } else if (projection_step == 3) {
                // O result + original hidden -> first residual, followed by
                // the second RMSNorm.  gbuf1 is reused for the norm weights.
                total_vector_tasks += run_cc8_vector_gbuf_task(
                    core0_task_stream,
                    core0_vector_input0_stream,
                    core0_vector_input1_stream,
                    core0_result_stream,
                    core1_task_stream,
                    core1_vector_input0_stream,
                    core1_vector_input1_stream,
                    core1_result_stream,
                    hidden0,
                    gbuf1,
                    hidden1,
                    CC8_OP_RESIDUAL_ADD,
                    CU8_MODE_RESIDUAL_ADD,
                    token_count,
                    HIDDEN_SIZE,
                    token_count,
                    token_count,
                    op == CC8_OP_ATTENTION_SUBLAYER
                );
                total_packets +=
                    token_count * ceildiv(HIDDEN_SIZE, CU_VEC_LANES);

                if (op != CC8_OP_ATTENTION_SUBLAYER) {
                    load_cc8_feature_gbuf(
                        gbuf1,
                        aux_port0,
                        aux_port0,
                        1,
                        1,
                        HIDDEN_SIZE,
                        cc8_ffn_norm_word_offset(layer_id)
                    );
                    total_vector_tasks += run_cc8_vector_gbuf_task(
                        core0_task_stream,
                        core0_vector_input0_stream,
                        core0_vector_input1_stream,
                        core0_result_stream,
                        core1_task_stream,
                        core1_vector_input0_stream,
                        core1_vector_input1_stream,
                        core1_result_stream,
                        hidden1,
                        gbuf1,
                        hidden0,
                        CC8_OP_RMSNORM,
                        CU8_MODE_RMSNORM,
                        token_count,
                        HIDDEN_SIZE,
                        1,
                        1,
                        false
                    );
                    total_packets +=
                        token_count * ceildiv(HIDDEN_SIZE, CU_VEC_LANES);
                }
            } else if (projection_step == 5) {
                // Gate may be overwritten once both Gate and Up packets have
                // been accepted; the packet handshake keeps this II=1.
                total_vector_tasks +=
                    run_cc8_vector_gbuf_task_inplace_binary(
                        core0_task_stream,
                        core0_vector_input0_stream,
                        core0_vector_input1_stream,
                        core0_result_stream,
                        core1_task_stream,
                        core1_vector_input0_stream,
                        core1_vector_input1_stream,
                        core1_result_stream,
                        gbuf0,
                        gbuf1,
                        CU8_MODE_SILU_MUL,
                        token_count,
                        INTERMEDIATE_SIZE,
                        false
                    );
                total_packets +=
                    token_count *
                    ceildiv(INTERMEDIATE_SIZE, CU_VEC_LANES);
            } else if (projection_step == 6) {
                // Down + first residual completes the layer and terminates
                // both compute sessions.
                total_vector_tasks +=
                    run_cc8_vector_gbuf_task_inplace_binary(
                        core0_task_stream,
                        core0_vector_input0_stream,
                        core0_vector_input1_stream,
                        core0_result_stream,
                        core1_task_stream,
                        core1_vector_input0_stream,
                        core1_vector_input1_stream,
                        core1_result_stream,
                        hidden0,
                        hidden1,
                        CU8_MODE_RESIDUAL_ADD,
                        token_count,
                        HIDDEN_SIZE,
                        true
                    );
                total_packets +=
                    token_count * ceildiv(HIDDEN_SIZE, CU_VEC_LANES);
            }
        }
        if (op == CC8_OP_ATTENTION_SUBLAYER) {
            store_cc8_gbuf_to_hbm(
                output_port0,
                output_port1,
                hidden1,
                HIDDEN_SIZE
            );
        } else {
            store_cc8_gbuf_to_hbm(
                output_port0,
                output_port1,
                hidden0,
                HIDDEN_SIZE
            );
        }
        status.output_waves = total_wave_slots;
        status.dispatched_mm_tasks = total_mm_tasks;
        status.dispatched_vector_tasks = total_vector_tasks;
        status.completed_output_packets = total_packets;
        status_stream.write(status);
        return;
    }
#endif

#if CC8_RESIDENT_LAYER_ONLY
    // The production resident-layer image deliberately excludes the legacy
    // per-operator scheduler.  This keeps the controller FSM and its global
    // enables proportional to one fixed layer pipeline instead of the union
    // of every diagnostic path.
    status.status = CC8_STATUS_BAD_OPERATOR;
    core0_task_stream.write(build_cc8_stop_task());
    core1_task_stream.write(build_cc8_stop_task());
    status_stream.write(status);
    return;
#endif

    if (op == CC8_OP_ATTN_PREFILL_BLOCK) {
        if (token_count == 0 || token_count > MM_STREAM_8X64_TOKENS) {
            status.status = CC8_STATUS_BAD_TOKEN_COUNT;
            core0_task_stream.write(build_cc8_stop_task());
            core1_task_stream.write(build_cc8_stop_task());
            status_stream.write(status);
            return;
        }
        if (position >= MAX_SEQ_LEN || position + token_count > MAX_SEQ_LEN) {
            status.status = CC8_STATUS_BAD_POSITION;
            core0_task_stream.write(build_cc8_stop_task());
            core1_task_stream.write(build_cc8_stop_task());
            status_stream.write(status);
            return;
        }

        const unsigned int context_len = position + token_count;
        const unsigned int tile_count = ceildiv(context_len, CC8_ATTN_TILE);
        const unsigned int tasks_per_core =
            tile_count * GQA_GROUP_SIZE * (1 + CC8_ATTN_PV_WAVES);
        if (tasks_per_core > CU8_MAX_TASKS_PER_LAUNCH) {
            status.status = CC8_STATUS_BAD_POSITION;
            core0_task_stream.write(build_cc8_stop_task());
            core1_task_stream.write(build_cc8_stop_task());
            status_stream.write(status);
            return;
        }

        cc8_prefill_query_block_t query0;
        cc8_prefill_query_block_t query1;
        fm_t state0_online_max
            [GQA_GROUP_SIZE][MM_STREAM_8X64_TOKENS];
        fm_t state1_online_max
            [GQA_GROUP_SIZE][MM_STREAM_8X64_TOKENS];
        fm_accum_t state0_online_sum
            [GQA_GROUP_SIZE][MM_STREAM_8X64_TOKENS];
        fm_accum_t state1_online_sum
            [GQA_GROUP_SIZE][MM_STREAM_8X64_TOKENS];
        cc8_flash_accumulator_t state0_accumulator[GQA_GROUP_SIZE];
        cc8_flash_accumulator_t state1_accumulator[GQA_GROUP_SIZE];
        cc8_attention_buffer_t qk0;
        cc8_attention_buffer_t qk1;
        cc8_attention_buffer_t prob0;
        cc8_attention_buffer_t prob1;
        cc8_attention_qk_packet_panel_t k_panel0;
        cc8_attention_qk_packet_panel_t k_panel1;
        cc8_attention_panel_t v_panel0;
        cc8_attention_panel_t v_panel1;
        cc8_current_kv_words_t dummy_current_v;
        #pragma HLS bind_storage variable=query0.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=query1.block type=ram_2p impl=bram
        #pragma HLS array_partition variable=query0.block complete dim=2
        #pragma HLS array_partition variable=query1.block complete dim=2
        #pragma HLS bind_storage variable=state0_accumulator type=ram_2p impl=bram
        #pragma HLS bind_storage variable=state1_accumulator type=ram_2p impl=bram
        #pragma HLS array_partition variable=state0_accumulator complete dim=1
        #pragma HLS array_partition variable=state1_accumulator complete dim=1
        #pragma HLS bind_storage variable=qk0.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=qk1.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=prob0.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=prob1.block type=ram_2p impl=bram
        #pragma HLS array_partition variable=qk0.block complete dim=1
        #pragma HLS array_partition variable=qk1.block complete dim=1
        #pragma HLS array_partition variable=prob0.block complete dim=1
        #pragma HLS array_partition variable=prob1.block complete dim=1
        #pragma HLS array_partition variable=k_panel0.packet complete dim=1
        #pragma HLS array_partition variable=k_panel1.packet complete dim=1
        #pragma HLS array_partition variable=v_panel0.word complete dim=0
        #pragma HLS array_partition variable=v_panel1.word complete dim=0
        #pragma HLS array_partition variable=dummy_current_v.word complete dim=0

        load_cc8_prefill_query_block(query0, input_port0, token_count);
        load_cc8_prefill_query_block(query1, input_port1, token_count);
        init_cc8_prefill_flash_state(
            state0_online_max, state0_online_sum, state0_accumulator
        );
        init_cc8_prefill_flash_state(
            state1_online_max, state1_online_sum, state1_accumulator
        );
        clear_cc8_current_kv_words(dummy_current_v);

        attention_prob_t old_scale0[MM_STREAM_8X64_TOKENS];
        attention_prob_t old_scale1[MM_STREAM_8X64_TOKENS];

        for (unsigned int tile_begin = 0;
             tile_begin < MAX_SEQ_LEN;
             tile_begin += CC8_ATTN_TILE) {
            #pragma HLS loop_tripcount min=1 max=ATTENTION_NUM_TILES
            if (tile_begin < context_len) {
                unsigned int current_tile_len = CC8_ATTN_TILE;
                if (tile_begin + current_tile_len > context_len) {
                    current_tile_len = context_len - tile_begin;
                }

                // Critical reuse boundary: one HBM K/V tile load feeds all
                // eight GQA query heads for this query block.
                load_cc8_k_attention_packet_panel(
                    k_panel0, k_panel1, kv_cache_k,
                    layer_id, tile_begin, current_tile_len
                );
                load_cc8_kv_attention_panel(
                    v_panel0, v_panel1, kv_cache_v,
                    layer_id, tile_begin, current_tile_len
                );

                for (unsigned int head = 0;
                     head < GQA_GROUP_SIZE;
                     head++) {
                    cu8_task_t qk_task = build_cc8_compute_task(
                        CU8_MODE_MM_SCALE,
                        HEAD_DIM,
                        token_count,
                        CC8_ATTN_TILE,
                        MM_STREAM_8X64_PACKETS_PER_BLOCK,
                        0,
                        head,
                        false
                    );
                    qk_task.output_scale = fm_t(ATTENTION_SCALE);
                    core0_task_stream.write(qk_task);
                    core1_task_stream.write(qk_task);
                    emit_cc8_prefill_qk_inputs(
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
                        query0, query1, k_panel0, k_panel1,
                        head, token_count
                    );
                    collect_cc8_separate_mm_results(
                        qk0, qk1,
                        core0_result_stream, core1_result_stream,
                        token_count, CC8_ATTN_TILE
                    );

                    compute_cc8_prefill_flash_probabilities_one_core(
                        prob0, qk0,
                        state0_online_max[head],
                        state0_online_sum[head],
                        old_scale0,
                        position, token_count,
                        tile_begin, current_tile_len
                    );
                    compute_cc8_prefill_flash_probabilities_one_core(
                        prob1, qk1,
                        state1_online_max[head],
                        state1_online_sum[head],
                        old_scale1,
                        position, token_count,
                        tile_begin, current_tile_len
                    );
                    scale_cc8_prefill_flash_accumulator_one_core(
                        state0_accumulator[head], old_scale0, token_count
                    );
                    scale_cc8_prefill_flash_accumulator_one_core(
                        state1_accumulator[head], old_scale1, token_count
                    );

                    for (unsigned int output_wave = 0;
                         output_wave < CC8_ATTN_PV_WAVES;
                         output_wave++) {
                        const bool last_task =
                            tile_begin + CC8_ATTN_TILE >= context_len &&
                            head + 1 == GQA_GROUP_SIZE &&
                            output_wave + 1 == CC8_ATTN_PV_WAVES;
                        cu8_task_t pv_task = build_cc8_compute_task(
                            CU8_MODE_MM_SCALE,
                            current_tile_len,
                            token_count,
                            MM_STREAM_8X64_OUTPUTS,
                            MM_STREAM_8X64_PACKETS_PER_BLOCK,
                            output_wave * MM_STREAM_8X64_OUTPUTS,
                            output_wave,
                            last_task
                        );
                        // The 16-bit activation payload carries Q2.14
                        // probabilities.  The existing DSP path sees the raw
                        // bits as Q8.8 (64x larger), so compensate at its
                        // result conversion without widening the stream.
                        pv_task.output_scale = fm_t(1.0 / 64.0);
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
                            prob0, prob1, v_panel0, v_panel1,
                            dummy_current_v, false, 0,
                            output_wave, current_tile_len
                        );
                        accumulate_cc8_prefill_flash_pv_one_core(
                            state0_accumulator[head],
                            core0_result_stream,
                            token_count
                        );
                        accumulate_cc8_prefill_flash_pv_one_core(
                            state1_accumulator[head],
                            core1_result_stream,
                            token_count
                        );
                    }
                }
            }
        }

        store_cc8_prefill_flash_output_one_core(
            output_port0, state0_online_sum, state0_accumulator, token_count
        );
        store_cc8_prefill_flash_output_one_core(
            output_port1, state1_online_sum, state1_accumulator, token_count
        );
        status.output_waves = CC8_ATTN_PV_WAVES;
        status.dispatched_mm_tasks = CC8_MM_CORE_COUNT * tasks_per_core;
        status.completed_output_packets =
            status.dispatched_mm_tasks * MM_STREAM_8X64_PACKETS_PER_BLOCK;
        status_stream.write(status);
        return;
    }

#if !defined(CC8_PREFILL_LAYER_ONLY)
    if (op == CC8_OP_ATTN_FLASH || op == CC8_OP_DECODE_SMOKE) {
        if (token_count != GQA_GROUP_SIZE) {
            status.status = CC8_STATUS_BAD_TOKEN_COUNT;
            core0_task_stream.write(build_cc8_stop_task());
            core1_task_stream.write(build_cc8_stop_task());
            status_stream.write(status);
            return;
        }
        if (position >= MAX_SEQ_LEN) {
            status.status = CC8_STATUS_BAD_POSITION;
            core0_task_stream.write(build_cc8_stop_task());
            core1_task_stream.write(build_cc8_stop_task());
            status_stream.write(status);
            return;
        }
        if (tile_len > CC8_ATTN_TILE) {
            status.status = CC8_STATUS_BAD_TILE_LENGTH;
            core0_task_stream.write(build_cc8_stop_task());
            core1_task_stream.write(build_cc8_stop_task());
            status_stream.write(status);
            return;
        }

        cc8_attention_buffer_t query0;
        cc8_attention_buffer_t query1;
        cc8_attention_buffer_t qk0;
        cc8_attention_buffer_t qk1;
        cc8_attention_buffer_t prob0;
        cc8_attention_buffer_t prob1;
        cc8_attention_panel_t k_panel0;
        cc8_attention_panel_t k_panel1;
        cc8_attention_panel_t v_panel0;
        cc8_attention_panel_t v_panel1;
        cc8_current_kv_words_t current_k_words;
        cc8_current_kv_words_t current_v_words;
        #pragma HLS bind_storage variable=query0.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=query1.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=qk0.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=qk1.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=prob0.block type=ram_2p impl=bram
        #pragma HLS bind_storage variable=prob1.block type=ram_2p impl=bram
        #pragma HLS array_partition variable=query0.block complete dim=1
        #pragma HLS array_partition variable=query1.block complete dim=1
        #pragma HLS array_partition variable=qk0.block complete dim=1
        #pragma HLS array_partition variable=qk1.block complete dim=1
        #pragma HLS array_partition variable=prob0.block complete dim=1
        #pragma HLS array_partition variable=prob1.block complete dim=1
        #pragma HLS array_partition variable=k_panel0.word complete dim=0
        #pragma HLS array_partition variable=k_panel1.word complete dim=0
        #pragma HLS array_partition variable=v_panel0.word complete dim=0
        #pragma HLS array_partition variable=v_panel1.word complete dim=0
        #pragma HLS array_partition variable=current_k_words.word complete dim=0
        #pragma HLS array_partition variable=current_v_words.word complete dim=0

        fm_t online_max0[MM_STREAM_8X64_TOKENS];
        fm_t online_max1[MM_STREAM_8X64_TOKENS];
        fm_t old_scale0[MM_STREAM_8X64_TOKENS];
        fm_t old_scale1[MM_STREAM_8X64_TOKENS];
        fm_accum_t online_sum0[MM_STREAM_8X64_TOKENS];
        fm_accum_t online_sum1[MM_STREAM_8X64_TOKENS];
        cc8_flash_accumulator_t acc0;
        cc8_flash_accumulator_t acc1;
        #pragma HLS array_partition variable=online_max0 complete dim=1
        #pragma HLS array_partition variable=online_max1 complete dim=1
        #pragma HLS array_partition variable=old_scale0 complete dim=1
        #pragma HLS array_partition variable=old_scale1 complete dim=1
        #pragma HLS array_partition variable=online_sum0 complete dim=1
        #pragma HLS array_partition variable=online_sum1 complete dim=1
        #pragma HLS array_partition variable=acc0.value complete dim=1
        #pragma HLS array_partition variable=acc0.value complete dim=3
        #pragma HLS array_partition variable=acc1.value complete dim=1
        #pragma HLS array_partition variable=acc1.value complete dim=3

        if (op == CC8_OP_DECODE_SMOKE) {
            load_cc8_current_kv_words(current_k_words, aux_port0);
            load_cc8_current_kv_words(current_v_words, aux_port1);
            store_cc8_current_kv_to_cache(
                kv_cache_k,
                kv_cache_v,
                current_k_words,
                current_v_words,
                layer_id,
                position
            );
        } else {
            clear_cc8_current_kv_words(current_k_words);
            clear_cc8_current_kv_words(current_v_words);
        }

        load_cc8_flat_rows(
            query0,
            input_port0,
            GQA_GROUP_SIZE,
            HEAD_DIM
        );
        load_cc8_flat_rows(
            query1,
            input_port1,
            GQA_GROUP_SIZE,
            HEAD_DIM
        );
        init_cc8_flash_state(
            online_max0,
            online_max1,
            online_sum0,
            online_sum1,
            acc0,
            acc1
        );

        unsigned int context_len = position + 1;
        unsigned int output_waves =
            ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS);
        unsigned int tile_count = ceildiv(context_len, CC8_ATTN_TILE);
        status.output_waves = output_waves;
        status.dispatched_mm_tasks =
            CC8_MM_CORE_COUNT * tile_count * (1 + output_waves);
        status.completed_output_packets =
            status.dispatched_mm_tasks *
            MM_STREAM_8X64_PACKETS_PER_BLOCK;

        for (unsigned int tile_begin = 0;
             tile_begin < MAX_SEQ_LEN;
             tile_begin += CC8_ATTN_TILE) {
            #pragma HLS loop_tripcount min=1 max=ATTENTION_NUM_TILES
            if (tile_begin < context_len) {
                unsigned int current_tile_len = CC8_ATTN_TILE;
                if (tile_begin + current_tile_len > context_len) {
                    current_tile_len = context_len - tile_begin;
                }
                bool decode_current_in_tile =
                    op == CC8_OP_DECODE_SMOKE &&
                    position >= tile_begin &&
                    position < tile_begin + current_tile_len;
                unsigned int current_local_pos = 0;
                unsigned int load_tile_len = current_tile_len;
                if (decode_current_in_tile) {
                    current_local_pos = position - tile_begin;
                    load_tile_len = current_local_pos;
                }
                load_cc8_kv_attention_panel(
                    k_panel0,
                    k_panel1,
                    kv_cache_k,
                    layer_id,
                    tile_begin,
                    load_tile_len
                );
                load_cc8_kv_attention_panel(
                    v_panel0,
                    v_panel1,
                    kv_cache_v,
                    layer_id,
                    tile_begin,
                    load_tile_len
                );

                cu8_task_t qk_task = build_cc8_compute_task(
                    CU8_MODE_MM_SCALE,
                    HEAD_DIM,
                    GQA_GROUP_SIZE,
                    CC8_ATTN_TILE,
                    MM_STREAM_8X64_PACKETS_PER_BLOCK,
                    0,
                    0,
                    false
                );
                qk_task.output_scale = fm_t(ATTENTION_SCALE);
                core0_task_stream.write(qk_task);
                core1_task_stream.write(qk_task);
                run_cc8_attention_qk_exchange(
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
                    query0,
                    query1,
                    k_panel0,
                    k_panel1,
                    current_k_words,
                    qk0,
                    qk1,
                    decode_current_in_tile,
                    current_local_pos,
                    GQA_GROUP_SIZE,
                    CC8_ATTN_TILE
                );

                compute_cc8_flash_probabilities_one_core(
                    prob0,
                    qk0,
                    online_max0,
                    online_sum0,
                    old_scale0,
                    current_tile_len
                );
                compute_cc8_flash_probabilities_one_core(
                    prob1,
                    qk1,
                    online_max1,
                    online_sum1,
                    old_scale1,
                    current_tile_len
                );
                scale_cc8_flash_accumulator_one_core(acc0, old_scale0);
                scale_cc8_flash_accumulator_one_core(acc1, old_scale1);

                for (unsigned int output_wave = 0;
                     output_wave < ceildiv(
                         CC8_ATTN_BUFFER_ELEMS,
                         MM_STREAM_8X64_OUTPUTS
                     );
                     output_wave++) {
                    if (output_wave < output_waves) {
                        bool last_tile =
                            tile_begin + CC8_ATTN_TILE >= context_len;
                        bool last_wave =
                            last_tile &&
                            output_wave + 1 == output_waves;
                        cu8_task_t pv_task = build_cc8_compute_task(
                            CU8_MODE_MM,
                            current_tile_len,
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
                            prob0,
                            prob1,
                            v_panel0,
                            v_panel1,
                            current_v_words,
                            decode_current_in_tile,
                            current_local_pos,
                            output_wave,
                            current_tile_len
                        );
                        accumulate_cc8_flash_pv_one_core(
                            acc0,
                            core0_result_stream
                        );
                        accumulate_cc8_flash_pv_one_core(
                            acc1,
                            core1_result_stream
                        );
                    }
                }
            }
        }

        store_cc8_flash_output_one_core(
            output_port0,
            acc0,
            online_sum0
        );
        store_cc8_flash_output_one_core(
            output_port1,
            acc1,
            online_sum1
        );
        status_stream.write(status);
        return;
    }
#endif

#if !defined(CC8_PREFILL_LAYER_ONLY)
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
        cc8_current_kv_words_t dummy_current_kv;
        #pragma HLS array_partition variable=panel0.word complete dim=0
        #pragma HLS array_partition variable=panel1.word complete dim=0
        #pragma HLS array_partition variable=dummy_current_kv.word complete dim=0
        clear_cc8_current_kv_words(dummy_current_kv);
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
            run_cc8_attention_qk_exchange(
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
                source0,
                source1,
                panel0,
                panel1,
                dummy_current_kv,
                destination0,
                destination1,
                false,
                0,
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
                    dummy_current_kv,
                    false,
                    0,
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
#endif

    unsigned int profile_debug_stage =
        tile_len != 0 ? ((profile_flags >> 1) & 0xfu) : 0u;
    unsigned int profile_core_mask =
        tile_len != 0 ? ((profile_flags >> 5) & 0x3u) : 0x3u;
    if (profile_core_mask == 0u) {
        profile_core_mask = 0x3u;
    }

    if (spec.uses_mm) {
        mm_projection_spec_t projection_spec;
        get_mm_projection_spec(spec.projection, projection_spec);
        weight_addr_t layer_base =
            weight_addr_t(layer_id) * weight_addr_t(LAYER_WEIGHT_SIZE);
        weight_addr_t weight_base =
            layer_base + projection_spec.weight_base;
        unsigned int output_waves =
            ceildiv(spec.out_dim, CC8_OUTPUTS_PER_WAVE);
        unsigned int profile_wave_begin = 0;
        unsigned int profile_wave_end = output_waves;
        unsigned int active_in_dim = spec.in_dim;
        bool zero_weight_stream = false;
        if (tile_len != 0) {
            unsigned int profile_wave_count = tile_len & 0xffffu;
            unsigned int profile_k_limit = tile_len >> 16;
            zero_weight_stream = (profile_flags & 1u) != 0u;
            profile_wave_begin = position;
            if (profile_wave_count == 0) {
                profile_wave_count =
                    profile_wave_begin < output_waves ?
                    output_waves - profile_wave_begin :
                    0;
            }
            profile_wave_end = position + profile_wave_count;
            if (profile_wave_begin > output_waves) {
                profile_wave_begin = output_waves;
            }
            if (profile_wave_end > output_waves) {
                profile_wave_end = output_waves;
            }
            if (profile_k_limit != 0 && profile_k_limit < active_in_dim) {
                unsigned int aligned_k =
                    (profile_k_limit / MM_PE_IN) * MM_PE_IN;
                if (aligned_k == 0) {
                    aligned_k = MM_PE_IN;
                }
                if (aligned_k > active_in_dim) {
                    aligned_k = active_in_dim;
                }
                active_in_dim = aligned_k;
            }
        }
        unsigned int active_output_waves =
            profile_wave_end > profile_wave_begin ?
            profile_wave_end - profile_wave_begin :
            0;
        unsigned int active_core_count =
            cc8_core_mask_count(profile_core_mask);
        status.output_waves = output_waves;
        status.dispatched_mm_tasks =
            active_output_waves * active_core_count;

        if (tile_len != 0 && profile_debug_stage == 1u) {
            status.dispatched_mm_tasks = 0;
            status.completed_output_packets = 0;
            core0_task_stream.write(build_cc8_stop_task());
            core1_task_stream.write(build_cc8_stop_task());
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
        clear_cc8_gbuf(gbuf1, spec.out_dim);

        if (tile_len != 0 && profile_debug_stage == 2u) {
            status.dispatched_mm_tasks = 0;
            status.completed_output_packets = 0;
            core0_task_stream.write(build_cc8_stop_task());
            core1_task_stream.write(build_cc8_stop_task());
            status_stream.write(status);
            return;
        }

        if (active_output_waves == 0) {
            core0_task_stream.write(build_cc8_stop_task());
            core1_task_stream.write(build_cc8_stop_task());
            status_stream.write(status);
            return;
        }

        // Keep the token dimension explicit across synthesis.  Flattening
        // the completely partitioned [token][block] GBUF into a pointer made
        // software simulation look correct, but RTL only preserved row zero
        // for multi-token MM projections.
        run_cc8_projection_waves(
            core0_task_stream,
            core0_activation_stream,
            core0_weight_stream0,
            core0_weight_stream1,
            core0_weight_stream2,
            core0_weight_stream3,
            core0_result_stream,
            core1_task_stream,
            core1_activation_stream,
            core1_weight_stream0,
            core1_weight_stream1,
            core1_weight_stream2,
            core1_weight_stream3,
            core1_result_stream,
            gbuf0,
            gbuf1,
            profile_wave_begin,
            profile_wave_end,
            token_count,
            active_in_dim,
            spec.in_dim,
            spec.out_dim,
            weight_base,
            profile_core_mask,
            zero_weight_stream,
            true,
            QWEN_WEIGHT_SHARD_ARGS
        );
        store_cc8_gbuf_to_hbm(
            output_port0,
            output_port1,
            gbuf1,
            spec.out_dim
        );
        status.completed_output_packets =
            active_output_waves *
            active_core_count *
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

    unsigned int blocks_per_token = ceildiv(spec.out_dim, CU_VEC_LANES);
    unsigned int core0_token_count =
        token_count < CC8_TOKENS_PER_DATA_PORT ?
        token_count : CC8_TOKENS_PER_DATA_PORT;
    unsigned int core1_token_count =
        token_count > CC8_TOKENS_PER_DATA_PORT ?
        token_count - CC8_TOKENS_PER_DATA_PORT : 0;
    unsigned int core0_packet_count =
        core0_token_count * blocks_per_token;
    unsigned int core1_packet_count =
        core1_token_count * blocks_per_token;

    cu8_task_t core0_task = core0_token_count == 0 ?
        build_cc8_stop_task() :
        build_cc8_compute_task(
            spec.compute_mode,
            0,
            core0_token_count,
            spec.in_dim,
            core0_packet_count,
            0,
            0,
            true
        );
    cu8_task_t core1_task = core1_token_count == 0 ?
        build_cc8_stop_task() :
        build_cc8_compute_task(
            spec.compute_mode,
            0,
            core1_token_count,
            spec.in_dim,
            core1_packet_count,
            0,
            0,
            true
        );
    core0_task_stream.write(core0_task);
    core1_task_stream.write(core1_task);

    run_cc8_generic_vector_exchange(
        output_port0,
        output_port1,
        core0_vector_input0_stream,
        core0_vector_input1_stream,
        core0_result_stream,
        core1_vector_input0_stream,
        core1_vector_input1_stream,
        core1_result_stream,
        gbuf0,
        gbuf1,
        op,
        token_count,
        core0_token_count,
        core1_token_count,
        rhs_token_slots,
        rhs_valid_tokens,
        spec.in_dim,
        core0_packet_count,
        core1_packet_count
    );
    status.dispatched_vector_tasks =
        (core0_token_count != 0 ? 1 : 0) +
        (core1_token_count != 0 ? 1 : 0);
    status.completed_output_packets =
        core0_packet_count + core1_packet_count;
    status_stream.write(status);
}
