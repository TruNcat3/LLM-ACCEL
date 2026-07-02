#include "control_cache_8x64.hpp"

#include <cmath>
#include <cstdio>

static fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT];
static fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT];
static fm_word_t input_port0[CC8_DATA_PORT_WORDS];
static fm_word_t input_port1[CC8_DATA_PORT_WORDS];
static fm_word_t aux_port0[CC8_DATA_PORT_WORDS];
static fm_word_t aux_port1[CC8_DATA_PORT_WORDS];

static wt_block_t shard0[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard1[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard2[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard3[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard4[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard5[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard6[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard7[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard8[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard9[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard10[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard11[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard12[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard13[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard14[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard15[WEIGHT_SHARD_BLOCKS];

static wt_block_t* shards[NUM_WEIGHT_SHARDS] = {
    shard0, shard1, shard2, shard3, shard4, shard5, shard6, shard7,
    shard8, shard9, shard10, shard11, shard12, shard13, shard14, shard15
};

struct cc8_test_streams_t {
    hls::stream<cu8_task_t> core0_task;
    hls::stream<mm_stream_8x64_activation_packet_t> core0_activation;
    hls::stream<mm_stream_8x64_weight_packet_t> core0_weight0;
    hls::stream<mm_stream_8x64_weight_packet_t> core0_weight1;
    hls::stream<mm_stream_8x64_weight_packet_t> core0_weight2;
    hls::stream<mm_stream_8x64_weight_packet_t> core0_weight3;
    hls::stream<cu_vec16_packet_t> core0_vector0;
    hls::stream<cu_vec16_packet_t> core0_vector1;
    hls::stream<cu_vec16_packet_t> core0_result;
    hls::stream<cu8_task_t> core1_task;
    hls::stream<mm_stream_8x64_activation_packet_t> core1_activation;
    hls::stream<mm_stream_8x64_weight_packet_t> core1_weight0;
    hls::stream<mm_stream_8x64_weight_packet_t> core1_weight1;
    hls::stream<mm_stream_8x64_weight_packet_t> core1_weight2;
    hls::stream<mm_stream_8x64_weight_packet_t> core1_weight3;
    hls::stream<cu_vec16_packet_t> core1_vector0;
    hls::stream<cu_vec16_packet_t> core1_vector1;
    hls::stream<cu_vec16_packet_t> core1_result;
    hls::stream<cc8_status_packet_t> status;
};

static void clear_data_ports() {
    for (unsigned int word = 0; word < CC8_DATA_PORT_WORDS; word++) {
        input_port0[word] = 0;
        input_port1[word] = 0;
        aux_port0[word] = 0;
        aux_port1[word] = 0;
    }
    for (unsigned int word = 0; word < CC8_FEATURE_WORDS_PER_PORT; word++) {
        output_port0[word] = 0;
        output_port1[word] = 0;
    }
}

static void set_feature_value(
    fm_word_t port0[CC8_DATA_PORT_WORDS],
    fm_word_t port1[CC8_DATA_PORT_WORDS],
    unsigned int token,
    unsigned int elem,
    fm_t value
) {
    unsigned int port = token / CC8_TOKENS_PER_DATA_PORT;
    unsigned int token_in_port =
        token - port * CC8_TOKENS_PER_DATA_PORT;
    unsigned int word_idx =
        token_in_port * CC8_FEATURE_WORDS_PER_TOKEN +
        elem / FM_BLOCK_SIZE;
    if (port == 0) {
        set_fm_word_lane(port0[word_idx], elem % FM_BLOCK_SIZE, value);
    } else {
        set_fm_word_lane(port1[word_idx], elem % FM_BLOCK_SIZE, value);
    }
}

static fm_t get_output_value(unsigned int token, unsigned int elem) {
    unsigned int port = token / CC8_TOKENS_PER_DATA_PORT;
    unsigned int token_in_port =
        token - port * CC8_TOKENS_PER_DATA_PORT;
    unsigned int word_idx =
        token_in_port * CC8_FEATURE_WORDS_PER_TOKEN +
        elem / FM_BLOCK_SIZE;
    return port == 0 ?
        unpack_fm_word_lane(output_port0[word_idx], elem % FM_BLOCK_SIZE) :
        unpack_fm_word_lane(output_port1[word_idx], elem % FM_BLOCK_SIZE);
}

static fm_t get_flat_output_value(
    const fm_word_t output[CC8_FEATURE_WORDS_PER_PORT],
    unsigned int row,
    unsigned int elem,
    unsigned int elem_count
) {
    unsigned int words_per_row = ceildiv(elem_count, FM_BLOCK_SIZE);
    unsigned int word =
        row * words_per_row + elem / FM_BLOCK_SIZE;
    return unpack_fm_word_lane(output[word], elem % FM_BLOCK_SIZE);
}

static void set_flat_row_value(
    fm_word_t words[CC8_DATA_PORT_WORDS],
    unsigned int row,
    unsigned int elem,
    unsigned int row_elems,
    fm_t value
) {
    unsigned int words_per_row = ceildiv(row_elems, FM_BLOCK_SIZE);
    unsigned int word = row * words_per_row + elem / FM_BLOCK_SIZE;
    set_fm_word_lane(
        words[word],
        elem % FM_BLOCK_SIZE,
        value
    );
}

static fm_t input_value(unsigned int token, unsigned int elem) {
    int raw = int((token * 11 + elem * 3) % 31) - 15;
    return fm_t(raw * 0.0625);
}

static fm_t aux_value(unsigned int token, unsigned int elem) {
    int raw = int((token * 7 + elem * 5 + 3) % 29) - 14;
    return fm_t(raw * 0.0625);
}

static fm_t result_value(unsigned int token, unsigned int elem) {
    int raw = int((token * 13 + elem * 7 + 5) % 47) - 23;
    return fm_t(raw * 0.03125);
}

static fm_t attention_input_value(
    unsigned int core,
    unsigned int row,
    unsigned int elem
) {
    int raw = int((core * 19 + row * 7 + elem * 3) % 29) - 14;
    return fm_t(raw * 0.0625);
}

static fm_t attention_panel_value(
    unsigned int core,
    unsigned int pos,
    unsigned int elem
) {
    int raw = int((core * 13 + pos * 5 + elem * 7 + 1) % 31) - 15;
    return fm_t(raw * 0.03125);
}

static fm_t attention_result_value(
    unsigned int core,
    unsigned int row,
    unsigned int elem
) {
    int raw = int((core * 17 + row * 11 + elem * 5 + 2) % 37) - 18;
    return fm_t(raw * 0.03125);
}

static wt_linear_t weight_value(
    unsigned int shard,
    weight_addr_t local_block,
    unsigned int lane
) {
    int raw = int(
        (shard * 17 + unsigned(local_block) * 11 + lane * 3) % 31
    ) - 15;
    return wt_linear_t(raw * 0.03125);
}

static void init_weights() {
    for (unsigned int shard = 0; shard < NUM_WEIGHT_SHARDS; shard++) {
        for (weight_addr_t block = 0;
             block < weight_addr_t(WEIGHT_SHARD_BLOCKS);
             block++) {
            wt_block_t packed = 0;
            for (unsigned int lane = 0; lane < WT_BLOCK_SIZE; lane++) {
                set_wt_block_lane(
                    packed,
                    lane,
                    weight_value(shard, block, lane)
                );
            }
            shards[shard][block] = packed;
        }
    }
}

static void call_control_cache(
    cc8_test_streams_t& streams,
    cc8_operator_t op,
    unsigned int token_count,
    unsigned int position,
    unsigned int tile_len
) {
    control_cache_8x64_dual_core(
        streams.core0_task,
        streams.core0_activation,
        streams.core0_weight0,
        streams.core0_weight1,
        streams.core0_weight2,
        streams.core0_weight3,
        streams.core0_vector0,
        streams.core0_vector1,
        streams.core0_result,
        streams.core1_task,
        streams.core1_activation,
        streams.core1_weight0,
        streams.core1_weight1,
        streams.core1_weight2,
        streams.core1_weight3,
        streams.core1_vector0,
        streams.core1_vector1,
        streams.core1_result,
        streams.status,
        output_port0,
        output_port1,
        input_port0,
        input_port1,
        aux_port0,
        aux_port1,
        unsigned(op),
        0,
        token_count,
        position,
        tile_len,
        shard0, shard1, shard2, shard3,
        shard4, shard5, shard6, shard7,
        shard8, shard9, shard10, shard11,
        shard12, shard13, shard14, shard15
    );
}

static void preload_mm_results(
    cc8_test_streams_t& streams,
    unsigned int elem_base0,
    unsigned int elem_base1,
    bool last_stream
) {
    for (unsigned int token = 0; token < MM_STREAM_8X64_TOKENS; token++) {
        for (unsigned int group = 0;
             group < MM_STREAM_8X64_WEIGHT_GROUPS;
             group++) {
            cu_vec16_packet_t packet0;
            cu_vec16_packet_t packet1;
            packet0.valid_mask = 0xffff;
            packet1.valid_mask = 0xffff;
            packet0.token_lane = token;
            packet1.token_lane = token;
            packet0.elem_base = elem_base0 + group * CU_VEC_LANES;
            packet1.elem_base = elem_base1 + group * CU_VEC_LANES;
            packet0.block_id = 0;
            packet1.block_id = 0;
            packet0.last_block =
                token + 1 == MM_STREAM_8X64_TOKENS &&
                group + 1 == MM_STREAM_8X64_WEIGHT_GROUPS;
            packet1.last_block = packet0.last_block;
            packet0.last_stream = last_stream && packet0.last_block;
            packet1.last_stream = last_stream && packet1.last_block;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                unsigned int elem0 = packet0.elem_base + lane;
                unsigned int elem1 = packet1.elem_base + lane;
                packet0.data[lane] = result_value(token, elem0);
                packet1.data[lane] = result_value(token, elem1);
            }
            streams.core0_result.write(packet0);
            streams.core1_result.write(packet1);
        }
    }
}

static wt_linear_t expected_weight(
    mm_projection_kind_t projection,
    unsigned int core,
    unsigned int group,
    unsigned int k,
    unsigned int out_lane
) {
    mm_projection_spec_t spec;
    get_mm_projection_spec(projection, spec);
    unsigned int in_tile_count = ceildiv(spec.in_dim, MM_PE_IN);
    unsigned int out_tile =
        core * MM_STREAM_8X64_WEIGHT_GROUPS + group;
    unsigned int in_tile = k / MM_PE_IN;
    unsigned int k_lane = k % MM_PE_IN;
    weight_addr_t global_tile =
        spec.weight_base / weight_addr_t(MM_TILE_WEIGHT_ELEMS) +
        weight_addr_t(out_tile) * weight_addr_t(in_tile_count) +
        weight_addr_t(in_tile);
    unsigned int shard_group = unsigned(global_tile & weight_addr_t(1));
    weight_addr_t local_block = global_tile >> 1;
    unsigned int lane_in_tile = out_lane * MM_PE_IN + k_lane;
    unsigned int tile_word = lane_in_tile / WT_BLOCK_SIZE;
    unsigned int word_lane = lane_in_tile % WT_BLOCK_SIZE;
    return weight_value(
        shard_group * MM_TILE_WEIGHT_BLOCKS + tile_word,
        local_block,
        word_lane
    );
}

static int check_mm_streams(
    cc8_test_streams_t& streams,
    unsigned int token_count,
    mm_projection_kind_t projection
) {
    int errors = 0;
    mm_projection_spec_t spec;
    get_mm_projection_spec(projection, spec);

    if (streams.core0_task.empty() || streams.core1_task.empty()) {
        std::printf("dual MM task stream underflow\n");
        return 1;
    }
    cu8_task_t task0 = streams.core0_task.read();
    cu8_task_t task1 = streams.core1_task.read();
    if (task0.k_count != spec.in_dim || task1.k_count != spec.in_dim ||
        task0.mode != CU8_MODE_MM || task1.mode != CU8_MODE_MM ||
        task0.elem_base != 0 || task1.elem_base != 64 ||
        task0.block_id != 0 || task1.block_id != 0 ||
        !task0.last_task || !task1.last_task) {
        std::printf("dual MM task metadata mismatch\n");
        errors++;
    }

    for (unsigned int k = 0; k < spec.in_dim; k++) {
        if (streams.core0_activation.empty() ||
            streams.core1_activation.empty()) {
            std::printf("activation stream underflow k=%u\n", k);
            return errors + 1;
        }
        mm_stream_8x64_activation_packet_t activation0 =
            streams.core0_activation.read();
        mm_stream_8x64_activation_packet_t activation1 =
            streams.core1_activation.read();
        for (unsigned int token = 0;
             token < MM_STREAM_8X64_TOKENS;
             token++) {
            fm_t expected =
                token < token_count ? input_value(token, k) : fm_t(0);
            if (activation0.data[token] != expected ||
                activation1.data[token] != expected) {
                std::printf(
                    "activation mismatch k=%u token=%u\n",
                    k,
                    token
                );
                errors++;
            }
        }

        mm_stream_8x64_weight_packet_t packets[CC8_MM_CORE_COUNT][4];
        packets[0][0] = streams.core0_weight0.read();
        packets[0][1] = streams.core0_weight1.read();
        packets[0][2] = streams.core0_weight2.read();
        packets[0][3] = streams.core0_weight3.read();
        packets[1][0] = streams.core1_weight0.read();
        packets[1][1] = streams.core1_weight1.read();
        packets[1][2] = streams.core1_weight2.read();
        packets[1][3] = streams.core1_weight3.read();
        for (unsigned int core = 0; core < CC8_MM_CORE_COUNT; core++) {
            for (unsigned int group = 0; group < 4; group++) {
                for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                    wt_linear_t expected = expected_weight(
                        projection,
                        core,
                        group,
                        k,
                        lane
                    );
                    if (packets[core][group].data[lane] != expected) {
                        std::printf(
                            "weight mismatch core=%u group=%u k=%u lane=%u\n",
                            core,
                            group,
                            k,
                            lane
                        );
                        errors++;
                    }
                }
            }
        }
    }

    if (!streams.core0_task.empty() || !streams.core1_task.empty() ||
        !streams.core0_activation.empty() ||
        !streams.core1_activation.empty()) {
        std::printf("dual MM streams contain extra packets\n");
        errors++;
    }
    return errors;
}

static int run_gate_projection_case() {
    clear_data_ports();
    unsigned int token_count = LINEAR_TOKEN_TILE_ACTIVE - 3;
    for (unsigned int token = 0; token < token_count; token++) {
        for (unsigned int elem = 0; elem < HIDDEN_SIZE; elem++) {
            set_feature_value(
                input_port0,
                input_port1,
                token,
                elem,
                input_value(token, elem)
            );
        }
    }

    cc8_test_streams_t streams;
    preload_mm_results(streams, 0, 64, true);
    call_control_cache(
        streams,
        CC8_OP_FFN_GATE,
        token_count,
        0,
        0
    );

    int errors = check_mm_streams(
        streams,
        token_count,
        MM_PROJECTION_FFN_GATE
    );
    for (unsigned int token = 0;
         token < LINEAR_TOKEN_TILE_ACTIVE;
         token++) {
        for (unsigned int elem = 0; elem < INTERMEDIATE_SIZE; elem++) {
            fm_t expected =
                token < token_count ?
                result_value(token, elem) :
                fm_t(0);
            if (get_output_value(token, elem) != expected) {
                std::printf(
                    "gate output mismatch token=%u elem=%u\n",
                    token,
                    elem
                );
                errors++;
            }
        }
    }

    if (streams.status.empty()) {
        std::printf("gate status missing\n");
        errors++;
    } else {
        cc8_status_packet_t status = streams.status.read();
        if (status.op != CC8_OP_FFN_GATE ||
            status.status != CC8_STATUS_OK ||
            status.output_waves != 1 ||
            status.dispatched_mm_tasks != 2 ||
            status.dispatched_vector_tasks != 0 ||
            status.completed_output_packets != 64) {
            std::printf("gate status mismatch\n");
            errors++;
        }
    }
    return errors;
}

static void preload_vector_results(
    hls::stream<cu_vec16_packet_t>& result_stream,
    unsigned int token_begin,
    unsigned int token_slots,
    unsigned int elem_count
) {
    unsigned int blocks = ceildiv(elem_count, CU_VEC_LANES);
    for (unsigned int token = token_begin;
         token < token_begin + token_slots;
         token++) {
        for (unsigned int block = 0; block < blocks; block++) {
            cu_vec16_packet_t packet;
            packet.valid_mask = 0xffff;
            packet.token_lane = token;
            packet.elem_base = block * CU_VEC_LANES;
            packet.block_id = 0;
            packet.last_block =
                token + 1 == token_begin + token_slots &&
                block + 1 == blocks;
            packet.last_stream = packet.last_block;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                packet.data[lane] = result_value(
                    token,
                    packet.elem_base + lane
                );
            }
            result_stream.write(packet);
        }
    }
}

static int check_silu_mul_inputs(
    hls::stream<cu_vec16_packet_t>& lhs_stream,
    hls::stream<cu_vec16_packet_t>& rhs_stream,
    unsigned int packets,
    unsigned int token_count,
    unsigned int core
) {
    int errors = 0;
    for (unsigned int packet_idx = 0; packet_idx < packets; packet_idx++) {
        cu_vec16_packet_t lhs = lhs_stream.read();
        cu_vec16_packet_t rhs = rhs_stream.read();
        for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
            unsigned int elem = lhs.elem_base + lane;
            bool valid = lhs.token_lane < token_count;
            fm_t expected_lhs = valid ?
                input_value(lhs.token_lane, elem) :
                fm_t(0);
            fm_t expected_rhs = valid ?
                aux_value(rhs.token_lane, elem) :
                fm_t(0);
            if (lhs.data[lane] != expected_lhs ||
                rhs.data[lane] != expected_rhs ||
                bool(lhs.valid_mask[lane]) != valid ||
                bool(rhs.valid_mask[lane]) != valid) {
                std::printf(
                    "SiLU-Mul input mismatch core=%u packet=%u lane=%u\n",
                    core,
                    packet_idx,
                    lane
                );
                errors++;
            }
        }
    }
    return errors;
}

static void preload_attention_mm_results(
    cc8_test_streams_t& streams,
    unsigned int elem_base
) {
    for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
        for (unsigned int group = 0;
             group < MM_STREAM_8X64_WEIGHT_GROUPS;
             group++) {
            cu_vec16_packet_t packet0;
            cu_vec16_packet_t packet1;
            packet0.valid_mask = 0xffff;
            packet1.valid_mask = 0xffff;
            packet0.token_lane = row;
            packet1.token_lane = row;
            packet0.elem_base = elem_base + group * CU_VEC_LANES;
            packet1.elem_base = packet0.elem_base;
            packet0.block_id = elem_base / MM_STREAM_8X64_OUTPUTS;
            packet1.block_id = packet0.block_id;
            packet0.last_block =
                row + 1 == MM_STREAM_8X64_TOKENS &&
                group + 1 == MM_STREAM_8X64_WEIGHT_GROUPS;
            packet1.last_block = packet0.last_block;
            packet0.last_stream = packet0.last_block;
            packet1.last_stream = packet1.last_block;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                unsigned int elem = packet0.elem_base + lane;
                packet0.data[lane] =
                    attention_result_value(0, row, elem);
                packet1.data[lane] =
                    attention_result_value(1, row, elem);
            }
            streams.core0_result.write(packet0);
            streams.core1_result.write(packet1);
        }
    }
}

static void preload_attention_vector_results(
    cc8_test_streams_t& streams,
    unsigned int elem_count
) {
    unsigned int blocks = ceildiv(elem_count, CU_VEC_LANES);
    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int block = 0; block < blocks; block++) {
            cu_vec16_packet_t packet0;
            cu_vec16_packet_t packet1;
            packet0.valid_mask = 0;
            packet1.valid_mask = 0;
            packet0.token_lane = row;
            packet1.token_lane = row;
            packet0.elem_base = block * CU_VEC_LANES;
            packet1.elem_base = packet0.elem_base;
            packet0.block_id = 0;
            packet1.block_id = 0;
            packet0.last_block =
                row + 1 == GQA_GROUP_SIZE && block + 1 == blocks;
            packet1.last_block = packet0.last_block;
            packet0.last_stream = packet0.last_block;
            packet1.last_stream = packet1.last_block;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                unsigned int elem = packet0.elem_base + lane;
                bool valid = elem < elem_count;
                packet0.valid_mask[lane] = valid;
                packet1.valid_mask[lane] = valid;
                packet0.data[lane] = valid ?
                    attention_result_value(0, row, elem) :
                    fm_t(0);
                packet1.data[lane] = valid ?
                    attention_result_value(1, row, elem) :
                    fm_t(0);
            }
            streams.core0_result.write(packet0);
            streams.core1_result.write(packet1);
        }
    }
}

static int run_attention_softmax_route_case() {
    clear_data_ports();
    const unsigned int tile_len = 8;
    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int elem = 0; elem < CC8_ATTN_TILE; elem++) {
            set_flat_row_value(
                input_port0,
                row,
                elem,
                CC8_ATTN_TILE,
                attention_input_value(0, row, elem)
            );
            set_flat_row_value(
                input_port1,
                row,
                elem,
                CC8_ATTN_TILE,
                attention_input_value(1, row, elem)
            );
        }
    }

    cc8_test_streams_t streams;
    preload_attention_vector_results(streams, tile_len);
    call_control_cache(
        streams,
        CC8_OP_SOFTMAX,
        GQA_GROUP_SIZE,
        0,
        tile_len
    );

    int errors = 0;
    unsigned int blocks = ceildiv(tile_len, CU_VEC_LANES);
    unsigned int packets = GQA_GROUP_SIZE * blocks;
    cu8_task_t task0 = streams.core0_task.read();
    cu8_task_t task1 = streams.core1_task.read();
    if (task0.mode != CU8_MODE_SOFTMAX ||
        task1.mode != CU8_MODE_SOFTMAX ||
        task0.token_count != GQA_GROUP_SIZE ||
        task1.token_count != GQA_GROUP_SIZE ||
        task0.elem_count != tile_len ||
        task1.elem_count != tile_len ||
        task0.packet_count != packets ||
        task1.packet_count != packets ||
        !task0.last_task || !task1.last_task) {
        std::printf("attention softmax generic task mismatch\n");
        errors++;
    }

    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int block = 0; block < blocks; block++) {
            cu_vec16_packet_t packet0 = streams.core0_vector0.read();
            cu_vec16_packet_t packet1 = streams.core1_vector0.read();
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                unsigned int elem = block * CU_VEC_LANES + lane;
                bool valid = elem < tile_len;
                fm_t expected0 = valid ?
                    attention_input_value(0, row, elem) :
                    fm_t(0);
                fm_t expected1 = valid ?
                    attention_input_value(1, row, elem) :
                    fm_t(0);
                if (packet0.data[lane] != expected0 ||
                    packet1.data[lane] != expected1 ||
                    bool(packet0.valid_mask[lane]) != valid ||
                    bool(packet1.valid_mask[lane]) != valid) {
                    std::printf(
                        "attention softmax input mismatch row=%u elem=%u\n",
                        row,
                        elem
                    );
                    errors++;
                }
            }
        }
    }

    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int elem = 0; elem < tile_len; elem++) {
            if (get_flat_output_value(
                    output_port0,
                    row,
                    elem,
                    tile_len
                ) != attention_result_value(0, row, elem) ||
                get_flat_output_value(
                    output_port1,
                    row,
                    elem,
                    tile_len
                ) != attention_result_value(1, row, elem)) {
                std::printf(
                    "attention softmax output mismatch row=%u elem=%u\n",
                    row,
                    elem
                );
                errors++;
            }
        }
    }

    cc8_status_packet_t status = streams.status.read();
    if (status.status != CC8_STATUS_OK ||
        status.dispatched_vector_tasks != CC8_MM_CORE_COUNT ||
        status.completed_output_packets != CC8_MM_CORE_COUNT * packets) {
        std::printf("attention softmax status mismatch\n");
        errors++;
    }
    return errors;
}

static int run_attention_qk_route_case() {
    clear_data_ports();
    const unsigned int tile_len = 8;
    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            set_flat_row_value(
                input_port0,
                row,
                elem,
                HEAD_DIM,
                attention_input_value(0, row, elem)
            );
            set_flat_row_value(
                input_port1,
                row,
                elem,
                HEAD_DIM,
                attention_input_value(1, row, elem)
            );
        }
    }
    for (unsigned int pos = 0; pos < tile_len; pos++) {
        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            set_flat_row_value(
                aux_port0,
                pos,
                elem,
                HEAD_DIM,
                attention_panel_value(0, pos, elem)
            );
            set_flat_row_value(
                aux_port1,
                pos,
                elem,
                HEAD_DIM,
                attention_panel_value(1, pos, elem)
            );
        }
    }

    cc8_test_streams_t streams;
    preload_attention_mm_results(streams, 0);
    call_control_cache(
        streams,
        CC8_OP_ATTN_QK,
        GQA_GROUP_SIZE,
        0,
        tile_len
    );

    int errors = 0;
    cu8_task_t task0 = streams.core0_task.read();
    cu8_task_t task1 = streams.core1_task.read();
    if (task0.mode != CU8_MODE_MM_SCALE ||
        task1.mode != CU8_MODE_MM_SCALE ||
        task0.k_count != HEAD_DIM ||
        task1.k_count != HEAD_DIM ||
        task0.output_scale != fm_t(ATTENTION_SCALE) ||
        task1.output_scale != fm_t(ATTENTION_SCALE) ||
        !task0.last_task || !task1.last_task) {
        std::printf("attention QK generic MM task mismatch\n");
        errors++;
    }

    for (unsigned int k = 0; k < HEAD_DIM; k++) {
        mm_stream_8x64_activation_packet_t a0 =
            streams.core0_activation.read();
        mm_stream_8x64_activation_packet_t a1 =
            streams.core1_activation.read();
        for (unsigned int row = 0;
             row < MM_STREAM_8X64_TOKENS;
             row++) {
            fm_t expected0 =
                row < GQA_GROUP_SIZE ?
                attention_input_value(0, row, k) :
                fm_t(0);
            fm_t expected1 =
                row < GQA_GROUP_SIZE ?
                attention_input_value(1, row, k) :
                fm_t(0);
            if (a0.data[row] != expected0 || a1.data[row] != expected1) {
                std::printf(
                    "attention QK activation mismatch k=%u row=%u\n",
                    k,
                    row
                );
                errors++;
            }
        }

        mm_stream_8x64_weight_packet_t weights0[4];
        mm_stream_8x64_weight_packet_t weights1[4];
        weights0[0] = streams.core0_weight0.read();
        weights0[1] = streams.core0_weight1.read();
        weights0[2] = streams.core0_weight2.read();
        weights0[3] = streams.core0_weight3.read();
        weights1[0] = streams.core1_weight0.read();
        weights1[1] = streams.core1_weight1.read();
        weights1[2] = streams.core1_weight2.read();
        weights1[3] = streams.core1_weight3.read();
        for (unsigned int group = 0; group < 4; group++) {
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                unsigned int pos = group * CU_VEC_LANES + lane;
                wt_linear_t expected0 =
                    pos < tile_len ?
                    wt_linear_t(attention_panel_value(0, pos, k)) :
                    wt_linear_t(0);
                wt_linear_t expected1 =
                    pos < tile_len ?
                    wt_linear_t(attention_panel_value(1, pos, k)) :
                    wt_linear_t(0);
                if (weights0[group].data[lane] != expected0 ||
                    weights1[group].data[lane] != expected1) {
                    std::printf(
                        "attention QK weight mismatch k=%u pos=%u\n",
                        k,
                        pos
                    );
                    errors++;
                }
            }
        }
    }

    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int elem = 0; elem < CC8_ATTN_TILE; elem++) {
            if (get_flat_output_value(
                    output_port0,
                    row,
                    elem,
                    CC8_ATTN_TILE
                ) != attention_result_value(0, row, elem) ||
                get_flat_output_value(
                    output_port1,
                    row,
                    elem,
                    CC8_ATTN_TILE
                ) != attention_result_value(1, row, elem)) {
                std::printf(
                    "attention QK output mismatch row=%u elem=%u\n",
                    row,
                    elem
                );
                errors++;
            }
        }
    }
    cc8_status_packet_t status = streams.status.read();
    if (status.status != CC8_STATUS_OK ||
        status.dispatched_mm_tasks != 2 ||
        status.output_waves != 1) {
        std::printf("attention QK status mismatch\n");
        errors++;
    }
    return errors;
}

static int run_attention_pv_route_case() {
    clear_data_ports();
    const unsigned int tile_len = 8;
    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int pos = 0; pos < CC8_ATTN_TILE; pos++) {
            set_flat_row_value(
                input_port0,
                row,
                pos,
                CC8_ATTN_TILE,
                attention_input_value(0, row, pos)
            );
            set_flat_row_value(
                input_port1,
                row,
                pos,
                CC8_ATTN_TILE,
                attention_input_value(1, row, pos)
            );
        }
    }
    for (unsigned int pos = 0; pos < tile_len; pos++) {
        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            set_flat_row_value(
                aux_port0,
                pos,
                elem,
                HEAD_DIM,
                attention_panel_value(0, pos, elem)
            );
            set_flat_row_value(
                aux_port1,
                pos,
                elem,
                HEAD_DIM,
                attention_panel_value(1, pos, elem)
            );
        }
    }

    cc8_test_streams_t streams;
    unsigned int output_waves =
        ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS);
    for (unsigned int wave = 0; wave < output_waves; wave++) {
        preload_attention_mm_results(
            streams,
            wave * MM_STREAM_8X64_OUTPUTS
        );
    }
    call_control_cache(
        streams,
        CC8_OP_ATTN_PV,
        GQA_GROUP_SIZE,
        0,
        tile_len
    );

    int errors = 0;
    for (unsigned int wave = 0; wave < output_waves; wave++) {
        cu8_task_t task0 = streams.core0_task.read();
        cu8_task_t task1 = streams.core1_task.read();
        bool expected_last = wave + 1 == output_waves;
        if (task0.mode != CU8_MODE_MM || task1.mode != CU8_MODE_MM ||
            task0.k_count != tile_len || task1.k_count != tile_len ||
            task0.elem_base != wave * MM_STREAM_8X64_OUTPUTS ||
            task1.elem_base != task0.elem_base ||
            task0.last_task != expected_last ||
            task1.last_task != expected_last) {
            std::printf("attention PV generic MM task mismatch wave=%u\n", wave);
            errors++;
        }
        for (unsigned int pos = 0; pos < tile_len; pos++) {
            mm_stream_8x64_activation_packet_t a0 =
                streams.core0_activation.read();
            mm_stream_8x64_activation_packet_t a1 =
                streams.core1_activation.read();
            for (unsigned int row = 0;
                 row < MM_STREAM_8X64_TOKENS;
                 row++) {
                fm_t expected0 =
                    row < GQA_GROUP_SIZE ?
                    attention_input_value(0, row, pos) :
                    fm_t(0);
                fm_t expected1 =
                    row < GQA_GROUP_SIZE ?
                    attention_input_value(1, row, pos) :
                    fm_t(0);
                if (a0.data[row] != expected0 ||
                    a1.data[row] != expected1) {
                    std::printf(
                        "attention PV activation mismatch pos=%u row=%u\n",
                        pos,
                        row
                    );
                    errors++;
                }
            }

            mm_stream_8x64_weight_packet_t weights0[4];
            mm_stream_8x64_weight_packet_t weights1[4];
            weights0[0] = streams.core0_weight0.read();
            weights0[1] = streams.core0_weight1.read();
            weights0[2] = streams.core0_weight2.read();
            weights0[3] = streams.core0_weight3.read();
            weights1[0] = streams.core1_weight0.read();
            weights1[1] = streams.core1_weight1.read();
            weights1[2] = streams.core1_weight2.read();
            weights1[3] = streams.core1_weight3.read();
            for (unsigned int group = 0; group < 4; group++) {
                for (unsigned int lane = 0;
                     lane < CU_VEC_LANES;
                     lane++) {
                    unsigned int elem =
                        wave * MM_STREAM_8X64_OUTPUTS +
                        group * CU_VEC_LANES +
                        lane;
                    wt_linear_t expected0 =
                        elem < HEAD_DIM ?
                        wt_linear_t(
                            attention_panel_value(0, pos, elem)
                        ) :
                        wt_linear_t(0);
                    wt_linear_t expected1 =
                        elem < HEAD_DIM ?
                        wt_linear_t(
                            attention_panel_value(1, pos, elem)
                        ) :
                        wt_linear_t(0);
                    if (weights0[group].data[lane] != expected0 ||
                        weights1[group].data[lane] != expected1) {
                        std::printf(
                            "attention PV weight mismatch pos=%u elem=%u\n",
                            pos,
                            elem
                        );
                        errors++;
                    }
                }
            }
        }
    }

    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            if (get_flat_output_value(
                    output_port0,
                    row,
                    elem,
                    HEAD_DIM
                ) != attention_result_value(0, row, elem) ||
                get_flat_output_value(
                    output_port1,
                    row,
                    elem,
                    HEAD_DIM
                ) != attention_result_value(1, row, elem)) {
                std::printf(
                    "attention PV output mismatch row=%u elem=%u\n",
                    row,
                    elem
                );
                errors++;
            }
        }
    }
    cc8_status_packet_t status = streams.status.read();
    if (status.status != CC8_STATUS_OK ||
        status.dispatched_mm_tasks != 2 * output_waves ||
        status.output_waves != output_waves) {
        std::printf("attention PV status mismatch\n");
        errors++;
    }
    return errors;
}

static int run_silu_mul_route_case() {
    clear_data_ports();
    unsigned int token_count = LINEAR_TOKEN_TILE_ACTIVE - 1;
    for (unsigned int token = 0; token < token_count; token++) {
        for (unsigned int elem = 0; elem < INTERMEDIATE_SIZE; elem++) {
            set_feature_value(
                input_port0,
                input_port1,
                token,
                elem,
                input_value(token, elem)
            );
            set_feature_value(
                aux_port0,
                aux_port1,
                token,
                elem,
                aux_value(token, elem)
            );
        }
    }

    cc8_test_streams_t streams;
    preload_vector_results(
        streams.core0_result,
        0,
        CC8_TOKENS_PER_DATA_PORT,
        INTERMEDIATE_SIZE
    );
    preload_vector_results(
        streams.core1_result,
        CC8_TOKENS_PER_DATA_PORT,
        CC8_TOKENS_PER_DATA_PORT,
        INTERMEDIATE_SIZE
    );
    call_control_cache(
        streams,
        CC8_OP_SILU_MUL,
        token_count,
        0,
        0
    );

    int errors = 0;
    if (streams.core0_task.empty() || streams.core1_task.empty()) {
        std::printf("SiLU-Mul compute task missing\n");
        return 1;
    }
    cu8_task_t task0 = streams.core0_task.read();
    cu8_task_t task1 = streams.core1_task.read();
    unsigned int packets_per_core =
        CC8_TOKENS_PER_DATA_PORT *
        ceildiv(INTERMEDIATE_SIZE, CU_VEC_LANES);
    if (task0.mode != CU8_MODE_SILU_MUL ||
        task1.mode != CU8_MODE_SILU_MUL ||
        task0.packet_count != packets_per_core ||
        task1.packet_count != packets_per_core ||
        task0.token_count != CC8_TOKENS_PER_DATA_PORT ||
        task1.token_count != CC8_TOKENS_PER_DATA_PORT ||
        !task0.last_task || !task1.last_task) {
        std::printf("SiLU-Mul task metadata mismatch\n");
        errors++;
    }

    errors += check_silu_mul_inputs(
        streams.core0_vector0,
        streams.core0_vector1,
        packets_per_core,
        token_count,
        0
    );
    errors += check_silu_mul_inputs(
        streams.core1_vector0,
        streams.core1_vector1,
        packets_per_core,
        token_count,
        1
    );

    for (unsigned int token = 0;
         token < LINEAR_TOKEN_TILE_ACTIVE;
         token++) {
        for (unsigned int elem = 0; elem < INTERMEDIATE_SIZE; elem++) {
            if (get_output_value(token, elem) != result_value(token, elem)) {
                std::printf(
                    "SiLU-Mul routed output mismatch token=%u elem=%u\n",
                    token,
                    elem
                );
                errors++;
            }
        }
    }
    if (!streams.core0_task.empty() || !streams.core1_task.empty() ||
        !streams.core0_vector0.empty() ||
        !streams.core0_vector1.empty() ||
        !streams.core1_vector0.empty() ||
        !streams.core1_vector1.empty()) {
        std::printf("SiLU-Mul stream count mismatch\n");
        errors++;
    }
    cc8_status_packet_t status = streams.status.read();
    if (status.status != CC8_STATUS_OK ||
        status.dispatched_vector_tasks != CC8_MM_CORE_COUNT ||
        status.completed_output_packets !=
            CC8_MM_CORE_COUNT * packets_per_core) {
        std::printf("SiLU-Mul status mismatch\n");
        errors++;
    }
    return errors;
}

static int check_operator_specs() {
    int errors = 0;
    const cc8_operator_t projection_ops[7] = {
        CC8_OP_Q_PROJECTION,
        CC8_OP_K_PROJECTION,
        CC8_OP_V_PROJECTION,
        CC8_OP_O_PROJECTION,
        CC8_OP_FFN_GATE,
        CC8_OP_FFN_UP,
        CC8_OP_FFN_DOWN
    };
    const mm_projection_kind_t projections[7] = {
        MM_PROJECTION_Q,
        MM_PROJECTION_K,
        MM_PROJECTION_V,
        MM_PROJECTION_ATTN_O,
        MM_PROJECTION_FFN_GATE,
        MM_PROJECTION_FFN_UP,
        MM_PROJECTION_FFN_DOWN
    };

    for (unsigned int i = 0; i < 7; i++) {
        cc8_operator_spec_t got;
        mm_projection_spec_t expected;
        get_mm_projection_spec(projections[i], expected);
        if (!cc8_get_operator_spec(projection_ops[i], got) ||
            !got.uses_mm || got.uses_vector ||
            got.projection != projections[i] ||
            got.in_dim != expected.in_dim ||
            got.out_dim != expected.out_dim) {
            std::printf("operator spec mismatch index=%u\n", i);
            errors++;
        }
    }

    cc8_operator_spec_t invalid;
    if (cc8_get_operator_spec(cc8_operator_t(255), invalid)) {
        std::printf("invalid operator unexpectedly accepted\n");
        errors++;
    }
    return errors;
}

int main() {
    init_weights();

    int errors = 0;
    errors += check_operator_specs();
    errors += run_gate_projection_case();
    errors += run_silu_mul_route_case();
    errors += run_attention_qk_route_case();
    errors += run_attention_softmax_route_case();
    errors += run_attention_pv_route_case();

    if (errors != 0) {
        std::printf("CONTROL CACHE 8X64 CSIM FAIL errors=%d\n", errors);
        return 1;
    }
    std::printf("CONTROL CACHE 8X64 CSIM PASS cases=6\n");
    return 0;
}
