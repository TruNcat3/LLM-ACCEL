#include "control_cache_8x64.hpp"

#include <cmath>
#include <cstdio>

static fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT];
static fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT];
static fm_word_t input_port0[CC8_DATA_PORT_WORDS];
static fm_word_t input_port1[CC8_DATA_PORT_WORDS];
static fm_word_t aux_port0[CC8_DATA_PORT_WORDS];
static fm_word_t aux_port1[CC8_DATA_PORT_WORDS];
static fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS];
static fm_word_t kv_cache_v[CC8_KV_CACHE_WORDS];

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
    for (unsigned int word = 0; word < CC8_KV_CACHE_WORDS; word++) {
        kv_cache_k[word] = 0;
        kv_cache_v[word] = 0;
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

static void set_kv_cache_value(
    fm_word_t words[CC8_KV_CACHE_WORDS],
    unsigned int layer,
    unsigned int position,
    unsigned int kv_head,
    unsigned int elem,
    fm_t value
) {
    unsigned int word_idx =
        (
            (
                layer * MAX_SEQ_LEN +
                position
            ) *
            NUM_KEY_VALUE_HEADS +
            kv_head
        ) *
        CC8_HEAD_WORDS +
        elem / FM_BLOCK_SIZE;
    set_fm_word_lane(words[word_idx], elem % FM_BLOCK_SIZE, value);
}

static void set_k_cache_value(
    fm_word_t words[CC8_KV_CACHE_WORDS],
    unsigned int layer,
    unsigned int position,
    unsigned int kv_head,
    unsigned int elem,
    fm_t value
) {
    set_kv_cache_value(
        words, layer, position, kv_head, elem, value
    );
    const unsigned int transposed_word =
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
        position / FM_BLOCK_SIZE;
    set_fm_word_lane(
        words[transposed_word], position % FM_BLOCK_SIZE, value
    );
}

static unsigned int get_kv_cache_word_index(
    unsigned int layer,
    unsigned int position,
    unsigned int kv_head,
    unsigned int word_idx
) {
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

static fm_t attention_history_k_value(
    unsigned int kv_head,
    unsigned int position,
    unsigned int elem
) {
    int raw = int((kv_head * 17 + position * 5 + elem * 3) % 63) - 31;
    return fm_t(raw * 0.03125);
}

static fm_t attention_history_v_value(
    unsigned int kv_head,
    unsigned int position,
    unsigned int elem
) {
    int raw = int((kv_head * 11 + position * 7 + elem * 5 + 9) % 63) - 31;
    return fm_t(raw * 0.03125);
}

static fm_t attention_current_k_value(
    unsigned int kv_head,
    unsigned int elem
) {
    return fm_t(2.0 + kv_head * 0.5 + elem * 0.015625);
}

static fm_t attention_current_v_value(
    unsigned int kv_head,
    unsigned int elem
) {
    return fm_t(-2.0 - kv_head * 0.5 - elem * 0.015625);
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
    unsigned int tile_len,
    unsigned int layer_id = 0
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
        layer_id,
        token_count,
        position,
        tile_len,
        shard0, shard1, shard2, shard3,
        shard4, shard5, shard6, shard7,
        shard8, shard9, shard10, shard11,
        shard12, shard13, shard14, shard15,
        kv_cache_k,
        kv_cache_v
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
    unsigned int out_lane,
    unsigned int output_wave = 0
) {
    mm_projection_spec_t spec;
    get_mm_projection_spec(projection, spec);
    unsigned int in_tile_count = ceildiv(spec.in_dim, MM_PE_IN);
    unsigned int out_tile =
        output_wave * CC8_OUTPUT_TILES_PER_WAVE +
        core * MM_STREAM_8X64_WEIGHT_GROUPS + group;
    if (out_tile * MM_PE_OUT >= spec.out_dim) {
        return wt_linear_t(0);
    }
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
                        if (errors < 8) {
                            std::printf(
                                "weight mismatch core=%u group=%u k=%u lane=%u got=%g expected=%g\n",
                                core,
                                group,
                                k,
                                lane,
                                double(packets[core][group].data[lane]),
                                double(expected)
                            );
                        }
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

static int check_mm_repeat_streams(
    cc8_test_streams_t& streams,
    unsigned int token_count,
    mm_projection_kind_t projection,
    unsigned int first_wave,
    unsigned int wave_count
) {
    int errors = 0;
    mm_projection_spec_t spec;
    get_mm_projection_spec(projection, spec);

    if (streams.core0_task.empty() || streams.core1_task.empty()) {
        std::printf("repeated MM task stream underflow\n");
        return 1;
    }
    cu8_task_t task0 = streams.core0_task.read();
    cu8_task_t task1 = streams.core1_task.read();
    unsigned int first_elem = first_wave * CC8_OUTPUTS_PER_WAVE;
    if (task0.mode != CU8_MODE_MM || task1.mode != CU8_MODE_MM ||
        task0.k_count != spec.in_dim || task1.k_count != spec.in_dim ||
        task0.elem_base != first_elem ||
        task1.elem_base != first_elem + MM_STREAM_8X64_OUTPUTS ||
        task0.block_id != first_wave || task1.block_id != first_wave ||
        task0.repeat_count != wave_count ||
        task1.repeat_count != wave_count ||
        task0.elem_stride != CC8_OUTPUTS_PER_WAVE ||
        task1.elem_stride != CC8_OUTPUTS_PER_WAVE ||
        task0.block_stride != 1 || task1.block_stride != 1 ||
        !task0.last_task || !task1.last_task) {
        std::printf("repeated MM task metadata mismatch\n");
        errors++;
    }

    for (unsigned int repeat = 0; repeat < wave_count; repeat++) {
        unsigned int output_wave = first_wave + repeat;
        for (unsigned int k = 0; k < spec.in_dim; k++) {
            if (streams.core0_activation.empty() ||
                streams.core1_activation.empty()) {
                std::printf(
                    "repeated activation underflow wave=%u k=%u\n",
                    output_wave,
                    k
                );
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
                    if (errors < 8) {
                        std::printf(
                            "repeated activation mismatch wave=%u k=%u token=%u\n",
                            output_wave,
                            k,
                            token
                        );
                    }
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
            for (unsigned int core = 0;
                 core < CC8_MM_CORE_COUNT;
                 core++) {
                for (unsigned int group = 0; group < 4; group++) {
                    for (unsigned int lane = 0;
                         lane < CU_VEC_LANES;
                         lane++) {
                        wt_linear_t expected = expected_weight(
                            projection,
                            core,
                            group,
                            k,
                            lane,
                            output_wave
                        );
                        if (packets[core][group].data[lane] != expected) {
                            if (errors < 8) {
                                std::printf(
                                    "repeated weight mismatch wave=%u core=%u group=%u k=%u lane=%u\n",
                                    output_wave,
                                    core,
                                    group,
                                    k,
                                    lane
                                );
                            }
                            errors++;
                        }
                    }
                }
            }
        }
    }

    if (!streams.core0_task.empty() || !streams.core1_task.empty() ||
        !streams.core0_activation.empty() ||
        !streams.core1_activation.empty()) {
        std::printf("repeated MM streams contain extra packets\n");
        errors++;
    }
    return errors;
}

static int run_gate_projection_repeat_case() {
#if !CC8_ENABLE_MM_WAVE_REPEAT
    return 0;
#else
    const unsigned int total_waves =
        ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE);
    if (total_waves < 3) {
        return 0;
    }

    clear_data_ports();
    const unsigned int token_count = LINEAR_TOKEN_TILE_ACTIVE - 2;
    const unsigned int first_wave = 1;
    const unsigned int wave_count = 3;
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
    for (unsigned int repeat = 0; repeat < wave_count; repeat++) {
        unsigned int wave = first_wave + repeat;
        preload_mm_results(
            streams,
            wave * CC8_OUTPUTS_PER_WAVE,
            wave * CC8_OUTPUTS_PER_WAVE + MM_STREAM_8X64_OUTPUTS,
            repeat + 1 == wave_count
        );
    }
    call_control_cache(
        streams,
        CC8_OP_FFN_GATE,
        token_count,
        first_wave,
        wave_count
    );

    int errors = check_mm_repeat_streams(
        streams,
        token_count,
        MM_PROJECTION_FFN_GATE,
        first_wave,
        wave_count
    );
    unsigned int first_elem = first_wave * CC8_OUTPUTS_PER_WAVE;
    unsigned int end_elem = (first_wave + wave_count) * CC8_OUTPUTS_PER_WAVE;
    if (end_elem > INTERMEDIATE_SIZE) {
        end_elem = INTERMEDIATE_SIZE;
    }
    for (unsigned int token = 0;
         token < LINEAR_TOKEN_TILE_ACTIVE;
         token++) {
        for (unsigned int elem = 0; elem < INTERMEDIATE_SIZE; elem++) {
            bool in_profile_range = elem >= first_elem && elem < end_elem;
            fm_t expected =
                token < token_count && in_profile_range ?
                result_value(token, elem) : fm_t(0);
            if (get_output_value(token, elem) != expected) {
                if (errors < 8) {
                    std::printf(
                        "repeated gate output mismatch token=%u elem=%u\n",
                        token,
                        elem
                    );
                }
                errors++;
            }
        }
    }

    if (streams.status.empty()) {
        std::printf("repeated gate status missing\n");
        errors++;
    } else {
        cc8_status_packet_t status = streams.status.read();
        if (status.op != CC8_OP_FFN_GATE ||
            status.status != CC8_STATUS_OK ||
            status.output_waves != total_waves ||
            status.dispatched_mm_tasks !=
                wave_count * CC8_MM_CORE_COUNT ||
            status.dispatched_vector_tasks != 0 ||
            status.completed_output_packets !=
                wave_count * CC8_MM_CORE_COUNT *
                    MM_STREAM_8X64_PACKETS_PER_BLOCK) {
            std::printf("repeated gate status mismatch\n");
            errors++;
        }
    }
    return errors;
#endif
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
        1
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
                token < token_count && elem < CC8_OUTPUTS_PER_WAVE ?
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
            status.output_waves !=
                ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE) ||
            status.dispatched_mm_tasks != 2 ||
            status.dispatched_vector_tasks != 0 ||
            status.completed_output_packets != 64) {
            std::printf("gate status mismatch\n");
            errors++;
        }
    }
    return errors;
}

static int run_down_projection_chunk_pipeline_case() {
    clear_data_ports();
    unsigned int token_count = LINEAR_TOKEN_TILE_ACTIVE - 2;
    for (unsigned int token = 0; token < token_count; token++) {
        for (unsigned int elem = 0; elem < INTERMEDIATE_SIZE; elem++) {
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
    // Profile exactly one output wave while keeping the complete FFN-down K
    // range.  The weight-chunk model profile gives 64 input tiles, so this
    // exercises steady-state producer/consumer backpressure at every FIFO
    // depth in the packed-block pipeline sweep.
    call_control_cache(
        streams,
        CC8_OP_FFN_DOWN,
        token_count,
        0,
        1
    );

    int errors = check_mm_streams(
        streams,
        token_count,
        MM_PROJECTION_FFN_DOWN
    );
    for (unsigned int token = 0;
         token < LINEAR_TOKEN_TILE_ACTIVE;
         token++) {
        for (unsigned int elem = 0; elem < HIDDEN_SIZE; elem++) {
            fm_t expected =
                token < token_count && elem < CC8_OUTPUTS_PER_WAVE ?
                result_value(token, elem) :
                fm_t(0);
            if (get_output_value(token, elem) != expected) {
                std::printf(
                    "down output mismatch token=%u elem=%u\n",
                    token,
                    elem
                );
                errors++;
            }
        }
    }

    if (streams.status.empty()) {
        std::printf("down status missing\n");
        errors++;
    } else {
        cc8_status_packet_t status = streams.status.read();
        if (status.op != CC8_OP_FFN_DOWN ||
            status.status != CC8_STATUS_OK ||
            status.output_waves != ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE) ||
            status.dispatched_mm_tasks != 2 ||
            status.dispatched_vector_tasks != 0 ||
            status.completed_output_packets != 64) {
            std::printf("down status mismatch\n");
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

static void preload_attention_constant_results(
    cc8_test_streams_t& streams,
    fm_t value,
    unsigned int elem_base,
    unsigned int block_id,
    bool last_stream
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
            packet0.block_id = block_id;
            packet1.block_id = block_id;
            packet0.last_block =
                row + 1 == MM_STREAM_8X64_TOKENS &&
                group + 1 == MM_STREAM_8X64_WEIGHT_GROUPS;
            packet1.last_block = packet0.last_block;
            packet0.last_stream = last_stream && packet0.last_block;
            packet1.last_stream = last_stream && packet1.last_block;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                packet0.data[lane] = value;
                packet1.data[lane] = value;
            }
            streams.core0_result.write(packet0);
            streams.core1_result.write(packet1);
        }
    }
}

static void preload_prefill_pv_results(
    cc8_test_streams_t& streams,
    unsigned int query_begin,
    unsigned int token_count,
    unsigned int tile_begin,
    unsigned int tile_len,
    unsigned int elem_base,
    unsigned int block_id,
    bool last_stream
) {
    for (unsigned int token = 0;
         token < MM_STREAM_8X64_TOKENS;
         token++) {
        unsigned int valid_count = 0;
        if (token < token_count && query_begin + token >= tile_begin) {
            valid_count = query_begin + token - tile_begin + 1;
            if (valid_count > tile_len) {
                valid_count = tile_len;
            }
        }
        for (unsigned int group = 0;
             group < MM_STREAM_8X64_WEIGHT_GROUPS;
             group++) {
            cu_vec16_packet_t packet0;
            cu_vec16_packet_t packet1;
            packet0.valid_mask = 0xffff;
            packet1.valid_mask = 0xffff;
            packet0.token_lane = token;
            packet1.token_lane = token;
            packet0.elem_base = elem_base + group * CU_VEC_LANES;
            packet1.elem_base = packet0.elem_base;
            packet0.block_id = block_id;
            packet1.block_id = block_id;
            packet0.last_block =
                token + 1 == MM_STREAM_8X64_TOKENS &&
                group + 1 == MM_STREAM_8X64_WEIGHT_GROUPS;
            packet1.last_block = packet0.last_block;
            packet0.last_stream = last_stream && packet0.last_block;
            packet1.last_stream = packet0.last_stream;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                packet0.data[lane] = fm_t(valid_count);
                packet1.data[lane] = fm_t(valid_count);
            }
            streams.core0_result.write(packet0);
            streams.core1_result.write(packet1);
        }
    }
}

static int check_attention_qk_inputs(
    hls::stream<mm_stream_8x64_activation_packet_t>& activation,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight3,
    unsigned int core,
    cc8_operator_t op,
    unsigned int position,
    unsigned int tile_begin,
    unsigned int tile_len
) {
    int errors = 0;
    for (unsigned int k = 0; k < HEAD_DIM; k++) {
        if (activation.empty() ||
            weight0.empty() ||
            weight1.empty() ||
            weight2.empty() ||
            weight3.empty()) {
            std::printf("attention QK input stream underflow core=%u k=%u\n", core, k);
            return 1;
        }
        mm_stream_8x64_activation_packet_t act = activation.read();
        mm_stream_8x64_weight_packet_t weights[4];
        weights[0] = weight0.read();
        weights[1] = weight1.read();
        weights[2] = weight2.read();
        weights[3] = weight3.read();

        for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
            fm_t expected = row < GQA_GROUP_SIZE ?
                fm_t(core == 0 ? 0.125 : 0.25) : fm_t(0);
            if (act.data[row] != expected) {
                if (errors < 8) {
                    std::printf(
                        "attention QK activation mismatch core=%u k=%u row=%u\n",
                        core,
                        k,
                        row
                    );
                }
                errors++;
            }
        }

        for (unsigned int group = 0; group < 4; group++) {
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                unsigned int local_pos = group * CU_VEC_LANES + lane;
                unsigned int absolute_pos = tile_begin + local_pos;
                wt_linear_t expected = wt_linear_t(0);
                if (local_pos < tile_len) {
                    bool use_current =
                        op == CC8_OP_DECODE_SMOKE &&
                        absolute_pos == position;
                    fm_t source = use_current ?
                        attention_current_k_value(core, k) :
                        attention_history_k_value(core, absolute_pos, k);
                    expected = wt_linear_t(source);
                }
                if (weights[group].data[lane] != expected) {
                    if (errors < 8) {
                        std::printf(
                            "attention QK weight mismatch core=%u k=%u pos=%u\n",
                            core,
                            k,
                            absolute_pos
                        );
                    }
                    errors++;
                }
            }
        }
    }
    return errors;
}

static int check_attention_pv_inputs(
    hls::stream<mm_stream_8x64_activation_packet_t>& activation,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight3,
    unsigned int core,
    cc8_operator_t op,
    unsigned int position,
    unsigned int tile_begin,
    unsigned int tile_len,
    unsigned int output_wave
) {
    int errors = 0;
    for (unsigned int local_pos = 0; local_pos < tile_len; local_pos++) {
        if (activation.empty() ||
            weight0.empty() ||
            weight1.empty() ||
            weight2.empty() ||
            weight3.empty()) {
            std::printf(
                "attention PV input stream underflow core=%u pos=%u wave=%u\n",
                core,
                local_pos,
                output_wave
            );
            return errors + 1;
        }
        mm_stream_8x64_activation_packet_t act = activation.read();
        mm_stream_8x64_weight_packet_t weights[4];
        weights[0] = weight0.read();
        weights[1] = weight1.read();
        weights[2] = weight2.read();
        weights[3] = weight3.read();

        for (unsigned int row = 0; row < MM_STREAM_8X64_TOKENS; row++) {
            fm_t expected = row < GQA_GROUP_SIZE ? fm_t(1) : fm_t(0);
            if (act.data[row] != expected) {
                if (errors < 8) {
                    std::printf(
                        "attention PV activation mismatch core=%u pos=%u row=%u wave=%u\n",
                        core,
                        local_pos,
                        row,
                        output_wave
                    );
                }
                errors++;
            }
        }

        unsigned int absolute_pos = tile_begin + local_pos;
        for (unsigned int group = 0; group < 4; group++) {
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                unsigned int elem =
                    output_wave * MM_STREAM_8X64_OUTPUTS +
                    group * CU_VEC_LANES + lane;
                wt_linear_t expected = wt_linear_t(0);
                if (elem < HEAD_DIM) {
                    bool use_current =
                        op == CC8_OP_DECODE_SMOKE &&
                        absolute_pos == position;
                    fm_t source = use_current ?
                        attention_current_v_value(core, elem) :
                        attention_history_v_value(core, absolute_pos, elem);
                    expected = wt_linear_t(source);
                }
                if (weights[group].data[lane] != expected) {
                    if (errors < 8) {
                        std::printf(
                            "attention PV weight mismatch core=%u pos=%u elem=%u wave=%u\n",
                            core,
                            absolute_pos,
                            elem,
                            output_wave
                        );
                    }
                    errors++;
                }
            }
        }
    }
    return errors;
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

static fm_t get_prefill_block_output_value(
    const fm_word_t output[CC8_FEATURE_WORDS_PER_PORT],
    unsigned int token,
    unsigned int head,
    unsigned int elem
) {
    constexpr unsigned int kWordsPerToken =
        GQA_GROUP_SIZE * CC8_HEAD_WORDS;
    const unsigned int word =
        token * kWordsPerToken +
        head * CC8_HEAD_WORDS + elem / FM_BLOCK_SIZE;
    return unpack_fm_word_lane(output[word], elem % FM_BLOCK_SIZE);
}

static int run_attention_prefill_block_case(
    unsigned int query_begin,
    unsigned int token_count
) {
    clear_data_ports();
    constexpr unsigned int kWordsPerHead =
        ceildiv(HEAD_DIM, FM_BLOCK_SIZE);
    for (unsigned int token = 0; token < token_count; token++) {
        for (unsigned int head = 0; head < GQA_GROUP_SIZE; head++) {
            for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
                const unsigned int word =
                    (token * GQA_GROUP_SIZE + head) * kWordsPerHead +
                    elem / FM_BLOCK_SIZE;
                set_fm_word_lane(
                    input_port0[word], elem % FM_BLOCK_SIZE,
                    fm_t(0.0625 * (head + 1))
                );
                set_fm_word_lane(
                    input_port1[word], elem % FM_BLOCK_SIZE,
                    fm_t(0.03125 * (head + 1))
                );
            }
        }
    }

    const unsigned int context_len = query_begin + token_count;
    for (unsigned int pos = 0; pos < context_len; pos++) {
        for (unsigned int kv_head = 0;
             kv_head < NUM_KEY_VALUE_HEADS;
             kv_head++) {
            for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
                set_k_cache_value(
                    kv_cache_k,
                    0,
                    pos,
                    kv_head,
                    elem,
                    attention_history_k_value(kv_head, pos, elem)
                );
                set_kv_cache_value(
                    kv_cache_v,
                    0,
                    pos,
                    kv_head,
                    elem,
                    attention_history_v_value(kv_head, pos, elem)
                );
            }
        }
    }

    cc8_test_streams_t streams;
    for (unsigned int tile_begin = 0;
         tile_begin < context_len;
         tile_begin += CC8_ATTN_TILE) {
        unsigned int tile_len = CC8_ATTN_TILE;
        if (tile_begin + tile_len > context_len) {
            tile_len = context_len - tile_begin;
        }
        for (unsigned int head = 0; head < GQA_GROUP_SIZE; head++) {
            preload_attention_constant_results(
                streams, fm_t(0), 0, head, false
            );
            for (unsigned int wave = 0;
                 wave < CC8_ATTN_PV_WAVES;
                 wave++) {
                const bool last =
                    tile_begin + CC8_ATTN_TILE >= context_len &&
                    head + 1 == GQA_GROUP_SIZE &&
                    wave + 1 == CC8_ATTN_PV_WAVES;
                preload_prefill_pv_results(
                    streams,
                    query_begin,
                    token_count,
                    tile_begin,
                    tile_len,
                    wave * MM_STREAM_8X64_OUTPUTS,
                    wave,
                    last
                );
            }
        }
    }

    call_control_cache(
        streams,
        CC8_OP_ATTN_PREFILL_BLOCK,
        token_count,
        query_begin,
        0
    );

    int errors = 0;
    for (unsigned int tile_begin = 0;
         tile_begin < context_len;
         tile_begin += CC8_ATTN_TILE) {
        const unsigned int tile_len =
            tile_begin + CC8_ATTN_TILE <= context_len ?
            CC8_ATTN_TILE : context_len - tile_begin;
        for (unsigned int head = 0; head < GQA_GROUP_SIZE; head++) {
            const cu8_task_t qk_task0 = streams.core0_task.read();
            const cu8_task_t qk_task1 = streams.core1_task.read();
            if (qk_task0.mode != CU8_MODE_MM_SCALE ||
                qk_task1.mode != CU8_MODE_MM_SCALE ||
                qk_task0.k_count != HEAD_DIM ||
                qk_task1.k_count != HEAD_DIM ||
                qk_task0.token_count != token_count ||
                qk_task1.token_count != token_count ||
                qk_task0.elem_count != CC8_ATTN_TILE ||
                qk_task1.elem_count != CC8_ATTN_TILE) {
                std::printf(
                    "prefill QK task mismatch tile=%u head=%u\n",
                    tile_begin,
                    head
                );
                errors++;
            }
            for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
                const mm_stream_8x64_activation_packet_t activation0 =
                    streams.core0_activation.read();
                const mm_stream_8x64_activation_packet_t activation1 =
                    streams.core1_activation.read();
                for (unsigned int token = 0;
                     token < MM_STREAM_8X64_TOKENS;
                     token++) {
                    const fm_t expected0 = token < token_count ?
                        fm_t(0.0625 * (head + 1)) : fm_t(0);
                    const fm_t expected1 = token < token_count ?
                        fm_t(0.03125 * (head + 1)) : fm_t(0);
                    if (activation0.data[token] != expected0 ||
                        activation1.data[token] != expected1) {
                        if (errors < 8) {
                            std::printf(
                                "prefill QK activation mismatch tile=%u head=%u elem=%u token=%u\n",
                                tile_begin, head, elem, token
                            );
                        }
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
                        const unsigned int local_pos =
                            group * CU_VEC_LANES + lane;
                        const unsigned int absolute_pos =
                            tile_begin + local_pos;
                        const wt_linear_t expected0 =
                            local_pos < tile_len ?
                            wt_linear_t(attention_history_k_value(
                                0, absolute_pos, elem
                            )) : wt_linear_t(0);
                        const wt_linear_t expected1 =
                            local_pos < tile_len ?
                            wt_linear_t(attention_history_k_value(
                                1, absolute_pos, elem
                            )) : wt_linear_t(0);
                        if (weights0[group].data[lane] != expected0 ||
                            weights1[group].data[lane] != expected1) {
                            if (errors < 8) {
                                std::printf(
                                    "prefill K^T packet mismatch tile=%u head=%u elem=%u pos=%u got=(%f,%f) expected=(%f,%f)\n",
                                    tile_begin,
                                    head,
                                    elem,
                                    absolute_pos,
                                    double(weights0[group].data[lane]),
                                    double(weights1[group].data[lane]),
                                    double(expected0),
                                    double(expected1)
                                );
                            }
                            errors++;
                        }
                    }
                }
            }

            for (unsigned int wave = 0;
                 wave < CC8_ATTN_PV_WAVES;
                 wave++) {
                const cu8_task_t pv_task0 = streams.core0_task.read();
                const cu8_task_t pv_task1 = streams.core1_task.read();
                if (pv_task0.mode != CU8_MODE_MM_SCALE ||
                    pv_task1.mode != CU8_MODE_MM_SCALE ||
                    pv_task0.k_count != tile_len ||
                    pv_task1.k_count != tile_len ||
                    pv_task0.output_scale != fm_t(1.0 / 64.0) ||
                    pv_task1.output_scale != fm_t(1.0 / 64.0) ||
                    pv_task0.elem_base !=
                        wave * MM_STREAM_8X64_OUTPUTS ||
                    pv_task1.elem_base != pv_task0.elem_base) {
                    std::printf(
                        "prefill PV task mismatch tile=%u head=%u wave=%u\n",
                        tile_begin,
                        head,
                        wave
                    );
                    errors++;
                }
                for (unsigned int local_pos = 0;
                     local_pos < tile_len;
                     local_pos++) {
                    (void)streams.core0_activation.read();
                    (void)streams.core1_activation.read();
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
                            const unsigned int output_elem =
                                wave * MM_STREAM_8X64_OUTPUTS +
                                group * CU_VEC_LANES + lane;
                            const unsigned int absolute_pos =
                                tile_begin + local_pos;
                            const wt_linear_t expected0 =
                                output_elem < HEAD_DIM ?
                                wt_linear_t(attention_history_v_value(
                                    0, absolute_pos, output_elem
                                )) : wt_linear_t(0);
                            const wt_linear_t expected1 =
                                output_elem < HEAD_DIM ?
                                wt_linear_t(attention_history_v_value(
                                    1, absolute_pos, output_elem
                                )) : wt_linear_t(0);
                            if (weights0[group].data[lane] != expected0 ||
                                weights1[group].data[lane] != expected1) {
                                if (errors < 8) {
                                    std::printf(
                                        "prefill PV weight mismatch tile=%u head=%u wave=%u pos=%u elem=%u\n",
                                        tile_begin,
                                        head,
                                        wave,
                                        absolute_pos,
                                        output_elem
                                    );
                                }
                                errors++;
                            }
                        }
                    }
                }
            }
        }
    }

    for (unsigned int token = 0; token < token_count; token++) {
        const fm_accum_t expected_sum =
            fm_accum_t(query_begin + token + 1);
        const fm_accum_t safe_sum =
            expected_sum < fm_accum_t(0.000244140625) ?
            fm_accum_t(0.000244140625) : expected_sum;
        const fm_t expected = fm_t(
            expected_sum * fm_accum_t(fm_t(fm_accum_t(1) / safe_sum))
        );
        for (unsigned int head = 0; head < GQA_GROUP_SIZE; head++) {
            for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
                const fm_t got0 = get_prefill_block_output_value(
                    output_port0, token, head, elem
                );
                const fm_t got1 = get_prefill_block_output_value(
                    output_port1, token, head, elem
                );
                if (got0 != expected || got1 != expected) {
                    if (errors < 8) {
                        std::printf(
                            "prefill block output mismatch q=%u token=%u head=%u elem=%u got=(%f,%f) expected=%f\n",
                            query_begin,
                            token,
                            head,
                            elem,
                            double(got0),
                            double(got1),
                            double(expected)
                        );
                    }
                    errors++;
                }
            }
        }
    }
    if (streams.status.empty()) {
        std::printf("prefill block missing status\n");
        return errors + 1;
    }
    const cc8_status_packet_t got_status = streams.status.read();
    const unsigned int expected_tasks =
        CC8_MM_CORE_COUNT * ceildiv(context_len, CC8_ATTN_TILE) *
        GQA_GROUP_SIZE * (1 + CC8_ATTN_PV_WAVES);
    if (got_status.status != CC8_STATUS_OK ||
        got_status.op != CC8_OP_ATTN_PREFILL_BLOCK ||
        got_status.token_count != token_count ||
        got_status.dispatched_mm_tasks != expected_tasks ||
        got_status.completed_output_packets !=
            expected_tasks * MM_STREAM_8X64_PACKETS_PER_BLOCK) {
        std::printf(
            "prefill block status mismatch q=%u tasks=%u expected=%u\n",
            query_begin,
            got_status.dispatched_mm_tasks,
            expected_tasks
        );
        errors++;
    }
    if (!streams.core0_result.empty() || !streams.core1_result.empty()) {
        std::printf("prefill block result stream was not fully consumed\n");
        errors++;
    }
    if (!streams.core0_task.empty() || !streams.core1_task.empty() ||
        !streams.core0_activation.empty() ||
        !streams.core1_activation.empty() ||
        !streams.core0_weight0.empty() ||
        !streams.core0_weight1.empty() ||
        !streams.core0_weight2.empty() ||
        !streams.core0_weight3.empty() ||
        !streams.core1_weight0.empty() ||
        !streams.core1_weight1.empty() ||
        !streams.core1_weight2.empty() ||
        !streams.core1_weight3.empty()) {
        std::printf("prefill block emitted stream was not fully consumed\n");
        errors++;
    }
    return errors;
}

static int run_attention_flash_route_case(
    cc8_operator_t op,
    unsigned int position
) {
    clear_data_ports();
    const unsigned int context_len = position + 1;
    const bool decode_smoke = op == CC8_OP_DECODE_SMOKE;
    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            set_flat_row_value(
                input_port0,
                row,
                elem,
                HEAD_DIM,
                fm_t(0.125)
            );
            set_flat_row_value(
                input_port1,
                row,
                elem,
                HEAD_DIM,
                fm_t(0.25)
            );
        }
    }
    if (decode_smoke) {
        for (unsigned int kv_head = 0;
             kv_head < NUM_KEY_VALUE_HEADS;
             kv_head++) {
            for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
                set_flat_row_value(
                    aux_port0,
                    kv_head,
                    elem,
                    HEAD_DIM,
                    attention_current_k_value(kv_head, elem)
                );
                set_flat_row_value(
                    aux_port1,
                    kv_head,
                    elem,
                    HEAD_DIM,
                    attention_current_v_value(kv_head, elem)
                );
            }
        }
    }
    for (unsigned int pos = 0; pos < context_len; pos++) {
        for (unsigned int kv_head = 0;
             kv_head < NUM_KEY_VALUE_HEADS;
             kv_head++) {
            for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
                set_kv_cache_value(
                    kv_cache_k,
                    0,
                    pos,
                    kv_head,
                    elem,
                    attention_history_k_value(kv_head, pos, elem)
                );
                set_kv_cache_value(
                    kv_cache_v,
                    0,
                    pos,
                    kv_head,
                    elem,
                    attention_history_v_value(kv_head, pos, elem)
                );
            }
        }
    }

    cc8_test_streams_t streams;
    const unsigned int output_waves =
        ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS);
    for (unsigned int tile_begin = 0;
         tile_begin < context_len;
         tile_begin += CC8_ATTN_TILE) {
        unsigned int current_tile_len = CC8_ATTN_TILE;
        if (tile_begin + current_tile_len > context_len) {
            current_tile_len = context_len - tile_begin;
        }
        preload_attention_constant_results(
            streams,
            fm_t(0),
            0,
            0,
            false
        );
        for (unsigned int output_wave = 0;
             output_wave < output_waves;
             output_wave++) {
            bool last_stream =
                tile_begin + current_tile_len == context_len &&
                output_wave + 1 == output_waves;
            preload_attention_constant_results(
                streams,
                fm_t(current_tile_len),
                output_wave * MM_STREAM_8X64_OUTPUTS,
                output_wave,
                last_stream
            );
        }
    }
    call_control_cache(
        streams,
        op,
        GQA_GROUP_SIZE,
        position,
        0
    );

    int errors = 0;
    if (decode_smoke) {
        for (unsigned int kv_head = 0;
             kv_head < NUM_KEY_VALUE_HEADS;
             kv_head++) {
            for (unsigned int word_idx = 0;
                 word_idx < CC8_HEAD_WORDS;
                 word_idx++) {
                unsigned int source_idx =
                    kv_head * CC8_HEAD_WORDS + word_idx;
                unsigned int cache_idx = get_kv_cache_word_index(
                    0,
                    position,
                    kv_head,
                    word_idx
                );
                if (kv_cache_k[cache_idx] != aux_port0[source_idx] ||
                    kv_cache_v[cache_idx] != aux_port1[source_idx]) {
                    std::printf(
                        "decode smoke KV cache mismatch head=%u word=%u\n",
                        kv_head,
                        word_idx
                    );
                    errors++;
                }
            }
        }
    }

    for (unsigned int tile_begin = 0;
         tile_begin < context_len;
         tile_begin += CC8_ATTN_TILE) {
        unsigned int current_tile_len = CC8_ATTN_TILE;
        if (tile_begin + current_tile_len > context_len) {
            current_tile_len = context_len - tile_begin;
        }

        if (streams.core0_task.empty() || streams.core1_task.empty()) {
            std::printf("attention flash QK task underflow tile=%u\n", tile_begin);
            return errors + 1;
        }
        cu8_task_t qk0 = streams.core0_task.read();
        cu8_task_t qk1 = streams.core1_task.read();
        if (qk0.mode != CU8_MODE_MM_SCALE ||
            qk1.mode != CU8_MODE_MM_SCALE ||
            qk0.k_count != HEAD_DIM ||
            qk1.k_count != HEAD_DIM ||
            qk0.token_count != GQA_GROUP_SIZE ||
            qk1.token_count != GQA_GROUP_SIZE ||
            qk0.elem_count != CC8_ATTN_TILE ||
            qk1.elem_count != CC8_ATTN_TILE ||
            qk0.packet_count != MM_STREAM_8X64_PACKETS_PER_BLOCK ||
            qk1.packet_count != MM_STREAM_8X64_PACKETS_PER_BLOCK ||
            qk0.elem_base != 0 ||
            qk1.elem_base != 0 ||
            qk0.block_id != 0 ||
            qk1.block_id != 0 ||
            qk0.output_scale != fm_t(ATTENTION_SCALE) ||
            qk1.output_scale != fm_t(ATTENTION_SCALE) ||
            qk0.last_task ||
            qk1.last_task) {
            std::printf("attention flash QK task mismatch tile=%u\n", tile_begin);
            errors++;
        }
        errors += check_attention_qk_inputs(
            streams.core0_activation,
            streams.core0_weight0,
            streams.core0_weight1,
            streams.core0_weight2,
            streams.core0_weight3,
            0,
            op,
            position,
            tile_begin,
            current_tile_len
        );
        errors += check_attention_qk_inputs(
            streams.core1_activation,
            streams.core1_weight0,
            streams.core1_weight1,
            streams.core1_weight2,
            streams.core1_weight3,
            1,
            op,
            position,
            tile_begin,
            current_tile_len
        );

        for (unsigned int output_wave = 0;
             output_wave < output_waves;
             output_wave++) {
            if (streams.core0_task.empty() || streams.core1_task.empty()) {
                std::printf(
                    "attention flash PV task underflow tile=%u wave=%u\n",
                    tile_begin,
                    output_wave
                );
                return errors + 1;
            }
            cu8_task_t pv0 = streams.core0_task.read();
            cu8_task_t pv1 = streams.core1_task.read();
            bool last_task =
                tile_begin + current_tile_len == context_len &&
                output_wave + 1 == output_waves;
            if (pv0.mode != CU8_MODE_MM ||
                pv1.mode != CU8_MODE_MM ||
                pv0.k_count != current_tile_len ||
                pv1.k_count != current_tile_len ||
                pv0.token_count != GQA_GROUP_SIZE ||
                pv1.token_count != GQA_GROUP_SIZE ||
                pv0.elem_count != MM_STREAM_8X64_OUTPUTS ||
                pv1.elem_count != MM_STREAM_8X64_OUTPUTS ||
                pv0.packet_count != MM_STREAM_8X64_PACKETS_PER_BLOCK ||
                pv1.packet_count != MM_STREAM_8X64_PACKETS_PER_BLOCK ||
                pv0.elem_base != output_wave * MM_STREAM_8X64_OUTPUTS ||
                pv1.elem_base != output_wave * MM_STREAM_8X64_OUTPUTS ||
                pv0.block_id != output_wave ||
                pv1.block_id != output_wave ||
                pv0.last_task != last_task ||
                pv1.last_task != last_task) {
                std::printf(
                    "attention flash PV task mismatch tile=%u wave=%u\n",
                    tile_begin,
                    output_wave
                );
                errors++;
            }
            errors += check_attention_pv_inputs(
                streams.core0_activation,
                streams.core0_weight0,
                streams.core0_weight1,
                streams.core0_weight2,
                streams.core0_weight3,
                0,
                op,
                position,
                tile_begin,
                current_tile_len,
                output_wave
            );
            errors += check_attention_pv_inputs(
                streams.core1_activation,
                streams.core1_weight0,
                streams.core1_weight1,
                streams.core1_weight2,
                streams.core1_weight3,
                1,
                op,
                position,
                tile_begin,
                current_tile_len,
                output_wave
            );
        }
    }

    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            fm_t got0 = get_flat_output_value(
                output_port0,
                row,
                elem,
                HEAD_DIM
            );
            fm_t got1 = get_flat_output_value(
                output_port1,
                row,
                elem,
                HEAD_DIM
            );
            if (got0 != fm_t(1) || got1 != fm_t(1)) {
                std::printf(
                    "attention flash output mismatch row=%u elem=%u\n",
                    row,
                    elem
                );
                errors++;
            }
        }
    }

    if (streams.status.empty()) {
        std::printf("attention flash status underflow\n");
        return errors + 1;
    }
    cc8_status_packet_t status = streams.status.read();
    if (status.op != op ||
        status.status != CC8_STATUS_OK ||
        status.output_waves != output_waves ||
        status.dispatched_mm_tasks !=
            2 * ceildiv(context_len, CC8_ATTN_TILE) * (1 + output_waves) ||
        status.completed_output_packets !=
            status.dispatched_mm_tasks * MM_STREAM_8X64_PACKETS_PER_BLOCK ||
        !status.last_task) {
        std::printf("attention flash status mismatch\n");
        errors++;
    }

    if (!streams.core0_task.empty() ||
        !streams.core1_task.empty() ||
        !streams.core0_activation.empty() ||
        !streams.core1_activation.empty() ||
        !streams.core0_weight0.empty() ||
        !streams.core0_weight1.empty() ||
        !streams.core0_weight2.empty() ||
        !streams.core0_weight3.empty() ||
        !streams.core1_weight0.empty() ||
        !streams.core1_weight1.empty() ||
        !streams.core1_weight2.empty() ||
        !streams.core1_weight3.empty() ||
        !streams.core0_vector0.empty() ||
        !streams.core0_vector1.empty() ||
        !streams.core1_vector0.empty() ||
        !streams.core1_vector1.empty() ||
        !streams.core0_result.empty() ||
        !streams.core1_result.empty() ||
        !streams.status.empty()) {
        std::printf("attention flash stream count mismatch\n");
        errors++;
    }
    return errors;
}

static int run_silu_mul_route_case(unsigned int token_count) {
    clear_data_ports();
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
    unsigned int core0_token_count =
        token_count < CC8_TOKENS_PER_DATA_PORT ?
        token_count : CC8_TOKENS_PER_DATA_PORT;
    unsigned int core1_token_count =
        token_count > CC8_TOKENS_PER_DATA_PORT ?
        token_count - CC8_TOKENS_PER_DATA_PORT : 0;
    preload_vector_results(
        streams.core0_result,
        0,
        core0_token_count,
        INTERMEDIATE_SIZE
    );
    preload_vector_results(
        streams.core1_result,
        CC8_TOKENS_PER_DATA_PORT,
        core1_token_count,
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
    unsigned int blocks_per_token =
        ceildiv(INTERMEDIATE_SIZE, CU_VEC_LANES);
    unsigned int core0_packet_count =
        core0_token_count * blocks_per_token;
    unsigned int core1_packet_count =
        core1_token_count * blocks_per_token;
    if (task0.mode != CU8_MODE_SILU_MUL ||
        task0.packet_count != core0_packet_count ||
        task0.token_count != core0_token_count ||
        !task0.last_task) {
        std::printf("SiLU-Mul task metadata mismatch\n");
        errors++;
    }
    if (core1_token_count != 0) {
        if (task1.mode != CU8_MODE_SILU_MUL ||
            task1.packet_count != core1_packet_count ||
            task1.token_count != core1_token_count ||
            !task1.last_task) {
            std::printf("SiLU-Mul core1 task metadata mismatch\n");
            errors++;
        }
    } else if (task1.mode != CU8_MODE_STOP || !task1.last_task) {
        std::printf("SiLU-Mul inactive core did not receive STOP\n");
        errors++;
    }

    errors += check_silu_mul_inputs(
        streams.core0_vector0,
        streams.core0_vector1,
        core0_packet_count,
        token_count,
        0
    );
    errors += check_silu_mul_inputs(
        streams.core1_vector0,
        streams.core1_vector1,
        core1_packet_count,
        token_count,
        1
    );

    for (unsigned int token = 0;
         token < LINEAR_TOKEN_TILE_ACTIVE;
         token++) {
        for (unsigned int elem = 0; elem < INTERMEDIATE_SIZE; elem++) {
            fm_t expected = token < token_count ?
                result_value(token, elem) : fm_t(0);
            if (get_output_value(token, elem) != expected) {
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
    unsigned int active_core_count =
        (core0_token_count != 0 ? 1 : 0) +
        (core1_token_count != 0 ? 1 : 0);
    if (status.status != CC8_STATUS_OK ||
        status.dispatched_vector_tasks != active_core_count ||
        status.completed_output_packets !=
            core0_packet_count + core1_packet_count) {
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
    cc8_operator_spec_t decode_smoke;
    if (!cc8_get_operator_spec(CC8_OP_DECODE_SMOKE, decode_smoke) ||
        decode_smoke.in_dim != HEAD_DIM ||
        decode_smoke.out_dim != HEAD_DIM ||
        decode_smoke.compute_mode != CU8_MODE_MM_SCALE) {
        std::printf("decode smoke operator spec mismatch\n");
        errors++;
    }
    return errors;
}

static int run_rejected_attention_call(
    unsigned int token_count,
    unsigned int layer_id,
    cc8_status_code_t expected_status
) {
    clear_data_ports();
    cc8_test_streams_t streams;
    call_control_cache(
        streams,
        CC8_OP_DECODE_SMOKE,
        token_count,
        0,
        1,
        layer_id
    );

    if (streams.core0_task.empty() ||
        streams.core1_task.empty() ||
        streams.status.empty()) {
        std::printf("rejected attention call did not terminate streams\n");
        return 1;
    }

    cu8_task_t task0 = streams.core0_task.read();
    cu8_task_t task1 = streams.core1_task.read();
    cc8_status_packet_t status = streams.status.read();
    if (task0.mode != CU8_MODE_STOP ||
        task1.mode != CU8_MODE_STOP ||
        !task0.last_task ||
        !task1.last_task ||
        status.status != expected_status ||
        !status.last_task) {
        std::printf("rejected attention call status mismatch\n");
        return 1;
    }
    return 0;
}

static int run_attention_argument_validation_cases() {
    int errors = 0;
    errors += run_rejected_attention_call(
        GQA_GROUP_SIZE,
        NUM_LAYERS,
        CC8_STATUS_BAD_LAYER
    );
    errors += run_rejected_attention_call(
        GQA_GROUP_SIZE - 1,
        0,
        CC8_STATUS_BAD_TOKEN_COUNT
    );
    return errors;
}

static int run_resident_decoder_layer_route_case() {
    clear_data_ports();
    for (unsigned int elem = 0; elem < HIDDEN_SIZE; elem++) {
        set_feature_value(
            input_port0,
            input_port1,
            0,
            elem,
            input_value(0, elem)
        );
        set_fm_word_lane(
            aux_port0[elem / FM_BLOCK_SIZE],
            elem % FM_BLOCK_SIZE,
            fm_t(1)
        );
        set_fm_word_lane(
            aux_port1[elem / FM_BLOCK_SIZE],
            elem % FM_BLOCK_SIZE,
            fm_t(1)
        );
    }
    for (unsigned int i = 0; i < CC8_ROPE_HALF_ELEMS; i++) {
        set_fm_word_lane(
            aux_port0[
                CC8_ROPE_COS_WORD_OFFSET + i / FM_BLOCK_SIZE
            ],
            i % FM_BLOCK_SIZE,
            fm_t(1)
        );
    }

    cc8_test_streams_t streams;
    preload_vector_results(streams.core0_result, 0, 1, HIDDEN_SIZE);

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
        unsigned int waves =
            ceildiv(projection.out_dim, CC8_OUTPUTS_PER_WAVE);
        for (unsigned int wave = 0; wave < waves; wave++) {
            preload_mm_results(
                streams,
                wave * CC8_OUTPUTS_PER_WAVE,
                wave * CC8_OUTPUTS_PER_WAVE + MM_STREAM_8X64_OUTPUTS,
                false
            );
        }
    }

    preload_attention_constant_results(streams, fm_t(0), 0, 0, false);
    for (unsigned int wave = 0;
         wave < ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS);
         wave++) {
        preload_attention_constant_results(
            streams,
            fm_t(0),
            wave * MM_STREAM_8X64_OUTPUTS,
            wave,
            false
        );
    }

    const mm_projection_kind_t tail_mm[4] = {
        MM_PROJECTION_ATTN_O,
        MM_PROJECTION_FFN_GATE,
        MM_PROJECTION_FFN_UP,
        MM_PROJECTION_FFN_DOWN
    };
    for (unsigned int projection_idx = 0;
         projection_idx < 4;
         projection_idx++) {
        if (projection_idx == 1) {
            // O is followed by residual and the second RMSNorm.
            preload_vector_results(
                streams.core0_result,
                0,
                1,
                HIDDEN_SIZE
            );
            preload_vector_results(
                streams.core0_result,
                0,
                1,
                HIDDEN_SIZE
            );
        }
        mm_projection_spec_t projection;
        get_mm_projection_spec(tail_mm[projection_idx], projection);
        unsigned int waves =
            ceildiv(projection.out_dim, CC8_OUTPUTS_PER_WAVE);
        for (unsigned int wave = 0; wave < waves; wave++) {
            preload_mm_results(
                streams,
                wave * CC8_OUTPUTS_PER_WAVE,
                wave * CC8_OUTPUTS_PER_WAVE + MM_STREAM_8X64_OUTPUTS,
                false
            );
        }
        if (projection_idx == 2) {
            // Gate and Up are followed by SiLU-Mul.
            preload_vector_results(
                streams.core0_result,
                0,
                1,
                INTERMEDIATE_SIZE
            );
        }
    }
    // Final residual is the only task that terminates core0; core1 receives
    // an explicit STOP because token 0 is mapped to core0.
    preload_vector_results(streams.core0_result, 0, 1, HIDDEN_SIZE);

    call_control_cache(
        streams,
        CC8_OP_DECODER_LAYER,
        1,
        0,
        0
    );

    int errors = 0;
    unsigned int core0_tasks = 0;
    bool saw_core0_last = false;
    while (!streams.core0_task.empty()) {
        cu8_task_t task = streams.core0_task.read();
        core0_tasks++;
        if (task.last_task) {
            if (saw_core0_last || !streams.core0_task.empty()) {
                errors++;
            }
            saw_core0_last = true;
        }
    }
    unsigned int core1_tasks = 0;
    bool saw_core1_stop = false;
    while (!streams.core1_task.empty()) {
        cu8_task_t task = streams.core1_task.read();
        core1_tasks++;
        if (task.mode == CU8_MODE_STOP) {
            saw_core1_stop = task.last_task && streams.core1_task.empty();
        } else if (task.last_task) {
            errors++;
        }
    }

    const unsigned int expected_mm_wave_slots =
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(KV_CHANNELS, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(KV_CHANNELS, CC8_OUTPUTS_PER_WAVE) +
        (1 + ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS)) +
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE);
    const unsigned int expected_vector_packets =
        4 * ceildiv(HIDDEN_SIZE, CU_VEC_LANES) +
        ceildiv(INTERMEDIATE_SIZE, CU_VEC_LANES);
    const unsigned int expected_packets =
        expected_mm_wave_slots * CC8_MM_CORE_COUNT *
            MM_STREAM_8X64_PACKETS_PER_BLOCK +
        expected_vector_packets;
    const unsigned int expected_core0_tasks = expected_mm_wave_slots + 5;
    const unsigned int expected_core1_tasks = expected_mm_wave_slots + 1;
    if (!saw_core0_last || !saw_core1_stop ||
        core0_tasks != expected_core0_tasks ||
        core1_tasks != expected_core1_tasks) {
        std::printf(
            "resident layer task sequence mismatch c0=%u/%u c1=%u/%u last0=%u stop1=%u\n",
            core0_tasks,
            expected_core0_tasks,
            core1_tasks,
            expected_core1_tasks,
            unsigned(saw_core0_last),
            unsigned(saw_core1_stop)
        );
        errors++;
    }
    if (streams.status.empty()) {
        std::printf("resident layer status missing\n");
        errors++;
    } else {
        cc8_status_packet_t status = streams.status.read();
        if (status.op != CC8_OP_DECODER_LAYER ||
            status.status != CC8_STATUS_OK ||
            status.token_count != 1 ||
            status.output_waves != expected_mm_wave_slots ||
            status.dispatched_mm_tasks !=
                expected_mm_wave_slots * CC8_MM_CORE_COUNT ||
            status.dispatched_vector_tasks != 5 ||
            status.completed_output_packets != expected_packets ||
            !status.last_task) {
            std::printf(
                "resident layer status mismatch waves=%u mm=%u vec=%u packets=%u\n",
                status.output_waves,
                status.dispatched_mm_tasks,
                status.dispatched_vector_tasks,
                status.completed_output_packets
            );
            errors++;
        }
    }
    for (unsigned int elem = 0; elem < HIDDEN_SIZE; elem++) {
        if (get_output_value(0, elem) != result_value(0, elem)) {
            if (errors < 8) {
                std::printf("resident layer output mismatch elem=%u\n", elem);
            }
            errors++;
        }
    }
    return errors;
}

static int check_resident_sublayer_route(
    cc8_test_streams_t& streams,
    cc8_operator_t expected_op,
    unsigned int expected_mm_wave_slots,
    unsigned int expected_vector_tasks,
    unsigned int expected_vector_packets,
    const char* label
) {
    int errors = 0;
    unsigned int core0_tasks = 0;
    bool saw_core0_last = false;
    while (!streams.core0_task.empty()) {
        cu8_task_t task = streams.core0_task.read();
        core0_tasks++;
        if (task.last_task) {
            if (saw_core0_last || !streams.core0_task.empty()) {
                errors++;
            }
            saw_core0_last = true;
        }
    }
    unsigned int core1_tasks = 0;
    bool saw_core1_stop = false;
    while (!streams.core1_task.empty()) {
        cu8_task_t task = streams.core1_task.read();
        core1_tasks++;
        if (task.mode == CU8_MODE_STOP) {
            saw_core1_stop = task.last_task && streams.core1_task.empty();
        } else if (task.last_task) {
            errors++;
        }
    }
    const unsigned int expected_core0_tasks =
        expected_mm_wave_slots + expected_vector_tasks;
    const unsigned int expected_core1_tasks =
        expected_mm_wave_slots + 1;
    if (!saw_core0_last || !saw_core1_stop ||
        core0_tasks != expected_core0_tasks ||
        core1_tasks != expected_core1_tasks) {
        std::printf(
            "%s task sequence mismatch c0=%u/%u c1=%u/%u last0=%u stop1=%u\n",
            label,
            core0_tasks,
            expected_core0_tasks,
            core1_tasks,
            expected_core1_tasks,
            unsigned(saw_core0_last),
            unsigned(saw_core1_stop)
        );
        errors++;
    }

    const unsigned int expected_packets =
        expected_mm_wave_slots * CC8_MM_CORE_COUNT *
            MM_STREAM_8X64_PACKETS_PER_BLOCK +
        expected_vector_packets;
    if (streams.status.empty()) {
        std::printf("%s status missing\n", label);
        errors++;
    } else {
        cc8_status_packet_t status = streams.status.read();
        if (status.op != expected_op ||
            status.status != CC8_STATUS_OK ||
            status.token_count != 1 ||
            status.output_waves != expected_mm_wave_slots ||
            status.dispatched_mm_tasks !=
                expected_mm_wave_slots * CC8_MM_CORE_COUNT ||
            status.dispatched_vector_tasks != expected_vector_tasks ||
            status.completed_output_packets != expected_packets ||
            !status.last_task) {
            std::printf(
                "%s status mismatch op=%u waves=%u mm=%u vec=%u packets=%u/%u\n",
                label,
                unsigned(status.op),
                status.output_waves,
                status.dispatched_mm_tasks,
                status.dispatched_vector_tasks,
                status.completed_output_packets,
                expected_packets
            );
            errors++;
        }
    }
    for (unsigned int elem = 0; elem < HIDDEN_SIZE; elem++) {
        if (get_output_value(0, elem) != result_value(0, elem)) {
            if (errors < 8) {
                std::printf("%s output mismatch elem=%u\n", label, elem);
            }
            errors++;
        }
    }
    return errors;
}

static int run_attention_sublayer_route_case() {
    clear_data_ports();
    for (unsigned int elem = 0; elem < HIDDEN_SIZE; elem++) {
        set_feature_value(
            input_port0,
            input_port1,
            0,
            elem,
            input_value(0, elem)
        );
        set_fm_word_lane(
            aux_port0[elem / FM_BLOCK_SIZE],
            elem % FM_BLOCK_SIZE,
            fm_t(1)
        );
    }
    for (unsigned int i = 0; i < CC8_ROPE_HALF_ELEMS; i++) {
        set_fm_word_lane(
            aux_port0[CC8_ROPE_COS_WORD_OFFSET + i / FM_BLOCK_SIZE],
            i % FM_BLOCK_SIZE,
            fm_t(1)
        );
    }

    cc8_test_streams_t streams;
    preload_vector_results(streams.core0_result, 0, 1, HIDDEN_SIZE);
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
            preload_mm_results(
                streams,
                wave * CC8_OUTPUTS_PER_WAVE,
                wave * CC8_OUTPUTS_PER_WAVE + MM_STREAM_8X64_OUTPUTS,
                false
            );
        }
    }
    preload_attention_constant_results(streams, fm_t(0), 0, 0, false);
    for (unsigned int wave = 0;
         wave < ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS);
         wave++) {
        preload_attention_constant_results(
            streams,
            fm_t(0),
            wave * MM_STREAM_8X64_OUTPUTS,
            wave,
            false
        );
    }
    mm_projection_spec_t o_projection;
    get_mm_projection_spec(MM_PROJECTION_ATTN_O, o_projection);
    for (unsigned int wave = 0;
         wave < ceildiv(o_projection.out_dim, CC8_OUTPUTS_PER_WAVE);
         wave++) {
        preload_mm_results(
            streams,
            wave * CC8_OUTPUTS_PER_WAVE,
            wave * CC8_OUTPUTS_PER_WAVE + MM_STREAM_8X64_OUTPUTS,
            false
        );
    }
    preload_vector_results(streams.core0_result, 0, 1, HIDDEN_SIZE);

    call_control_cache(streams, CC8_OP_ATTENTION_SUBLAYER, 1, 0, 0);
    const unsigned int expected_waves =
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE) +
        2 * ceildiv(KV_CHANNELS, CC8_OUTPUTS_PER_WAVE) +
        1 + ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS) +
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE);
    return check_resident_sublayer_route(
        streams,
        CC8_OP_ATTENTION_SUBLAYER,
        expected_waves,
        2,
        2 * ceildiv(HIDDEN_SIZE, CU_VEC_LANES),
        "attention sublayer"
    );
}

static int run_ffn_sublayer_route_case() {
    clear_data_ports();
    for (unsigned int elem = 0; elem < HIDDEN_SIZE; elem++) {
        set_feature_value(
            input_port0,
            input_port1,
            0,
            elem,
            input_value(0, elem)
        );
        set_fm_word_lane(
            aux_port1[elem / FM_BLOCK_SIZE],
            elem % FM_BLOCK_SIZE,
            fm_t(1)
        );
    }

    cc8_test_streams_t streams;
    preload_vector_results(streams.core0_result, 0, 1, HIDDEN_SIZE);
    const mm_projection_kind_t ffn[3] = {
        MM_PROJECTION_FFN_GATE,
        MM_PROJECTION_FFN_UP,
        MM_PROJECTION_FFN_DOWN
    };
    for (unsigned int projection_idx = 0;
         projection_idx < 3;
         projection_idx++) {
        mm_projection_spec_t projection;
        get_mm_projection_spec(ffn[projection_idx], projection);
        const unsigned int waves =
            ceildiv(projection.out_dim, CC8_OUTPUTS_PER_WAVE);
        for (unsigned int wave = 0; wave < waves; wave++) {
            preload_mm_results(
                streams,
                wave * CC8_OUTPUTS_PER_WAVE,
                wave * CC8_OUTPUTS_PER_WAVE + MM_STREAM_8X64_OUTPUTS,
                false
            );
        }
        if (projection_idx == 1) {
            preload_vector_results(
                streams.core0_result,
                0,
                1,
                INTERMEDIATE_SIZE
            );
        }
    }
    preload_vector_results(streams.core0_result, 0, 1, HIDDEN_SIZE);

    call_control_cache(streams, CC8_OP_FFN_SUBLAYER, 1, 0, 0);
    const unsigned int expected_waves =
        2 * ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE);
    return check_resident_sublayer_route(
        streams,
        CC8_OP_FFN_SUBLAYER,
        expected_waves,
        3,
        2 * ceildiv(HIDDEN_SIZE, CU_VEC_LANES) +
            ceildiv(INTERMEDIATE_SIZE, CU_VEC_LANES),
        "ffn sublayer"
    );
}

static int run_final_norm_route_case() {
    clear_data_ports();
    for (unsigned int elem = 0; elem < HIDDEN_SIZE; elem++) {
        set_feature_value(
            input_port0,
            input_port1,
            0,
            elem,
            input_value(0, elem)
        );
        set_fm_word_lane(
            aux_port0[elem / FM_BLOCK_SIZE],
            elem % FM_BLOCK_SIZE,
            fm_t(1)
        );
    }
    cc8_test_streams_t streams;
    preload_vector_results(streams.core0_result, 0, 1, HIDDEN_SIZE);
    call_control_cache(streams, CC8_OP_FINAL_NORM, 1, 0, 0);
    return check_resident_sublayer_route(
        streams,
        CC8_OP_FINAL_NORM,
        0,
        1,
        ceildiv(HIDDEN_SIZE, CU_VEC_LANES),
        "final norm"
    );
}

int main() {
#ifdef CC8_PREFILL_LAYER_TEST_ONLY
    init_weights();
    int errors = 0;
    errors += run_gate_projection_case();
    errors += run_down_projection_chunk_pipeline_case();
    errors += run_silu_mul_route_case(LINEAR_TOKEN_TILE_ACTIVE - 1);
    errors += run_attention_prefill_block_case(0, MM_STREAM_8X64_TOKENS);
    if (errors != 0) {
        std::printf("CONTROL CACHE PREFILL LAYER CSIM FAIL errors=%d\n", errors);
        return 1;
    }
    std::printf("CONTROL CACHE PREFILL LAYER CSIM PASS cases=4\n");
    return 0;
#elif defined(CC8_PREFILL_BLOCK_TEST_ONLY)
    int errors = 0;
    errors += run_attention_prefill_block_case(0, MM_STREAM_8X64_TOKENS);
    if (MAX_SEQ_LEN > 1024) {
        // This is the first context length that exceeds the legacy 32-tile,
        // 256-task launch bound and mirrors D@1024 in the hw_emu sweep.
        errors += run_attention_prefill_block_case(1024, 1);
    }
    errors += run_attention_prefill_block_case(
        MAX_SEQ_LEN - MM_STREAM_8X64_TOKENS,
        MM_STREAM_8X64_TOKENS
    );
    if (MAX_SEQ_LEN >= CC8_ATTN_TILE + MM_STREAM_8X64_TOKENS) {
        errors += run_attention_prefill_block_case(
            CC8_ATTN_TILE,
            MM_STREAM_8X64_TOKENS
        );
    }
    if (errors != 0) {
        std::printf("CONTROL CACHE PREFILL BLOCK CSIM FAIL errors=%d\n", errors);
        return 1;
    }
    std::printf(
        "CONTROL CACHE PREFILL BLOCK CSIM PASS cases=%u\n",
        MAX_SEQ_LEN > 1024 ? 4u : 3u
    );
    return 0;
#else
    init_weights();

    int errors = 0;
    errors += check_operator_specs();
    errors += run_gate_projection_case();
    errors += run_gate_projection_repeat_case();
    errors += run_down_projection_chunk_pipeline_case();
    errors += run_silu_mul_route_case(LINEAR_TOKEN_TILE_ACTIVE - 1);
    errors += run_silu_mul_route_case(1);
    errors += run_attention_qk_route_case();
    errors += run_attention_softmax_route_case();
    errors += run_attention_pv_route_case();
    errors += run_attention_prefill_block_case(0, MM_STREAM_8X64_TOKENS);
    errors += run_attention_prefill_block_case(
        MAX_SEQ_LEN - MM_STREAM_8X64_TOKENS,
        MM_STREAM_8X64_TOKENS
    );
    if (MAX_SEQ_LEN >= CC8_ATTN_TILE + MM_STREAM_8X64_TOKENS) {
        errors += run_attention_prefill_block_case(
            CC8_ATTN_TILE,
            MM_STREAM_8X64_TOKENS
        );
    }
    errors += run_attention_flash_route_case(CC8_OP_ATTN_FLASH, 1);
    errors += run_attention_flash_route_case(CC8_OP_DECODE_SMOKE, 0);
    errors += run_attention_flash_route_case(
        CC8_OP_DECODE_SMOKE,
        MAX_SEQ_LEN - 1
    );
    errors += run_attention_argument_validation_cases();
    errors += run_resident_decoder_layer_route_case();
    errors += run_attention_sublayer_route_case();
    errors += run_ffn_sublayer_route_case();
    errors += run_final_norm_route_case();

    if (errors != 0) {
        std::printf("CONTROL CACHE 8X64 CSIM FAIL errors=%d\n", errors);
        return 1;
    }
    std::printf("CONTROL CACHE 8X64 CSIM PASS cases=21\n");
    return 0;
#endif
}
