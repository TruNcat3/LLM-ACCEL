#include "closed_loop_8x64_cosim.hpp"

#include <cstdio>

static fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT];
static fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT];
static fm_word_t status_output[1];
static fm_word_t input_port0[CC8_DATA_PORT_WORDS];
static fm_word_t input_port1[CC8_DATA_PORT_WORDS];
static fm_word_t aux_port0[CC8_DATA_PORT_WORDS];
static fm_word_t aux_port1[CC8_DATA_PORT_WORDS];
static wt_block_t weight_shards[NUM_WEIGHT_SHARDS][WEIGHT_SHARD_BLOCKS];
static fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS];
static fm_word_t kv_cache_v[CC8_KV_CACHE_WORDS];

static unsigned int test_kv_cache_word_index(
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

static fm_t get_flat_output_value(
    const fm_word_t words[CC8_FEATURE_WORDS_PER_PORT],
    unsigned int row,
    unsigned int elem
) {
    const unsigned int words_per_row = ceildiv(HEAD_DIM, FM_BLOCK_SIZE);
    const unsigned int word_idx =
        row * words_per_row + elem / FM_BLOCK_SIZE;
    return unpack_fm_word_lane(words[word_idx], elem % FM_BLOCK_SIZE);
}

static void clear_test_data() {
    for (unsigned int word = 0; word < CC8_FEATURE_WORDS_PER_PORT; word++) {
        output_port0[word] = 0;
        output_port1[word] = 0;
    }
    status_output[0] = 0;
    for (unsigned int word = 0; word < CC8_DATA_PORT_WORDS; word++) {
        input_port0[word] = 0;
        input_port1[word] = 0;
        aux_port0[word] = 0;
        aux_port1[word] = 0;
    }
    for (unsigned int shard = 0; shard < NUM_WEIGHT_SHARDS; shard++) {
        for (unsigned int word = 0; word < WEIGHT_SHARD_BLOCKS; word++) {
            weight_shards[shard][word] = 0;
        }
    }
    for (unsigned int word = 0; word < CC8_KV_CACHE_WORDS; word++) {
        kv_cache_k[word] = 0;
        kv_cache_v[word] = 0;
    }
}

static void set_current_v() {
    for (unsigned int kv_head = 0;
         kv_head < NUM_KEY_VALUE_HEADS;
         kv_head++) {
        const fm_t value = kv_head == 0 ? fm_t(1) : fm_t(2);
        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            const unsigned int word_idx =
                kv_head * CC8_HEAD_WORDS + elem / FM_BLOCK_SIZE;
            set_fm_word_lane(
                aux_port1[word_idx],
                elem % FM_BLOCK_SIZE,
                value
            );
        }
    }
}

static int check_outputs() {
    int errors = 0;
    for (unsigned int row = 0; row < GQA_GROUP_SIZE; row++) {
        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            const fm_t got0 = get_flat_output_value(output_port0, row, elem);
            const fm_t got1 = get_flat_output_value(output_port1, row, elem);
            if (got0 != fm_t(0.5) || got1 != fm_t(1)) {
                if (errors < 8) {
                    std::printf(
                        "CLOSED LOOP output mismatch row=%u elem=%u got0=%f got1=%f\n",
                        row,
                        elem,
                        double(got0),
                        double(got1)
                    );
                }
                errors++;
            }
        }
    }
    return errors;
}

static int check_kv_cache(unsigned int layer, unsigned int position) {
    int errors = 0;
    for (unsigned int kv_head = 0;
         kv_head < NUM_KEY_VALUE_HEADS;
         kv_head++) {
        for (unsigned int word_idx = 0;
             word_idx < CC8_HEAD_WORDS;
             word_idx++) {
            const unsigned int source_idx =
                kv_head * CC8_HEAD_WORDS + word_idx;
            const unsigned int cache_idx = test_kv_cache_word_index(
                layer,
                position,
                kv_head,
                word_idx
            );
            if (kv_cache_k[cache_idx] != aux_port0[source_idx] ||
                kv_cache_v[cache_idx] != aux_port1[source_idx]) {
                if (errors < 8) {
                    std::printf(
                        "CLOSED LOOP KV mismatch head=%u word=%u\n",
                        kv_head,
                        word_idx
                    );
                }
                errors++;
            }
        }
    }
    return errors;
}

static int check_status() {
    const fm_word_t word = status_output[0];
    const unsigned int op = word.range(31, 0).to_uint();
    const unsigned int status = word.range(63, 32).to_uint();
    const unsigned int token_count = word.range(95, 64).to_uint();
    const unsigned int output_waves = word.range(127, 96).to_uint();
    const unsigned int dispatched_mm_tasks =
        word.range(159, 128).to_uint();
    const unsigned int dispatched_vector_tasks =
        word.range(191, 160).to_uint();
    const unsigned int completed_output_packets =
        word.range(223, 192).to_uint();
    const bool last_task = word[224];

    if (op != unsigned(CC8_OP_DECODE_SMOKE) ||
        status != unsigned(CC8_STATUS_OK) ||
        token_count != GQA_GROUP_SIZE ||
        output_waves != ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS) ||
        dispatched_mm_tasks != 4 ||
        dispatched_vector_tasks != 0 ||
        completed_output_packets !=
            4 * MM_STREAM_8X64_PACKETS_PER_BLOCK ||
        !last_task) {
        std::printf(
            "CLOSED LOOP status mismatch op=%u status=%u tokens=%u waves=%u mm=%u vec=%u packets=%u last=%u\n",
            op,
            status,
            token_count,
            output_waves,
            dispatched_mm_tasks,
            dispatched_vector_tasks,
            completed_output_packets,
            unsigned(last_task)
        );
        return 1;
    }
    return 0;
}

static void call_closed_loop_raw(
    cc8_operator_t op,
    unsigned int token_count,
    unsigned int position,
    unsigned int tile_len
) {
#ifdef CC8_CLOSED_LOOP_NK_COSIM
    cc8_closed_loop_nk_cosim(
#else
    cc8_closed_loop_inner_cosim(
#endif
        output_port0,
        output_port1,
        status_output,
        input_port0,
        input_port1,
        aux_port0,
        aux_port1,
        unsigned(op),
        0,
        token_count,
        position,
        tile_len,
        weight_shards[0],
        weight_shards[1],
        weight_shards[2],
        weight_shards[3],
        weight_shards[4],
        weight_shards[5],
        weight_shards[6],
        weight_shards[7],
        weight_shards[8],
        weight_shards[9],
        weight_shards[10],
        weight_shards[11],
        weight_shards[12],
        weight_shards[13],
        weight_shards[14],
        weight_shards[15],
        kv_cache_k,
        kv_cache_v
    );
}

static void call_closed_loop(
    cc8_operator_t op,
    unsigned int token_count,
    unsigned int position
) {
    call_closed_loop_raw(op, token_count, position, 0);
}

#ifdef CC8_CLOSED_LOOP_MM_REPEAT_COSIM
static void initialize_mm_repeat_outputs() {
    fm_word_t sentinel = ~fm_word_t(0);
    for (unsigned int word = 0;
         word < CC8_FEATURE_WORDS_PER_PORT;
         word++) {
        output_port0[word] = sentinel;
        output_port1[word] = sentinel;
    }
}

static int check_mm_repeat_outputs() {
    int errors = 0;
    for (unsigned int word = 0;
         word < CC8_FEATURE_WORDS_PER_PORT;
         word++) {
        if (output_port0[word] != fm_word_t(0)) {
            if (errors < 8) {
                std::printf(
                    "CLOSED LOOP MM repeat output0 mismatch word=%u\n",
                    word
                );
            }
            errors++;
        }
        if (output_port1[word] != fm_word_t(0)) {
            if (errors < 8) {
                std::printf(
                    "CLOSED LOOP MM repeat output1 mismatch word=%u\n",
                    word
                );
            }
            errors++;
        }
    }
    return errors;
}

static int check_mm_repeat_status() {
    const fm_word_t word = status_output[0];
    const unsigned int op = word.range(31, 0).to_uint();
    const unsigned int status = word.range(63, 32).to_uint();
    const unsigned int token_count = word.range(95, 64).to_uint();
    const unsigned int output_waves = word.range(127, 96).to_uint();
    const unsigned int dispatched_mm_tasks =
        word.range(159, 128).to_uint();
    const unsigned int dispatched_vector_tasks =
        word.range(191, 160).to_uint();
    const unsigned int completed_output_packets =
        word.range(223, 192).to_uint();
    const bool last_task = word[224];
    const unsigned int expected_waves =
        ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE);
    const unsigned int expected_packets =
        expected_waves *
        CC8_MM_CORE_COUNT *
        MM_STREAM_8X64_PACKETS_PER_BLOCK;
    if (op != unsigned(CC8_OP_FFN_GATE) ||
        status != unsigned(CC8_STATUS_OK) ||
        token_count != LINEAR_TOKEN_TILE_ACTIVE ||
        output_waves != expected_waves ||
        dispatched_mm_tasks != expected_waves * CC8_MM_CORE_COUNT ||
        dispatched_vector_tasks != 0 ||
        completed_output_packets != expected_packets ||
        !last_task) {
        std::printf(
            "CLOSED LOOP MM repeat status mismatch op=%u status=%u tokens=%u waves=%u mm=%u vec=%u packets=%u last=%u\n",
            op,
            status,
            token_count,
            output_waves,
            dispatched_mm_tasks,
            dispatched_vector_tasks,
            completed_output_packets,
            unsigned(last_task)
        );
        return 1;
    }
    return 0;
}
#endif

#ifdef CC8_CLOSED_LOOP_VECTOR_COSIM
static fm_word_t vector_output_sentinel() {
    return ~fm_word_t(0);
}

static void initialize_vector_outputs() {
    fm_word_t sentinel = vector_output_sentinel();
    for (unsigned int word = 0; word < CC8_FEATURE_WORDS_PER_PORT; word++) {
        output_port0[word] = sentinel;
        output_port1[word] = sentinel;
    }
}

static fm_t vector_lhs_value(unsigned int token, unsigned int elem) {
    return fm_t((token + 1) * 16 + (elem & 15)) / fm_t(16);
}

static fm_t vector_rhs_value(unsigned int token, unsigned int elem) {
    return fm_t((token + 2) * 8 + (elem & 7)) / fm_t(8);
}

static unsigned int vector_elem_count(cc8_operator_t op) {
    return op == CC8_OP_SILU_MUL ? INTERMEDIATE_SIZE : HIDDEN_SIZE;
}

static void initialize_vector_inputs(cc8_operator_t op) {
    unsigned int elem_count = vector_elem_count(op);
    for (unsigned int token = 0;
         token < LINEAR_TOKEN_TILE_ACTIVE;
         token++) {
        unsigned int local_token = token % CC8_TOKENS_PER_DATA_PORT;
        for (unsigned int elem = 0; elem < elem_count; elem++) {
            unsigned int port_word =
                local_token * CC8_FEATURE_WORDS_PER_TOKEN +
                elem / FM_BLOCK_SIZE;
            unsigned int lane = elem % FM_BLOCK_SIZE;
            fm_t lhs = op == CC8_OP_RESIDUAL_ADD ?
                vector_lhs_value(token, elem) : fm_t(0);
            fm_t rhs = vector_rhs_value(token, elem);
            if (token < CC8_TOKENS_PER_DATA_PORT) {
                set_fm_word_lane(input_port0[port_word], lane, lhs);
                set_fm_word_lane(aux_port0[port_word], lane, rhs);
            } else {
                set_fm_word_lane(input_port1[port_word], lane, lhs);
                set_fm_word_lane(aux_port1[port_word], lane, rhs);
            }
        }
    }
}

static int check_vector_outputs(
    cc8_operator_t op,
    unsigned int token_count
) {
    int errors = 0;
    unsigned int words_per_token =
        ceildiv(vector_elem_count(op), FM_BLOCK_SIZE);
    unsigned int elem_count = vector_elem_count(op);
    fm_word_t sentinel = vector_output_sentinel();
    for (unsigned int token = 0;
         token < LINEAR_TOKEN_TILE_ACTIVE;
         token++) {
        unsigned int local_token = token % CC8_TOKENS_PER_DATA_PORT;
        for (unsigned int word_idx = 0;
             word_idx < CC8_FEATURE_WORDS_PER_TOKEN;
             word_idx++) {
            unsigned int port_word =
                local_token * CC8_FEATURE_WORDS_PER_TOKEN + word_idx;
            fm_word_t got = token < CC8_TOKENS_PER_DATA_PORT ?
                output_port0[port_word] : output_port1[port_word];
            bool word_should_change =
                token < token_count && word_idx < words_per_token;
            if (!word_should_change && got != sentinel) {
                if (errors < 8) {
                    std::printf(
                        "CLOSED LOOP vector sentinel mismatch op=%u token=%u word=%u\n",
                        unsigned(op),
                        token,
                        word_idx
                    );
                }
                errors++;
            }
            if (word_should_change) {
                for (unsigned int lane = 0; lane < FM_BLOCK_SIZE; lane++) {
                    unsigned int elem = word_idx * FM_BLOCK_SIZE + lane;
                    if (elem < elem_count) {
                        fm_t expected = fm_t(0);
                        if (op == CC8_OP_RESIDUAL_ADD) {
                            expected = fm_t(
                                vector_lhs_value(token, elem) +
                                vector_rhs_value(token, elem)
                            );
                        }
                        fm_t actual = unpack_fm_word_lane(got, lane);
                        if (actual != expected) {
                            if (errors < 8) {
                                std::printf(
                                    "CLOSED LOOP vector data mismatch op=%u token=%u elem=%u got=%f expected=%f\n",
                                    unsigned(op),
                                    token,
                                    elem,
                                    double(actual),
                                    double(expected)
                                );
                            }
                            errors++;
                        }
                    }
                }
            }
        }
    }
    return errors;
}

static int check_vector_status(
    cc8_operator_t expected_op,
    unsigned int token_count
) {
    const fm_word_t word = status_output[0];
    const unsigned int op = word.range(31, 0).to_uint();
    const unsigned int status = word.range(63, 32).to_uint();
    const unsigned int got_tokens = word.range(95, 64).to_uint();
    const unsigned int dispatched_vector_tasks =
        word.range(191, 160).to_uint();
    const unsigned int completed_output_packets =
        word.range(223, 192).to_uint();
    const bool last_task = word[224];
    const unsigned int expected_tasks =
        token_count <= CC8_TOKENS_PER_DATA_PORT ? 1 : 2;
    const unsigned int expected_packets =
        token_count * ceildiv(vector_elem_count(expected_op), CU_VEC_LANES);
    if (op != unsigned(expected_op) ||
        status != unsigned(CC8_STATUS_OK) ||
        got_tokens != token_count ||
        dispatched_vector_tasks != expected_tasks ||
        completed_output_packets != expected_packets ||
        !last_task) {
        std::printf(
            "CLOSED LOOP vector status mismatch expected_op=%u tokens=%u op=%u status=%u got_tokens=%u vec=%u packets=%u last=%u\n",
            unsigned(expected_op),
            token_count,
            op,
            status,
            got_tokens,
            dispatched_vector_tasks,
            completed_output_packets,
            unsigned(last_task)
        );
        return 1;
    }
    return 0;
}
#endif

#ifdef CC8_CLOSED_LOOP_RESIDENT_LAYER_COSIM
static void initialize_resident_layer_aux() {
    for (unsigned int elem = 0; elem < HIDDEN_SIZE; elem++) {
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
}

static unsigned int resident_layer_wave_slots(unsigned int position) {
    const unsigned int attention_tiles =
        ceildiv(position + 1, CC8_ATTN_TILE);
    const unsigned int attention_waves =
        ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS);
    return
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(KV_CHANNELS, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(KV_CHANNELS, CC8_OUTPUTS_PER_WAVE) +
        attention_tiles * (1 + attention_waves) +
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE) +
        ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE);
}

static int check_resident_layer_result(unsigned int position) {
    int errors = 0;
    for (unsigned int elem = 0; elem < HIDDEN_SIZE; elem++) {
        const unsigned int word = elem / FM_BLOCK_SIZE;
        const unsigned int lane = elem % FM_BLOCK_SIZE;
        if (unpack_fm_word_lane(output_port0[word], lane) != fm_t(0)) {
            if (errors < 8) {
                std::printf(
                    "CLOSED LOOP resident output mismatch elem=%u\n",
                    elem
                );
            }
            errors++;
        }
    }

    const fm_word_t word = status_output[0];
    const unsigned int op = word.range(31, 0).to_uint();
    const unsigned int status = word.range(63, 32).to_uint();
    const unsigned int token_count = word.range(95, 64).to_uint();
    const unsigned int waves = word.range(127, 96).to_uint();
    const unsigned int mm_tasks = word.range(159, 128).to_uint();
    const unsigned int vector_tasks = word.range(191, 160).to_uint();
    const unsigned int packets = word.range(223, 192).to_uint();
    const bool last_task = word[224];
    const unsigned int expected_waves = resident_layer_wave_slots(position);
    const unsigned int expected_packets =
        expected_waves * CC8_MM_CORE_COUNT *
            MM_STREAM_8X64_PACKETS_PER_BLOCK +
        4 * ceildiv(HIDDEN_SIZE, CU_VEC_LANES) +
        ceildiv(INTERMEDIATE_SIZE, CU_VEC_LANES);
    if (op != unsigned(CC8_OP_DECODER_LAYER) ||
        status != unsigned(CC8_STATUS_OK) ||
        token_count != 1 ||
        waves != expected_waves ||
        mm_tasks != expected_waves * CC8_MM_CORE_COUNT ||
        vector_tasks != 5 ||
        packets != expected_packets ||
        !last_task) {
        std::printf(
            "CLOSED LOOP resident status mismatch op=%u status=%u tokens=%u waves=%u/%u mm=%u vec=%u packets=%u/%u last=%u\n",
            op,
            status,
            token_count,
            waves,
            expected_waves,
            mm_tasks,
            vector_tasks,
            packets,
            expected_packets,
            unsigned(last_task)
        );
        errors++;
    }
    return errors;
}
#endif

int main() {
#ifdef CC8_CLOSED_LOOP_RESIDENT_LAYER_COSIM
    clear_test_data();
    initialize_resident_layer_aux();
    constexpr unsigned int position = 0;
    call_closed_loop(CC8_OP_DECODER_LAYER, 1, position);
    int errors = check_resident_layer_result(position);
    std::printf(
        "CLOSED LOOP 8X64 RESIDENT LAYER RTL COSIM %s\n",
        errors == 0 ? "PASS" : "FAIL"
    );
    return errors == 0 ? 0 : 1;
#elif defined(CC8_CLOSED_LOOP_MM_REPEAT_COSIM)
    static_assert(
        ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE) == 2,
        "MM repeat closed-loop cosim requires exactly two output waves"
    );
    clear_test_data();
    initialize_mm_repeat_outputs();
    const unsigned int encoded_token_count =
        LINEAR_TOKEN_TILE_ACTIVE |
        (1u << 16) |
        (0x3u << 21);
    const unsigned int encoded_tile_len = 2;
    call_closed_loop_raw(
        CC8_OP_FFN_GATE,
        encoded_token_count,
        0,
        encoded_tile_len
    );
    int errors = 0;
    errors += check_mm_repeat_outputs();
    errors += check_mm_repeat_status();
    std::printf(
        "CLOSED LOOP 8X64 NK MM REPEAT RTL COSIM %s waves=2\n",
        errors == 0 ? "PASS" : "FAIL"
    );
    return errors == 0 ? 0 : 1;
#elif defined(CC8_CLOSED_LOOP_VECTOR_COSIM)
    const cc8_operator_t operator_cases[3] = {
        CC8_OP_RESIDUAL_ADD,
        CC8_OP_SILU_MUL,
        CC8_OP_RMSNORM
    };
    const unsigned int token_cases[4] = {1, 4, 5, 8};
    int errors = 0;
    for (unsigned int op_idx = 0; op_idx < 3; op_idx++) {
        for (unsigned int case_idx = 0; case_idx < 4; case_idx++) {
            cc8_operator_t op = operator_cases[op_idx];
            unsigned int token_count = token_cases[case_idx];
            clear_test_data();
            initialize_vector_outputs();
            initialize_vector_inputs(op);
            call_closed_loop(op, token_count, 0);
            errors += check_vector_outputs(op, token_count);
            errors += check_vector_status(op, token_count);
        }
    }
    std::printf(
        "CLOSED LOOP 8X64 VECTOR RTL COSIM %s cases=12\n",
        errors == 0 ? "PASS" : "FAIL"
    );
    return errors == 0 ? 0 : 1;
#else
    clear_test_data();
    set_current_v();

    const unsigned int layer = 0;
    const unsigned int position = 1;
    call_closed_loop(CC8_OP_DECODE_SMOKE, GQA_GROUP_SIZE, position);

    int errors = 0;
    errors += check_outputs();
    errors += check_kv_cache(layer, position);
    errors += check_status();

    std::printf(
        "CLOSED LOOP 8X64 RTL COSIM %s\n",
        errors == 0 ? "PASS" : "FAIL"
    );
    return errors == 0 ? 0 : 1;
#endif
}
