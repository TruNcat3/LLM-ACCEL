#include "vitis_stream_8x64.hpp"

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

struct cc8_nk_test_streams_t {
    hls::stream<cu8_nk_task_word_t> core0_task_stream;
    hls::stream<cu8_nk_activation_word_t> core0_activation_stream;
    hls::stream<cu8_nk_weight_word_t> core0_weight_stream0;
    hls::stream<cu8_nk_weight_word_t> core0_weight_stream1;
    hls::stream<cu8_nk_weight_word_t> core0_weight_stream2;
    hls::stream<cu8_nk_weight_word_t> core0_weight_stream3;
    hls::stream<cu8_nk_vector_word_t> core0_vector_input0_stream;
    hls::stream<cu8_nk_vector_word_t> core0_vector_input1_stream;
    hls::stream<cu8_nk_vector_word_t> core0_result_stream;
    hls::stream<cu8_nk_task_word_t> core1_task_stream;
    hls::stream<cu8_nk_activation_word_t> core1_activation_stream;
    hls::stream<cu8_nk_weight_word_t> core1_weight_stream0;
    hls::stream<cu8_nk_weight_word_t> core1_weight_stream1;
    hls::stream<cu8_nk_weight_word_t> core1_weight_stream2;
    hls::stream<cu8_nk_weight_word_t> core1_weight_stream3;
    hls::stream<cu8_nk_vector_word_t> core1_vector_input0_stream;
    hls::stream<cu8_nk_vector_word_t> core1_vector_input1_stream;
    hls::stream<cu8_nk_vector_word_t> core1_result_stream;
    hls::stream<cu8_nk_status_word_t> status_stream;
};

static void clear_ports() {
    for (unsigned int word = 0; word < CC8_FEATURE_WORDS_PER_PORT; word++) {
        output_port0[word] = 0;
        output_port1[word] = 0;
    }
    for (unsigned int word = 0; word < CC8_DATA_PORT_WORDS; word++) {
        input_port0[word] = 0;
        input_port1[word] = 0;
        aux_port0[word] = 0;
        aux_port1[word] = 0;
    }
    for (unsigned int word = 0; word < CC8_KV_CACHE_WORDS; word++) {
        kv_cache_k[word] = 0;
        kv_cache_v[word] = 0;
    }
}

static void set_flat_row_value(
    fm_word_t words[CC8_DATA_PORT_WORDS],
    unsigned int row,
    unsigned int elem,
    unsigned int row_elems,
    fm_t value
) {
    unsigned int words_per_row = ceildiv(row_elems, FM_BLOCK_SIZE);
    unsigned int word_idx = row * words_per_row + elem / FM_BLOCK_SIZE;
    set_fm_word_lane(words[word_idx], elem % FM_BLOCK_SIZE, value);
}

static fm_t get_flat_output_value(
    const fm_word_t words[CC8_FEATURE_WORDS_PER_PORT],
    unsigned int row,
    unsigned int elem,
    unsigned int row_elems
) {
    unsigned int words_per_row = ceildiv(row_elems, FM_BLOCK_SIZE);
    unsigned int word_idx = row * words_per_row + elem / FM_BLOCK_SIZE;
    return unpack_fm_word_lane(words[word_idx], elem % FM_BLOCK_SIZE);
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

static void call_control_cache(
    cc8_nk_test_streams_t& streams,
    cc8_operator_t op,
    unsigned int token_count,
    unsigned int position,
    unsigned int tile_len
) {
    control_cache_8x64_dual_core_nk(
        streams.core0_task_stream,
        streams.core0_activation_stream,
        streams.core0_weight_stream0,
        streams.core0_weight_stream1,
        streams.core0_weight_stream2,
        streams.core0_weight_stream3,
        streams.core0_vector_input0_stream,
        streams.core0_vector_input1_stream,
        streams.core0_result_stream,
        streams.core1_task_stream,
        streams.core1_activation_stream,
        streams.core1_weight_stream0,
        streams.core1_weight_stream1,
        streams.core1_weight_stream2,
        streams.core1_weight_stream3,
        streams.core1_vector_input0_stream,
        streams.core1_vector_input1_stream,
        streams.core1_result_stream,
        streams.status_stream,
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
        shard0,
        shard1,
        shard2,
        shard3,
        shard4,
        shard5,
        shard6,
        shard7,
        shard8,
        shard9,
        shard10,
        shard11,
        shard12,
        shard13,
        shard14,
        shard15,
        kv_cache_k,
        kv_cache_v
    );
}

static void write_constant_result_packets(
    hls::stream<cu8_nk_vector_word_t>& result_stream,
    fm_t value,
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
            packet.elem_base = group * CU_VEC_LANES;
            packet.block_id = 0;
            packet.last_block =
                token + 1 == MM_STREAM_8X64_TOKENS &&
                group + 1 == MM_STREAM_8X64_WEIGHT_GROUPS;
            packet.last_stream = last_stream && packet.last_block;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                packet.data[lane] = value;
            }
            result_stream.write(pack_cu8_nk_vector(packet));
        }
    }
}

static int drain_mm_axis_inputs(
    hls::stream<cu8_nk_activation_word_t>& activation_stream,
    hls::stream<cu8_nk_weight_word_t>& weight_stream0,
    hls::stream<cu8_nk_weight_word_t>& weight_stream1,
    hls::stream<cu8_nk_weight_word_t>& weight_stream2,
    hls::stream<cu8_nk_weight_word_t>& weight_stream3,
    unsigned int packets
) {
    int errors = 0;
    for (unsigned int packet = 0; packet < packets; packet++) {
        if (activation_stream.empty() ||
            weight_stream0.empty() ||
            weight_stream1.empty() ||
            weight_stream2.empty() ||
            weight_stream3.empty()) {
            return errors + 1;
        }
        activation_stream.read();
        weight_stream0.read();
        weight_stream1.read();
        weight_stream2.read();
        weight_stream3.read();
    }
    return errors;
}

static void write_zero_result_packets(
    hls::stream<cu8_nk_vector_word_t>& result_stream,
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
                packet.data[lane] = fm_t(0);
            }
            result_stream.write(pack_cu8_nk_vector(packet));
        }
    }
}

static int drain_zero_vector_inputs(
    hls::stream<cu8_nk_vector_word_t>& lhs_stream,
    hls::stream<cu8_nk_vector_word_t>& rhs_stream,
    unsigned int packets,
    unsigned int token_begin,
    unsigned int token_slots,
    unsigned int elem_count
) {
    int errors = 0;
    unsigned int blocks = ceildiv(elem_count, CU_VEC_LANES);
    for (unsigned int packet_idx = 0; packet_idx < packets; packet_idx++) {
        if (lhs_stream.empty() || rhs_stream.empty()) {
            std::printf("CONTROL CACHE 8X64 NK vector input missing\n");
            return errors + 1;
        }
        cu_vec16_packet_t lhs = unpack_cu8_nk_vector(lhs_stream.read());
        cu_vec16_packet_t rhs = unpack_cu8_nk_vector(rhs_stream.read());
        unsigned int token = token_begin + packet_idx / blocks;
        unsigned int block = packet_idx % blocks;
        bool last =
            token + 1 == token_begin + token_slots &&
            block + 1 == blocks;
        if (lhs.token_lane != token ||
            rhs.token_lane != token ||
            lhs.elem_base != block * CU_VEC_LANES ||
            rhs.elem_base != block * CU_VEC_LANES ||
            lhs.last_block != last ||
            rhs.last_block != last ||
            lhs.last_stream != last ||
            rhs.last_stream != last) {
            errors++;
        }
        for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
            if (lhs.data[lane] != fm_t(0) ||
                rhs.data[lane] != fm_t(0)) {
                errors++;
            }
        }
    }
    return errors;
}

static int run_nop_case() {
    clear_ports();
    cc8_nk_test_streams_t streams;
    call_control_cache(streams, CC8_OP_NOP, 1, 0, 0);

    int errors = 0;
    if (streams.core0_task_stream.empty() ||
        streams.core1_task_stream.empty()) {
        errors++;
    } else {
        cu8_task_t task0 =
            unpack_cu8_nk_task(streams.core0_task_stream.read());
        cu8_task_t task1 =
            unpack_cu8_nk_task(streams.core1_task_stream.read());
        if (task0.mode != CU8_MODE_STOP ||
            task1.mode != CU8_MODE_STOP ||
            !task0.last_task ||
            !task1.last_task) {
            errors++;
        }
    }
    if (streams.status_stream.empty()) {
        errors++;
    } else {
        cc8_status_packet_t status =
            unpack_cu8_nk_status(streams.status_stream.read());
        if (status.op != CC8_OP_NOP ||
            status.status != CC8_STATUS_OK ||
            !status.last_task) {
            errors++;
        }
    }
    return errors;
}

static int run_residual_add_case(unsigned int token_count) {
    clear_ports();
    cc8_nk_test_streams_t streams;
    const unsigned int blocks_per_token =
        ceildiv(HIDDEN_SIZE, CU_VEC_LANES);
    const unsigned int core0_token_count =
        token_count < CC8_TOKENS_PER_DATA_PORT ?
        token_count : CC8_TOKENS_PER_DATA_PORT;
    const unsigned int core1_token_count =
        token_count > CC8_TOKENS_PER_DATA_PORT ?
        token_count - CC8_TOKENS_PER_DATA_PORT : 0;
    const unsigned int core0_packet_count =
        core0_token_count * blocks_per_token;
    const unsigned int core1_packet_count =
        core1_token_count * blocks_per_token;

    write_zero_result_packets(
        streams.core0_result_stream,
        0,
        core0_token_count,
        HIDDEN_SIZE
    );
    write_zero_result_packets(
        streams.core1_result_stream,
        CC8_TOKENS_PER_DATA_PORT,
        core1_token_count,
        HIDDEN_SIZE
    );

    call_control_cache(streams, CC8_OP_RESIDUAL_ADD, token_count, 0, 0);

    int errors = 0;
    if (streams.core0_task_stream.empty() ||
        streams.core1_task_stream.empty()) {
        std::printf("CONTROL CACHE 8X64 NK residual task missing\n");
        return 1;
    }

    cu8_task_t task0 =
        unpack_cu8_nk_task(streams.core0_task_stream.read());
    cu8_task_t task1 =
        unpack_cu8_nk_task(streams.core1_task_stream.read());
    if (task0.mode != CU8_MODE_RESIDUAL_ADD ||
        task0.packet_count != core0_packet_count ||
        task0.token_count != core0_token_count ||
        !task0.last_task) {
        errors++;
    }
    if (core1_token_count != 0) {
        if (task1.mode != CU8_MODE_RESIDUAL_ADD ||
            task1.packet_count != core1_packet_count ||
            task1.token_count != core1_token_count ||
            !task1.last_task) {
            errors++;
        }
    } else if (task1.mode != CU8_MODE_STOP || !task1.last_task) {
        errors++;
    }

    errors += drain_zero_vector_inputs(
        streams.core0_vector_input0_stream,
        streams.core0_vector_input1_stream,
        core0_packet_count,
        0,
        core0_token_count,
        HIDDEN_SIZE
    );
    errors += drain_zero_vector_inputs(
        streams.core1_vector_input0_stream,
        streams.core1_vector_input1_stream,
        core1_packet_count,
        CC8_TOKENS_PER_DATA_PORT,
        core1_token_count,
        HIDDEN_SIZE
    );

    for (unsigned int token = 0; token < token_count; token++) {
        for (unsigned int elem = 0; elem < HIDDEN_SIZE; elem++) {
            if (get_output_value(token, elem) != fm_t(0)) {
                errors++;
            }
        }
    }

    if (streams.status_stream.empty()) {
        errors++;
    } else {
        cc8_status_packet_t status =
            unpack_cu8_nk_status(streams.status_stream.read());
        if (status.op != CC8_OP_RESIDUAL_ADD ||
            status.status != CC8_STATUS_OK ||
            status.dispatched_mm_tasks != 0 ||
            status.dispatched_vector_tasks !=
                (core1_token_count == 0 ? 1 : CC8_MM_CORE_COUNT) ||
            status.completed_output_packets !=
                core0_packet_count + core1_packet_count ||
            !status.last_task) {
            errors++;
        }
    }

    if (!streams.core0_task_stream.empty() ||
        !streams.core1_task_stream.empty() ||
        !streams.core0_vector_input0_stream.empty() ||
        !streams.core0_vector_input1_stream.empty() ||
        !streams.core1_vector_input0_stream.empty() ||
        !streams.core1_vector_input1_stream.empty()) {
        errors++;
    }
    return errors;
}

static int run_flash_attention_case(cc8_operator_t op) {
    clear_ports();
    cc8_nk_test_streams_t streams;
    const unsigned int position = 1;
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
        for (unsigned int elem = 0; elem < HEAD_DIM; elem++) {
            set_flat_row_value(
                aux_port0,
                0,
                elem,
                HEAD_DIM,
                fm_t(0.5)
            );
            set_flat_row_value(
                aux_port0,
                1,
                elem,
                HEAD_DIM,
                fm_t(0.75)
            );
            set_flat_row_value(
                aux_port1,
                0,
                elem,
                HEAD_DIM,
                fm_t(1.0)
            );
            set_flat_row_value(
                aux_port1,
                1,
                elem,
                HEAD_DIM,
                fm_t(1.25)
            );
        }
    }

    write_constant_result_packets(
        streams.core0_result_stream,
        fm_t(0),
        false
    );
    write_constant_result_packets(
        streams.core1_result_stream,
        fm_t(0),
        false
    );
    write_constant_result_packets(
        streams.core0_result_stream,
        fm_t(2),
        true
    );
    write_constant_result_packets(
        streams.core1_result_stream,
        fm_t(2),
        true
    );

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
                unsigned int cache_idx = test_kv_cache_word_index(
                    0,
                    position,
                    kv_head,
                    word_idx
                );
                if (kv_cache_k[cache_idx] != aux_port0[source_idx] ||
                    kv_cache_v[cache_idx] != aux_port1[source_idx]) {
                    errors++;
                }
            }
        }
    }

    if (streams.core0_task_stream.empty() ||
        streams.core1_task_stream.empty()) {
        std::printf("CONTROL CACHE 8X64 NK flash QK task missing\n");
        return 1;
    }
    cu8_task_t qk0 =
        unpack_cu8_nk_task(streams.core0_task_stream.read());
    cu8_task_t qk1 =
        unpack_cu8_nk_task(streams.core1_task_stream.read());
    if (qk0.mode != CU8_MODE_MM_SCALE ||
        qk1.mode != CU8_MODE_MM_SCALE ||
        qk0.k_count != HEAD_DIM ||
        qk1.k_count != HEAD_DIM ||
        qk0.elem_count != CC8_ATTN_TILE ||
        qk1.elem_count != CC8_ATTN_TILE ||
        qk0.last_task ||
        qk1.last_task) {
        errors++;
    }
    errors += drain_mm_axis_inputs(
        streams.core0_activation_stream,
        streams.core0_weight_stream0,
        streams.core0_weight_stream1,
        streams.core0_weight_stream2,
        streams.core0_weight_stream3,
        HEAD_DIM
    );
    errors += drain_mm_axis_inputs(
        streams.core1_activation_stream,
        streams.core1_weight_stream0,
        streams.core1_weight_stream1,
        streams.core1_weight_stream2,
        streams.core1_weight_stream3,
        HEAD_DIM
    );

    if (streams.core0_task_stream.empty() ||
        streams.core1_task_stream.empty()) {
        std::printf("CONTROL CACHE 8X64 NK flash PV task missing\n");
        return errors + 1;
    }
    cu8_task_t pv0 =
        unpack_cu8_nk_task(streams.core0_task_stream.read());
    cu8_task_t pv1 =
        unpack_cu8_nk_task(streams.core1_task_stream.read());
    if (pv0.mode != CU8_MODE_MM ||
        pv1.mode != CU8_MODE_MM ||
        pv0.k_count != context_len ||
        pv1.k_count != context_len ||
        pv0.elem_count != MM_STREAM_8X64_OUTPUTS ||
        pv1.elem_count != MM_STREAM_8X64_OUTPUTS ||
        !pv0.last_task ||
        !pv1.last_task) {
        errors++;
    }
    errors += drain_mm_axis_inputs(
        streams.core0_activation_stream,
        streams.core0_weight_stream0,
        streams.core0_weight_stream1,
        streams.core0_weight_stream2,
        streams.core0_weight_stream3,
        context_len
    );
    errors += drain_mm_axis_inputs(
        streams.core1_activation_stream,
        streams.core1_weight_stream0,
        streams.core1_weight_stream1,
        streams.core1_weight_stream2,
        streams.core1_weight_stream3,
        context_len
    );

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
                errors++;
            }
        }
    }

    if (streams.status_stream.empty()) {
        errors++;
    } else {
        cc8_status_packet_t status =
            unpack_cu8_nk_status(streams.status_stream.read());
        if (status.op != op ||
            status.status != CC8_STATUS_OK ||
            status.output_waves != ceildiv(
                HEAD_DIM,
                MM_STREAM_8X64_OUTPUTS
            ) ||
            status.dispatched_mm_tasks != 4 ||
            status.completed_output_packets !=
                4 * MM_STREAM_8X64_PACKETS_PER_BLOCK ||
            !status.last_task) {
            errors++;
        }
    }

    if (!streams.core0_task_stream.empty() ||
        !streams.core1_task_stream.empty() ||
        !streams.core0_result_stream.empty() ||
        !streams.core1_result_stream.empty()) {
        errors++;
    }
    return errors;
}

int main() {
    int errors = 0;
    {
        cc8_status_packet_t expected;
        expected.op = CC8_OP_DECODER_LAYER;
        expected.status = CC8_STATUS_OK;
        expected.token_count = 1;
        expected.output_waves = 233;
        expected.dispatched_mm_tasks =
            2 * CU8_MAX_TASKS_PER_LAUNCH;
        expected.dispatched_vector_tasks = 5;
        expected.completed_output_packets =
            expected.dispatched_mm_tasks *
            MM_STREAM_8X64_PACKETS_PER_BLOCK;
        expected.last_task = true;
        cc8_status_packet_t got = unpack_cu8_nk_status(
            pack_cu8_nk_status(expected)
        );
        if (got.op != expected.op ||
            got.status != expected.status ||
            got.token_count != expected.token_count ||
            got.output_waves != expected.output_waves ||
            got.dispatched_mm_tasks != expected.dispatched_mm_tasks ||
            got.dispatched_vector_tasks != expected.dispatched_vector_tasks ||
            got.completed_output_packets !=
                expected.completed_output_packets ||
            got.last_task != expected.last_task) {
            std::printf("CONTROL CACHE 8X64 NK status roundtrip failed\n");
            errors++;
        }
    }
    std::printf("CONTROL CACHE 8X64 NK running nop case\n");
    errors += run_nop_case();
#ifdef CC8_NK_ENABLE_FLASH_CSIM_CASE
    std::printf("CONTROL CACHE 8X64 NK running flash attention case\n");
    errors += run_flash_attention_case(CC8_OP_ATTN_FLASH);
    std::printf("CONTROL CACHE 8X64 NK running decode smoke case\n");
    errors += run_flash_attention_case(CC8_OP_DECODE_SMOKE);
#endif
#ifdef CC8_NK_ENABLE_VECTOR_CSIM_CASE
    std::printf("CONTROL CACHE 8X64 NK running residual full case\n");
    errors += run_residual_add_case(LINEAR_TOKEN_TILE_ACTIVE);
    std::printf("CONTROL CACHE 8X64 NK running residual single-token case\n");
    errors += run_residual_add_case(1);
#endif

    std::printf(
        "CONTROL CACHE 8X64 NK CSIM %s\n",
        errors == 0 ? "PASS" : "FAIL"
    );
    return errors == 0 ? 0 : 1;
}
