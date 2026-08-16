#include "vitis_stream_8x64.hpp"

#include <cstdio>

static void write_vec_packet(
    hls::stream<cu8_nk_vector_word_t>& stream,
    fm_t value,
    unsigned int token_lane,
    unsigned int elem_base,
    unsigned int block_id,
    bool last_block,
    bool last_stream
) {
    cu_vec16_packet_t packet;
    packet.valid_mask = 0xffff;
    packet.token_lane = token_lane;
    packet.elem_base = elem_base;
    packet.block_id = block_id;
    packet.last_block = last_block;
    packet.last_stream = last_stream;
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        packet.data[lane] = value;
    }
    stream.write(pack_cu8_nk_vector(packet));
}

static void write_zero_vec_packet(
    hls::stream<cu8_nk_vector_word_t>& stream,
    unsigned int token_lane,
    unsigned int elem_base,
    unsigned int block_id,
    bool last_block,
    bool last_stream
) {
    write_vec_packet(
        stream,
        fm_t(0),
        token_lane,
        elem_base,
        block_id,
        last_block,
        last_stream
    );
}

static int check_vec_packet(
    const cu_vec16_packet_t& packet,
    fm_t expected,
    unsigned int token_lane,
    unsigned int elem_base,
    unsigned int block_id,
    bool last_block,
    bool last_stream
) {
    int errors = 0;
    if (packet.valid_mask != 0xffff ||
        packet.token_lane != token_lane ||
        packet.elem_base != elem_base ||
        packet.block_id != block_id ||
        packet.last_block != last_block ||
        packet.last_stream != last_stream) {
        errors++;
    }
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        if (packet.data[lane] != expected) {
            errors++;
        }
    }
    return errors;
}

static int check_zero_vec_packet(
    const cu_vec16_packet_t& packet,
    unsigned int token_lane,
    unsigned int elem_base,
    unsigned int block_id,
    bool last_block,
    bool last_stream
) {
    return check_vec_packet(
        packet,
        fm_t(0),
        token_lane,
        elem_base,
        block_id,
        last_block,
        last_stream
    );
}

static void write_zero_mm_inputs(
    hls::stream<cu8_nk_activation_word_t>& activation_stream,
    hls::stream<cu8_nk_weight_word_t>& weight_stream0,
    hls::stream<cu8_nk_weight_word_t>& weight_stream1,
    hls::stream<cu8_nk_weight_word_t>& weight_stream2,
    hls::stream<cu8_nk_weight_word_t>& weight_stream3,
    unsigned int k_count
) {
    mm_stream_8x64_activation_packet_t activation;
    mm_stream_8x64_weight_packet_t weight;
    for (unsigned int token = 0; token < MM_STREAM_8X64_TOKENS; token++) {
        activation.data[token] = fm_t(0);
    }
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        weight.data[lane] = wt_linear_t(0);
    }

    for (unsigned int k = 0; k < k_count; k++) {
        activation_stream.write(pack_cu8_nk_activation(activation));
        weight_stream0.write(pack_cu8_nk_weight(weight));
        weight_stream1.write(pack_cu8_nk_weight(weight));
        weight_stream2.write(pack_cu8_nk_weight(weight));
        weight_stream3.write(pack_cu8_nk_weight(weight));
    }
}

static int run_silu_mul_case() {
    hls::stream<cu8_nk_vector_word_t> out_stream;
    hls::stream<cu8_nk_task_word_t> task_stream;
    hls::stream<cu8_nk_activation_word_t> activation_stream;
    hls::stream<cu8_nk_weight_word_t> weight_stream0;
    hls::stream<cu8_nk_weight_word_t> weight_stream1;
    hls::stream<cu8_nk_weight_word_t> weight_stream2;
    hls::stream<cu8_nk_weight_word_t> weight_stream3;
    hls::stream<cu8_nk_vector_word_t> vector_input0_stream;
    hls::stream<cu8_nk_vector_word_t> vector_input1_stream;

    cu8_task_t task;
    task.mode = CU8_MODE_SILU_MUL;
    task.result_policy = CU8_RESULT_RELEASE;
    task.k_count = 0;
    task.token_count = 1;
    task.elem_count = CU_VEC_LANES;
    task.packet_count = 1;
    task.elem_base = 32;
    task.block_id = 7;
    task.repeat_count = 1;
    task.elem_stride = 0;
    task.block_stride = 0;
    task.output_scale = fm_t(1);
    task.last_task = true;
    task_stream.write(pack_cu8_nk_task(task));

    write_zero_vec_packet(
        vector_input0_stream,
        3,
        32,
        7,
        true,
        true
    );
    write_zero_vec_packet(
        vector_input1_stream,
        3,
        32,
        7,
        true,
        true
    );

    compute_core_8x64_unified_nk(
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

    int errors = 0;
    if (out_stream.empty()) {
        std::printf("COMPUTE CORE 8X64 NK SILU_MUL FAIL missing output\n");
        return 1;
    }
    errors += check_zero_vec_packet(
        unpack_cu8_nk_vector(out_stream.read()),
        3,
        32,
        7,
        true,
        true
    );
    if (!out_stream.empty()) {
        errors++;
    }
    return errors;
}

static int run_bypass_case() {
    hls::stream<cu8_nk_vector_word_t> out_stream;
    hls::stream<cu8_nk_task_word_t> task_stream;
    hls::stream<cu8_nk_activation_word_t> activation_stream;
    hls::stream<cu8_nk_weight_word_t> weight_stream0;
    hls::stream<cu8_nk_weight_word_t> weight_stream1;
    hls::stream<cu8_nk_weight_word_t> weight_stream2;
    hls::stream<cu8_nk_weight_word_t> weight_stream3;
    hls::stream<cu8_nk_vector_word_t> vector_input0_stream;
    hls::stream<cu8_nk_vector_word_t> vector_input1_stream;

    cu8_task_t task;
    task.mode = CU8_MODE_RESIDUAL_ADD;
    task.result_policy = CU8_RESULT_BYPASS;
    task.k_count = 0;
    task.token_count = 1;
    task.elem_count = CU_VEC_LANES;
    task.packet_count = 1;
    task.elem_base = 48;
    task.block_id = 17;
    task.repeat_count = 1;
    task.elem_stride = 0;
    task.block_stride = 0;
    task.output_scale = fm_t(1);
    task.last_task = true;
    task_stream.write(pack_cu8_nk_task(task));

    write_vec_packet(
        vector_input0_stream,
        fm_t(7),
        2,
        48,
        17,
        true,
        true
    );
    write_vec_packet(
        vector_input1_stream,
        fm_t(9),
        2,
        48,
        17,
        true,
        true
    );

    compute_core_8x64_unified_nk(
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

    int errors = 0;
    if (out_stream.empty()) {
        std::printf("COMPUTE CORE 8X64 NK BYPASS FAIL missing output\n");
        return 1;
    }
    errors += check_vec_packet(
        unpack_cu8_nk_vector(out_stream.read()),
        fm_t(7),
        2,
        48,
        17,
        true,
        true
    );
    if (!out_stream.empty()) {
        errors++;
    }
    if (!vector_input0_stream.empty() || !vector_input1_stream.empty()) {
        errors++;
    }
    return errors;
}

static int run_mm_zero_case() {
    hls::stream<cu8_nk_vector_word_t> out_stream;
    hls::stream<cu8_nk_task_word_t> task_stream;
    hls::stream<cu8_nk_activation_word_t> activation_stream;
    hls::stream<cu8_nk_weight_word_t> weight_stream0;
    hls::stream<cu8_nk_weight_word_t> weight_stream1;
    hls::stream<cu8_nk_weight_word_t> weight_stream2;
    hls::stream<cu8_nk_weight_word_t> weight_stream3;
    hls::stream<cu8_nk_vector_word_t> vector_input0_stream;
    hls::stream<cu8_nk_vector_word_t> vector_input1_stream;

    const unsigned int k_count = 4;
    const unsigned int block_id = 11;

    cu8_task_t task;
    task.mode = CU8_MODE_MM;
    task.result_policy = CU8_RESULT_RELEASE;
    task.k_count = k_count;
    task.token_count = MM_STREAM_8X64_TOKENS;
    task.elem_count = MM_STREAM_8X64_OUTPUTS;
    task.packet_count = MM_STREAM_8X64_PACKETS_PER_BLOCK;
    task.elem_base = 0;
    task.block_id = block_id;
    task.repeat_count = 1;
    task.elem_stride = 0;
    task.block_stride = 0;
    task.output_scale = fm_t(1);
    task.last_task = true;
    task_stream.write(pack_cu8_nk_task(task));
    write_zero_mm_inputs(
        activation_stream,
        weight_stream0,
        weight_stream1,
        weight_stream2,
        weight_stream3,
        k_count
    );

    compute_core_8x64_unified_nk(
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

    int errors = 0;
    for (unsigned int packet_idx = 0;
         packet_idx < MM_STREAM_8X64_PACKETS_PER_BLOCK;
         packet_idx++) {
        if (out_stream.empty()) {
            std::printf(
                "COMPUTE CORE 8X64 NK MM FAIL missing packet %u\n",
                packet_idx
            );
            return errors + 1;
        }
        unsigned int token =
            packet_idx / MM_STREAM_8X64_WEIGHT_GROUPS;
        unsigned int group =
            packet_idx % MM_STREAM_8X64_WEIGHT_GROUPS;
        bool last =
            packet_idx + 1 == MM_STREAM_8X64_PACKETS_PER_BLOCK;
        errors += check_zero_vec_packet(
            unpack_cu8_nk_vector(out_stream.read()),
            token,
            group * CU_VEC_LANES,
            block_id,
            last,
            last
        );
    }
    if (!out_stream.empty()) {
        errors++;
    }
    return errors;
}

static int run_mm_repeat_zero_case() {
    hls::stream<cu8_nk_vector_word_t> out_stream;
    hls::stream<cu8_nk_task_word_t> task_stream;
    hls::stream<cu8_nk_activation_word_t> activation_stream;
    hls::stream<cu8_nk_weight_word_t> weight_stream0;
    hls::stream<cu8_nk_weight_word_t> weight_stream1;
    hls::stream<cu8_nk_weight_word_t> weight_stream2;
    hls::stream<cu8_nk_weight_word_t> weight_stream3;
    hls::stream<cu8_nk_vector_word_t> vector_input0_stream;
    hls::stream<cu8_nk_vector_word_t> vector_input1_stream;

    const unsigned int k_count = 4;
    const unsigned int repeat_count = 2;
    const unsigned int elem_base = 128;
    const unsigned int block_id = 21;

    cu8_task_t task;
    task.mode = CU8_MODE_MM;
    task.result_policy = CU8_RESULT_RELEASE;
    task.k_count = k_count;
    task.token_count = MM_STREAM_8X64_TOKENS;
    task.elem_count = MM_STREAM_8X64_OUTPUTS;
    task.packet_count = MM_STREAM_8X64_PACKETS_PER_BLOCK;
    task.elem_base = elem_base;
    task.block_id = block_id;
    task.repeat_count = repeat_count;
    task.elem_stride = MM_STREAM_8X64_OUTPUTS;
    task.block_stride = 1;
    task.output_scale = fm_t(1);
    task.last_task = true;
    task_stream.write(pack_cu8_nk_task(task));

    for (unsigned int repeat = 0; repeat < repeat_count; repeat++) {
        write_zero_mm_inputs(
            activation_stream,
            weight_stream0,
            weight_stream1,
            weight_stream2,
            weight_stream3,
            k_count
        );
    }

    compute_core_8x64_unified_nk(
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

    int errors = 0;
    for (unsigned int repeat = 0; repeat < repeat_count; repeat++) {
        for (unsigned int packet_idx = 0;
             packet_idx < MM_STREAM_8X64_PACKETS_PER_BLOCK;
             packet_idx++) {
            if (out_stream.empty()) {
                std::printf(
                    "COMPUTE CORE 8X64 NK MM REPEAT FAIL missing repeat %u packet %u\n",
                    repeat,
                    packet_idx
                );
                return errors + 1;
            }
            unsigned int token =
                packet_idx / MM_STREAM_8X64_WEIGHT_GROUPS;
            unsigned int group =
                packet_idx % MM_STREAM_8X64_WEIGHT_GROUPS;
            bool last_block =
                packet_idx + 1 == MM_STREAM_8X64_PACKETS_PER_BLOCK;
            bool last_stream =
                repeat + 1 == repeat_count && last_block;
            errors += check_zero_vec_packet(
                unpack_cu8_nk_vector(out_stream.read()),
                token,
                elem_base +
                    repeat * MM_STREAM_8X64_OUTPUTS +
                    group * CU_VEC_LANES,
                block_id + repeat,
                last_block,
                last_stream
            );
        }
    }
    if (!out_stream.empty()) {
        errors++;
    }
    if (!activation_stream.empty() ||
        !weight_stream0.empty() ||
        !weight_stream1.empty() ||
        !weight_stream2.empty() ||
        !weight_stream3.empty()) {
        errors++;
    }
    return errors;
}

static int run_two_mm_task_zero_case() {
    hls::stream<cu8_nk_vector_word_t> out_stream;
    hls::stream<cu8_nk_task_word_t> task_stream;
    hls::stream<cu8_nk_activation_word_t> activation_stream;
    hls::stream<cu8_nk_weight_word_t> weight_stream0;
    hls::stream<cu8_nk_weight_word_t> weight_stream1;
    hls::stream<cu8_nk_weight_word_t> weight_stream2;
    hls::stream<cu8_nk_weight_word_t> weight_stream3;
    hls::stream<cu8_nk_vector_word_t> vector_input0_stream;
    hls::stream<cu8_nk_vector_word_t> vector_input1_stream;

    cu8_task_t qk_task;
    qk_task.mode = CU8_MODE_MM_SCALE;
    qk_task.result_policy = CU8_RESULT_RELEASE;
    // Exercise consecutive variable-length MM tasks.  The first task spans
    // the full small-profile hidden width while the second uses one input
    // tile; this catches fixed-MAX loop shells and task-boundary leftovers.
    qk_task.k_count = HIDDEN_SIZE;
    qk_task.token_count = GQA_GROUP_SIZE;
    qk_task.elem_count = CC8_ATTN_TILE;
    qk_task.packet_count = MM_STREAM_8X64_PACKETS_PER_BLOCK;
    qk_task.elem_base = 0;
    qk_task.block_id = 3;
    qk_task.repeat_count = 1;
    qk_task.elem_stride = 0;
    qk_task.block_stride = 0;
    qk_task.output_scale = fm_t(ATTENTION_SCALE);
    qk_task.last_task = false;

    cu8_task_t pv_task;
    pv_task.mode = CU8_MODE_MM;
    pv_task.result_policy = CU8_RESULT_RELEASE;
    pv_task.k_count = MM_PE_IN;
    pv_task.token_count = GQA_GROUP_SIZE;
    pv_task.elem_count = MM_STREAM_8X64_OUTPUTS;
    pv_task.packet_count = MM_STREAM_8X64_PACKETS_PER_BLOCK;
    pv_task.elem_base = 0;
    pv_task.block_id = 4;
    pv_task.repeat_count = 1;
    pv_task.elem_stride = 0;
    pv_task.block_stride = 0;
    pv_task.output_scale = fm_t(1);
    pv_task.last_task = true;

    task_stream.write(pack_cu8_nk_task(qk_task));
    write_zero_mm_inputs(
        activation_stream,
        weight_stream0,
        weight_stream1,
        weight_stream2,
        weight_stream3,
        qk_task.k_count
    );
    task_stream.write(pack_cu8_nk_task(pv_task));
    write_zero_mm_inputs(
        activation_stream,
        weight_stream0,
        weight_stream1,
        weight_stream2,
        weight_stream3,
        pv_task.k_count
    );

    compute_core_8x64_unified_nk(
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

    int errors = 0;
    for (unsigned int task_idx = 0; task_idx < 2; task_idx++) {
        const unsigned int block_id = task_idx == 0 ?
            qk_task.block_id :
            pv_task.block_id;
        for (unsigned int packet_idx = 0;
             packet_idx < MM_STREAM_8X64_PACKETS_PER_BLOCK;
             packet_idx++) {
            if (out_stream.empty()) {
                std::printf(
                    "COMPUTE CORE 8X64 NK TWO-MM FAIL missing task %u packet %u\n",
                    task_idx,
                    packet_idx
                );
                return errors + 1;
            }
            unsigned int token =
                packet_idx / MM_STREAM_8X64_WEIGHT_GROUPS;
            unsigned int group =
                packet_idx % MM_STREAM_8X64_WEIGHT_GROUPS;
            bool last_block =
                packet_idx + 1 == MM_STREAM_8X64_PACKETS_PER_BLOCK;
            bool last_stream = task_idx == 1 && last_block;
            errors += check_zero_vec_packet(
                unpack_cu8_nk_vector(out_stream.read()),
                token,
                group * CU_VEC_LANES,
                block_id,
                last_block,
                last_stream
            );
        }
    }
    if (!out_stream.empty()) {
        errors++;
    }
    return errors;
}

static int run_long_task_bound_case() {
    hls::stream<cu8_nk_vector_word_t> out_stream;
    hls::stream<cu8_nk_task_word_t> task_stream;
    hls::stream<cu8_nk_activation_word_t> activation_stream;
    hls::stream<cu8_nk_weight_word_t> weight_stream0;
    hls::stream<cu8_nk_weight_word_t> weight_stream1;
    hls::stream<cu8_nk_weight_word_t> weight_stream2;
    hls::stream<cu8_nk_weight_word_t> weight_stream3;
    hls::stream<cu8_nk_vector_word_t> vector_input0_stream;
    hls::stream<cu8_nk_vector_word_t> vector_input1_stream;

    // Mode 15 intentionally exercises the vector switch default without
    // consuming data.  The final task is placed at the exact configured RTL
    // bound, proving that a 2048-token attention launch can be fully drained.
    for (unsigned int task_idx = 0;
         task_idx < CU8_MAX_TASKS_PER_LAUNCH;
         task_idx++) {
        cu8_task_t task;
        task.mode = static_cast<cu8_mode_t>(15);
        task.result_policy = CU8_RESULT_RELEASE;
        task.k_count = 0;
        task.token_count = 0;
        task.elem_count = 0;
        task.packet_count = 0;
        task.elem_base = 0;
        task.block_id = task_idx;
        task.repeat_count = 1;
        task.elem_stride = 0;
        task.block_stride = 0;
        task.output_scale = fm_t(1);
        task.last_task = task_idx + 1 == CU8_MAX_TASKS_PER_LAUNCH;
        task_stream.write(pack_cu8_nk_task(task));
    }

    compute_core_8x64_unified_nk(
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

    if (!task_stream.empty() || !out_stream.empty()) {
        std::printf(
            "COMPUTE CORE 8X64 NK LONG TASK BOUND FAIL max_tasks=%u\n",
            CU8_MAX_TASKS_PER_LAUNCH
        );
        return 1;
    }
    std::printf(
        "COMPUTE CORE 8X64 NK LONG TASK BOUND PASS max_tasks=%u\n",
        CU8_MAX_TASKS_PER_LAUNCH
    );
    return 0;
}

int main() {
    int errors = 0;
    errors += run_silu_mul_case();
    errors += run_bypass_case();
    errors += run_mm_zero_case();
    errors += run_mm_repeat_zero_case();
    errors += run_two_mm_task_zero_case();
    if (CU8_MAX_TASKS_PER_LAUNCH > 256) {
        errors += run_long_task_bound_case();
    }

    std::printf(
        "COMPUTE CORE 8X64 NK CSIM %s\n",
        errors == 0 ? "PASS" : "FAIL"
    );
    return errors == 0 ? 0 : 1;
}
