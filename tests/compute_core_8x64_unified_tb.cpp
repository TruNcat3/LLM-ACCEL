#include "compute_core_8x64_unified.hpp"

#include <cmath>
#include <cstdio>

static cu8_task_t make_task(
    cu8_mode_t mode,
    unsigned int k_count,
    unsigned int packet_count,
    unsigned int block_id,
    bool last_task
) {
    cu8_task_t task;
    task.mode = mode;
    task.k_count = k_count;
    task.token_count = 1;
    task.elem_count = CU_VEC_LANES;
    task.packet_count = packet_count;
    task.elem_base = 0;
    task.block_id = block_id;
    task.repeat_count = 1;
    task.elem_stride = 0;
    task.block_stride = 0;
    task.output_scale = fm_t(1);
    task.last_task = last_task;
    return task;
}

static void emit_mm_inputs(
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3,
    unsigned int k_count
) {
    for (unsigned int k = 0; k < k_count; k++) {
        mm_stream_8x64_activation_packet_t activation;
        mm_stream_8x64_weight_packet_t weight;
        for (unsigned int token = 0; token < MM_STREAM_8X64_TOKENS; token++) {
            activation.data[token] = fm_t(0.25);
        }
        for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
            weight.data[lane] = wt_linear_t(0.125);
        }
        activation_stream.write(activation);
        weight_stream0.write(weight);
        weight_stream1.write(weight);
        weight_stream2.write(weight);
        weight_stream3.write(weight);
    }
}

static cu_vec16_packet_t make_vector_packet(
    fm_t value,
    unsigned int block_id,
    bool last_stream
) {
    cu_vec16_packet_t packet;
    packet.valid_mask = 0xffff;
    packet.token_lane = 0;
    packet.elem_base = 0;
    packet.block_id = block_id;
    packet.last_block = true;
    packet.last_stream = last_stream;
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        packet.data[lane] = value;
    }
    return packet;
}

int main() {
    hls::stream<cu_vec16_packet_t> out_stream;
    hls::stream<cu8_task_t> task_stream;
    hls::stream<mm_stream_8x64_activation_packet_t> activation_stream;
    hls::stream<mm_stream_8x64_weight_packet_t> weight_stream0;
    hls::stream<mm_stream_8x64_weight_packet_t> weight_stream1;
    hls::stream<mm_stream_8x64_weight_packet_t> weight_stream2;
    hls::stream<mm_stream_8x64_weight_packet_t> weight_stream3;
    hls::stream<cu_vec16_packet_t> vector_input0_stream;
    hls::stream<cu_vec16_packet_t> vector_input1_stream;

    task_stream.write(make_task(CU8_MODE_MM, 16, 32, 0, false));
    emit_mm_inputs(
        activation_stream,
        weight_stream0,
        weight_stream1,
        weight_stream2,
        weight_stream3,
        16
    );

    cu8_task_t scaled_mm =
        make_task(CU8_MODE_MM_SCALE, 16, 32, 1, false);
    scaled_mm.output_scale = fm_t(ATTENTION_SCALE);
    task_stream.write(scaled_mm);
    emit_mm_inputs(
        activation_stream,
        weight_stream0,
        weight_stream1,
        weight_stream2,
        weight_stream3,
        16
    );

    cu8_task_t repeated_mm =
        make_task(CU8_MODE_MM, 16, 32, 10, false);
    repeated_mm.elem_base = 128;
    repeated_mm.repeat_count = 2;
    repeated_mm.elem_stride = MM_STREAM_8X64_OUTPUTS;
    repeated_mm.block_stride = 1;
    task_stream.write(repeated_mm);
    for (unsigned int repeat = 0; repeat < 2; repeat++) {
        emit_mm_inputs(
            activation_stream,
            weight_stream0,
            weight_stream1,
            weight_stream2,
            weight_stream3,
            16
        );
    }

    task_stream.write(make_task(CU8_MODE_MM_SILU, 16, 32, 12, false));
    emit_mm_inputs(
        activation_stream,
        weight_stream0,
        weight_stream1,
        weight_stream2,
        weight_stream3,
        16
    );

    task_stream.write(make_task(CU8_MODE_SILU_MUL, 0, 1, 20, false));
    vector_input0_stream.write(make_vector_packet(fm_t(0), 20, false));
    vector_input1_stream.write(make_vector_packet(fm_t(3), 20, false));

    task_stream.write(make_task(CU8_MODE_RESIDUAL_ADD, 0, 1, 21, false));
    vector_input0_stream.write(make_vector_packet(fm_t(1), 21, false));
    vector_input1_stream.write(make_vector_packet(fm_t(2), 21, false));

    task_stream.write(make_task(CU8_MODE_SILU, 0, 1, 22, false));
    vector_input0_stream.write(make_vector_packet(fm_t(0.5), 22, false));

    cu8_task_t rmsnorm_task =
        make_task(CU8_MODE_RMSNORM, 0, 1, 23, false);
    rmsnorm_task.token_count = 1;
    rmsnorm_task.elem_count = CU_VEC_LANES;
    task_stream.write(rmsnorm_task);
    vector_input0_stream.write(make_vector_packet(fm_t(1), 23, false));
    vector_input1_stream.write(make_vector_packet(fm_t(1), 23, false));

    cu8_task_t softmax_task =
        make_task(CU8_MODE_SOFTMAX, 0, 1, 24, true);
    softmax_task.token_count = 1;
    softmax_task.elem_count = CU_VEC_LANES;
    task_stream.write(softmax_task);
    vector_input0_stream.write(make_vector_packet(fm_t(0), 24, true));

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

    int errors = 0;
    const fm_t linear_expected = fm_t(0.5);
    const fm_t qk_expected = fm_t(0.5 * ATTENTION_SCALE);
    for (unsigned int task_idx = 0; task_idx < 2; task_idx++) {
        fm_t expected = task_idx == 0 ? linear_expected : qk_expected;
        for (unsigned int packet = 0;
             packet < MM_STREAM_8X64_PACKETS_PER_BLOCK;
             packet++) {
            cu_vec16_packet_t got = out_stream.read();
            if (got.block_id != task_idx) {
                std::printf(
                    "unified MM block mismatch task=%u packet=%u\n",
                    task_idx,
                    packet
                );
                errors++;
            }
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                if (std::fabs(
                        float(got.data[lane]) -
                        float(expected)
                    ) > 0.004f) {
                    std::printf(
                        "unified MM value mismatch task=%u packet=%u lane=%u\n",
                        task_idx,
                        packet,
                        lane
                    );
                    errors++;
                }
            }
        }
    }

    for (unsigned int repeat = 0; repeat < 2; repeat++) {
        for (unsigned int packet = 0;
             packet < MM_STREAM_8X64_PACKETS_PER_BLOCK;
             packet++) {
            cu_vec16_packet_t got = out_stream.read();
            if (got.block_id != 10 + repeat ||
                got.elem_base <
                    128 + repeat * MM_STREAM_8X64_OUTPUTS ||
                got.elem_base >=
                    128 + (repeat + 1) * MM_STREAM_8X64_OUTPUTS) {
                std::printf(
                    "unified repeated MM metadata mismatch repeat=%u packet=%u\n",
                    repeat,
                    packet
                );
                errors++;
            }
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                if (std::fabs(
                        float(got.data[lane]) -
                        float(linear_expected)
                    ) > 0.004f) {
                    std::printf(
                        "unified repeated MM value mismatch repeat=%u packet=%u lane=%u\n",
                        repeat,
                        packet,
                        lane
                    );
                    errors++;
                }
            }
        }
    }

    const float mm_silu_expected =
        0.5f / (1.0f + std::exp(-0.5f));
    for (unsigned int packet = 0;
         packet < MM_STREAM_8X64_PACKETS_PER_BLOCK;
         packet++) {
        cu_vec16_packet_t got = out_stream.read();
        if (got.block_id != 12) {
            std::printf(
                "unified MM-SiLU block mismatch packet=%u\n",
                packet
            );
            errors++;
        }
        for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
            if (std::fabs(
                    float(got.data[lane]) - mm_silu_expected
                ) > 0.03f) {
                std::printf(
                    "unified MM-SiLU value mismatch packet=%u lane=%u\n",
                    packet,
                    lane
                );
                errors++;
            }
        }
    }

    cu_vec16_packet_t silu_mul = out_stream.read();
    cu_vec16_packet_t add = out_stream.read();
    cu_vec16_packet_t silu = out_stream.read();
    cu_vec16_packet_t rmsnorm = out_stream.read();
    cu_vec16_packet_t softmax = out_stream.read();
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        if (silu_mul.data[lane] != fm_t(0)) {
            std::printf("unified SiLU-Mul mismatch lane=%u\n", lane);
            errors++;
        }
        if (add.data[lane] != fm_t(3)) {
            std::printf("unified Add mismatch lane=%u\n", lane);
            errors++;
        }
        if (std::fabs(
                float(silu.data[lane]) - mm_silu_expected
            ) > 0.03f) {
            std::printf("unified SiLU mismatch lane=%u\n", lane);
            errors++;
        }
        if (std::fabs(float(rmsnorm.data[lane]) - 1.0f) > 0.08f) {
            std::printf(
                "unified RMSNorm mismatch lane=%u got=%f\n",
                lane,
                float(rmsnorm.data[lane])
            );
            errors++;
        }
        if (std::fabs(float(softmax.data[lane]) - 0.0625f) > 0.02f) {
            std::printf(
                "unified softmax mismatch lane=%u got=%f\n",
                lane,
                float(softmax.data[lane])
            );
            errors++;
        }
    }

    if (!out_stream.empty() || !task_stream.empty() ||
        !activation_stream.empty() || !weight_stream0.empty() ||
        !weight_stream1.empty() || !weight_stream2.empty() ||
        !weight_stream3.empty() || !vector_input0_stream.empty() ||
        !vector_input1_stream.empty()) {
        std::printf("unified core stream contains extra packets\n");
        errors++;
    }

    if (errors != 0) {
        std::printf("COMPUTE CORE 8X64 UNIFIED CSIM FAIL errors=%d\n", errors);
        return 1;
    }
    std::printf(
        "COMPUTE CORE 8X64 UNIFIED CSIM PASS tasks=9 mm_blocks=5\n"
    );
    return 0;
}
