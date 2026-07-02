#include "vitis_stream_8x64.hpp"

#include <cstdio>

int main() {
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
    task.mode = CU8_MODE_SILU;
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

    cu_vec16_packet_t input;
    input.valid_mask = 0xffff;
    input.token_lane = 3;
    input.elem_base = 32;
    input.block_id = 7;
    input.last_block = true;
    input.last_stream = true;
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        input.data[lane] = fm_t(0);
    }
    vector_input0_stream.write(pack_cu8_nk_vector(input));

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

    if (out_stream.empty()) {
        std::printf("COMPUTE CORE 8X64 NK CSIM FAIL missing output\n");
        return 1;
    }
    cu_vec16_packet_t output =
        unpack_cu8_nk_vector(out_stream.read());
    int errors = 0;
    if (output.valid_mask != input.valid_mask ||
        output.token_lane != input.token_lane ||
        output.elem_base != input.elem_base ||
        output.block_id != input.block_id ||
        !output.last_block ||
        !output.last_stream) {
        errors++;
    }
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        if (output.data[lane] != fm_t(0)) {
            errors++;
        }
    }
    if (!out_stream.empty()) {
        errors++;
    }

    std::printf(
        "COMPUTE CORE 8X64 NK CSIM %s\n",
        errors == 0 ? "PASS" : "FAIL"
    );
    return errors == 0 ? 0 : 1;
}
