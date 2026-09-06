#include "mm_stream_8x64_int4x4_block.hpp"

#include <cstdio>

static ap_int<4> activation_value(
    unsigned int token, unsigned int k, unsigned int salt) {
    if (k == 0 && token < 8) {
        static const int edge[8] = {-8, 7, -1, 0, 1, 2, -7, 4};
        return ap_int<4>(edge[(token + salt) & 7]);
    }
    return ap_int<4>(int((token * 5 + k * 3 + salt * 7) % 15) - 7);
}

static ap_int<4> weight_value(
    unsigned int output, unsigned int k, unsigned int salt) {
    return ap_int<4>(int((output * 3 + k * 5 + salt * 2) % 15) - 7);
}

static int reference_value(
    unsigned int token, unsigned int output,
    unsigned int k_count, unsigned int salt) {
    int sum = 0;
    for (unsigned int k = 0; k < k_count; k++) {
        sum += activation_value(token, k, salt).to_int() *
               weight_value(output, k, salt).to_int();
    }
    return sum;
}

static int run_case(
    const char* name, unsigned int k_count, unsigned int elem_base,
    unsigned int block_id, unsigned int salt, bool last_stream) {
    hls::stream<mm_stream_8x64_int4x4_output_word_t> out_stream;
    hls::stream<mm_stream_quantized_task_word_t> task_stream;
    hls::stream<mm_stream_8x64_int4x4_activation_word_t> activation_stream;
    hls::stream<mm_stream_8x64_int4x4_weight_word_t> weight_stream;

    mm_stream_quantized_task_t task;
    task.k_count = k_count;
    task.elem_base = elem_base;
    task.block_id = block_id;
    task.last_stream = last_stream;
    task.activation_scale = fm_t(0.125);
    task.weight_scale = fm_t(0.25);
    task_stream.write(pack_mm_stream_quantized_task(task));

    for (unsigned int k = 0; k < k_count; k++) {
        mm_stream_8x64_int4x4_activation_word_t activation_word = 0;
        mm_stream_8x64_int4x4_weight_word_t weight_word = 0;
        for (unsigned int token = 0;
             token < MM_STREAM_8X64_INT4X4_TOKENS; token++) {
            activation_word.range(token * 4 + 3, token * 4) =
                activation_value(token, k, salt);
        }
        for (unsigned int output = 0;
             output < MM_STREAM_8X64_INT4X4_OUTPUTS; output++) {
            weight_word.range(output * 4 + 3, output * 4) =
                weight_value(output, k, salt);
        }
        activation_stream.write(activation_word);
        weight_stream.write(weight_word);
    }

    compute_mm_stream_8x64_int4x4_block_nk(
        out_stream, task_stream, activation_stream, weight_stream);

    int errors = 0;
    for (unsigned int token = 0;
         token < MM_STREAM_8X64_INT4X4_TOKENS; token++) {
        for (unsigned int group = 0;
             group < MM_STREAM_8X64_INT4X4_GROUPS; group++) {
            if (out_stream.empty()) {
                std::printf("%s missing token=%u group=%u\n", name, token, group);
                return errors + 1;
            }
            const mm_stream_8x64_int4x4_output_word_t output =
                out_stream.read();
            const bool expected_last =
                token == MM_STREAM_8X64_INT4X4_TOKENS - 1 &&
                group == MM_STREAM_8X64_INT4X4_GROUPS - 1;
            if (output.range(399, 384) != 0xffff ||
                output.range(407, 400).to_uint() != token ||
                output.range(423, 408).to_uint() !=
                    elem_base + group * MM_STREAM_8X64_INT4X4_LANES_PER_GROUP ||
                output.range(439, 424).to_uint() != block_id ||
                bool(output[440]) != expected_last ||
                bool(output[441]) != (last_stream && expected_last)) {
                std::printf("%s metadata mismatch token=%u group=%u\n",
                            name, token, group);
                errors++;
            }
            for (unsigned int lane = 0;
                 lane < MM_STREAM_8X64_INT4X4_LANES_PER_GROUP; lane++) {
                const unsigned int out =
                    group * MM_STREAM_8X64_INT4X4_LANES_PER_GROUP + lane;
                const int got =
                    unpack_mm_stream_8x64_int4x4_output(output, lane).to_int();
                const int expected =
                    reference_value(token, out, k_count, salt);
                if (got != expected) {
                    std::printf(
                        "%s mismatch token=%u out=%u got=%d expected=%d\n",
                        name, token, out, got, expected);
                    errors++;
                    return errors;
                }
            }
        }
    }
    if (!out_stream.empty()) {
        std::printf("%s emitted extra output\n", name);
        errors++;
    }
    return errors;
}

int main() {
    int errors = 0;
    errors += run_case("w4a4_k16", 16, 0, 1, 0, false);
    errors += run_case("w4a4_k64", 64, 64, 2, 1, false);
    errors += run_case("w4a4_k128_last", 128, 128, 3, 2, true);
    if (errors != 0) {
        std::printf("MM STREAM 8X64 INT4X4 BLOCK TB FAIL errors=%d\n", errors);
        return 1;
    }
    std::printf(
        "MM STREAM 8X64 INT4X4 BLOCK TB PASS cases=3 tokens=%u outputs=%u "
        "physical_dsps=%u packing_factor=%u logical_products_per_k=%u\n",
        MM_STREAM_8X64_INT4X4_TOKENS,
        MM_STREAM_8X64_INT4X4_OUTPUTS,
        MM_STREAM_8X64_INT4X4_PHYSICAL_DSPS,
        MM_STREAM_8X64_INT4X4_PACKING_FACTOR,
        MM_STREAM_8X64_INT4X4_LOGICAL_PRODUCTS_PER_K);
    return 0;
}
