#include "mm_stream_4x128_int8x8_block.hpp"

#include <cstdio>

static ap_int<8> activation_value(
    unsigned int token, unsigned int k, unsigned int salt) {
    if (k == 0) {
        static const int edge[4] = {-128, 127, -1, 0};
        return ap_int<8>(edge[(token + salt) & 3]);
    }
    return ap_int<8>(int((token * 17 + k * 5 + salt * 11) % 255) - 127);
}

static ap_int<8> weight_value(
    unsigned int output, unsigned int k, unsigned int salt) {
    return ap_int<8>(int((output * 7 + k * 3 + salt * 13) % 255) - 127);
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
    hls::stream<mm_stream_4x128_int8x8_output_word_t> out_stream;
    hls::stream<mm_stream_quantized_task_word_t> task_stream;
    hls::stream<mm_stream_4x128_int8x8_activation_word_t> activation_stream;
    hls::stream<mm_stream_4x128_int8x8_weight_word_t> weight_stream0;
    hls::stream<mm_stream_4x128_int8x8_weight_word_t> weight_stream1;
    hls::stream<mm_stream_4x128_int8x8_weight_word_t> weight_stream2;
    hls::stream<mm_stream_4x128_int8x8_weight_word_t> weight_stream3;

    mm_stream_quantized_task_t task;
    task.k_count = k_count;
    task.elem_base = elem_base;
    task.block_id = block_id;
    task.last_stream = last_stream;
    task.activation_scale = fm_t(0.03125);
    task.weight_scale = fm_t(0.0625);
    task_stream.write(pack_mm_stream_quantized_task(task));

    for (unsigned int k = 0; k < k_count; k++) {
        mm_stream_4x128_int8x8_activation_word_t activation_word = 0;
        mm_stream_4x128_int8x8_weight_word_t weight_words[4];
        #pragma HLS array_partition variable=weight_words complete dim=1
        for (unsigned int stream = 0; stream < 4; stream++) {
            weight_words[stream] = 0;
        }
        for (unsigned int token = 0;
             token < MM_STREAM_4X128_INT8X8_TOKENS; token++) {
            activation_word.range(token * 8 + 7, token * 8) =
                activation_value(token, k, salt);
        }
        for (unsigned int output = 0;
             output < MM_STREAM_4X128_INT8X8_OUTPUTS; output++) {
            const unsigned int stream = output / 32;
            const unsigned int local = output % 32;
            weight_words[stream].range(local * 8 + 7, local * 8) =
                weight_value(output, k, salt);
        }
        activation_stream.write(activation_word);
        weight_stream0.write(weight_words[0]);
        weight_stream1.write(weight_words[1]);
        weight_stream2.write(weight_words[2]);
        weight_stream3.write(weight_words[3]);
    }

    compute_mm_stream_4x128_int8x8_block_nk(
        out_stream, task_stream, activation_stream,
        weight_stream0, weight_stream1, weight_stream2, weight_stream3);

    int errors = 0;
    for (unsigned int token = 0;
         token < MM_STREAM_4X128_INT8X8_TOKENS; token++) {
        for (unsigned int group = 0;
             group < MM_STREAM_4X128_INT8X8_OUTPUT_GROUPS; group++) {
            if (out_stream.empty()) {
                std::printf("%s missing token=%u group=%u\n", name, token, group);
                return errors + 1;
            }
            const mm_stream_4x128_int8x8_output_word_t output =
                out_stream.read();
            const bool expected_last =
                token == MM_STREAM_4X128_INT8X8_TOKENS - 1 &&
                group == MM_STREAM_4X128_INT8X8_OUTPUT_GROUPS - 1;
            if (output.range(527, 512) != 0xffff ||
                output.range(535, 528).to_uint() != token ||
                output.range(551, 536).to_uint() !=
                    elem_base + group * MM_STREAM_4X128_INT8X8_LANES_PER_GROUP ||
                output.range(567, 552).to_uint() != block_id ||
                bool(output[568]) != expected_last ||
                bool(output[569]) != (last_stream && expected_last)) {
                std::printf("%s metadata mismatch token=%u group=%u\n",
                            name, token, group);
                errors++;
            }
            for (unsigned int lane = 0;
                 lane < MM_STREAM_4X128_INT8X8_LANES_PER_GROUP; lane++) {
                const unsigned int out =
                    group * MM_STREAM_4X128_INT8X8_LANES_PER_GROUP + lane;
                const int got =
                    unpack_mm_stream_4x128_int8x8_output(output, lane).to_int();
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
    errors += run_case("w8a8_k16", 16, 0, 4, 0, false);
    errors += run_case("w8a8_k64", 64, 128, 5, 1, false);
    errors += run_case("w8a8_k128_last", 128, 256, 6, 2, true);
    if (errors != 0) {
        std::printf("MM STREAM 4X128 INT8X8 BLOCK TB FAIL errors=%d\n", errors);
        return 1;
    }
    std::printf(
        "MM STREAM 4X128 INT8X8 BLOCK TB PASS cases=3 tokens=%u outputs=%u "
        "weight_streams=%u physical_dsps=%u logical_products_per_k=%u\n",
        MM_STREAM_4X128_INT8X8_TOKENS,
        MM_STREAM_4X128_INT8X8_OUTPUTS,
        MM_STREAM_4X128_INT8X8_WEIGHT_STREAMS,
        MM_STREAM_4X128_INT8X8_PHYSICAL_DSPS,
        MM_STREAM_4X128_INT8X8_LOGICAL_PRODUCTS_PER_K);
    return 0;
}
