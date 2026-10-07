#include "quantized_w8_layer_runtime.hpp"
#include "quantized_layer_protocol_fixture.hpp"

#include <cassert>
#include <iostream>
#include <vector>

namespace {

struct runtime_fixture_t {
    hls::stream<mm_stream_quantized_task_word_t> task[4];
    hls::stream<quantized_w8_activation_word_t> activation[4];
    hls::stream<quantized_w8_weight_word_t> weight[4][4];
    hls::stream<quantized_w8_output_word_t> output[4];
    hls::stream<quantized_vector_word_t> vector[4][2];
    std::vector<quantized_w8_weight_word_t> weight_memory[4][4];
    quantized_w8_projection_streams_t streams{};
    quantized_w8_weight_memories_t memories{};

    runtime_fixture_t() {
        streams.vector = {&vector[0][0], &vector[0][1], &vector[1][0], &vector[1][1],
            &vector[2][0], &vector[2][1], &vector[3][0], &vector[3][1]};
        const std::size_t words = quantized_layer_weight_words_per_shard();
        for (unsigned int cu = 0; cu < 4; ++cu) {
            streams.task[cu] = &task[cu];
            streams.activation[cu] = &activation[cu];
            streams.output[cu] = &output[cu];
            for (unsigned int port = 0; port < 4; ++port) {
                streams.weight[cu][port] = &weight[cu][port];
                weight_memory[cu][port].resize(words);
                memories.weight[cu][port] = weight_memory[cu][port].data();
            }
        }
    }
};

template <typename StreamT>
unsigned int drain(StreamT& stream) {
    unsigned int count = 0;
    while (!stream.empty()) {
        (void)stream.read();
        ++count;
    }
    return count;
}

}  // namespace

int main() {
    static_assert(HIDDEN_SIZE == 64, "runtime TB uses QWEN_TEST_SMALL");
    static_assert(INTERMEDIATE_SIZE == 128, "runtime TB uses QWEN_TEST_SMALL");
    runtime_fixture_t fixture;
    const quantized_layer_task_t layer_task =
        make_quantized_prefill_block_task(
            0, 0, 2, 0, 0, 1, QUANTIZED_ACTIVATION_INT8,
            QUANTIZED_WEIGHT_INT8, 0, 0, MAX_SEQ_LEN,
            QUANTIZED_LAYER_W8_TOKEN_BLOCK);

    preload_quantized_layer_protocol_outputs<quantized_w8_projection_policy>(
        fixture.output, layer_task, true);

    std::vector<mm_input_block_t> hidden_input(
        2 * QUANTIZED_W8_HIDDEN_BLOCKS);
    std::vector<mm_input_block_t> hidden_output(
        2 * QUANTIZED_W8_HIDDEN_BLOCKS);
    for (unsigned int token = 0; token < 2; ++token) {
        for (unsigned int element = 0; element < HIDDEN_SIZE; ++element) {
            mm_input_block_t& word = hidden_input[
                quantized_w8_hidden_word_index(
                    token, element / QUANTIZED_W8_RESIDENT_LANES_PER_WORD)];
            set_mm_input_block_lane(
                word, element % QUANTIZED_W8_RESIDENT_LANES_PER_WORD,
                fm_t(token + 1));
        }
    }
    std::vector<mm_input_block_t> key_cache(quantized_w8_kv_cache_words());
    std::vector<mm_input_block_t> value_cache(quantized_w8_kv_cache_words());
    std::vector<mm_input_block_t> norm_memory(
        2 * QUANTIZED_W8_HIDDEN_BLOCKS);
    for (unsigned int norm = 0; norm < 2; ++norm) {
        for (unsigned int element = 0; element < HIDDEN_SIZE; ++element) {
            mm_input_block_t& word = norm_memory[
                quantized_w8_norm_word_index(
                    0, norm,
                    element / QUANTIZED_W8_RESIDENT_LANES_PER_WORD)];
            set_mm_input_block_lane(
                word, element % QUANTIZED_W8_RESIDENT_LANES_PER_WORD,
                fm_t(1));
        }
    }
    std::vector<mm_input_block_t> rope_memory(
        2 * 2 * QUANTIZED_W8_ROPE_BLOCKS);
    for (unsigned int position = 0; position < 2; ++position) {
        for (unsigned int element = 0; element < HEAD_DIM / 2; ++element) {
            mm_input_block_t& cosine = rope_memory[
                quantized_w8_rope_word_index(
                    position, 0,
                    element / QUANTIZED_W8_RESIDENT_LANES_PER_WORD)];
            set_mm_input_block_lane(
                cosine, element % QUANTIZED_W8_RESIDENT_LANES_PER_WORD,
                fm_t(1));
        }
    }
    quantized_w8_scale_word_t scale_memory[1]{};
    for (unsigned int projection = 0;
         projection < QUANTIZED_PROJECTION_COUNT; ++projection) {
        const quant_scale_t one = 1;
        scale_memory[0].range((projection + 1) * 16 - 1, projection * 16) =
            one.range(15, 0);
    }

    run_quantized_w8_layer_block(
        layer_task, hidden_input.data(), hidden_output.data(),
        key_cache.data(), value_cache.data(), norm_memory.data(),
        rope_memory.data(), scale_memory, fixture.streams, fixture.memories);

    for (unsigned int token = 0; token < 2; ++token) {
        for (unsigned int element = 0; element < HIDDEN_SIZE; ++element) {
            assert(unpack_mm_input_block_lane(
                       hidden_output[quantized_w8_hidden_word_index(
                           token,
                           element / QUANTIZED_W8_RESIDENT_LANES_PER_WORD)],
                       element % QUANTIZED_W8_RESIDENT_LANES_PER_WORD) ==
                   fm_t(token + 1));
        }
    }
    for (unsigned int cu = 0; cu < 4; ++cu) {
        const unsigned int expected_tasks =
            quantized_projection_tasks_per_block_for_cu(cu) +
            quantized_w8_attention_tasks_for_block_per_cu(layer_task, cu) +
            quantized_vector_layer_tasks_for_cu(
                4, layer_task.query_tokens, cu);
        unsigned int observed_tasks = 0;
        unsigned int qk_tasks = 0;
        unsigned int pv_tasks = 0;
        while (!fixture.task[cu].empty()) {
            const mm_stream_quantized_task_t task =
                unpack_mm_stream_quantized_task(fixture.task[cu].read());
            ++observed_tasks;
            qk_tasks += task.compute_mode ==
                MM_STREAM_QUANTIZED_MODE_ATTENTION_QK;
            pv_tasks += task.compute_mode ==
                MM_STREAM_QUANTIZED_MODE_ATTENTION_PV;
        }
        assert(observed_tasks == expected_tasks);
        assert(qk_tasks == 1 && pv_tasks == 1);
        assert(fixture.output[cu].empty());
        (void)drain(fixture.vector[cu][0]);
        (void)drain(fixture.vector[cu][1]);
        (void)drain(fixture.activation[cu]);
        for (unsigned int port = 0; port < 4; ++port) {
            (void)drain(fixture.weight[cu][port]);
        }
    }

    std::cout << "QUANTIZED W8 LAYER RUNTIME PASS seeded_protocol_only=1 "
              << "tokens=2 projections=7 attention_tiles=1 "
              << "hidden_resident=1 kv_resident=1\n";
    return 0;
}
