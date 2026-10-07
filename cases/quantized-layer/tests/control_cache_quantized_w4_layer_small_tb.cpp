#include "control_cache_quantized_w4_layer.hpp"
#include "quantized_layer_protocol_fixture.hpp"

#include <cstdlib>
#include <iostream>

namespace {
using task_stream_t = hls::stream<mm_stream_quantized_task_word_t>;
using activation_stream_t = hls::stream<quantized_w4_activation_word_t>;
using weight_stream_t = hls::stream<quantized_w4_weight_word_t>;
using output_stream_t = hls::stream<quantized_w4_output_word_t>;

constexpr unsigned int kCus = QUANTIZED_LAYER_COMPUTE_CUS;
constexpr unsigned int kHiddenWords = MAX_SEQ_LEN * QUANTIZED_W4_HIDDEN_BLOCKS;
constexpr unsigned int kKvWords = NUM_LAYERS * MAX_SEQ_LEN *
    QUANTIZED_W4_KV_BLOCKS;
constexpr unsigned int kNormWords = NUM_LAYERS * 2 *
    QUANTIZED_W4_HIDDEN_BLOCKS;
constexpr unsigned int kRopeWords = MAX_SEQ_LEN * 2 * QUANTIZED_W4_ROPE_BLOCKS;
constexpr unsigned int kWeightWords = QUANTIZED_W4_WEIGHT_MEMORY_DEPTH;
constexpr unsigned int kOutputWordsPerTask =
    MM_STREAM_8X128_INT4X4_TOKENS * MM_STREAM_8X128_INT4X4_OUTPUT_GROUPS;

void fail(const char* message) {
    std::cerr << "Q4 full-layer CoSim FAIL: " << message << "\n";
    std::exit(EXIT_FAILURE);
}

void check_request(
    task_stream_t tasks[kCus],
    activation_stream_t activations[kCus],
    weight_stream_t weights[kCus][QUANTIZED_W4_WEIGHT_STREAMS_PER_CU],
    output_stream_t outputs[kCus],
    hls::stream<quantized_vector_word_t> vectors[kCus][2],
    unsigned int sequence_length,
    unsigned int request_position,
    unsigned int expected_phase,
    unsigned int block_size) {
    for (unsigned int cu = 0; cu < kCus; ++cu) {
        const unsigned int expected_tasks = quantized_w4_layer_tasks_for_cu(
            sequence_length, request_position, cu,
            expected_phase == QUANTIZED_LAYER_PHASE_DECODE, block_size);
        if (tasks[cu].size() != expected_tasks || !outputs[cu].empty()) {
            fail("task or output stream count mismatch");
        }

        unsigned int expected_words = 0;
        unsigned int expected_vector0 = 0, expected_vector1 = 0;
        unsigned int valid_token_histogram[QUANTIZED_LAYER_MAX_TOKEN_BLOCK + 1] = {};
        for (unsigned int index = 0; index < expected_tasks; ++index) {
            const mm_stream_quantized_task_t task =
                unpack_mm_stream_quantized_task(tasks[cu].read());
            const bool position_aligned = task.request_position >=
                    request_position &&
                task.request_position < request_position + sequence_length &&
                ((task.request_position - request_position) % block_size) == 0;
            const unsigned int token_offset =
                task.request_position - request_position;
            const unsigned int remaining_tokens = sequence_length - token_offset;
            const unsigned int block_valid_tokens =
                remaining_tokens < block_size ? remaining_tokens : block_size;
            const bool is_vector = task.compute_mode >= MM_STREAM_QUANTIZED_MODE_RMSNORM;
            const bool is_decode_attention = expected_phase == QUANTIZED_LAYER_PHASE_DECODE &&
                (task.compute_mode == MM_STREAM_QUANTIZED_MODE_ATTENTION_QK ||
                 task.compute_mode == MM_STREAM_QUANTIZED_MODE_ATTENTION_PV);
            const unsigned int vector_elements =
                task.compute_mode == MM_STREAM_QUANTIZED_MODE_SILU_MUL ?
                    INTERMEDIATE_SIZE : HIDDEN_SIZE;
            const unsigned int expected_valid_tokens = is_vector ?
                quantized_vector_task_rows(
                    8, block_valid_tokens, cu, task.compute_mode,
                    vector_elements) :
                (is_decode_attention ? (GQA_GROUP_SIZE < 8 ? GQA_GROUP_SIZE : 8) : block_valid_tokens);
            const unsigned int expected_k_count = is_vector &&
                    quantized_vector_split_columns(
                        block_valid_tokens, task.compute_mode) ?
                quantized_vector_blocks_for_cu(vector_elements, cu) *
                    CU_VEC_LANES : vector_elements;
            if (task.phase != expected_phase || !position_aligned ||
                task.valid_tokens != expected_valid_tokens ||
                task.k_count == 0 ||
                (is_vector && task.k_count != expected_k_count)) {
                std::cerr << "task mismatch cu=" << cu << " index=" << index
                          << " phase=" << task.phase
                          << " request_position=" << task.request_position
                          << " valid_tokens=" << task.valid_tokens
                          << " k_count=" << task.k_count
                          << " expected_phase=" << expected_phase
                          << " block_size=" << block_size << "\n";
                fail("task semantic metadata mismatch");
            }
            // Attention rows denote heads in D1; other rows denote tokens.
            ++valid_token_histogram[is_decode_attention ? 1 : task.valid_tokens];
            if (is_vector) {
                const unsigned int packets = ceildiv(task.k_count, CU_VEC_LANES);
                expected_vector0 += task.valid_tokens * packets;
                expected_vector1 += task.compute_mode == MM_STREAM_QUANTIZED_MODE_RMSNORM ?
                    packets : task.valid_tokens * packets;
            } else expected_words += task.k_count;
        }
        if (expected_phase == QUANTIZED_LAYER_PHASE_DECODE) {
            if (valid_token_histogram[1] != expected_tasks) {
                fail("decode token histogram mismatch");
            }
        }

        if (vectors[cu][0].size() != expected_vector0 ||
            vectors[cu][1].size() != expected_vector1 || !outputs[cu].empty())
            fail("vector operand/result count mismatch");
        while (!vectors[cu][0].empty()) vectors[cu][0].read();
        while (!vectors[cu][1].empty()) vectors[cu][1].read();
        if (activations[cu].size() != expected_words) {
            fail("activation stream count mismatch");
        }
        while (!activations[cu].empty()) activations[cu].read();
        for (unsigned int port = 0;
             port < QUANTIZED_W4_WEIGHT_STREAMS_PER_CU; ++port) {
            if (weights[cu][port].size() != expected_words) {
                fail("weight stream count mismatch");
            }
            while (!weights[cu][port].empty()) weights[cu][port].read();
        }
    }
}
}  // namespace

int main() {
    std::cerr << "CONTROL Q4 FULL-LAYER COSIM TB START\n";
    task_stream_t tasks[kCus];
    activation_stream_t activations[kCus];
    weight_stream_t weights[kCus][QUANTIZED_W4_WEIGHT_STREAMS_PER_CU];
    output_stream_t outputs[kCus];
    hls::stream<quantized_vector_word_t> vectors[kCus][2];

    static mm_input_block_t hidden_input[kHiddenWords] = {};
    static mm_input_block_t hidden_output[kHiddenWords] = {};
    static mm_input_block_t key_cache[kKvWords] = {};
    static mm_input_block_t value_cache[kKvWords] = {};
    static mm_input_block_t norm_memory[kNormWords] = {};
    static mm_input_block_t rope_memory[kRopeWords] = {};
    static quantized_w4_scale_word_t scale_memory[NUM_LAYERS] = {};
    static quantized_w4_weight_word_t weight_memory[kCus]
        [QUANTIZED_W4_WEIGHT_STREAMS_PER_CU][kWeightWords] = {};

    constexpr unsigned int kPrefillTokens = 18;
    constexpr unsigned int kPrefillBlockSize = 8;
    for (unsigned int block = 0; block < quantized_layer_block_count(kPrefillTokens, kPrefillBlockSize); ++block) {
        const auto task = make_quantized_prefill_block_task(
            0, 0, kPrefillTokens, block, 0, 1, QUANTIZED_ACTIVATION_INT4,
            QUANTIZED_WEIGHT_INT4, 0, 0, MAX_SEQ_LEN, kPrefillBlockSize);
        preload_quantized_layer_protocol_outputs<quantized_w4_projection_policy>(outputs, task);
    }
    control_cache_quantized_w4_layer(
        tasks[0], tasks[1], tasks[2], tasks[3], activations[0], activations[1],
        activations[2], activations[3], weights[0][0], weights[0][1],
        weights[1][0], weights[1][1], weights[2][0], weights[2][1],
        weights[3][0], weights[3][1], outputs[0], outputs[1], outputs[2],
        outputs[3], vectors[0][0], vectors[0][1], vectors[1][0], vectors[1][1],
        vectors[2][0], vectors[2][1], vectors[3][0], vectors[3][1],
        hidden_input, hidden_output, key_cache, value_cache,
        norm_memory, rope_memory, scale_memory, weight_memory[0][0],
        weight_memory[0][1], weight_memory[1][0], weight_memory[1][1],
        weight_memory[2][0], weight_memory[2][1], weight_memory[3][0],
        weight_memory[3][1], 0,
        kPrefillTokens, 0, 0, QUANTIZED_LAYER_OP_PREFILL,
        kPrefillBlockSize);
    check_request(tasks, activations, weights, outputs, vectors, kPrefillTokens, 0,
                  QUANTIZED_LAYER_PHASE_PREFILL, kPrefillBlockSize);

    std::cout << "CONTROL Q4 FULL-LAYER COSIM TB PASS seeded_protocol_only=1 prefill_tokens="
              << kPrefillTokens << " block_size=" << kPrefillBlockSize
              << " tail_tokens=2 cus=4 "
              << "deadlock_detection=enabled\n";
    return EXIT_SUCCESS;
}
