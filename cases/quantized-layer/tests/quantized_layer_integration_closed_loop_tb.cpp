#include "quantized_layer_integration_closed_loop.hpp"

#include "compute_core_quantized_w4_unified.hpp"
#include "quantized_layer_test_io.hpp"
#include "quantized_w4_attention_schedule.hpp"
#include "quantized_w4_projection_engine.hpp"

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef QUANTIZED_CSIM_FEEDBACK
#error This closed-loop fixture requires QUANTIZED_CSIM_FEEDBACK for C simulation
#endif

// The CSim reference links the production controller object. In RTL CoSim
// the host testbench links only the generated adapter, so leave that optional
// reference weak and never enter it when the replay marker is set by Tcl.
#if defined(__GNUC__)
#pragma weak control_cache_quantized_w4_layer
#endif

namespace {

constexpr unsigned int kCus = QUANTIZED_LAYER_COMPUTE_CUS;
constexpr unsigned int kWeightPorts = QUANTIZED_W4_WEIGHT_STREAMS_PER_CU;
constexpr unsigned int kHiddenWords = QUANTIZED_W4_HIDDEN_MEMORY_DEPTH;
constexpr unsigned int kKvWords = QUANTIZED_W4_KV_MEMORY_DEPTH;
constexpr unsigned int kNormWords = QUANTIZED_W4_NORM_MEMORY_DEPTH;
constexpr unsigned int kRopeWords = QUANTIZED_W4_ROPE_MEMORY_DEPTH;
constexpr unsigned int kWeightWords = QUANTIZED_W4_WEIGHT_MEMORY_DEPTH;
constexpr unsigned int kBlockSize = QUANTIZED_LAYER_W4_TOKEN_BLOCK;

struct stream_fixture_t {
    hls::stream<mm_stream_quantized_task_word_t> task[kCus];
    hls::stream<quantized_w4_activation_word_t> activation[kCus];
    hls::stream<quantized_w4_weight_word_t> weight[kCus][kWeightPorts];
    hls::stream<quantized_w4_output_word_t> output[kCus];
    hls::stream<quantized_vector_word_t> vector[kCus][2];
};

struct memory_fixture_t {
    std::vector<mm_input_block_t> hidden_input;
    std::vector<mm_input_block_t> hidden_output;
    std::vector<mm_input_block_t> key_cache;
    std::vector<mm_input_block_t> value_cache;
    std::vector<mm_input_block_t> norm_memory;
    std::vector<mm_input_block_t> rope_memory;
    std::vector<quantized_w4_scale_word_t> scale_memory;
    std::array<std::array<std::vector<quantized_w4_weight_word_t>, kWeightPorts>, kCus>
        weight_memory;

    memory_fixture_t()
        : hidden_input(kHiddenWords), hidden_output(kHiddenWords),
          key_cache(kKvWords), value_cache(kKvWords),
          norm_memory(kNormWords), rope_memory(kRopeWords),
          scale_memory(NUM_LAYERS) {
        for (auto& cu : weight_memory)
            for (auto& port : cu)
                port.resize(kWeightWords);
        reset();
    }

    void reset() {
        quantized_layer_test_io::fill_hidden(hidden_input, MAX_SEQ_LEN, 3);
        quantized_layer_test_io::fill_norm(norm_memory);
        quantized_layer_test_io::fill_rope(rope_memory, MAX_SEQ_LEN);
        quantized_layer_test_io::fill_random_weights<4>(weight_memory);
        std::fill(hidden_output.begin(), hidden_output.end(), mm_input_block_t(0));
        std::fill(key_cache.begin(), key_cache.end(), mm_input_block_t(0));
        std::fill(value_cache.begin(), value_cache.end(), mm_input_block_t(0));
        std::fill(scale_memory.begin(), scale_memory.end(),
                  quantized_w4_scale_word_t(0));
    }
};

struct snapshot_t {
    std::string label;
    std::vector<mm_input_block_t> hidden;
    std::vector<mm_input_block_t> key;
    std::vector<mm_input_block_t> value;
};

unsigned int feedback_calls[kCus] = {};

void fail(const std::string& message) {
    throw std::runtime_error("quantized layer integration: " + message);
}

// The controller requests a result only after it has emitted the matching
// task/operands. Consume exactly one task and invoke the production CU; the
// RTL branch below runs the same CU concurrently through HLS dataflow.
void quantized_csim_feedback_impl(
    quantized_w4_projection_streams_t& streams, unsigned int cu) {
    if (cu >= kCus || streams.task[cu]->empty())
        fail("feedback requested without a task");

    const mm_stream_quantized_task_word_t packed = streams.task[cu]->read();
    hls::stream<mm_stream_quantized_task_word_t> one_task;
    one_task.write(packed);
    hls::stream<quantized_vector_word_t>* vectors[kCus][2] = {
        {streams.vector.input0a, streams.vector.input0b},
        {streams.vector.input1a, streams.vector.input1b},
        {streams.vector.input2a, streams.vector.input2b},
        {streams.vector.input3a, streams.vector.input3b}};

    compute_core_quantized_w4_unified_nk(
        *streams.output[cu], one_task, *streams.activation[cu],
        *streams.weight[cu][0], *streams.weight[cu][1],
        *vectors[cu][0], *vectors[cu][1], 1);
    ++feedback_calls[cu];
    if (streams.output[cu]->empty())
        fail("real W4 compute produced no result");
}

} // namespace

void quantized_csim_feedback(
    quantized_w4_projection_streams_t& streams, unsigned int cu) {
    quantized_csim_feedback_impl(streams, cu);
}

namespace {

void clear_feedback_calls() {
    for (unsigned int cu = 0; cu < kCus; ++cu)
        feedback_calls[cu] = 0;
}

bool rtl_replay_enabled() {
    const char* value = std::getenv("QUANTIZED_LAYER_RTL_REPLAY");
    return value != nullptr && std::string(value) == "1";
}

void check_feedback_calls(unsigned int sequence_length,
                          unsigned int request_position,
                          unsigned int request_op,
                          const char* label) {
    const bool decode = request_op == QUANTIZED_LAYER_OP_DECODE;
    for (unsigned int cu = 0; cu < kCus; ++cu) {
        const unsigned int expected = quantized_w4_layer_tasks_for_cu(
            sequence_length, request_position, cu, decode, kBlockSize);
        if (feedback_calls[cu] != expected) {
            fail(std::string(label) + " feedback count mismatch cu=" +
                 std::to_string(cu) + " expected=" + std::to_string(expected) +
                 " actual=" + std::to_string(feedback_calls[cu]));
        }
    }
}

void check_controller_streams(stream_fixture_t& streams,
                              const char* label) {
    for (unsigned int cu = 0; cu < kCus; ++cu) {
        if (!streams.task[cu].empty() || !streams.activation[cu].empty() ||
            !streams.output[cu].empty() || !streams.vector[cu][0].empty() ||
            !streams.vector[cu][1].empty()) {
            fail(std::string(label) + " left task/operand/result streams");
        }
        for (unsigned int port = 0; port < kWeightPorts; ++port)
            if (!streams.weight[cu][port].empty())
                fail(std::string(label) + " left weight stream data");
    }
}

void invoke_controller(memory_fixture_t& memory,
                       unsigned int sequence_length,
                       unsigned int request_position,
                       unsigned int kv_context_length,
                       unsigned int request_op) {
    std::cout << "QLI_PHASE_START path=reference rows=" << sequence_length
              << " position=" << request_position << " op=" << request_op << std::endl;
    stream_fixture_t streams;
    clear_feedback_calls();
    control_cache_quantized_w4_layer(
        streams.task[0], streams.task[1], streams.task[2], streams.task[3],
        streams.activation[0], streams.activation[1],
        streams.activation[2], streams.activation[3],
        streams.weight[0][0], streams.weight[0][1],
        streams.weight[1][0], streams.weight[1][1],
        streams.weight[2][0], streams.weight[2][1],
        streams.weight[3][0], streams.weight[3][1],
        streams.output[0], streams.output[1],
        streams.output[2], streams.output[3],
        streams.vector[0][0], streams.vector[0][1],
        streams.vector[1][0], streams.vector[1][1],
        streams.vector[2][0], streams.vector[2][1],
        streams.vector[3][0], streams.vector[3][1],
        memory.hidden_input.data(), memory.hidden_output.data(),
        memory.key_cache.data(), memory.value_cache.data(),
        memory.norm_memory.data(), memory.rope_memory.data(),
        memory.scale_memory.data(),
        memory.weight_memory[0][0].data(), memory.weight_memory[0][1].data(),
        memory.weight_memory[1][0].data(), memory.weight_memory[1][1].data(),
        memory.weight_memory[2][0].data(), memory.weight_memory[2][1].data(),
        memory.weight_memory[3][0].data(), memory.weight_memory[3][1].data(),
        0, sequence_length, request_position, kv_context_length, request_op,
        kBlockSize);
    check_feedback_calls(sequence_length, request_position, request_op,
                          "controller reference");
    check_controller_streams(streams, "controller reference");
}

void invoke_dut(memory_fixture_t& memory,
                unsigned int sequence_length,
                unsigned int request_position,
                unsigned int kv_context_length,
                unsigned int request_op) {
    std::cout << "QLI_PHASE_START path=dut rows=" << sequence_length
              << " position=" << request_position << " op=" << request_op << std::endl;
    clear_feedback_calls();
    const unsigned int status = quantized_layer_integration_closed_loop(
        memory.hidden_input.data(), memory.hidden_output.data(),
        memory.key_cache.data(), memory.value_cache.data(),
        memory.norm_memory.data(), memory.rope_memory.data(),
        memory.scale_memory.data(),
        memory.weight_memory[0][0].data(), memory.weight_memory[0][1].data(),
        memory.weight_memory[1][0].data(), memory.weight_memory[1][1].data(),
        memory.weight_memory[2][0].data(), memory.weight_memory[2][1].data(),
        memory.weight_memory[3][0].data(), memory.weight_memory[3][1].data(),
        0, sequence_length, request_position, kv_context_length, request_op,
        kBlockSize);
    if (status != 0)
        fail("DUT returned nonzero status");
    // RTL CoSim executes the four synthesized CUs, not the C-model feedback
    // hook. Keep the full tensor comparison active there, but only require
    // callback accounting for the standalone C model.
    if (!rtl_replay_enabled()) {
        check_feedback_calls(sequence_length, request_position, request_op,
                             "DUT C model");
    }
}

snapshot_t capture(const memory_fixture_t& memory, const char* label,
                   unsigned int hidden_tokens, unsigned int kv_positions) {
    snapshot_t snapshot;
    snapshot.label = label;
    const std::size_t hidden_words =
        std::size_t(hidden_tokens) * QUANTIZED_W4_HIDDEN_BLOCKS;
    const std::size_t kv_words =
        std::size_t(kv_positions) * QUANTIZED_W4_KV_BLOCKS;
    snapshot.hidden.assign(memory.hidden_output.begin(),
                            memory.hidden_output.begin() + hidden_words);
    snapshot.key.assign(memory.key_cache.begin(),
                        memory.key_cache.begin() + kv_words);
    snapshot.value.assign(memory.value_cache.begin(),
                          memory.value_cache.begin() + kv_words);
    return snapshot;
}

std::vector<snapshot_t> run_suite(bool use_dut) {
    memory_fixture_t memory;
    std::vector<snapshot_t> snapshots;

    if (use_dut)
        invoke_dut(memory, 10, 0, 0, QUANTIZED_LAYER_OP_PREFILL);
    else
        invoke_controller(memory, 10, 0, 0, QUANTIZED_LAYER_OP_PREFILL);
    snapshots.push_back(capture(memory, "prefill_p10_tail2", 10, 10));

    // Decode source rows are deliberately independent from the prefill
    // output. KV state remains live so the D1 calls test append and reuse.
    quantized_layer_test_io::fill_hidden(memory.hidden_input, 1, 19);
    if (use_dut)
        invoke_dut(memory, 1, 10, 10, QUANTIZED_LAYER_OP_DECODE);
    else
        invoke_controller(memory, 1, 10, 10, QUANTIZED_LAYER_OP_DECODE);
    snapshots.push_back(capture(memory, "decode_d1_pos10", 1, 11));

    if (use_dut)
        invoke_dut(memory, 1, 11, 11, QUANTIZED_LAYER_OP_DECODE);
    else
        invoke_controller(memory, 1, 11, 11, QUANTIZED_LAYER_OP_DECODE);
    snapshots.push_back(capture(memory, "decode_d1_repeat_pos11", 1, 12));
    return snapshots;
}

std::string word_to_hex(const mm_input_block_t& word) {
    return word.to_string(16);
}

mm_input_block_t parse_hex_word(const std::string& text) {
    mm_input_block_t word = 0;
    for (char character : text) {
        if (character == 'x' || character == 'X')
            continue;
        unsigned int digit = 0;
        if (character >= '0' && character <= '9') digit = character - '0';
        else if (character >= 'a' && character <= 'f') digit = character - 'a' + 10;
        else if (character >= 'A' && character <= 'F') digit = character - 'A' + 10;
        else fail("invalid expected-dump hexadecimal word");
        word = (word << 4) | digit;
    }
    return word;
}

void write_expected(const std::string& path,
                    const std::vector<snapshot_t>& snapshots) {
    std::ofstream output(path.c_str());
    if (!output) fail("cannot create expected dump: " + path);
    output << "QLI_EXPECTED_V1 " << snapshots.size() << '\n';
    for (const auto& snapshot : snapshots) {
        output << "snapshot " << snapshot.label << ' '
               << snapshot.hidden.size() << ' ' << snapshot.key.size() << '\n';
        for (const auto& word : snapshot.hidden)
            output << "H " << word_to_hex(word) << '\n';
        for (const auto& word : snapshot.key)
            output << "K " << word_to_hex(word) << '\n';
        for (const auto& word : snapshot.value)
            output << "V " << word_to_hex(word) << '\n';
    }
    if (!output) fail("failed writing expected dump: " + path);
}

std::vector<snapshot_t> read_expected(const std::string& path) {
    std::ifstream input(path.c_str());
    if (!input) fail("cannot read expected dump: " + path);
    std::string magic;
    std::size_t snapshot_count = 0;
    if (!(input >> magic >> snapshot_count) || magic != "QLI_EXPECTED_V1")
        fail("invalid expected dump header");
    std::vector<snapshot_t> snapshots(snapshot_count);
    for (auto& snapshot : snapshots) {
        std::string tag;
        std::size_t hidden_count = 0, kv_count = 0;
        if (!(input >> tag >> snapshot.label >> hidden_count >> kv_count) ||
            tag != "snapshot")
            fail("invalid expected snapshot header");
        snapshot.hidden.resize(hidden_count);
        snapshot.key.resize(kv_count);
        snapshot.value.resize(kv_count);
        for (auto& word : snapshot.hidden) {
            std::string word_tag, text;
            if (!(input >> word_tag >> text) || word_tag != "H")
                fail("invalid expected hidden word");
            word = parse_hex_word(text);
        }
        for (auto& word : snapshot.key) {
            std::string word_tag, text;
            if (!(input >> word_tag >> text) || word_tag != "K")
                fail("invalid expected key word");
            word = parse_hex_word(text);
        }
        for (auto& word : snapshot.value) {
            std::string word_tag, text;
            if (!(input >> word_tag >> text) || word_tag != "V")
                fail("invalid expected value word");
            word = parse_hex_word(text);
        }
    }
    return snapshots;
}

void compare_words(const std::vector<mm_input_block_t>& expected,
                   const std::vector<mm_input_block_t>& actual,
                   const std::string& label, const char* tensor) {
    if (expected.size() != actual.size())
        fail(label + " " + tensor + " length mismatch");
    for (std::size_t index = 0; index < expected.size(); ++index)
        if (expected[index] != actual[index]) {
            fail(label + " " + tensor + " mismatch at word " +
                 std::to_string(index) + " expected=" +
                 word_to_hex(expected[index]) + " actual=" +
                 word_to_hex(actual[index]));
        }
}

void compare_suites(const std::vector<snapshot_t>& expected,
                    const std::vector<snapshot_t>& actual) {
    if (expected.size() != actual.size()) fail("snapshot count mismatch");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (expected[index].label != actual[index].label)
            fail("snapshot label mismatch at index " + std::to_string(index));
        compare_words(expected[index].hidden, actual[index].hidden,
                      expected[index].label, "hidden");
        compare_words(expected[index].key, actual[index].key,
                      expected[index].label, "key");
        compare_words(expected[index].value, actual[index].value,
                      expected[index].label, "value");
    }
}

bool file_exists(const std::string& path) {
    std::ifstream input(path.c_str());
    return input.good();
}

} // namespace

int main() try {
    const char* environment_path = std::getenv("QUANTIZED_LAYER_EXPECTED_DUMP");
    const std::string expected_path = environment_path == nullptr ? "" :
        std::string(environment_path);

    std::vector<snapshot_t> expected;
    const bool have_expected = !expected_path.empty() && file_exists(expected_path);
    if (!have_expected) {
        expected = run_suite(false);
        if (!expected_path.empty()) write_expected(expected_path, expected);
        std::cout << "QLI_REFERENCE_GENERATED snapshots=" << expected.size()
                  << " path=" << (expected_path.empty() ? "<memory>" : expected_path)
                  << " real_unified_compute=1 seeded_outputs=0\n";
    } else {
        expected = read_expected(expected_path);
    }

    const std::vector<snapshot_t> actual = run_suite(true);
    compare_suites(expected, actual);
    std::cout << "QLI_TRANSACTION caseID=P10_tail2_then_D1_repeat status=PASS"
              << " hidden=H" << HIDDEN_SIZE << " intermediate=I" << INTERMEDIATE_SIZE
              << " seq_capacity=" << MAX_SEQ_LEN << " block_size=" << kBlockSize
              << " cus=4 exact_snapshots=" << actual.size()
              << " seeded_outputs=0 deadlock_detection=enabled\n";
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << "QLI FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
}
