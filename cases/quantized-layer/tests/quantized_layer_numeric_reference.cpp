// Complete layer P/D C model for post-HW-Emu numerical comparison. This runs
// production controller and actual unified compute C functions cooperatively;
// independent arithmetic tests remain separate, as does finite-FIFO RTL CoSim.
#ifndef QUANTIZED_CSIM_FEEDBACK
#error Build this C reference with QUANTIZED_CSIM_FEEDBACK enabled
#endif
#ifdef QUANTIZED_ALIGNMENT_W8
#include "control_cache_quantized_w8_layer.hpp"
#include "compute_core_quantized_w8_unified.hpp"
#define CONTROLLER control_cache_quantized_w8_layer
#define COMPUTE compute_core_quantized_w8_unified_nk
#define TASK_COUNT quantized_w8_layer_tasks_for_cu
using policy = quantized_w8_projection_policy;
constexpr unsigned int bits = 8;
#else
#include "control_cache_quantized_w4_layer.hpp"
#include "compute_core_quantized_w4_unified.hpp"
#define CONTROLLER control_cache_quantized_w4_layer
#define COMPUTE compute_core_quantized_w4_unified_nk
#define TASK_COUNT quantized_w4_layer_tasks_for_cu
using policy = quantized_w4_projection_policy;
constexpr unsigned int bits = 4;
#endif
#include "quantized_layer_test_io.hpp"
#include <array>
#include <iostream>
#include <vector>

static unsigned int compute_calls[4] = {};

void quantized_csim_feedback(policy::streams_t& streams, unsigned int cu) {
    if (cu >= 4 || streams.task[cu]->empty())
        throw std::runtime_error("Controller requested output without a compute task");
    auto& v = streams.vector;
    hls::stream<quantized_vector_word_t>* vectors[4][2] = {
        {v.input0a, v.input0b}, {v.input1a, v.input1b},
        {v.input2a, v.input2b}, {v.input3a, v.input3b}};
    COMPUTE(*streams.output[cu], *streams.task[cu], *streams.activation[cu],
        *streams.weight[cu][0], *streams.weight[cu][1],
#ifdef QUANTIZED_ALIGNMENT_W8
        *streams.weight[cu][2], *streams.weight[cu][3],
#endif
        *vectors[cu][0], *vectors[cu][1], 1);
    ++compute_calls[cu];
    if (streams.output[cu]->empty())
        throw std::runtime_error("Compute task produced no result");
}

int main(int argc, char** argv) try {
    unsigned int prefill = 10, block = policy::token_rows;
    std::string mode = "random", output_path;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::runtime_error("Missing option value");
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--output") output_path = value;
        else if (key == "--weights") mode = value;
        else if (key == "--prefill" || key == "--block-size") {
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error("Invalid integer option");
            const unsigned long number = std::stoul(value);
            if (number > MAX_SEQ_LEN) throw std::runtime_error("Option exceeds model capacity");
            (key == "--prefill" ? prefill : block) = number;
        } else throw std::runtime_error("Unknown option: " + key);
    }
    if (output_path.empty() || prefill == 0 || prefill >= MAX_SEQ_LEN ||
        block == 0 || block > policy::token_rows || (mode != "zero" && mode != "random"))
        throw std::runtime_error("Invalid output/workload parameters");
    std::ofstream out(output_path);
    if (!out) throw std::runtime_error("Cannot create reference dump");
    quantized_layer_test_io::dump_header(out, bits, prefill, block, mode);
    const std::size_t hidden_blocks = (HIDDEN_SIZE + 15) / 16;
    const std::size_t kv_blocks = (KV_CHANNELS + 15) / 16;
    const std::size_t hidden_words = prefill * hidden_blocks;
    const std::size_t kv_words = (prefill + 1) * kv_blocks;
    std::vector<mm_input_block_t> input(hidden_words), output(hidden_words),
        key(kv_words), value(kv_words), norm(2 * hidden_blocks),
        rope((prefill + 1) * 2 * ((HEAD_DIM / 2 + 15) / 16));
    std::vector<ap_uint<128>> scales(1, ap_uint<128>(0));
    std::array<std::array<std::vector<policy::weight_word_t>, policy::weight_ports>, 4> memory;
    for (auto& cu : memory)
        for (auto& port : cu) port.resize(quantized_layer_weight_words_per_shard());
    quantized_layer_test_io::fill_hidden(input, prefill, 3);
    quantized_layer_test_io::fill_norm(norm);
    quantized_layer_test_io::fill_rope(rope, prefill + 1);
    if (mode == "random") quantized_layer_test_io::fill_random_weights<bits>(memory);
    hls::stream<mm_stream_quantized_task_word_t> tasks[4];
    hls::stream<policy::activation_word_t> activation[4];
    hls::stream<policy::weight_word_t> weights[4][policy::weight_ports];
    hls::stream<policy::output_word_t> results[4];
    hls::stream<quantized_vector_word_t> vectors[4][2];
    auto run = [&](bool decode) {
        const unsigned int sequence = decode ? 1 : prefill;
        const unsigned int position = decode ? prefill : 0;
        std::fill(std::begin(compute_calls), std::end(compute_calls), 0);
        std::cout << "C_MODEL_PHASE_START bits=" << bits << " decode=" << decode
                  << " hidden=" << HIDDEN_SIZE << " prefill=" << prefill << std::endl;
        CONTROLLER(tasks[0], tasks[1], tasks[2], tasks[3],
            activation[0], activation[1], activation[2], activation[3],
            weights[0][0], weights[0][1],
#ifdef QUANTIZED_ALIGNMENT_W8
            weights[0][2], weights[0][3],
#endif
            weights[1][0], weights[1][1],
#ifdef QUANTIZED_ALIGNMENT_W8
            weights[1][2], weights[1][3],
#endif
            weights[2][0], weights[2][1],
#ifdef QUANTIZED_ALIGNMENT_W8
            weights[2][2], weights[2][3],
#endif
            weights[3][0], weights[3][1],
#ifdef QUANTIZED_ALIGNMENT_W8
            weights[3][2], weights[3][3],
#endif
            results[0], results[1], results[2], results[3],
            vectors[0][0], vectors[0][1], vectors[1][0], vectors[1][1],
            vectors[2][0], vectors[2][1], vectors[3][0], vectors[3][1],
            input.data(), output.data(), key.data(), value.data(), norm.data(),
            rope.data(), scales.data(), memory[0][0].data(), memory[0][1].data(),
#ifdef QUANTIZED_ALIGNMENT_W8
            memory[0][2].data(), memory[0][3].data(),
#endif
            memory[1][0].data(), memory[1][1].data(),
#ifdef QUANTIZED_ALIGNMENT_W8
            memory[1][2].data(), memory[1][3].data(),
#endif
            memory[2][0].data(), memory[2][1].data(),
#ifdef QUANTIZED_ALIGNMENT_W8
            memory[2][2].data(), memory[2][3].data(),
#endif
            memory[3][0].data(), memory[3][1].data(),
#ifdef QUANTIZED_ALIGNMENT_W8
            memory[3][2].data(), memory[3][3].data(),
#endif
            0, sequence, position, position,
            decode ? QUANTIZED_LAYER_OP_DECODE : QUANTIZED_LAYER_OP_PREFILL, block);
        for (unsigned int cu = 0; cu < 4; ++cu) {
            if (compute_calls[cu] != TASK_COUNT(sequence, position, cu, decode, block) ||
                !tasks[cu].empty() || !activation[cu].empty() || !results[cu].empty() ||
                !vectors[cu][0].empty() || !vectors[cu][1].empty())
                throw std::runtime_error("Full-layer feedback task/stream accounting failed");
            for (auto& port : weights[cu])
                if (!port.empty()) throw std::runtime_error("Unconsumed weight data");
        }
        if (mode == "zero") {
            for (std::size_t word = 0; word < sequence * hidden_blocks; ++word)
                if (input[word] != output[word]) throw std::runtime_error("Zero-weight residual mismatch");
            for (std::size_t word = 0; word < kv_words; ++word)
                if (key[word] != 0 || value[word] != 0) throw std::runtime_error("Zero-weight KV mismatch");
        }
        std::cout << "C_MODEL_PHASE_COMPLETE bits=" << bits << " decode=" << decode
                  << " tasks=" << compute_calls[0] << ',' << compute_calls[1] << ','
                  << compute_calls[2] << ',' << compute_calls[3] << std::endl;
    };
    run(false);
    quantized_layer_test_io::dump_words(out, "prefill_hidden", output, hidden_words);
    quantized_layer_test_io::fill_hidden(input, 1, 19);
    run(true);
    quantized_layer_test_io::dump_words(out, "decode_hidden", output, hidden_blocks);
    quantized_layer_test_io::dump_words(out, "key_cache", key, kv_words);
    quantized_layer_test_io::dump_words(out, "value_cache", value, kv_words);
    std::cout << "C_MODEL_REFERENCE_COMPLETE real_unified_compute=1 seeded_outputs=0"
                 " rtl_validation_pending=1\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "C model failed: " << error.what() << '\n';
    return 1;
}
