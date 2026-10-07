#ifndef LLM_FPGA_QUANTIZED_LAYER_TEST_IO_HPP
#define LLM_FPGA_QUANTIZED_LAYER_TEST_IO_HPP

// Host/test utilities only; no accelerator-side dependency or arithmetic.
#include "hardware.hpp"
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>

namespace quantized_layer_test_io {
template <typename Buffer>
void fill_hidden(Buffer& hidden, unsigned int tokens, unsigned int salt) {
    std::fill(hidden.begin(), hidden.end(), mm_input_block_t(0));
    const unsigned int blocks = (HIDDEN_SIZE + 15) / 16;
    for (unsigned int token = 0; token < tokens; ++token)
        for (unsigned int block = 0; block < blocks; ++block) {
            mm_input_block_t word = 0;
            for (unsigned int lane = 0; lane < 16; ++lane) {
                const unsigned int element = block * 16 + lane;
                const int raw = int((token * 13 + element * 7 + salt) % 31) - 15;
                set_mm_input_block_lane(word, lane,
                    element < HIDDEN_SIZE ? fm_t(fm_t(raw) / fm_t(16)) : fm_t(0));
            }
            hidden[std::size_t(token) * blocks + block] = word;
        }
}

template <typename Buffer>
void fill_norm(Buffer& norm) {
    for (auto& word : norm) {
        word = 0;
        for (unsigned int lane = 0; lane < 16; ++lane)
            set_mm_input_block_lane(word, lane, fm_t(1));
    }
}

template <typename Buffer>
void fill_rope(Buffer& rope, unsigned int positions) {
    std::fill(rope.begin(), rope.end(), mm_input_block_t(0));
    const unsigned int blocks = (HEAD_DIM / 2 + 15) / 16;
    for (unsigned int position = 0; position < positions; ++position)
        for (unsigned int block = 0; block < blocks; ++block) {
            mm_input_block_t word = 0;
            for (unsigned int lane = 0; lane < 16; ++lane)
                set_mm_input_block_lane(word, lane, fm_t(1));
            rope[std::size_t(position) * 2 * blocks + block] = word;
        }
}

template <unsigned int Bits, typename Weights>
void fill_random_weights(Weights& weights) {
    static_assert(Bits == 4 || Bits == 8, "Supported quantized formats");
    for (unsigned int cu = 0; cu < weights.size(); ++cu)
        for (unsigned int port = 0; port < weights[cu].size(); ++port)
            for (std::size_t index = 0; index < weights[cu][port].size(); ++index) {
                auto& word = weights[cu][port][index];
                word = 0;
                for (unsigned int lane = 0; lane < 256 / Bits; ++lane) {
                    const int value = int((index * 3 + lane * 5 + cu * 7 + port * 11) %
                        ((1u << Bits) - 1)) - ((1 << (Bits - 1)) - 1);
                    word.range(lane * Bits + Bits - 1, lane * Bits) = ap_int<Bits>(value);
                }
            }
}

inline void dump_header(std::ostream& out, unsigned int bits, unsigned int prefill,
                        unsigned int block, const std::string& weights) {
    out << "quantized_layer_dump_v1\n"
        << "bits=" << bits << " hidden=" << HIDDEN_SIZE << " ffn=" << INTERMEDIATE_SIZE
        << " q_heads=" << NUM_ATTENTION_HEADS << " kv_heads=" << NUM_KEY_VALUE_HEADS
        << " head_dim=" << HEAD_DIM << " prefill=" << prefill << " block=" << block
        << " weights=" << weights << " fixture=deterministic_v1 layer=0\n";
}

template <typename Buffer>
void dump_words(std::ostream& out, const char* tensor, const Buffer& buffer,
                std::size_t count) {
    if (count > buffer.size()) throw std::runtime_error("Dump exceeds tensor size");
    for (std::size_t word = 0; word < count; ++word)
        out << tensor << '\t' << word << '\t' << buffer[word].to_string(16) << '\n';
    out.flush();
    if (!out) throw std::runtime_error("Failed to write numerical output dump");
}
} // namespace quantized_layer_test_io
#endif
