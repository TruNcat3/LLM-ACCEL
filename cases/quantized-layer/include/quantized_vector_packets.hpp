#ifndef LLM_FPGA_QUANTIZED_VECTOR_PACKETS_HPP
#define LLM_FPGA_QUANTIZED_VECTOR_PACKETS_HPP

#include "vitis_stream_8x64.hpp"
#include "mm_stream_quantized_nk.hpp"

using quantized_vector_word_t = cu8_nk_vector_word_t;

// Explicit members preserve independent AXIS ports in HLS 2022.2.
struct quantized_vector_streams_t {
    hls::stream<quantized_vector_word_t>* input0a;
    hls::stream<quantized_vector_word_t>* input0b;
    hls::stream<quantized_vector_word_t>* input1a;
    hls::stream<quantized_vector_word_t>* input1b;
    hls::stream<quantized_vector_word_t>* input2a;
    hls::stream<quantized_vector_word_t>* input2b;
    hls::stream<quantized_vector_word_t>* input3a;
    hls::stream<quantized_vector_word_t>* input3b;
};

inline void write_quantized_vector_operand(
    quantized_vector_streams_t& streams, unsigned int cu,
    unsigned int operand, const quantized_vector_word_t& word) {
    #pragma HLS inline
    if (cu == 0) {
        if (operand == 0) streams.input0a->write(word);
        else streams.input0b->write(word);
    } else if (cu == 1) {
        if (operand == 0) streams.input1a->write(word);
        else streams.input1b->write(word);
    } else if (cu == 2) {
        if (operand == 0) streams.input2a->write(word);
        else streams.input2b->write(word);
    } else {
        if (operand == 0) streams.input3a->write(word);
        else streams.input3b->write(word);
    }
}

inline unsigned int quantized_vector_rows_for_cu(
    unsigned int physical_rows, unsigned int valid_rows, unsigned int cu) {
    #pragma HLS inline
    const unsigned int slots = (physical_rows + 3) / 4;
    const unsigned int begin = cu * slots;
    if (cu >= 4 || begin >= valid_rows) return 0;
    return valid_rows - begin < slots ? valid_rows - begin : slots;
}

#endif
