#include "control_cache_8x64.hpp"
#include "vitis_stream_8x64.hpp"

static fm_word_t pack_cc8_status(const cc8_status_packet_t& status) {
    fm_word_t word = 0;
    word.range(31, 0) = static_cast<unsigned int>(status.op);
    word.range(63, 32) = static_cast<unsigned int>(status.status);
    word.range(95, 64) = status.token_count;
    word.range(127, 96) = status.output_waves;
    word.range(159, 128) = status.dispatched_mm_tasks;
    word.range(191, 160) = status.dispatched_vector_tasks;
    word.range(223, 192) = status.completed_output_packets;
    word[224] = status.last_task;
    return word;
}

void cc8_status_sink(
    hls::stream<cc8_status_packet_t>& status_stream,
    fm_word_t status_output[1]
) {
    #pragma HLS interface axis port=status_stream
    #pragma HLS interface m_axi port=status_output offset=slave bundle=status depth=1 max_widen_bitwidth=512
    #pragma HLS interface s_axilite port=status_output bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
    #pragma HLS inline off

    status_output[0] = pack_cc8_status(status_stream.read());
}

void cc8_status_sink_nk(
    hls::stream<cu8_nk_status_word_t>& status_stream,
    fm_word_t status_output[1]
) {
    #pragma HLS interface axis port=status_stream
    #pragma HLS interface m_axi port=status_output offset=slave bundle=status depth=1 max_widen_bitwidth=512
    #pragma HLS interface s_axilite port=status_output bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
    #pragma HLS inline off

    fm_word_t word = 0;
    word.range(CU8_NK_STATUS_BITS - 1, 0) = status_stream.read();
    status_output[0] = word;
}
