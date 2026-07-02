#ifndef LLM_FPGA_VITIS_STREAM_8X64_HPP
#define LLM_FPGA_VITIS_STREAM_8X64_HPP

#include "control_cache_8x64.hpp"

constexpr unsigned int CU8_NK_TASK_BITS = 352;
constexpr unsigned int CU8_NK_ACTIVATION_BITS = 128;
constexpr unsigned int CU8_NK_WEIGHT_BITS = 256;
constexpr unsigned int CU8_NK_VECTOR_BITS = 416;
constexpr unsigned int CU8_NK_STATUS_BITS = 256;

using cu8_nk_task_word_t = ap_uint<CU8_NK_TASK_BITS>;
using cu8_nk_activation_word_t = ap_uint<CU8_NK_ACTIVATION_BITS>;
using cu8_nk_weight_word_t = ap_uint<CU8_NK_WEIGHT_BITS>;
using cu8_nk_vector_word_t = ap_uint<CU8_NK_VECTOR_BITS>;
using cu8_nk_status_word_t = ap_uint<CU8_NK_STATUS_BITS>;

inline cu8_nk_task_word_t pack_cu8_nk_task(const cu8_task_t& task) {
    #pragma HLS inline
    cu8_nk_task_word_t word = 0;
    word.range(31, 0) = static_cast<unsigned int>(task.mode);
    word.range(63, 32) = task.k_count;
    word.range(95, 64) = task.token_count;
    word.range(127, 96) = task.elem_count;
    word.range(159, 128) = task.packet_count;
    word.range(191, 160) = task.elem_base;
    word.range(223, 192) = task.block_id;
    word.range(255, 224) = task.repeat_count;
    word.range(287, 256) = task.elem_stride;
    word.range(319, 288) = task.block_stride;
    word.range(335, 320) =
        task.output_scale.range(fm_t::width - 1, 0);
    word[336] = task.last_task;
    return word;
}

inline cu8_task_t unpack_cu8_nk_task(const cu8_nk_task_word_t& word) {
    #pragma HLS inline
    cu8_task_t task;
    task.mode = cu8_mode_t(word.range(31, 0).to_uint());
    task.k_count = word.range(63, 32).to_uint();
    task.token_count = word.range(95, 64).to_uint();
    task.elem_count = word.range(127, 96).to_uint();
    task.packet_count = word.range(159, 128).to_uint();
    task.elem_base = word.range(191, 160).to_uint();
    task.block_id = word.range(223, 192).to_uint();
    task.repeat_count = word.range(255, 224).to_uint();
    task.elem_stride = word.range(287, 256).to_uint();
    task.block_stride = word.range(319, 288).to_uint();
    task.output_scale.range(fm_t::width - 1, 0) =
        word.range(335, 320);
    task.last_task = word[336];
    return task;
}

inline cu8_nk_activation_word_t pack_cu8_nk_activation(
    const mm_stream_8x64_activation_packet_t& packet
) {
    #pragma HLS inline
    cu8_nk_activation_word_t word = 0;
    for (unsigned int lane = 0; lane < MM_STREAM_8X64_TOKENS; lane++) {
        #pragma HLS unroll
        word.range((lane + 1) * fm_t::width - 1, lane * fm_t::width) =
            packet.data[lane].range(fm_t::width - 1, 0);
    }
    return word;
}

inline mm_stream_8x64_activation_packet_t unpack_cu8_nk_activation(
    const cu8_nk_activation_word_t& word
) {
    #pragma HLS inline
    mm_stream_8x64_activation_packet_t packet;
    for (unsigned int lane = 0; lane < MM_STREAM_8X64_TOKENS; lane++) {
        #pragma HLS unroll
        packet.data[lane].range(fm_t::width - 1, 0) =
            word.range((lane + 1) * fm_t::width - 1, lane * fm_t::width);
    }
    return packet;
}

inline cu8_nk_weight_word_t pack_cu8_nk_weight(
    const mm_stream_8x64_weight_packet_t& packet
) {
    #pragma HLS inline
    cu8_nk_weight_word_t word = 0;
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        #pragma HLS unroll
        word.range(
            (lane + 1) * wt_linear_t::width - 1,
            lane * wt_linear_t::width
        ) = packet.data[lane].range(wt_linear_t::width - 1, 0);
    }
    return word;
}

inline mm_stream_8x64_weight_packet_t unpack_cu8_nk_weight(
    const cu8_nk_weight_word_t& word
) {
    #pragma HLS inline
    mm_stream_8x64_weight_packet_t packet;
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        #pragma HLS unroll
        packet.data[lane].range(wt_linear_t::width - 1, 0) =
            word.range(
                (lane + 1) * wt_linear_t::width - 1,
                lane * wt_linear_t::width
            );
    }
    return packet;
}

inline cu8_nk_vector_word_t pack_cu8_nk_vector(
    const cu_vec16_packet_t& packet
) {
    #pragma HLS inline
    cu8_nk_vector_word_t word = 0;
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        #pragma HLS unroll
        word.range((lane + 1) * fm_t::width - 1, lane * fm_t::width) =
            packet.data[lane].range(fm_t::width - 1, 0);
    }
    word.range(271, 256) = packet.valid_mask;
    word.range(303, 272) = packet.token_lane;
    word.range(335, 304) = packet.elem_base;
    word.range(367, 336) = packet.block_id;
    word[368] = packet.last_block;
    word[369] = packet.last_stream;
    return word;
}

inline cu_vec16_packet_t unpack_cu8_nk_vector(
    const cu8_nk_vector_word_t& word
) {
    #pragma HLS inline
    cu_vec16_packet_t packet;
    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        #pragma HLS unroll
        packet.data[lane].range(fm_t::width - 1, 0) =
            word.range((lane + 1) * fm_t::width - 1, lane * fm_t::width);
    }
    packet.valid_mask = word.range(271, 256);
    packet.token_lane = word.range(303, 272).to_uint();
    packet.elem_base = word.range(335, 304).to_uint();
    packet.block_id = word.range(367, 336).to_uint();
    packet.last_block = word[368];
    packet.last_stream = word[369];
    return packet;
}

inline cu8_nk_status_word_t pack_cu8_nk_status(
    const cc8_status_packet_t& status
) {
    #pragma HLS inline
    cu8_nk_status_word_t word = 0;
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

void compute_core_8x64_unified_nk(
    hls::stream<cu8_nk_vector_word_t>& out_stream,
    hls::stream<cu8_nk_task_word_t>& task_stream,
    hls::stream<cu8_nk_activation_word_t>& activation_stream,
    hls::stream<cu8_nk_weight_word_t>& weight_stream0,
    hls::stream<cu8_nk_weight_word_t>& weight_stream1,
    hls::stream<cu8_nk_weight_word_t>& weight_stream2,
    hls::stream<cu8_nk_weight_word_t>& weight_stream3,
    hls::stream<cu8_nk_vector_word_t>& vector_input0_stream,
    hls::stream<cu8_nk_vector_word_t>& vector_input1_stream
);

void cc8_status_sink_nk(
    hls::stream<cu8_nk_status_word_t>& status_stream,
    fm_word_t status_output[1]
);

void control_cache_8x64_dual_core_nk(
    hls::stream<cu8_nk_task_word_t>& core0_task_stream,
    hls::stream<cu8_nk_activation_word_t>& core0_activation_stream,
    hls::stream<cu8_nk_weight_word_t>& core0_weight_stream0,
    hls::stream<cu8_nk_weight_word_t>& core0_weight_stream1,
    hls::stream<cu8_nk_weight_word_t>& core0_weight_stream2,
    hls::stream<cu8_nk_weight_word_t>& core0_weight_stream3,
    hls::stream<cu8_nk_vector_word_t>& core0_vector_input0_stream,
    hls::stream<cu8_nk_vector_word_t>& core0_vector_input1_stream,
    hls::stream<cu8_nk_vector_word_t>& core0_result_stream,
    hls::stream<cu8_nk_task_word_t>& core1_task_stream,
    hls::stream<cu8_nk_activation_word_t>& core1_activation_stream,
    hls::stream<cu8_nk_weight_word_t>& core1_weight_stream0,
    hls::stream<cu8_nk_weight_word_t>& core1_weight_stream1,
    hls::stream<cu8_nk_weight_word_t>& core1_weight_stream2,
    hls::stream<cu8_nk_weight_word_t>& core1_weight_stream3,
    hls::stream<cu8_nk_vector_word_t>& core1_vector_input0_stream,
    hls::stream<cu8_nk_vector_word_t>& core1_vector_input1_stream,
    hls::stream<cu8_nk_vector_word_t>& core1_result_stream,
    hls::stream<cu8_nk_status_word_t>& status_stream,
    fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT],
    const fm_word_t input_port0[CC8_DATA_PORT_WORDS],
    const fm_word_t input_port1[CC8_DATA_PORT_WORDS],
    const fm_word_t aux_port0[CC8_DATA_PORT_WORDS],
    const fm_word_t aux_port1[CC8_DATA_PORT_WORDS],
    unsigned int operator_kind,
    unsigned int layer_id,
    unsigned int token_count,
    unsigned int position,
    unsigned int tile_len,
    QWEN_WEIGHT_SHARD_PARAMS
);

#endif
