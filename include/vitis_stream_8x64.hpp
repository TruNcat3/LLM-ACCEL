#ifndef LLM_FPGA_VITIS_STREAM_8X64_HPP
#define LLM_FPGA_VITIS_STREAM_8X64_HPP

#include "control_cache_8x64.hpp"

constexpr unsigned int CU8_NK_TASK_BITS = 160;
constexpr unsigned int CU8_NK_ACTIVATION_BITS = 128;
constexpr unsigned int CU8_NK_WEIGHT_BITS = 256;
constexpr unsigned int CU8_NK_VECTOR_BITS = 416;
constexpr unsigned int CU8_NK_STATUS_BITS = 64;

using cu8_nk_task_word_t = ap_uint<CU8_NK_TASK_BITS>;
using cu8_nk_activation_word_t = ap_uint<CU8_NK_ACTIVATION_BITS>;
using cu8_nk_weight_word_t = ap_uint<CU8_NK_WEIGHT_BITS>;
using cu8_nk_vector_word_t = ap_uint<CU8_NK_VECTOR_BITS>;
using cu8_nk_status_word_t = ap_uint<CU8_NK_STATUS_BITS>;

inline cu8_nk_task_word_t pack_cu8_nk_task(const cu8_task_t& task) {
    #pragma HLS inline
    static_assert(CU8_NK_TASK_BITS >= 160, "compact 8x64 task word needs 160 bits");
    static_assert(MAX_LINEAR_IN_DIM < 65536, "task k_count/elem fields use 16-bit compact encoding");
    static_assert(MAX_LINEAR_OUT_DIM < 65536, "task k_count/elem fields use 16-bit compact encoding");
    static_assert(CU_STREAM_MAX_PACKETS < 65536, "task packet_count uses 16-bit compact encoding");
    static_assert(CU8_MAX_MM_REPEATS < 65536, "task repeat_count uses 16-bit compact encoding");
    cu8_nk_task_word_t word = 0;
    word.range(3, 0) = static_cast<unsigned int>(task.mode);
    word[4] = task.last_task;
    word.range(6, 5) = static_cast<unsigned int>(task.result_policy);
    word.range(15, 8) = task.token_count;
    word.range(31, 16) = task.k_count;
    word.range(47, 32) = task.elem_count;
    word.range(63, 48) = task.packet_count;
    word.range(79, 64) = task.elem_base;
    word.range(95, 80) = task.block_id;
    word.range(111, 96) = task.repeat_count;
    word.range(127, 112) = task.elem_stride;
    word.range(143, 128) = task.block_stride;
    word.range(159, 144) =
        task.output_scale.range(fm_t::width - 1, 0);
    return word;
}

inline cu8_task_t unpack_cu8_nk_task(const cu8_nk_task_word_t& word) {
    #pragma HLS inline
    cu8_task_t task;
    task.mode = cu8_mode_t(word.range(3, 0).to_uint());
    task.last_task = word[4];
    task.result_policy =
        cu8_result_policy_t(word.range(6, 5).to_uint());
    task.token_count = word.range(15, 8).to_uint();
    task.k_count = word.range(31, 16).to_uint();
    task.elem_count = word.range(47, 32).to_uint();
    task.packet_count = word.range(63, 48).to_uint();
    task.elem_base = word.range(79, 64).to_uint();
    task.block_id = word.range(95, 80).to_uint();
    task.repeat_count = word.range(111, 96).to_uint();
    task.elem_stride = word.range(127, 112).to_uint();
    task.block_stride = word.range(143, 128).to_uint();
    task.output_scale.range(fm_t::width - 1, 0) =
        word.range(159, 144);
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
    static_assert(CU8_NK_STATUS_BITS >= 64, "compact 8x64 status word needs 64 bits");
    static_assert(LINEAR_TOKEN_TILE_ACTIVE < 128, "status token_count uses 7-bit compact encoding");
    static_assert(ceildiv(MAX_LINEAR_OUT_DIM, CC8_OUTPUTS_PER_WAVE) < 256, "status output_waves uses 8-bit compact encoding");
    static_assert(2 * ceildiv(MAX_LINEAR_OUT_DIM, CC8_OUTPUTS_PER_WAVE) < 256, "status dispatched_mm_tasks uses 8-bit compact encoding");
    static_assert(2 * CU_STREAM_MAX_PACKETS < 65536, "status completed_output_packets uses 16-bit compact encoding");
    cu8_nk_status_word_t word = 0;
    static_assert(CC8_OP_DECODER_LAYER < 32, "status op uses 5-bit encoding");
    static_assert(2 * CU8_MAX_TASKS_PER_LAUNCH < 1024,
                  "status MM task count uses 10-bit encoding");
    word.range(4, 0) = static_cast<unsigned int>(status.op);
    word.range(8, 5) = static_cast<unsigned int>(status.status);
    word[9] = status.last_task;
    word.range(16, 10) = status.token_count;
    word.range(24, 17) = status.output_waves;
    word.range(34, 25) = status.dispatched_mm_tasks;
    word.range(42, 35) = status.dispatched_vector_tasks;
    word.range(58, 43) = status.completed_output_packets;
    return word;
}

inline cc8_status_packet_t unpack_cu8_nk_status(
    const cu8_nk_status_word_t& word
) {
    #pragma HLS inline
    cc8_status_packet_t status;
    status.op = cc8_operator_t(word.range(4, 0).to_uint());
    status.status = cc8_status_code_t(word.range(8, 5).to_uint());
    status.last_task = word[9];
    status.token_count = word.range(16, 10).to_uint();
    status.output_waves = word.range(24, 17).to_uint();
    status.dispatched_mm_tasks = word.range(34, 25).to_uint();
    status.dispatched_vector_tasks = word.range(42, 35).to_uint();
    status.completed_output_packets = word.range(58, 43).to_uint();
    return status;
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
    QWEN_WEIGHT_SHARD_PARAMS,
    fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS],
    fm_word_t kv_cache_v[CC8_KV_CACHE_WORDS]
);

#endif
