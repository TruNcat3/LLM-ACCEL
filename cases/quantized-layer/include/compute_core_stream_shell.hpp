#ifndef LLM_FPGA_COMPUTE_CORE_STREAM_SHELL_HPP
#define LLM_FPGA_COMPUTE_CORE_STREAM_SHELL_HPP

#include "mm_stream_8x64.hpp"

constexpr unsigned int CU_CORE_K_TILE = 16;
constexpr unsigned int CU_CORE_K_BUFFER_TILES = 2;
constexpr unsigned int CU_CORE_INPUT_BUFFER_DEPTH =
    CU_CORE_K_TILE * CU_CORE_K_BUFFER_TILES;
constexpr unsigned int CU_CORE_RESULT_BUFFER_DEPTH =
    MM_STREAM_8X64_PACKETS_PER_BLOCK;

enum cu_core_mode_t {
    CU_CORE_MODE_MM = 0,
    CU_CORE_MODE_MM_SILU = 1,
    CU_CORE_MODE_SILU = 2
};

struct cu_core_task_t {
    cu_core_mode_t mode;
    unsigned int k_count;
    unsigned int packet_count;
    unsigned int elem_base;
    unsigned int block_id;
    bool last_stream;
};

void compute_core_stream_shell(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_core_task_t>& task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3,
    hls::stream<cu_vec16_packet_t>& vector_stream
);

#endif
