#ifndef LLM_FPGA_COMPUTE_STREAM_HPP
#define LLM_FPGA_COMPUTE_STREAM_HPP

#include "hardware.hpp"
#include <hls_stream.h>

constexpr unsigned int CU_VEC_LANES = 16;
constexpr unsigned int CU_MV_LANES = 16;
#ifndef CU_NL_LANES_CONFIG
#define CU_NL_LANES_CONFIG 1
#endif
constexpr unsigned int CU_NL_LANES = CU_NL_LANES_CONFIG;
constexpr unsigned int CU_STREAM_MAX_PACKETS =
    LINEAR_TOKEN_TILE_ACTIVE * MAX_LINEAR_OUT_BLOCKS;
constexpr unsigned int CU_STREAM_MAX_VALUES =
    CU_STREAM_MAX_PACKETS * CU_VEC_LANES;
constexpr unsigned int CU_STREAM_MAX_FM_WORDS =
    ceildiv(CU_STREAM_MAX_VALUES, FM_BLOCK_SIZE);
constexpr unsigned int CU_MV_Q_FM_WORDS =
    ceildiv(HEAD_DIM, FM_BLOCK_SIZE);
constexpr unsigned int CU_MV_CACHE_FM_WORDS =
    ceildiv(MAX_SEQ_LEN * HEAD_DIM, FM_BLOCK_SIZE);
constexpr unsigned int CU_MV_MAX_PACKETS =
    MAX_SEQ_LEN * MAX_LINEAR_IN_BLOCKS;
constexpr unsigned int CU_SOFTMAX_ROWS = 8;
constexpr unsigned int CU_SOFTMAX_ELEMS = 64;
constexpr unsigned int CU_SOFTMAX_PACKETS =
    CU_SOFTMAX_ELEMS / CU_VEC_LANES;

static_assert(CU_VEC_LANES == MM_PE_OUT, "compute stream packets carry one 16-lane vector block");
static_assert(CU_NL_LANES >= 1, "nonlinear engine needs at least one physical lane");
static_assert(CU_NL_LANES <= CU_VEC_LANES, "nonlinear lanes cannot exceed packet lanes");
static_assert(CU_VEC_LANES % CU_NL_LANES == 0, "nonlinear lanes must divide the 16-lane packet");
static_assert(CU_SOFTMAX_PACKETS == 4, "8x64 softmax expects four packets per row");

enum cu_stream_op_t {
    CU_OP_CAST_ACCUM = 0,
    CU_OP_SILU = 1,
    CU_OP_MUL = 2,
    CU_OP_ADD = 3,
    CU_OP_RMS_SUMSQ = 4,
    CU_OP_RMS_APPLY = 5,
    CU_OP_MV_DOT = 6,
    CU_OP_MV_WEIGHTED_SUM = 7,
    CU_OP_SILU_MUL = 8,
    CU_OP_RMSNORM = 9
};

struct cu_vec16_packet_t {
    fm_t data[CU_VEC_LANES];
    ap_uint<CU_VEC_LANES> valid_mask;
    unsigned int token_lane;
    unsigned int elem_base;
    unsigned int block_id;
    bool last_block;
    bool last_stream;
};

struct cu_accum16_packet_t {
    fm_accum_t data[CU_VEC_LANES];
    ap_uint<CU_VEC_LANES> valid_mask;
    unsigned int token_lane;
    unsigned int elem_base;
    unsigned int block_id;
    bool last_block;
    bool last_stream;
};

void cast_accum_to_fm_stream(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_accum16_packet_t>& in_stream,
    unsigned int packet_count
);

void stream_silu_mul(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& gate_stream,
    hls::stream<cu_vec16_packet_t>& up_stream,
    unsigned int packet_count
);

void stream_silu(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    unsigned int packet_count
);

void stream_add_residual(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& lhs_stream,
    hls::stream<cu_vec16_packet_t>& rhs_stream,
    unsigned int packet_count
);

void stream_rmsnorm(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    const wt_norm_t weights[MAX_LINEAR_OUT_DIM],
    unsigned int elem_count
);

void stream_attention_mv_softmax(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& q_stream,
    hls::stream<cu_vec16_packet_t>& k_stream,
    hls::stream<cu_vec16_packet_t>& v_stream,
    unsigned int tile_len,
    unsigned int head_dim
);

void stream_softmax_rows(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    unsigned int row_count,
    unsigned int elem_count
);

void compute_stream_nonlinear_unit_synth_stub(
    fm_t sink[CU_VEC_LANES],
    fm_t gate_in[CU_VEC_LANES],
    fm_t up_in[CU_VEC_LANES],
    wt_norm_t norm_weights[CU_VEC_LANES]
);

void compute_stream_silu_mul_perf_stub(
    fm_word_t sink_words[1],
    const fm_word_t gate_words[CU_STREAM_MAX_FM_WORDS],
    const fm_word_t up_words[CU_STREAM_MAX_FM_WORDS],
    unsigned int token_count,
    unsigned int elem_count
);

void compute_stream_mv_softmax_perf_stub(
    fm_word_t sink_words[1],
    const fm_word_t q_words[CU_MV_Q_FM_WORDS],
    const fm_word_t k_words[CU_MV_CACHE_FM_WORDS],
    const fm_word_t v_words[CU_MV_CACHE_FM_WORDS],
    unsigned int tile_len,
    unsigned int head_dim
);

#endif
