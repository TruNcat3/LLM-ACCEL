#ifndef LLM_FPGA_HARDWARE_HPP
#define LLM_FPGA_HARDWARE_HPP

#include "datatypes.hpp"
#include "util.hpp"
#include <hls_stream.h>
#include <hls_vector.h>

constexpr unsigned int AXI_XFER_BIT_WIDTH = 512;
constexpr unsigned int FM_BLOCK_SIZE = AXI_XFER_BIT_WIDTH / fm_t::width;
constexpr unsigned int WT_BLOCK_SIZE = AXI_XFER_BIT_WIDTH / wt_linear_t::width;

constexpr unsigned int MM_PE_IN = 16;
constexpr unsigned int MM_PE_OUT = 16;
constexpr unsigned int MM_INPUT_BLOCK_BIT_WIDTH = MM_PE_IN * fm_t::width;
constexpr unsigned int MM_PE_COUNT = MM_PE_IN * MM_PE_OUT;
constexpr unsigned int MM_PARALLEL_MACS = MM_PE_COUNT;
constexpr unsigned int MM_CORE_COUNT_ACTIVE = 1;
constexpr unsigned int MM_CORE_COUNT_TARGET = 8;
constexpr unsigned int MM_CORE_COUNT_MAX_PLAN = 16;
constexpr unsigned int MM_CORE_COUNT_NEXT_EXPLORE = 16;
constexpr unsigned int MM_TARGET_PARALLEL_MACS = MM_PARALLEL_MACS * MM_CORE_COUNT_TARGET;
constexpr unsigned int MM_NEXT_PARALLEL_MACS = MM_PARALLEL_MACS * MM_CORE_COUNT_NEXT_EXPLORE;
constexpr unsigned int LINEAR_TOKEN_TILE_ACTIVE = MM_CORE_COUNT_TARGET;
constexpr unsigned int LINEAR_TOKEN_TILE_MAX_PLAN = MM_CORE_COUNT_MAX_PLAN;
constexpr unsigned int MM_REFERENCE_DSP_COUNT_2_PER_MAC = MM_PE_COUNT * 2;
constexpr unsigned int MM_REFERENCE_DSP_COUNT_1_PER_MAC = MM_PE_COUNT;
constexpr unsigned int MM_TARGET_DSP_COUNT_1_PER_MAC = MM_TARGET_PARALLEL_MACS;
constexpr unsigned int MM_NEXT_DSP_COUNT_1_PER_MAC = MM_NEXT_PARALLEL_MACS;
constexpr unsigned int MM_CONTROLLER_CORE_COUNT = MM_CORE_COUNT_NEXT_EXPLORE;
constexpr unsigned int MM_CONTROLLER_PARALLEL_MACS = MM_NEXT_PARALLEL_MACS;
constexpr unsigned int MM_CONTROLLER_TOKEN_LANES = LINEAR_TOKEN_TILE_ACTIVE;
constexpr unsigned int MM_CONTROLLER_OUT_TILE_PARALLEL = MM_CONTROLLER_CORE_COUNT / MM_CONTROLLER_TOKEN_LANES;
constexpr unsigned int MM_GLOBAL_CACHE_MAX_CHUNKS = ceildiv(MAX_SEQ_LEN, LINEAR_TOKEN_TILE_ACTIVE);
constexpr unsigned int MM_GLOBAL_CACHE_TOKEN_SLOTS = MM_GLOBAL_CACHE_MAX_CHUNKS * LINEAR_TOKEN_TILE_ACTIVE;
constexpr unsigned int MM_TIMING_TARGET_PS = 2000;
constexpr unsigned int MM_TILE_WEIGHT_ELEMS = MM_PE_IN * MM_PE_OUT;
constexpr unsigned int MM_TILE_WEIGHT_BLOCKS = MM_TILE_WEIGHT_ELEMS / WT_BLOCK_SIZE;
constexpr unsigned int LINEAR_TILE_PING_PONG_DEPTH = 2;
constexpr unsigned int LINEAR_OUTPUT_STREAM_DEPTH = 2;
constexpr unsigned int HBM_PORT_BIT_WIDTH = 512;
constexpr unsigned int HBM_PORT_TARGET_MHZ = 450;
constexpr double HBM_PORT_BW_GBPS = 28.8;
constexpr unsigned int U50_XRT_USABLE_HBM_PORTS = 28;
constexpr unsigned int SHARED_WEIGHT_KV_PORT_PLAN = 24;
constexpr unsigned int MAX_LINEAR_IN_DIM = max_const(HIDDEN_SIZE, INTERMEDIATE_SIZE);
constexpr unsigned int MAX_LINEAR_OUT_DIM = max_const(HIDDEN_SIZE, INTERMEDIATE_SIZE);
constexpr unsigned int MAX_LINEAR_IN_BLOCKS = ceildiv(MAX_LINEAR_IN_DIM, MM_PE_IN);
constexpr unsigned int MAX_LINEAR_OUT_BLOCKS = ceildiv(MAX_LINEAR_OUT_DIM, MM_PE_OUT);

static_assert(MM_PE_IN == 16, "v1 preserves the Edge-MoE 16-lane input tile");
static_assert(MM_PE_OUT == 16, "v1 preserves the Edge-MoE 16-lane output tile");
static_assert(MM_INPUT_BLOCK_BIT_WIDTH == 256, "MM global-cache input block should be one 256-bit word");
static_assert(MM_CORE_COUNT_ACTIVE >= 1, "at least one MM core must be active");
static_assert(MM_CORE_COUNT_TARGET >= MM_CORE_COUNT_ACTIVE, "target MM core count tracks the next architecture step");
static_assert(MM_CORE_COUNT_TARGET <= MM_CORE_COUNT_MAX_PLAN, "target MM core count must stay within the planned 8-16 range");
static_assert(MM_CORE_COUNT_NEXT_EXPLORE == 16, "next controller exploration targets 16 shared MM cores");
static_assert(MM_NEXT_DSP_COUNT_1_PER_MAC == 4096, "16 16x16 Fix16 cores should map to 4096 DSP at 1 DSP/MAC");
static_assert(LINEAR_TOKEN_TILE_ACTIVE == MM_CORE_COUNT_TARGET, "token tile lanes map one-to-one to target MM core lanes");
static_assert(MM_CONTROLLER_CORE_COUNT % MM_CONTROLLER_TOKEN_LANES == 0, "controller cores must form token-lane groups");
static_assert(MM_CONTROLLER_OUT_TILE_PARALLEL == 2, "16-core exploration computes two output tiles per token lane");
static_assert(SHARED_WEIGHT_KV_PORT_PLAN <= U50_XRT_USABLE_HBM_PORTS, "shared weight/KV ports must stay in HBM[0:27]");
static_assert(WT_BLOCK_SIZE == PACKED_WEIGHT_BLOCK_ELEMS, "weight block type must match packed model layout");
static_assert(MM_TILE_WEIGHT_ELEMS == PACKED_WEIGHT_TILE_ELEMS, "MM tile shape must match packed model layout");
static_assert(MM_TILE_WEIGHT_BLOCKS == PACKED_WEIGHT_BLOCKS_PER_TILE, "MM tile block count must match packed model layout");
static_assert(WEIGHT_SHARD_GROUPS == 2, "v2 streamer expects two alternating HBM shard groups");
static_assert(HIDDEN_SIZE % MM_PE_IN == 0, "Qwen hidden size must align to the MM input tile");
static_assert(HIDDEN_SIZE % MM_PE_OUT == 0, "Qwen hidden size must align to the MM output tile");
static_assert(INTERMEDIATE_SIZE % MM_PE_IN == 0, "Qwen FFN size must align to the MM input tile");
static_assert(INTERMEDIATE_SIZE % MM_PE_OUT == 0, "Qwen FFN size must align to the MM output tile");
static_assert(KV_CHANNELS % MM_PE_OUT == 0, "Qwen KV projection size must align to the MM output tile");

typedef hls::vector<fm_t, FM_BLOCK_SIZE> fm_block_t;
typedef ap_uint<AXI_XFER_BIT_WIDTH> fm_word_t;
typedef ap_uint<AXI_XFER_BIT_WIDTH> wt_block_t;
typedef hls::vector<fm_t, MM_PE_IN> linear_in_t;
typedef ap_uint<MM_INPUT_BLOCK_BIT_WIDTH> mm_input_block_t;
typedef hls::vector<fm_accum_t, MM_PE_OUT> linear_out_t;

struct wt_linear_block_t {
    wt_linear_t data[MM_PE_OUT][MM_PE_IN];

    wt_linear_t* operator[](unsigned int row) {
        return data[row];
    }

    const wt_linear_t* operator[](unsigned int row) const {
        return data[row];
    }
};

inline wt_linear_t unpack_wt_block_lane(const wt_block_t& block, unsigned int lane) {
    #pragma HLS inline
    ap_uint<wt_linear_t::width> raw = block.range((lane + 1) * wt_linear_t::width - 1, lane * wt_linear_t::width);
    wt_linear_t value;
    value.range(wt_linear_t::width - 1, 0) = raw;
    return value;
}

inline void set_wt_block_lane(wt_block_t& block, unsigned int lane, wt_linear_t value) {
    ap_uint<wt_linear_t::width> raw = value.range(wt_linear_t::width - 1, 0);
    block.range((lane + 1) * wt_linear_t::width - 1, lane * wt_linear_t::width) = raw;
}

inline fm_t unpack_mm_input_block_lane(const mm_input_block_t& block, unsigned int lane) {
    #pragma HLS inline
    ap_uint<fm_t::width> raw = block.range((lane + 1) * fm_t::width - 1, lane * fm_t::width);
    fm_t value;
    value.range(fm_t::width - 1, 0) = raw;
    return value;
}

inline void set_mm_input_block_lane(mm_input_block_t& block, unsigned int lane, fm_t value) {
    #pragma HLS inline
    ap_uint<fm_t::width> raw = value.range(fm_t::width - 1, 0);
    block.range((lane + 1) * fm_t::width - 1, lane * fm_t::width) = raw;
}

inline fm_t unpack_fm_word_lane(const fm_word_t& word, unsigned int lane) {
    #pragma HLS inline
    ap_uint<fm_t::width> raw = word.range((lane + 1) * fm_t::width - 1, lane * fm_t::width);
    fm_t value;
    value.range(fm_t::width - 1, 0) = raw;
    return value;
}

inline void set_fm_word_lane(fm_word_t& word, unsigned int lane, fm_t value) {
    #pragma HLS inline
    ap_uint<fm_t::width> raw = value.range(fm_t::width - 1, 0);
    word.range((lane + 1) * fm_t::width - 1, lane * fm_t::width) = raw;
}

inline mm_input_block_t pack_linear_in_block(const linear_in_t& block) {
    #pragma HLS inline
    mm_input_block_t packed = 0;
    for (unsigned int lane = 0; lane < MM_PE_IN; lane++) {
        #pragma HLS unroll
        set_mm_input_block_lane(packed, lane, block[lane]);
    }
    return packed;
}

inline linear_in_t unpack_mm_input_block(const mm_input_block_t& packed) {
    #pragma HLS inline
    linear_in_t block;
    for (unsigned int lane = 0; lane < MM_PE_IN; lane++) {
        #pragma HLS unroll
        block[lane] = unpack_mm_input_block_lane(packed, lane);
    }
    return block;
}

template <typename T>
T min_value(T a, T b) {
    return (a < b) ? a : b;
}

template <typename T>
T max_value(T a, T b) {
    return (a > b) ? a : b;
}

#endif
