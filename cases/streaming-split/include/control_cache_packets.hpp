// The ap_uint<2144> output packet exceeds the default ap_int maximum of 1024;
// define the override before every ap_int/ap_axi include.
#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W 4096
#endif

#ifndef LLM_FPGA_CC_CONTROL_CACHE_PACKETS_HPP
#define LLM_FPGA_CC_CONTROL_CACHE_PACKETS_HPP
// Shared packet/dimension definitions, ap_axiu stream types, and pack/unpack helpers.
// Used by control_cache_core, V8-2_s, and the Vitis software-emulation wrapper.
// UG1393: AXIS streams between Vitis kernels use ap_axiu<N,0,0,0>, not a custom
// structure. Pack/unpack each packet manually as ap_uint<N> in ap_axiu.data.
#include "kernel_cc_qwen.hpp"   // fm_t, fm_accum_t, wt_block_t
#include "model_config.hpp"
#include "ap_axi_sdata.h"

#define INPUT_DIM         16     // D=16: 16-wide reduction, 4x fewer operations (2048/16=128), full Q output (group 32 = Q 32)
#define OUTPUT_DIM        64
#define NUM_CORES         2
#define NUM_LANES         1      // Fixed at 2048 DSP: NUM_CORES x NUM_LANES x INPUT_DIM x OUTPUT_DIM = 2 x 1 x 16 x 64
#define TOTAL_LANES       2      // NUM_CORES × NUM_LANES
#define NUM_TILES         16
#define WT_BLOCKS_PER_LANE 32    // INPUT_DIM×OUTPUT_DIM/32 = 16×64/32

struct compute_input_packet_t {
    fm_t data[INPUT_DIM];
    unsigned int seq_id, tile_id, lane_id;
};
struct compute_output_packet_t {
    fm_accum_t results[OUTPUT_DIM];
    unsigned int seq_id, tile_id, lane_id;
};

// AXIS stream type between Vitis kernels (ap_axiu: TDATA=N bits, no side channels).
// Vitis hardware limits a stream to 1024 bits, so split the 2144-bit output:
//   out_lo: results[0..31]  = 32×32 = 1024 bit
//   out_hi: results[32..63] + seq/tile/lane = 32 x 32 + 3 x 32 = 1184 bits,
//   which still exceeds the limit. Therefore out_lo carries results[0..31]
//   and out_hi carries results[32..63], each exactly 1024 bits. Metadata
//   (seq/tile/lane) is implicit in packet order because the controller knows
//   the packet index.
// The input packet grows from 160 to 512 bits: one packet contains one tile's
// 8 lanes x INPUT_DIM=4 = 32 lane-major elements with no metadata. This reduces
// reads from 128 to 16 cycles while retaining 8-lane compute parallelism.
typedef ap_axiu<512, 0, 0, 0>   in_pkt_axis;
typedef ap_axiu<512, 0, 0, 0>   weight_axis;
typedef ap_axiu<1024, 0, 0, 0>  out_lo_axis;   // results[0..31]
typedef ap_axiu<1024, 0, 0, 0>  out_hi_axis;   // results[32..63]

// ---- pack/unpack: struct ↔ ap_uint (TDATA) ----
// input packet 160b: [63:0]=data[0..3], [95:64]=seq, [127:96]=tile, [159:128]=lane
inline ap_uint<160> pack_in(const compute_input_packet_t& p) {
    ap_uint<160> r = 0;
    for (int i = 0; i < INPUT_DIM; i++) r.range(i*16+15, i*16) = p.data[i].range(15, 0);
    r.range(95, 64) = p.seq_id;
    r.range(127, 96) = p.tile_id;
    r.range(159, 128) = p.lane_id;
    return r;
}
inline compute_input_packet_t unpack_in(const ap_uint<160>& r) {
    compute_input_packet_t p;
    for (int i = 0; i < INPUT_DIM; i++) p.data[i].range(15, 0) = r.range(i*16+15, i*16);
    p.seq_id = r.range(95, 64);
    p.tile_id = r.range(127, 96);
    p.lane_id = r.range(159, 128);
    return p;
}

// ---- Wide input packet (512 bits): one tile x 8 lanes x INPUT_DIM,
// lane-major, with no metadata. ----
// bit[(lane*INPUT_DIM+i)*16 + 15 : (lane*INPUT_DIM+i)*16] = data[lane][i]
// tile_id/lane_id are implicit: tile follows packet order 0..NUM_TILES-1,
// while lanes 0..7 occupy separate 64-bit slots inside each packet.
struct compute_input_packet_wide_t {
    fm_t data[TOTAL_LANES][INPUT_DIM];
};
inline ap_uint<512> pack_in_wide(const compute_input_packet_wide_t& p) {
    ap_uint<512> r = 0;
    for (int lane = 0; lane < TOTAL_LANES; lane++)
        for (int i = 0; i < INPUT_DIM; i++)
            r.range((lane*INPUT_DIM + i)*16 + 15, (lane*INPUT_DIM + i)*16) = p.data[lane][i].range(15, 0);
    return r;
}
inline compute_input_packet_wide_t unpack_in_wide(const ap_uint<512>& r) {
    compute_input_packet_wide_t p;
    for (int lane = 0; lane < TOTAL_LANES; lane++)
        for (int i = 0; i < INPUT_DIM; i++)
            p.data[lane][i].range(15, 0) = r.range((lane*INPUT_DIM + i)*16 + 15, (lane*INPUT_DIM + i)*16);
    return p;
}

// Output is split across two 1024-bit streams, each carrying 32 fm_accum_t values.
inline ap_uint<1024> pack_out_lo_raw(const fm_accum_t results[OUTPUT_DIM]) {
    ap_uint<1024> r = 0;
    for (int i = 0; i < 32; i++) r.range(i*32+31, i*32) = results[i].range(31, 0);
    return r;
}
inline ap_uint<1024> pack_out_hi_raw(const fm_accum_t results[OUTPUT_DIM]) {
    ap_uint<1024> r = 0;
    for (int i = 0; i < 32; i++) r.range(i*32+31, i*32) = results[32+i].range(31, 0);
    return r;
}
inline void unpack_out(const ap_uint<1024>& lo, const ap_uint<1024>& hi, fm_accum_t results[OUTPUT_DIM]) {
    for (int i = 0; i < 32; i++) results[i].range(31, 0) = lo.range(i*32+31, i*32);
    for (int i = 0; i < 32; i++) results[32+i].range(31, 0) = hi.range(i*32+31, i*32);
}

// weight 512b: wt_block = 32 × wt_linear_t(16b)
inline ap_uint<512> pack_w(const wt_block_t& w) {
    ap_uint<512> r = 0;
    for (int i = 0; i < 32; i++) r.range(i*16+15, i*16) = w[i].range(15, 0);
    return r;
}
inline wt_block_t unpack_w(const ap_uint<512>& r) {
    wt_block_t w;
    for (int i = 0; i < 32; i++) w[i].range(15, 0) = r.range(i*16+15, i*16);
    return w;
}

// ============================================================================
// operator_program: the controller's operation sequence (M1 op_loop).
// ============================================================================
// Flat uint32 array with six fields per operation (OP_TASK_STRIDE). The Host
// constructs it and the controller reads op*STRIDE+field. op_kind
// (Q/K/V/O/Gate/Up/Down) is folded into weight_offset, precomputed by the Host
// from model offsets. op_ctrl selects activation and is forwarded to V8-2_s.
#define OP_TASK_STRIDE    6
// input_source / output_dest values.
#define SRC_HBM           0   // Load gbuf_in from hidden_in[input_hbm_offset].
#define SRC_PREV_GBUF     1   // Reuse previous output (M3+ feedback placeholder; not implemented).
#define DST_HBM           0   // Write gbuf_out to hidden_out[output_hbm_offset].
#define DST_GBUF_FEEDBACK 1   // Keep on chip for the next op (M3+ placeholder; not implemented).
// op_ctrl bit mask, aligned with V8-2_s unified_activation:
//   NONE=0x00 RELU=0x01 GELU=0x02 SILU=0x03 SOFTMAX=0x04 LAYERNORM=0x05, ENABLE_BIAS=0x80
struct op_task_t {
    unsigned int op_ctrl;            // Activation+bias mask forwarded to V8-2_s.
    unsigned int weight_offset;      // wt_block offset of this operation in weight_hbm.
    unsigned int input_source;       // SRC_HBM / SRC_PREV_GBUF
    unsigned int input_hbm_offset;   // fm_t element offset in hidden_in for SRC_HBM.
    unsigned int output_dest;        // DST_HBM / DST_GBUF_FEEDBACK
    unsigned int output_hbm_offset;  // fm_accum_t element offset in hidden_out for DST_HBM.
};
// cc -> V8-2_s control stream: one 32-bit op_ctrl packet per operation.
typedef ap_axiu<32, 0, 0, 0> ctrl_axis;

#endif
