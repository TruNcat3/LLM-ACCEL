// output packet ap_uint<2144> 超过 ap_int 默认 max 1024, 必须在所有 ap_int/ap_axi include 前定义
#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W 4096
#endif

#ifndef LLM_FPGA_CC_CONTROL_CACHE_PACKETS_HPP
#define LLM_FPGA_CC_CONTROL_CACHE_PACKETS_HPP
// 共享 packet/维度定义 + ap_axiu stream 类型 + pack/unpack。
// control_cache_core + V8-2_s + Vitis sw_emu wrapper 都用。
// ug1393: Vitis kernel 间 axis stream 用 ap_axiu<N,0,0,0> (非自定义 struct),
// packet 在 stream 边界手动 pack/unpack 成 ap_uint<N> 塞进 ap_axiu.data。
#include "kernel_cc_qwen.hpp"   // fm_t, fm_accum_t, wt_block_t
#include "model_config.hpp"
#include "ap_axi_sdata.h"

#define INPUT_DIM         16     // D=16: reduction 16, 减 op 数 4× (2048/16=128) + Q output 100% (group 32=Q 32)
#define OUTPUT_DIM        64
#define NUM_CORES         2
#define NUM_LANES         1      // DSP 固定 2048: NUM_CORES×NUM_LANES×INPUT_DIM×OUTPUT_DIM = 2×1×16×64
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

// Vitis kernel 间 axis stream 类型 (ap_axiu: TDATA=N bit, 无 side-channel)
// Vitis hw 限制 stream ≤1024 bit, output 2144bit 拆 2 条:
//   out_lo: results[0..31]  = 32×32 = 1024 bit
//   out_hi: results[32..63] + seq/tile/lane = 32×32 + 3×32 = 1184 bit → 仍超限!
//   → 改为: out_lo: results[0..31] = 1024 bit, out_hi: results[32..63] = 1024 bit
//     metadata (seq/tile/lane) 隐含在 packet 顺序中 (cc 知道 pkt index)
// input packet 加宽 (160→512): 1 packet = 1 tile 的 8 lane × INPUT_DIM=4 = 32 元素 (lane-major, 零 metadata)
// read 128→16cyc, 保 8 lane 并行 (对齐 compute 消费, 避免流式化破坏并行)
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

// ---- input wide packet (512-bit): 1 tile × 8 lane × INPUT_DIM, lane-major, 零 metadata ----
// bit[(lane*INPUT_DIM+i)*16 + 15 : (lane*INPUT_DIM+i)*16] = data[lane][i]
// tile_id/lane_id 全隐含 (tile=packet 序 0..NUM_TILES-1, lane=packet 内 0..7 各占 64-bit slot)
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

// output: 拆 2 条 1024-bit (各 32 个 fm_accum_t)
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
// operator_program: cc 编排算子序列 (M1 op_loop)
// ============================================================================
// 平坦 uint32 数组, 每 op 6 个字段 (OP_TASK_STRIDE). host 构造, cc 按 op*STRIDE+field 读.
// op_kind (Q/K/V/O/Gate/Up/Down) 折叠进 weight_offset (host 按 model offset 算好填入),
// op_ctrl (激活选择) 经 ctrl_stream 透传给 V8-2_s.
#define OP_TASK_STRIDE    6
// input_source / output_dest 取值
#define SRC_HBM           0   // 从 hidden_in[input_hbm_offset] 载入 gbuf_in
#define SRC_PREV_GBUF     1   // 复用上一 op 输出 (M3+ feedback, 当前留桩未实现)
#define DST_HBM           0   // gbuf_out 写 hidden_out[output_hbm_offset]
#define DST_GBUF_FEEDBACK 1   // 留片上供下 op (M3+, 当前留桩未实现)
// op_ctrl 位掩码 (与 V8-2_s unified_activation 对齐):
//   NONE=0x00 RELU=0x01 GELU=0x02 SILU=0x03 SOFTMAX=0x04 LAYERNORM=0x05, ENABLE_BIAS=0x80
struct op_task_t {
    unsigned int op_ctrl;            // 激活+bias 位掩码, 透传 V8-2_s
    unsigned int weight_offset;      // 本 op 权重在 weight_hbm 的 wt_block 偏移
    unsigned int input_source;       // SRC_HBM / SRC_PREV_GBUF
    unsigned int input_hbm_offset;   // SRC_HBM 时 hidden_in 的 fm_t 元素偏移
    unsigned int output_dest;        // DST_HBM / DST_GBUF_FEEDBACK
    unsigned int output_hbm_offset;  // DST_HBM 时 hidden_out 的 fm_accum_t 元素偏移
};
// cc → V8-2_s 控制流: 每 op 发 1 个 32-bit op_ctrl 包
typedef ap_axiu<32, 0, 0, 0> ctrl_axis;

#endif
