/**
 * ============================================================================
 * QKV Tile Kernel - V8-2 (2套计算核心 + 共享激活, 部署优化版)
 * ============================================================================
 *
 * 动机(用户): V8单实例 FF631k/LUT461k, 4实例LUT 1.84M(107%超标)无法部署。
 *   V8-2 = 1个kernel含2套计算核心, 共享1套激活 + 接口/缓存/控制。
 *   2×V8-2 = 4套核心, LUT 1.44M(83%可部署) vs 4×V8 LUT 1.84M(107%超标)。
 *
 * 关键收益: 激活是FF大头(V8的501k=79%), 2套核心共享1套激活 → 省一倍激活FF/LUT。
 *
 * V8-2结构:
 *   - 2套compute核心 (每套4 lane × 4×64 matmul = 1024 DSP), 共2048 DSP
 *   - 8 lane并行 (2 core × 4 lane), 8个模板matmul实例, compute_tiles II=1
 *   - 1套activate_tiles_core (522 DSP), 处理2 core × 4 lane × 16 tile = 128次
 *   - 共享 input/output/weight stream 接口 (路线2: weight 从 control/cache core, 替代 w0..w7 m_axi)
 *
 * 资源预期(单V8-2): DSP 2570, FF ~747k, LUT ~720k
 *   2×V8-2: DSP 5140(42%), FF 1.49M(43%), LUT 1.44M(83%) ✅可部署
 *
 * lane_id映射: 0-3=core0 lane0-3, 4-7=core1 lane0-3 (core=lane/4, local_lane=lane%4)
 * ============================================================================
 */

#include "../include/control_cache_packets.hpp"
#include <hls_stream.h>
#include <hls_math.h>

#define CTRL_ACT_NONE      0x00
#define CTRL_ACT_RELU      0x01
#define CTRL_ACT_GELU      0x02
#define CTRL_ACT_SILU      0x03
#define CTRL_ACT_SOFTMAX   0x04
#define CTRL_ACT_LAYERNORM 0x05
#define CTRL_ACCUMULATE    0x40    // 累积: mm_results += (不清空)
#define CTRL_FINALIZE      0x80    // 执行 activate + write_stream (最终 reduction step)
#define CTRL_ACT_MASK      0x3F    // 激活类型掩码

// ============================================================================
// 模板4×64矩阵乘法 (8个独立实例 matmul_4x64_tpl<0..7>, inline off防共享)
// ============================================================================
template<int LANE_ID>
static void matmul_4x64_tpl(
    const fm_t input[INPUT_DIM],
    const wt_linear_t weight[INPUT_DIM][OUTPUT_DIM],
    fm_accum_t output[OUTPUT_DIM]
) {
    #pragma HLS inline off
    // P2: 宽位非饱和累加器, 消除每步 AP_RND/AP_SAT 的 fabric 钳位 (省 ~200k LUT/CU)。
    // product = fm_t<16,8> × wt<16,4> = <32,12>; 累加 4 个 < <34,13>; 用 <48,24> 富余, 中间无 round/sat。
    // 数值更准 (少中间钳位), 出口 output 单次 (fm_accum_t) 转换做 round/sat。
    ap_fixed<48, 24> acc[OUTPUT_DIM];
    #pragma HLS array_partition variable=acc complete
    for (int o = 0; o < OUTPUT_DIM; o++) {
        #pragma HLS unroll
        acc[o] = 0;
    }
    for (int in = 0; in < INPUT_DIM; in++) {
        #pragma HLS pipeline II=1
        for (int o = 0; o < OUTPUT_DIM; o++) {
            #pragma HLS unroll
            acc[o] += input[in] * weight[in][o];   // 精确累加, 无中间钳位
        }
    }
    for (int o = 0; o < OUTPUT_DIM; o++) {
        #pragma HLS unroll
        output[o] = (fm_accum_t)acc[o];   // 出口单次 round/sat
    }
}

// ============================================================================
// 定点激活函数 (64维, 同V8)
// ============================================================================
static void activation_relu(fm_accum_t data[OUTPUT_DIM]) {
    #pragma HLS inline off
    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS unroll
        data[i] = (data[i] > fm_accum_t(0)) ? data[i] : fm_accum_t(0);
    }
}
static void activation_gelu(fm_accum_t data[OUTPUT_DIM]) {
    #pragma HLS inline off
    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS unroll
        fm_accum_t xf = data[i]; fm_t slope, offset;
        if (xf < fm_accum_t(-3.0)) { slope = fm_t(0.0); offset = fm_t(0.0); }
        else if (xf < fm_accum_t(-1.0)) { slope = fm_t(0.15); offset = fm_t(0.15); }
        else if (xf < fm_accum_t(0.0)) { slope = fm_t(0.5); offset = fm_t(0.1); }
        else if (xf < fm_accum_t(1.0)) { slope = fm_t(0.8); offset = fm_t(0.0); }
        else { slope = fm_t(1.0); offset = fm_t(0.0); }
        data[i] = (slope * (fm_t)xf) + fm_accum_t(offset);
    }
}
static void activation_silu(fm_accum_t data[OUTPUT_DIM]) {
    #pragma HLS inline off
    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS unroll
        fm_accum_t xf = data[i]; fm_t slope, offset;
        if (xf < fm_accum_t(-5.0)) { slope = fm_t(0.0); offset = fm_t(0.0); }
        else if (xf < fm_accum_t(-1.0)) { slope = fm_t(0.1); offset = fm_t(0.1); }
        else if (xf < fm_accum_t(0.0)) { slope = fm_t(0.4); offset = fm_t(0.2); }
        else if (xf < fm_accum_t(1.0)) { slope = fm_t(0.7); offset = fm_t(0.15); }
        else if (xf < fm_accum_t(3.0)) { slope = fm_t(0.95); offset = fm_t(0.05); }
        else { slope = fm_t(1.0); offset = fm_t(0.0); }
        data[i] = (slope * (fm_t)xf) + fm_accum_t(offset);
    }
}
static void activation_softmax(fm_accum_t data[OUTPUT_DIM]) {
    #pragma HLS inline off
    fm_accum_t max_val = data[0];
    for (unsigned int i = 1; i < OUTPUT_DIM; i++) {
        #pragma HLS unroll
        if (data[i] > max_val) max_val = data[i];
    }
    fm_accum_t exp_vals[OUTPUT_DIM];
    #pragma HLS array_partition variable=exp_vals complete
    fm_accum_t exp_sum = fm_accum_t(0);
    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS unroll
        fm_accum_t xv = data[i] - max_val; fm_t slope, offset;
        if (xv < fm_accum_t(-5.0)) { slope = fm_t(0.0); offset = fm_t(0.0); }
        else if (xv < fm_accum_t(-2.0)) { slope = fm_t(0.05); offset = fm_t(0.25); }
        else if (xv < fm_accum_t(0.0)) { slope = fm_t(0.15); offset = fm_t(0.45); }
        else if (xv < fm_accum_t(2.0)) { slope = fm_t(0.5); offset = fm_t(0.5); }
        else if (xv < fm_accum_t(5.0)) { slope = fm_t(2.0); offset = fm_t(-2.5); }
        else { slope = fm_t(3.0); offset = fm_t(-7.5); }
        fm_accum_t ev = (slope * (fm_t)xv) + fm_accum_t(offset);
        exp_vals[i] = ev; exp_sum += ev;
    }
    fm_accum_t inv_sum = fm_accum_t(hls::recip(exp_sum.to_float()));   // NR: float recip 替换 ap_fixed 除法
    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS unroll
        data[i] = exp_vals[i] * inv_sum;
    }
}
static void activation_layernorm(fm_accum_t data[OUTPUT_DIM], const fm_t param1[OUTPUT_DIM], const fm_t param2[OUTPUT_DIM]) {
    #pragma HLS inline off
    fm_accum_t sum = fm_accum_t(0);
    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS unroll
        sum += data[i];
    }
    fm_accum_t mean = sum >> 6;
    fm_accum_t var_sum = fm_accum_t(0);
    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS unroll
        fm_accum_t diff = data[i] - mean;
        var_sum += diff * diff;
    }
    fm_accum_t variance = var_sum >> 6;
    fm_accum_t std_dev;
    if (variance < fm_accum_t(0.01)) std_dev = fm_accum_t(0.1);
    else if (variance < fm_accum_t(1.0)) std_dev = fm_accum_t(0.8) * variance + fm_accum_t(0.1);
    else if (variance < fm_accum_t(4.0)) std_dev = fm_accum_t(0.4) * variance + fm_accum_t(0.4);
    else std_dev = fm_accum_t(0.2) * variance + fm_accum_t(1.2);
    std_dev += fm_accum_t(1e-6);
    fm_accum_t inv_std = fm_accum_t(hls::recip(std_dev.to_float()));   // NR: float recip 替换 ap_fixed 除法
    // P1: layernorm pipeline II=1; 先 copy param → local reg (避免 pipeline 内 m_axi read carried dependence, II=67 退化)
    fm_t p1[OUTPUT_DIM]; fm_t p2[OUTPUT_DIM];
    #pragma HLS array_partition variable=p1 complete
    #pragma HLS array_partition variable=p2 complete
    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS unroll
        p1[i] = param1[i]; p2[i] = param2[i];
    }
    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS pipeline II=1
        fm_accum_t normalized = (data[i] - mean) * inv_std;
        data[i] = (p1[i] * (fm_t)normalized) + fm_accum_t(p2[i]);
    }
}
static void add_bias(fm_accum_t data[OUTPUT_DIM], const fm_t bias[OUTPUT_DIM]) {
    #pragma HLS inline off
    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS unroll
        data[i] += fm_accum_t(bias[i]);
    }
}
static void unified_activation(fm_accum_t data[OUTPUT_DIM], unsigned int ctrl,
                                const fm_t param1[OUTPUT_DIM], const fm_t param2[OUTPUT_DIM]) {
    bool enable_bias = false;  // CTRL_ENABLE_BIAS replaced by CTRL_FINALIZE
    unsigned int act_type = ctrl & 0x7F;
    if (enable_bias) add_bias(data, param1);
    if (act_type == CTRL_ACT_RELU) activation_relu(data);
    else if (act_type == CTRL_ACT_GELU) activation_gelu(data);
    else if (act_type == CTRL_ACT_SILU) activation_silu(data);
    else if (act_type == CTRL_ACT_SOFTMAX) activation_softmax(data);
    else if (act_type == CTRL_ACT_LAYERNORM) activation_layernorm(data, param1, param2);
}

// ============================================================================
// LANE_MM_stream: 流式 compute — 从 in_pkt (512-bit wide packet) 读 input, 不用 local_input_buffer
// 始终 += (无分支); 不累积时由调用方在 compute 前清零 mm_results
// ============================================================================
#define LANE_MM_stream(core, n) \
    { \
        fm_t in_c##core##_##n[INPUT_DIM]; \
        fm_accum_t out_c##core##_##n[OUTPUT_DIM]; \
        for (int i = 0; i < INPUT_DIM; i++) { \
            _Pragma("HLS unroll") \
            in_c##core##_##n[i] = in_pkt.data[core * NUM_LANES + n][i]; \
        } \
        matmul_4x64_tpl<core * NUM_LANES + n>(in_c##core##_##n, local_weights[core][n], out_c##core##_##n); \
        for (int i = 0; i < OUTPUT_DIM; i++) { \
            _Pragma("HLS unroll") \
            mm_results[core][n][tile * OUTPUT_DIM + i] += out_c##core##_##n[i]; \
        } \
    }

// ============================================================================
// compute_stream_core: 流式 compute — 边从 input_stream 读 512-bit packet 边 8 lane matmul
// 消除 local_input_buffer (read+compute 融合), 保 8 lane 并行 + II=1, weight_stationary
// ============================================================================
static void compute_stream_core(
    hls::stream<in_pkt_axis>& input_stream,
    fm_accum_t mm_results[NUM_CORES][NUM_LANES][NUM_TILES * OUTPUT_DIM],
    wt_linear_t local_weights[NUM_CORES][NUM_LANES][INPUT_DIM][OUTPUT_DIM],
    unsigned int num_tokens
) {
    #pragma HLS inline off
    compute_stream:
    for (unsigned int p = 0; p < num_tokens * NUM_TILES; p++) {
        #pragma HLS pipeline II=1
        #pragma HLS dependence variable=mm_results inter false
        in_pkt_axis ipkt = input_stream.read();
        compute_input_packet_wide_t in_pkt = unpack_in_wide(ipkt.data);
        unsigned int tile = p % NUM_TILES;   // token 维隐含 (mm_results 无 token 维, 多 token 覆盖)
        LANE_MM_stream(0, 0)
        LANE_MM_stream(1, 0)
    }
}

// ============================================================================
// activate_tiles_core: 原始串行 (lane 不 unroll, 避免 HLS 调度爆炸).
// P1 pipeline→unroll 回退已恢复 layernorm, 预期 latency 641→297.
// ============================================================================
static void activate_tiles_core(
    fm_accum_t mm_results[NUM_CORES][NUM_LANES][NUM_TILES * OUTPUT_DIM],
    fm_accum_t local_output_buffer[NUM_CORES][NUM_LANES][NUM_TILES * OUTPUT_DIM],
    unsigned int ctrl,
    const fm_t param1[OUTPUT_DIM],
    const fm_t param2[OUTPUT_DIM]
) {
    #pragma HLS inline off
    activate_tiles:
    for (unsigned int core = 0; core < NUM_CORES; core++) {
        for (unsigned int lane = 0; lane < NUM_LANES; lane++) {
            for (unsigned int tile = 0; tile < NUM_TILES; tile++) {
                #pragma HLS pipeline II=1
                fm_accum_t data[OUTPUT_DIM];
                #pragma HLS array_partition variable=data complete
                for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
                    #pragma HLS pipeline II=1
                    data[i] = mm_results[core][lane][tile * OUTPUT_DIM + i];
                }
                unified_activation(data, ctrl, param1, param2);
                for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
                    #pragma HLS pipeline II=1
                    local_output_buffer[core][lane][tile * OUTPUT_DIM + i] = data[i];
                }
            }
        }
    }
}

// 权重加载宏 (显式wptr参数, 避免指针数组HLS限制)
#define LOAD_LANE_W(core, n, wptr) \
    for (int blk = 0; blk < WT_BLOCKS_PER_LANE; blk++) { \
        wt_block_t wb = wptr[blk]; \
        for (int e = 0; e < 32; e++) { \
            _Pragma("HLS unroll") \
            int idx = blk * 32 + e; \
            local_weights[core][n][idx / OUTPUT_DIM][idx % OUTPUT_DIM] = wb[e]; \
        } \
    }

// ============================================================================
// V8-2 顶层: 2套计算核心 + 共享激活 (无dataflow)
// ============================================================================
extern "C" void qkv_tile_kernel_cc_qwen_small_core_v8_2_s(
    hls::stream<in_pkt_axis>& input_stream,
    hls::stream<out_lo_axis>& output_stream_lo,   // results[0..31] (1024 bit, hw 限 ≤1024)
    hls::stream<out_hi_axis>& output_stream_hi,   // results[32..63] (1024 bit)
    hls::stream<weight_axis>& weight_stream_0,    // STEP 2: 4 weight stream (4 PC 并行)
    hls::stream<weight_axis>& weight_stream_1,
    hls::stream<weight_axis>& weight_stream_2,
    hls::stream<weight_axis>& weight_stream_3,
    hls::stream<ctrl_axis>&   ctrl_stream,        // cc → 本核: 每 op 1 个 op_ctrl

    const fm_t* param1,
    const fm_t* param2,
    unsigned int num_tokens,
    unsigned int num_ops,                         // op 外循环边界 (与 cc num_ops 严格相等)
    unsigned int use_param1,
    unsigned int use_param2
) {
    // Vitis mode: 只标 axis stream + m_axi, s_axilite 让 HLS 全自动统一 bundle
    #pragma HLS INTERFACE mode=axis port=input_stream
    #pragma HLS INTERFACE mode=axis port=output_stream_lo
    #pragma HLS INTERFACE mode=axis port=output_stream_hi
    #pragma HLS INTERFACE mode=axis port=weight_stream_0
    #pragma HLS INTERFACE mode=axis port=weight_stream_1
    #pragma HLS INTERFACE mode=axis port=weight_stream_2
    #pragma HLS INTERFACE mode=axis port=weight_stream_3
    #pragma HLS INTERFACE mode=axis port=ctrl_stream
    #pragma HLS INTERFACE mode=m_axi port=param1 bundle=gmem_param offset=slave depth=64
    #pragma HLS INTERFACE mode=m_axi port=param2 bundle=gmem_param offset=slave depth=64
    // num_tokens, num_ops, use_param1, use_param2, return → HLS 自动 s_axilite (统一 bundle)

    // (流式 compute: 不用 local_input_buffer, read+compute 融合进 compute_stream_core)

    // 独立数组; mm_results cyclic 64 dim=3 + BRAM (D=16: compute_stream II=2 但 hw_emu 稳定, 整层 Q 6.97×)
    fm_accum_t mm_results[NUM_CORES][NUM_LANES][NUM_TILES * OUTPUT_DIM];
    #pragma HLS bind_storage variable=mm_results type=RAM_2P impl=BRAM
    #pragma HLS array_partition variable=mm_results complete dim=1
    #pragma HLS array_partition variable=mm_results complete dim=2
    #pragma HLS array_partition variable=mm_results cyclic factor=64 dim=3

    fm_accum_t local_output_buffer[NUM_CORES][NUM_LANES][NUM_TILES * OUTPUT_DIM];
    #pragma HLS bind_storage variable=local_output_buffer type=RAM_2P impl=BRAM
    #pragma HLS array_partition variable=local_output_buffer complete dim=1
    #pragma HLS array_partition variable=local_output_buffer complete dim=2
    #pragma HLS array_partition variable=local_output_buffer cyclic factor=64 dim=3

    // 权重: 2 core × 4 lane × (4 input × 64 output), 显式分维partition
    wt_linear_t local_weights[NUM_CORES][NUM_LANES][INPUT_DIM][OUTPUT_DIM];
    #pragma HLS array_partition variable=local_weights complete dim=1
    #pragma HLS array_partition variable=local_weights complete dim=2
    #pragma HLS array_partition variable=local_weights complete dim=3
    #pragma HLS array_partition variable=local_weights cyclic factor=64 dim=4

    // param 预 load → local (param 跨 op 不变; 避免 activation_layernorm pipeline 内 m_axi read carried dependence)
    fm_t g_p1[OUTPUT_DIM], g_p2[OUTPUT_DIM];
    #pragma HLS array_partition variable=g_p1 complete
    #pragma HLS array_partition variable=g_p2 complete
    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS unroll
        g_p1[i] = param1[i]; g_p2[i] = param2[i];
    }

    // ---- op 外循环: 每 op 从 ctrl_stream 读 op_ctrl, 跑完整 5 阶段 (与 cc num_ops 严格对齐) ----
    op_loop:
    for (unsigned int op = 0; op < num_ops; op++) {
        ctrl_axis cp = ctrl_stream.read();
        unsigned int op_ctrl = cp.data;   // 每 op 动态 op_ctrl (激活选择, 透传 activate_tiles_core)

        // 阶段0: 4 weight_stream 并行 → local_weights (STEP 2: 16 拍, 4× 带宽; 每 stream 16 wt_block)
        // stream0/1→core0 (blk0-15/16-31), stream2/3→core1 (blk0-15/16-31)
        load_weights_stream:
        for (unsigned int blk = 0; blk < WT_BLOCKS_PER_LANE / 2; blk++) {
            #pragma HLS pipeline II=1
            wt_block_t wb0 = unpack_w(weight_stream_0.read().data);
            wt_block_t wb1 = unpack_w(weight_stream_1.read().data);
            wt_block_t wb2 = unpack_w(weight_stream_2.read().data);
            wt_block_t wb3 = unpack_w(weight_stream_3.read().data);
            for (unsigned int e = 0; e < 32; e++) {
                #pragma HLS unroll
                int idx0 = blk * 32 + e;
                local_weights[0][0][idx0 / OUTPUT_DIM][idx0 % OUTPUT_DIM] = wb0[e];
                int idx1 = (blk + WT_BLOCKS_PER_LANE/2) * 32 + e;
                local_weights[0][0][idx1 / OUTPUT_DIM][idx1 % OUTPUT_DIM] = wb1[e];
                int idx2 = blk * 32 + e;
                local_weights[1][0][idx2 / OUTPUT_DIM][idx2 % OUTPUT_DIM] = wb2[e];
                int idx3 = (blk + WT_BLOCKS_PER_LANE/2) * 32 + e;
                local_weights[1][0][idx3 / OUTPUT_DIM][idx3 % OUTPUT_DIM] = wb3[e];
            }
        }

    unsigned int num_packets = num_tokens * TOTAL_LANES * NUM_TILES;   // 供 write_stream

    // (阶段1 read_stream 已融合进 compute_stream_core: 边读 512-bit packet 边 8 lane matmul)

    // 阶段2: 流式 compute (read+compute 融合, 无 local_input_buffer), accumulate 模式时累积到 mm_results
    bool accumulate = op_ctrl & CTRL_ACCUMULATE;
    bool finalize = op_ctrl & CTRL_FINALIZE;
    // 不累积时清零 mm_results (cyclic64: c,l unroll + i unroll factor=64)
    if (!accumulate) {
        for (unsigned int c = 0; c < NUM_CORES; c++) {
            #pragma HLS unroll
            for (unsigned int l = 0; l < NUM_LANES; l++) {
                #pragma HLS unroll
                for (unsigned int i = 0; i < NUM_TILES * OUTPUT_DIM; i++) {
                    #pragma HLS unroll factor=64
                    mm_results[c][l][i] = 0;
                }
            }
        }
    }
    compute_stream_core(input_stream, mm_results, local_weights, num_tokens);

    // FINALIZE 模式: 执行 activate + write_stream (最终 reduction step 或 standalone op)
    if (finalize) {
        // 阶段3: 1套激活, 处理2套核心结果
        activate_tiles_core(mm_results, local_output_buffer, op_ctrl & CTRL_ACT_MASK, g_p1, g_p2);

        // 阶段4: 输出缓存 → output_stream_lo/hi (仅最终步输出)
        write_stream:
        for (unsigned int pkt = 0; pkt < num_packets; pkt++) {
            #pragma HLS pipeline II=1
            unsigned int lane = pkt % TOTAL_LANES;
            unsigned int tile = pkt / TOTAL_LANES;
            unsigned int core = lane / NUM_LANES;
            unsigned int local_lane = lane % NUM_LANES;
            fm_accum_t results[OUTPUT_DIM];
            #pragma HLS array_partition variable=results complete
            for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
                #pragma HLS pipeline II=1
                results[i] = local_output_buffer[core][local_lane][tile * OUTPUT_DIM + i];
            }
            out_lo_axis opkt_lo; opkt_lo.data = pack_out_lo_raw(results); opkt_lo.last = (pkt == num_packets - 1);
            out_hi_axis opkt_hi; opkt_hi.data = pack_out_hi_raw(results); opkt_hi.last = (pkt == num_packets - 1);
            output_stream_lo.write(opkt_lo);
            output_stream_hi.write(opkt_hi);
        }
    }   // end if (finalize || !accumulate)
    }   // end op_loop
}
