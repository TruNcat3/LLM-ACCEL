/**
 * ============================================================================
 * control_cache_core — operator_program 编排器 (dataflow + 512-bit store)
 * ============================================================================
 * op_loop + 512-bit 打包写 + dataflow 任务级流水。
 * cc 拆成 3 个 dataflow process, 经 hls::stream 连成标准 DAG (无 array 参数, 无死锁):
 *
 *   cc_dispatch ──iparam──▶ cc_input_path  (load_in + send ctrl/weight/input)
 *        │                   │  (独占 gmem0 hidden_in + gmem2 weight_hbm)
 *        └──oparam──▶ cc_output_path (collect + store_out 512-bit)
 *                            │  (独占 gmem1 hidden_out)
 *   cc_input_path ⇄ cc_output_path 经 V8-2_s stream 天然同步 (外部 kernel)
 *
 * 任务级流水: store_out(op) 与 load_in+send(op+1) 重叠 (经 dispatch stream 喂参数)。
 *
 * dataflow 规范性 (绕开已知陷阱):
 *   - canonical region: 只含函数调用 + 内部 hls::stream (无 array 参数 → 无 214-114/200-471)
 *   - op 参数经 stream 分发 (非 array) → 无 200-1449 (吞吐) / 200-1614 (cosim 死锁)
 *   - 3 process 各独占 m_axi bundle: dispatch=gmem3, input=gmem0/gmem2, output=gmem1
 *     → 无 bundle 共享 (HLS 200-984 根因)
 *
 * 回退点: kernel/control_cache_core_packed_v1.cpp.bak (打包写版, 无 dataflow, 已三关验证)
 * ============================================================================
 */
#include "../include/control_cache_packets.hpp"
#include <hls_stream.h>

#define WT_PER_V82        (TOTAL_LANES * WT_BLOCKS_PER_LANE)   // 64 wt_block per V8-2_s
#define PKT_PER_TOKEN     (TOTAL_LANES * NUM_TILES)            // 128
#define MAX_TOKENS        2
#define MAX_OPS           512

// op 参数包 (dispatch → input/output path 经 hls::stream 分发)
struct op_param_t {
    unsigned int ctrl;     // op_ctrl (激活+bias, 透传 V8-2_s)
    unsigned int woff;     // weight_offset
    unsigned int isrc;     // input_source
    unsigned int ioff;     // input_hbm_offset
    unsigned int ofinal;   // FINALIZE 标志 (ctrl & 0x80)
    unsigned int odest;    // output_dest
    unsigned int ooff;     // output_hbm_offset (fm_accum_t 元素语义)
};

// ---- cc_dispatch: 读 op_program (独占 gmem3), 每个op打包分发到 iparam/oparam stream ----
static void cc_dispatch(const unsigned int* op_program, unsigned int num_ops,
                        hls::stream<op_param_t>& iparam_stream,
                        hls::stream<op_param_t>& oparam_stream) {
    dispatch:
    for (unsigned int op = 0; op < num_ops; op++) {
        #pragma HLS pipeline II=1
        op_param_t p;
        unsigned int base = op * OP_TASK_STRIDE;
        p.ctrl   = op_program[base + 0];
        p.woff   = op_program[base + 1];
        p.isrc   = op_program[base + 2];
        p.ioff   = op_program[base + 3];
        p.ofinal = (p.ctrl & 0x80) ? 1 : 0;   // CTRL_FINALIZE
        p.odest  = op_program[base + 4];
        p.ooff   = op_program[base + 5];
        iparam_stream.write(p);
        oparam_stream.write(p);
    }
}

// ---- cc_input_path: 从 iparam 收 op 参数 + load_in + send ctrl/weight/input ----
//    独占 gmem0 hidden_in + gmem2 weight_hbm
static void cc_input_path(
    const fm_t* hidden_in,
    const wt_block_t* weight_hbm_0, const wt_block_t* weight_hbm_1,
    const wt_block_t* weight_hbm_2, const wt_block_t* weight_hbm_3,
    hls::stream<op_param_t>& iparam_stream,
    unsigned int num_ops, unsigned int ntok,
    hls::stream<ctrl_axis>& ctrl_stream,
    hls::stream<weight_axis>& weight_stream_0,
    hls::stream<weight_axis>& weight_stream_1,
    hls::stream<weight_axis>& weight_stream_2,
    hls::stream<weight_axis>& weight_stream_3,
    hls::stream<in_pkt_axis>& to_compute_0) {

    fm_t gbuf_in[MAX_TOKENS][NUM_TILES][INPUT_DIM];
    #pragma HLS bind_storage variable=gbuf_in type=RAM_2P impl=BRAM
    #pragma HLS array_partition variable=gbuf_in complete dim=3
    unsigned int prev_src = 0xFFFFFFFF, prev_off = 0xFFFFFFFF;

    ip_op_loop:
    for (unsigned int op = 0; op < num_ops; op++) {
        op_param_t p = iparam_stream.read();

        // (a) 输入载入 (复用逻辑: SRC_HBM 且首op/输入变了才 reload)
        if (p.isrc == SRC_HBM) {
            bool need_reload = (op == 0) || (p.isrc != prev_src) || (p.ioff != prev_off);
            if (need_reload) {
                load_in:
                for (unsigned int t = 0; t < ntok; t++) {
                    for (unsigned int a = 0; a < NUM_TILES * INPUT_DIM; a++) {
                        #pragma HLS pipeline II=1
                        gbuf_in[t][a / INPUT_DIM][a % INPUT_DIM] =
                            hidden_in[p.ioff + t * NUM_TILES * INPUT_DIM + a];
                    }
                }
            }
            prev_src = p.isrc; prev_off = p.ioff;
        }

        // (b) 发 ctrl (op_ctrl 透传 V8-2_s), 必须在 weight/input 前
        ctrl_axis cp; cp.data = p.ctrl; cp.last = (op == num_ops - 1); cp.keep = -1;
        ctrl_stream.write(cp);

        // (c) 发 weight (STEP 2: 4 m_axi 并行 + 4 stream, 16 拍 4× 带宽; 每 m_axi 16 wt_block)
        send_weight:
        for (unsigned int b = 0; b < WT_PER_V82 / 4; b++) {
            #pragma HLS pipeline II=1
            wt_block_t wb0 = weight_hbm_0[p.woff/4 + b];
            wt_block_t wb1 = weight_hbm_1[p.woff/4 + b];
            wt_block_t wb2 = weight_hbm_2[p.woff/4 + b];
            wt_block_t wb3 = weight_hbm_3[p.woff/4 + b];
            weight_axis wp0; wp0.data = pack_w(wb0); wp0.last = (b == WT_PER_V82/4 - 1); wp0.keep = -1;
            weight_axis wp1; wp1.data = pack_w(wb1); wp1.last = (b == WT_PER_V82/4 - 1); wp1.keep = -1;
            weight_axis wp2; wp2.data = pack_w(wb2); wp2.last = (b == WT_PER_V82/4 - 1); wp2.keep = -1;
            weight_axis wp3; wp3.data = pack_w(wb3); wp3.last = (b == WT_PER_V82/4 - 1); wp3.keep = -1;
            weight_stream_0.write(wp0);
            weight_stream_1.write(wp1);
            weight_stream_2.write(wp2);
            weight_stream_3.write(wp3);
        }

        // (d) 发 input (512-bit wide: 每 tile 1 packet, 8 lane × INPUT_DIM; broadcast 同 input)
        send_input:
        for (unsigned int t = 0; t < ntok; t++) {
            for (unsigned int tile = 0; tile < NUM_TILES; tile++) {
                #pragma HLS pipeline II=1
                compute_input_packet_wide_t p;
                for (unsigned int lane = 0; lane < TOTAL_LANES; lane++) {
                    #pragma HLS unroll
                    for (unsigned int i = 0; i < INPUT_DIM; i++) {
                        #pragma HLS unroll
                        p.data[lane][i] = gbuf_in[t][tile][i];   // broadcast: 8 lane 同 input (weight_stationary)
                    }
                }
                in_pkt_axis ip; ip.data = pack_in_wide(p); ip.last = (t == ntok-1 && tile == NUM_TILES-1);
                to_compute_0.write(ip);
            }
        }
    }
}

// ---- cc_output_path: 从 oparam 收 op 参数 + collect + store_out (512-bit 打包写) ----
//    独占 gmem1 hidden_out
static void cc_output_path(
    ap_uint<512>* hidden_out,
    hls::stream<op_param_t>& oparam_stream,
    unsigned int num_ops, unsigned int ntok,
    hls::stream<out_lo_axis>& from_compute_0_lo,
    hls::stream<out_hi_axis>& from_compute_0_hi) {

    ap_uint<512> gbuf_out_w[MAX_TOKENS][2][PKT_PER_TOKEN][OUTPUT_DIM/16];
    #pragma HLS bind_storage variable=gbuf_out_w type=RAM_2P impl=BRAM
    #pragma HLS array_partition variable=gbuf_out_w complete dim=2
    #pragma HLS array_partition variable=gbuf_out_w complete dim=4

    op_op_loop:
    for (unsigned int op = 0; op < num_ops; op++) {
        op_param_t p = oparam_stream.read();
        // 仅 FINALIZE op 收+写 (V8-2_s accumulate 模式不写 output stream, 避免流死锁)
        if (p.ofinal) {
            // (e) 收 output → gbuf_out_w (512-bit 字打包)
            collect:
            for (unsigned int t = 0; t < ntok; t++) {
                for (unsigned int pkt = 0; pkt < PKT_PER_TOKEN; pkt++) {
                    #pragma HLS pipeline II=1
                    out_lo_axis lo0 = from_compute_0_lo.read();
                    out_hi_axis hi0 = from_compute_0_hi.read();
                    fm_accum_t res0[OUTPUT_DIM];
                    #pragma HLS array_partition variable=res0 complete
                    unpack_out(lo0.data, hi0.data, res0);
                    for (unsigned int w = 0; w < OUTPUT_DIM/16; w++) {
                        #pragma HLS unroll
                        ap_uint<512> wd = 0;
                        for (unsigned int e = 0; e < 16; e++) {
                            #pragma HLS unroll
                            wd.range(e*32+31, e*32) = res0[w*16 + e].range(31, 0);
                        }
                        gbuf_out_w[t][0][pkt][w] = wd;
                    }
                }
            }
            // (f) 路由输出 (512-bit 打包写: ooff 保持 host 的 fm_accum_t 语义, /16 转字偏移)
            if (p.odest == DST_HBM) {
                store_out:
                for (unsigned int t = 0; t < ntok; t++) {
                    for (unsigned int idx = 0; idx < PKT_PER_TOKEN * (OUTPUT_DIM/16); idx++) {
                        #pragma HLS pipeline II=1
                        hidden_out[p.ooff/16 + t * PKT_PER_TOKEN * (OUTPUT_DIM/16) + idx] =
                            gbuf_out_w[t][0][idx / (OUTPUT_DIM/16)][idx % (OUTPUT_DIM/16)];
                    }
                }
            }
        }
    }
}

extern "C" void control_cache_core(
    const fm_t* hidden_in,
    ap_uint<512>* hidden_out,
    const wt_block_t* weight_hbm_0, const wt_block_t* weight_hbm_1,
    const wt_block_t* weight_hbm_2, const wt_block_t* weight_hbm_3,

    hls::stream<in_pkt_axis>&  to_compute_0,
    hls::stream<weight_axis>&  weight_stream_0,
    hls::stream<weight_axis>&  weight_stream_1,
    hls::stream<weight_axis>&  weight_stream_2,
    hls::stream<weight_axis>&  weight_stream_3,
    hls::stream<out_lo_axis>& from_compute_0_lo,
    hls::stream<out_hi_axis>& from_compute_0_hi,
    hls::stream<ctrl_axis>&   ctrl_stream,          // cc → V8-2_s: 每 op 1 个 op_ctrl

    const unsigned int* op_program,                 // m_axi: num_ops×6 uint (op_task_t 平坦)
    unsigned int num_ops,
    unsigned int num_tokens
) {
    #pragma HLS INTERFACE mode=m_axi port=hidden_in bundle=gmem0 offset=slave
    #pragma HLS INTERFACE mode=m_axi port=hidden_out bundle=gmem1 offset=slave
    #pragma HLS INTERFACE mode=m_axi port=weight_hbm_0 bundle=gmem2 offset=slave
    #pragma HLS INTERFACE mode=m_axi port=weight_hbm_1 bundle=gmem2b offset=slave
    #pragma HLS INTERFACE mode=m_axi port=weight_hbm_2 bundle=gmem2c offset=slave
    #pragma HLS INTERFACE mode=m_axi port=weight_hbm_3 bundle=gmem2d offset=slave
    #pragma HLS INTERFACE mode=m_axi port=op_program bundle=gmem3 offset=slave
    #pragma HLS INTERFACE mode=axis port=to_compute_0
    #pragma HLS INTERFACE mode=axis port=weight_stream_0
    #pragma HLS INTERFACE mode=axis port=weight_stream_1
    #pragma HLS INTERFACE mode=axis port=weight_stream_2
    #pragma HLS INTERFACE mode=axis port=weight_stream_3
    #pragma HLS INTERFACE mode=axis port=from_compute_0_lo
    #pragma HLS INTERFACE mode=axis port=from_compute_0_hi
    #pragma HLS INTERFACE mode=axis port=ctrl_stream
    #pragma HLS INTERFACE mode=s_axilite port=hidden_in bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=hidden_out bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=weight_hbm_0 bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=weight_hbm_1 bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=weight_hbm_2 bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=weight_hbm_3 bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=op_program bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=num_ops bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=num_tokens bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    unsigned int ntok = num_tokens;
    if (ntok == 0) ntok = 1;
    if (ntok > MAX_TOKENS) ntok = MAX_TOKENS;

    // dataflow 内部 stream: dispatch → input_path / output_path (op 参数分发)
    hls::stream<op_param_t> iparam_stream("iparam_stream");
    hls::stream<op_param_t> oparam_stream("oparam_stream");
    #pragma HLS stream variable=iparam_stream depth=4
    #pragma HLS stream variable=oparam_stream depth=4

    // canonical dataflow region: 3 函数调用 + 内部 stream (无 array 参数, 无死锁)
    #pragma HLS dataflow
    cc_dispatch(op_program, num_ops, iparam_stream, oparam_stream);
    cc_input_path(hidden_in, weight_hbm_0, weight_hbm_1, weight_hbm_2, weight_hbm_3,
                  iparam_stream, num_ops, ntok,
                  ctrl_stream, weight_stream_0, weight_stream_1, weight_stream_2, weight_stream_3,
                  to_compute_0);
    cc_output_path(hidden_out, oparam_stream, num_ops, ntok,
                   from_compute_0_lo, from_compute_0_hi);
}
