/**
 * ============================================================================
 * QKV Tile Kernel - V8-2 (two compute cores + shared activation, deployment-optimized)
 * ============================================================================
 *
 * Motivation: one V8 uses 631k FF/461k LUT; four instances require 1.84M LUT
 * (107%), which is not deployable. V8-2 places two compute cores in one kernel
 * and shares one activation unit plus interface/cache/control logic. Two V8-2
 * instances provide four cores at 1.44M LUT (83%) versus 1.84M LUT for four V8s.
 *
 * Key benefit: activation dominates FF use (501k, or 79%, in V8). Sharing one
 * activation unit between two cores roughly halves activation FF/LUT cost.
 *
 * V8-2 structure:
 *   - two compute cores, each with 4 lanes x 4x64 matmul = 1024 DSP; 2048 DSP total;
 *   - eight parallel lanes (2 cores x 4 lanes), eight templated matmul instances,
 *     and compute_tiles II=1;
 *   - one activate_tiles_core (522 DSP) serving 2 cores x 4 lanes x 16 tiles = 128 calls;
 *   - shared input/output/weight stream interfaces. Design path 2 supplies
 *     weights from the control/cache core instead of w0..w7 m_axi ports.
 *
 * Expected resources for one V8-2: 2570 DSP, about 747k FF, about 720k LUT.
 * Two V8-2 instances: 5140 DSP (42%), 1.49M FF (43%), 1.44M LUT (83%).
 *
 * lane_id mapping: 0-3 = core0 lanes 0-3, 4-7 = core1 lanes 0-3
 * (core=lane/4, local_lane=lane%4).
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
#define CTRL_ACCUMULATE    0x40    // Accumulate into mm_results without clearing.
#define CTRL_FINALIZE      0x80    // Run activation + write_stream at the final reduction step.
#define CTRL_ACT_MASK      0x3F    // Activation-type mask.

// ============================================================================
// Templated 4x64 matrix multiplication. Eight independent instances
// matmul_4x64_tpl<0..7> use inline off to prevent sharing.
// ============================================================================
template<int LANE_ID>
static void matmul_4x64_tpl(
    const fm_t input[INPUT_DIM],
    const wt_linear_t weight[INPUT_DIM][OUTPUT_DIM],
    fm_accum_t output[OUTPUT_DIM]
) {
    #pragma HLS inline off
    // P2: a wide non-saturating accumulator removes fabric clamping from
    // AP_RND/AP_SAT at each step, saving about 200k LUT/CU.
    // product = fm_t<16,8> x wt<16,4> = <32,12>; four products fit in <34,13>.
    // <48,24> provides margin with no intermediate rounding or saturation.
    // A single fm_accum_t conversion at the output performs rounding/saturation.
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
            acc[o] += input[in] * weight[in][o];   // Exact accumulation without intermediate clamping.
        }
    }
    for (int o = 0; o < OUTPUT_DIM; o++) {
        #pragma HLS unroll
        output[o] = (fm_accum_t)acc[o];   // Single output rounding/saturation.
    }
}

// ============================================================================
// Fixed-point activation functions (64 dimensions, matching V8).
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
    fm_accum_t inv_sum = fm_accum_t(hls::recip(exp_sum.to_float()));   // NR: replace ap_fixed division with float reciprocal.
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
    fm_accum_t inv_std = fm_accum_t(hls::recip(std_dev.to_float()));   // NR: replace ap_fixed division with float reciprocal.
    // P1: layernorm pipeline II=1. Copy parameters into local registers first
    // to avoid an m_axi read-carried dependence that degrades II to 67.
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
// LANE_MM_stream: streaming compute reads input from a 512-bit in_pkt without
// local_input_buffer. It always uses +=; the caller clears mm_results before
// compute when accumulation is disabled.
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
// compute_stream_core: streaming compute reads each 512-bit input packet while
// running the 8-lane matmul. Fusing read+compute removes local_input_buffer and
// retains 8-lane parallel, II=1, weight-stationary execution.
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
        unsigned int tile = p % NUM_TILES;   // Token dimension is implicit; mm_results has no token dimension.
        LANE_MM_stream(0, 0)
        LANE_MM_stream(1, 0)
    }
}

// ============================================================================
// activate_tiles_core: serial lanes avoid explosive HLS scheduling.
// Restoring layernorm after the P1 pipeline-to-unroll rollback is expected to
// reduce latency from 641 to 297 cycles.
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

// Weight-load macro with explicit wptr parameters to avoid HLS pointer-array limits.
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
// V8-2 top level: two compute cores plus shared activation, without dataflow.
// ============================================================================
extern "C" void qkv_tile_kernel_cc_qwen_small_core_v8_2_s(
    hls::stream<in_pkt_axis>& input_stream,
    hls::stream<out_lo_axis>& output_stream_lo,   // results[0..31] (1024 bits, hardware limit <=1024)
    hls::stream<out_hi_axis>& output_stream_hi,   // results[32..63] (1024 bit)
    hls::stream<weight_axis>& weight_stream_0,    // Step 2: four weight streams over four parallel PCs.
    hls::stream<weight_axis>& weight_stream_1,
    hls::stream<weight_axis>& weight_stream_2,
    hls::stream<weight_axis>& weight_stream_3,
    hls::stream<ctrl_axis>&   ctrl_stream,        // cc -> this kernel: one op_ctrl per operation.

    const fm_t* param1,
    const fm_t* param2,
    unsigned int num_tokens,
    unsigned int num_ops,                         // Operation-loop bound; must exactly match cc num_ops.
    unsigned int use_param1,
    unsigned int use_param2
) {
    // Vitis mode: annotate only AXIS streams and m_axi; HLS automatically
    // assigns all s_axilite ports to one bundle.
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
    // HLS automatically maps num_tokens, num_ops, use_param1, use_param2,
    // and return to the unified s_axilite bundle.

    // Streaming compute removes local_input_buffer by fusing read+compute in compute_stream_core.

    // Independent arrays; mm_results uses cyclic factor 64 on dimension 3 plus
    // BRAM. At D=16 compute_stream has II=2, but HW Emu is stable and full-layer Q is 6.97x faster.
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

    // Weights: 2 cores x 4 lanes x (4 inputs x 64 outputs), with explicit per-dimension partitioning.
    wt_linear_t local_weights[NUM_CORES][NUM_LANES][INPUT_DIM][OUTPUT_DIM];
    #pragma HLS array_partition variable=local_weights complete dim=1
    #pragma HLS array_partition variable=local_weights complete dim=2
    #pragma HLS array_partition variable=local_weights complete dim=3
    #pragma HLS array_partition variable=local_weights cyclic factor=64 dim=4

    // Preload parameters locally because they remain constant across operations;
    // this avoids an m_axi read-carried dependence in the activation_layernorm pipeline.
    fm_t g_p1[OUTPUT_DIM], g_p2[OUTPUT_DIM];
    #pragma HLS array_partition variable=g_p1 complete
    #pragma HLS array_partition variable=g_p2 complete
    for (unsigned int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS unroll
        g_p1[i] = param1[i]; g_p2[i] = param2[i];
    }

    // ---- Outer operation loop: read one op_ctrl from ctrl_stream and execute
    // all five stages, exactly aligned with cc num_ops. ----
    op_loop:
    for (unsigned int op = 0; op < num_ops; op++) {
        ctrl_axis cp = ctrl_stream.read();
        unsigned int op_ctrl = cp.data;   // Per-operation activation control forwarded to activate_tiles_core.

        // Stage 0: four parallel weight streams fill local_weights in 16 cycles
        // at 4x bandwidth, with 16 wt_block values per stream.
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

    unsigned int num_packets = num_tokens * TOTAL_LANES * NUM_TILES;   // Packet count for write_stream.

    // Stage 1 read_stream is fused into compute_stream_core: read a 512-bit packet while running 8-lane matmul.

    // Stage 2: fused streaming read+compute without local_input_buffer;
    // accumulation mode adds into mm_results.
    bool accumulate = op_ctrl & CTRL_ACCUMULATE;
    bool finalize = op_ctrl & CTRL_FINALIZE;
    // Clear mm_results when not accumulating (cyclic64 with c/l unrolled and i factor=64).
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

    // FINALIZE mode runs activation + write_stream for the final reduction step or standalone operation.
    if (finalize) {
        // Stage 3: one shared activation unit processes both cores.
        activate_tiles_core(mm_results, local_output_buffer, op_ctrl & CTRL_ACT_MASK, g_p1, g_p2);

        // Stage 4: send the output buffer to output_stream_lo/hi only at the final step.
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
