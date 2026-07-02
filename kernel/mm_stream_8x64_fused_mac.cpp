#include "mm_stream_8x64.hpp"

typedef ap_int<48> mm_stream_8x64_dsp_accum_t;
typedef ap_fixed<48, 28, AP_TRN, AP_WRAP> mm_stream_8x64_dsp_fixed_t;

struct mm_stream_8x64_dsp_bank_t {
    mm_stream_8x64_dsp_accum_t
        value[MM_STREAM_8X64_TOKENS][MM_STREAM_8X64_OUTPUTS];
};

static mm_stream_8x64_dsp_accum_t mm_stream_8x64_dsp_madd(
    fm_t activation,
    wt_linear_t weight,
    mm_stream_8x64_dsp_accum_t accumulated
) {
    #pragma HLS inline
    ap_int<16> activation_raw = activation.range(15, 0);
    ap_int<16> weight_raw = weight.range(15, 0);
    return mm_stream_8x64_dsp_accum_t(
        activation_raw * weight_raw + accumulated
    );
}

static void rotate_mm_stream_8x64_dsp_accumulators(
    mm_stream_8x64_dsp_bank_t& slot0,
    mm_stream_8x64_dsp_bank_t& slot1,
    mm_stream_8x64_dsp_bank_t& slot2,
    mm_stream_8x64_dsp_bank_t& slot3,
    const mm_stream_8x64_activation_packet_t& activation,
    const mm_stream_8x64_weight_packet_t& weight0,
    const mm_stream_8x64_weight_packet_t& weight1,
    const mm_stream_8x64_weight_packet_t& weight2,
    const mm_stream_8x64_weight_packet_t& weight3
) {
    #pragma HLS inline
    #pragma HLS array_partition variable=slot0.value complete dim=0
    #pragma HLS array_partition variable=slot1.value complete dim=0
    #pragma HLS array_partition variable=slot2.value complete dim=0
    #pragma HLS array_partition variable=slot3.value complete dim=0
    #pragma HLS array_partition variable=activation.data complete dim=1
    #pragma HLS array_partition variable=weight0.data complete dim=1
    #pragma HLS array_partition variable=weight1.data complete dim=1
    #pragma HLS array_partition variable=weight2.data complete dim=1
    #pragma HLS array_partition variable=weight3.data complete dim=1

    for (unsigned int token = 0; token < MM_STREAM_8X64_TOKENS; token++) {
        #pragma HLS unroll
        fm_t a = activation.data[token];
        for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
            #pragma HLS unroll
            const unsigned int out0 = lane;
            const unsigned int out1 = CU_VEC_LANES + lane;
            const unsigned int out2 = 2 * CU_VEC_LANES + lane;
            const unsigned int out3 = 3 * CU_VEC_LANES + lane;

            mm_stream_8x64_dsp_accum_t next0 = mm_stream_8x64_dsp_madd(
                a, weight0.data[lane], slot0.value[token][out0]
            );
            mm_stream_8x64_dsp_accum_t next1 = mm_stream_8x64_dsp_madd(
                a, weight1.data[lane], slot0.value[token][out1]
            );
            mm_stream_8x64_dsp_accum_t next2 = mm_stream_8x64_dsp_madd(
                a, weight2.data[lane], slot0.value[token][out2]
            );
            mm_stream_8x64_dsp_accum_t next3 = mm_stream_8x64_dsp_madd(
                a, weight3.data[lane], slot0.value[token][out3]
            );

            slot0.value[token][out0] = slot1.value[token][out0];
            slot0.value[token][out1] = slot1.value[token][out1];
            slot0.value[token][out2] = slot1.value[token][out2];
            slot0.value[token][out3] = slot1.value[token][out3];
            slot1.value[token][out0] = slot2.value[token][out0];
            slot1.value[token][out1] = slot2.value[token][out1];
            slot1.value[token][out2] = slot2.value[token][out2];
            slot1.value[token][out3] = slot2.value[token][out3];
            slot2.value[token][out0] = slot3.value[token][out0];
            slot2.value[token][out1] = slot3.value[token][out1];
            slot2.value[token][out2] = slot3.value[token][out2];
            slot2.value[token][out3] = slot3.value[token][out3];
            slot3.value[token][out0] = next0;
            slot3.value[token][out1] = next1;
            slot3.value[token][out2] = next2;
            slot3.value[token][out3] = next3;
        }
    }
}

static fm_accum_t mm_stream_8x64_dsp_to_fm_accum(
    mm_stream_8x64_dsp_accum_t raw
) {
    #pragma HLS inline
    mm_stream_8x64_dsp_fixed_t fixed;
    fixed.range(47, 0) = raw.range(47, 0);
    return fm_accum_t(fixed);
}

template <unsigned int GROUP>
static void emit_mm_stream_8x64_dsp_group(
    hls::stream<cu_accum16_packet_t>& out_stream,
    const mm_stream_8x64_dsp_bank_t& slot0,
    const mm_stream_8x64_dsp_bank_t& slot1,
    const mm_stream_8x64_dsp_bank_t& slot2,
    const mm_stream_8x64_dsp_bank_t& slot3,
    const mm_stream_8x64_task_t& task,
    unsigned int token
) {
    #pragma HLS inline
    #pragma HLS array_partition variable=slot0.value complete dim=0
    #pragma HLS array_partition variable=slot1.value complete dim=0
    #pragma HLS array_partition variable=slot2.value complete dim=0
    #pragma HLS array_partition variable=slot3.value complete dim=0

    cu_accum16_packet_t packet;
    packet.valid_mask = 0xFFFF;
    packet.token_lane = token;
    packet.elem_base = task.elem_base + GROUP * CU_VEC_LANES;
    packet.block_id = task.block_id;
    packet.last_block =
        token == MM_STREAM_8X64_TOKENS - 1 &&
        GROUP == MM_STREAM_8X64_WEIGHT_GROUPS - 1;
    packet.last_stream = task.last_stream && packet.last_block;

    for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
        #pragma HLS unroll
        constexpr unsigned int group_base = GROUP * CU_VEC_LANES;
        unsigned int out = group_base + lane;
        mm_stream_8x64_dsp_accum_t pair01 =
            slot0.value[token][out] + slot1.value[token][out];
        mm_stream_8x64_dsp_accum_t pair23 =
            slot2.value[token][out] + slot3.value[token][out];
        mm_stream_8x64_dsp_accum_t total = pair01 + pair23;
        packet.data[lane] = mm_stream_8x64_dsp_to_fm_accum(total);
    }
    out_stream.write(packet);
}

void compute_mm_stream_8x64_fused_mac_engine(
    hls::stream<cu_accum16_packet_t>& out_stream,
    const mm_stream_8x64_task_t& task,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3
) {
    #pragma HLS inline off

    mm_stream_8x64_dsp_bank_t slot0;
    mm_stream_8x64_dsp_bank_t slot1;
    mm_stream_8x64_dsp_bank_t slot2;
    mm_stream_8x64_dsp_bank_t slot3;
    #pragma HLS array_partition variable=slot0.value complete dim=0
    #pragma HLS array_partition variable=slot1.value complete dim=0
    #pragma HLS array_partition variable=slot2.value complete dim=0
    #pragma HLS array_partition variable=slot3.value complete dim=0

    for (unsigned int token = 0; token < MM_STREAM_8X64_TOKENS; token++) {
        #pragma HLS unroll
        for (unsigned int out = 0; out < MM_STREAM_8X64_OUTPUTS; out++) {
            #pragma HLS unroll
            slot0.value[token][out] = 0;
            slot1.value[token][out] = 0;
            slot2.value[token][out] = 0;
            slot3.value[token][out] = 0;
        }
    }

    for (unsigned int k = 0; k < task.k_count; k++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=16 max=MAX_LINEAR_IN_DIM avg=HIDDEN_SIZE
        mm_stream_8x64_activation_packet_t activation = activation_stream.read();
        mm_stream_8x64_weight_packet_t weight0 = weight_stream0.read();
        mm_stream_8x64_weight_packet_t weight1 = weight_stream1.read();
        mm_stream_8x64_weight_packet_t weight2 = weight_stream2.read();
        mm_stream_8x64_weight_packet_t weight3 = weight_stream3.read();
        rotate_mm_stream_8x64_dsp_accumulators(
            slot0,
            slot1,
            slot2,
            slot3,
            activation,
            weight0,
            weight1,
            weight2,
            weight3
        );
    }

    for (unsigned int token = 0; token < MM_STREAM_8X64_TOKENS; token++) {
        #pragma HLS pipeline II=4
        emit_mm_stream_8x64_dsp_group<0>(
            out_stream, slot0, slot1, slot2, slot3, task, token
        );
        emit_mm_stream_8x64_dsp_group<1>(
            out_stream, slot0, slot1, slot2, slot3, task, token
        );
        emit_mm_stream_8x64_dsp_group<2>(
            out_stream, slot0, slot1, slot2, slot3, task, token
        );
        emit_mm_stream_8x64_dsp_group<3>(
            out_stream, slot0, slot1, slot2, slot3, task, token
        );
    }
}

void compute_mm_stream_8x64_fused_mac_core(
    hls::stream<cu_accum16_packet_t>& out_stream,
    hls::stream<mm_stream_8x64_task_t>& task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& weight_stream3
) {
    #pragma HLS interface axis port=out_stream
    #pragma HLS interface axis port=task_stream
    #pragma HLS interface axis port=activation_stream
    #pragma HLS interface axis port=weight_stream0
    #pragma HLS interface axis port=weight_stream1
    #pragma HLS interface axis port=weight_stream2
    #pragma HLS interface axis port=weight_stream3
    #pragma HLS interface ap_ctrl_hs port=return
    #pragma HLS inline off

    mm_stream_8x64_task_t task = task_stream.read();
    compute_mm_stream_8x64_fused_mac_engine(
        out_stream,
        task,
        activation_stream,
        weight_stream0,
        weight_stream1,
        weight_stream2,
        weight_stream3
    );
}
