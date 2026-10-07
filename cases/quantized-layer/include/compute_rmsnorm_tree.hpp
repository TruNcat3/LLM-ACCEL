#ifndef LLM_FPGA_COMPUTE_RMSNORM_TREE_HPP
#define LLM_FPGA_COMPUTE_RMSNORM_TREE_HPP

#include "compute_core_8x64_unified.hpp"
#include <hls_math.h>

// RMSNorm is independently selectable from the other nonlinear engines.  A
// value of one deliberately leaves the original scalar implementation in
// compute_core_vector.hpp as the production default.
#ifndef CU_RMS_LANES_CONFIG
#define CU_RMS_LANES_CONFIG 1
#endif

#if CU_RMS_LANES_CONFIG != 1 && CU_RMS_LANES_CONFIG != 2 && \
    CU_RMS_LANES_CONFIG != 4
#error "CU_RMS_LANES_CONFIG must be 1, 2, or 4"
#endif

constexpr unsigned int CU_RMS_BANK_DEPTH =
    ceildiv(MAX_LINEAR_OUT_DIM, static_cast<unsigned int>(CU_RMS_LANES_CONFIG));

using cu_rms_sum_t = ap_ufixed<48, 32, AP_RND, AP_SAT>;
using cu_rms_math_t = ap_fixed<40, 16>;

// The production tree intentionally shares the arithmetic proof used by the
// completed lane probe: each Fix16 Q8.8 value has an exact raw square of a
// 16-bit input (represented in an unsigned 31-bit result), and the Q32.16
// sum preserves all of those fractional bits.
static_assert(fm_t::width == 16 && fm_t::iwidth == 8,
              "raw RMS reduction requires signed Q8.8 inputs");
static_assert(fm_accum_t::width == 32 && fm_accum_t::iwidth == 16,
              "RMS reference cast must preserve Q8.8 exactly");
static_assert(cu_rms_sum_t::width == 48 && cu_rms_sum_t::iwidth == 32,
              "RMS sum must have 16 fractional bits");

static ap_uint<31> cu_rms_raw_square(fm_t value) {
    #pragma HLS inline
    ap_int<16> raw;
    raw.range(15, 0) = value.range(15, 0);
    ap_int<32> square = raw * raw;
    #pragma HLS bind_op variable=square op=mul impl=dsp latency=1
    return ap_uint<31>(square.range(30, 0));
}

template <unsigned int LANES>
static ap_uint<33> cu_rms_raw_group_sum(const ap_uint<31> square[LANES]) {
    #pragma HLS inline
    if (LANES == 1) {
        return square[0];
    }
    const ap_uint<32> pair0 = ap_uint<32>(square[0]) + square[1];
    if (LANES == 2) {
        return pair0;
    }
    const ap_uint<32> pair1 = ap_uint<32>(square[2]) + square[3];
    return ap_uint<33>(pair0) + pair1;
}

static cu_rms_sum_t cu_rms_add_raw_group(
    cu_rms_sum_t sum,
    ap_uint<33> group
) {
    #pragma HLS inline
    const ap_uint<49> total = ap_uint<49>(sum.range(47, 0)) + group;
    const ap_uint<48> saturated = total[48]
        ? ~ap_uint<48>(0) : ap_uint<48>(total.range(47, 0));
    cu_rms_sum_t result;
    result.range(47, 0) = saturated;
    return result;
}

// Recursive bank accesses keep the bank index static after lane unrolling;
// this is what makes each cyclic bank a separate BRAM rather than a muxed
// monolithic array.  The packet offset handles non-aligned packet bases.
template <unsigned int LANES, unsigned int BANK>
static void cu_rms_store_weight_bank_at(
    wt_norm_t weight_bank[LANES][CU_RMS_BANK_DEPTH],
    const cu_vec16_packet_t& packet,
    unsigned int group,
    unsigned int elem_count,
    unsigned int offset
) {
    #pragma HLS inline
    const unsigned int lane = (BANK + LANES - offset) & (LANES - 1);
    const unsigned int packet_lane = group * LANES + lane;
    const unsigned int elem = packet.elem_base + packet_lane;
    if (elem < elem_count && packet.valid_mask[packet_lane]) {
        weight_bank[BANK][elem / LANES] =
            wt_norm_t(packet.data[packet_lane]);
    }
}

template <unsigned int LANES, unsigned int BANK>
struct cu_rms_weight_banks {
    static void run(
        wt_norm_t weight_bank[LANES][CU_RMS_BANK_DEPTH],
        const cu_vec16_packet_t& packet,
        unsigned int group,
        unsigned int elem_count,
        unsigned int offset
    ) {
        #pragma HLS inline
        cu_rms_store_weight_bank_at<LANES, BANK>(
            weight_bank, packet, group, elem_count, offset);
        cu_rms_weight_banks<LANES, BANK + 1>::run(
            weight_bank, packet, group, elem_count, offset);
    }
};

template <unsigned int LANES>
struct cu_rms_weight_banks<LANES, LANES> {
    static void run(
        wt_norm_t[LANES][CU_RMS_BANK_DEPTH],
        const cu_vec16_packet_t&,
        unsigned int,
        unsigned int,
        unsigned int
    ) {
        #pragma HLS inline
    }
};

template <unsigned int LANES>
static void cu_rms_tree_preload_packet(
    wt_norm_t weight_bank[LANES][CU_RMS_BANK_DEPTH],
    const cu_vec16_packet_t& packet,
    unsigned int elem_count
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=packet.data complete dim=1

    const unsigned int offset = packet.elem_base & (LANES - 1);
    for (unsigned int group = 0;
         group < CU_VEC_LANES / LANES;
         group++) {
        #pragma HLS pipeline II=1
        cu_rms_weight_banks<LANES, 0>::run(
            weight_bank, packet, group, elem_count, offset);
    }
}

template <unsigned int LANES, unsigned int BANK>
static void cu_rms_store_cache_bank_at(
    fm_t cache_bank[LANES][CU_RMS_BANK_DEPTH],
    const cu_vec16_packet_t& packet,
    unsigned int group,
    unsigned int elem_count,
    unsigned int offset
) {
    #pragma HLS inline
    const unsigned int lane = (BANK + LANES - offset) & (LANES - 1);
    const unsigned int packet_lane = group * LANES + lane;
    const unsigned int elem = packet.elem_base + packet_lane;
    if (elem < elem_count && packet.valid_mask[packet_lane]) {
        cache_bank[BANK][elem / LANES] = packet.data[packet_lane];
    }
}

template <unsigned int LANES, unsigned int BANK>
struct cu_rms_cache_banks {
    static void run(
        fm_t cache_bank[LANES][CU_RMS_BANK_DEPTH],
        const cu_vec16_packet_t& packet,
        unsigned int group,
        unsigned int elem_count,
        unsigned int offset
    ) {
        #pragma HLS inline
        cu_rms_store_cache_bank_at<LANES, BANK>(
            cache_bank, packet, group, elem_count, offset);
        cu_rms_cache_banks<LANES, BANK + 1>::run(
            cache_bank, packet, group, elem_count, offset);
    }
};

template <unsigned int LANES>
struct cu_rms_cache_banks<LANES, LANES> {
    static void run(
        fm_t[LANES][CU_RMS_BANK_DEPTH],
        const cu_vec16_packet_t&,
        unsigned int,
        unsigned int,
        unsigned int
    ) {
        #pragma HLS inline
    }
};

template <unsigned int LANES>
static void cu_rms_tree_accumulate_packet(
    fm_t cache_bank[LANES][CU_RMS_BANK_DEPTH],
    const cu_vec16_packet_t& packet,
    unsigned int elem_count,
    cu_rms_sum_t& sum_sq
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=packet.data complete dim=1

    const unsigned int offset = packet.elem_base & (LANES - 1);
    for (unsigned int group = 0;
         group < CU_VEC_LANES / LANES;
         group++) {
        #pragma HLS pipeline II=1
        cu_rms_cache_banks<LANES, 0>::run(
            cache_bank, packet, group, elem_count, offset);

        ap_uint<31> square[LANES];
        #pragma HLS array_partition variable=square complete dim=1
        for (unsigned int lane = 0; lane < LANES; lane++) {
            #pragma HLS unroll
            const unsigned int packet_lane = group * LANES + lane;
            const unsigned int elem = packet.elem_base + packet_lane;
            const ap_uint<31> product =
                cu_rms_raw_square(packet.data[packet_lane]);
            square[lane] =
                (elem < elem_count && packet.valid_mask[packet_lane])
                ? product : ap_uint<31>(0);
        }
        sum_sq = cu_rms_add_raw_group(
            sum_sq, cu_rms_raw_group_sum<LANES>(square));
    }
}

template <unsigned int LANES, unsigned int BANK>
static void cu_rms_read_bank_at(
    fm_t cache_values[LANES],
    wt_norm_t weight_values[LANES],
    const fm_t cache_bank[LANES][CU_RMS_BANK_DEPTH],
    const wt_norm_t weight_bank[LANES][CU_RMS_BANK_DEPTH],
    const cu_vec16_packet_t& packet,
    unsigned int group,
    unsigned int elem_count,
    unsigned int offset
) {
    #pragma HLS inline
    const unsigned int lane = (BANK + LANES - offset) & (LANES - 1);
    const unsigned int packet_lane = group * LANES + lane;
    const unsigned int elem = packet.elem_base + packet_lane;
    if (elem < elem_count && packet.valid_mask[packet_lane]) {
        cache_values[BANK] = cache_bank[BANK][elem / LANES];
        weight_values[BANK] = weight_bank[BANK][elem / LANES];
    } else {
        cache_values[BANK] = fm_t(0);
        weight_values[BANK] = wt_norm_t(0);
    }
}

template <unsigned int LANES, unsigned int BANK>
struct cu_rms_read_banks {
    static void run(
        fm_t cache_values[LANES],
        wt_norm_t weight_values[LANES],
        const fm_t cache_bank[LANES][CU_RMS_BANK_DEPTH],
        const wt_norm_t weight_bank[LANES][CU_RMS_BANK_DEPTH],
        const cu_vec16_packet_t& packet,
        unsigned int group,
        unsigned int elem_count,
        unsigned int offset
    ) {
        #pragma HLS inline
        cu_rms_read_bank_at<LANES, BANK>(
            cache_values, weight_values, cache_bank, weight_bank,
            packet, group, elem_count, offset);
        cu_rms_read_banks<LANES, BANK + 1>::run(
            cache_values, weight_values, cache_bank, weight_bank,
            packet, group, elem_count, offset);
    }
};

template <unsigned int LANES>
struct cu_rms_read_banks<LANES, LANES> {
    static void run(
        fm_t[LANES],
        wt_norm_t[LANES],
        const fm_t[LANES][CU_RMS_BANK_DEPTH],
        const wt_norm_t[LANES][CU_RMS_BANK_DEPTH],
        const cu_vec16_packet_t&,
        unsigned int,
        unsigned int,
        unsigned int
    ) {
        #pragma HLS inline
    }
};

template <unsigned int LANES>
static void cu_rms_tree_emit_packet(
    cu_vec16_packet_t& out_packet,
    const fm_t cache_bank[LANES][CU_RMS_BANK_DEPTH],
    const wt_norm_t weight_bank[LANES][CU_RMS_BANK_DEPTH],
    unsigned int elem_count,
    const cu_rms_math_t& inv_rms
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=out_packet.data complete dim=1

    const unsigned int offset = out_packet.elem_base & (LANES - 1);
    for (unsigned int group = 0;
         group < CU_VEC_LANES / LANES;
         group++) {
        #pragma HLS pipeline II=1
        fm_t cache_values[LANES];
        wt_norm_t weight_values[LANES];
        #pragma HLS array_partition variable=cache_values complete dim=1
        #pragma HLS array_partition variable=weight_values complete dim=1
        cu_rms_read_banks<LANES, 0>::run(
            cache_values, weight_values, cache_bank, weight_bank,
            out_packet, group, elem_count, offset);
        for (unsigned int lane = 0; lane < LANES; lane++) {
            #pragma HLS unroll
            const unsigned int packet_lane = group * LANES + lane;
            const unsigned int elem = out_packet.elem_base + packet_lane;
            const unsigned int bank = (offset + lane) & (LANES - 1);
            fm_t value = fm_t(0);
            if (elem < elem_count && out_packet.valid_mask[packet_lane]) {
                value = cache_values[bank] * inv_rms * weight_values[bank];
            }
            out_packet.data[packet_lane] = value;
        }
    }
}

template <unsigned int LANES>
static void cu_rms_tree_row(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    const wt_norm_t weight_bank[LANES][CU_RMS_BANK_DEPTH],
    unsigned int elem_count
) {
    #pragma HLS inline off

    fm_t cache_bank[LANES][CU_RMS_BANK_DEPTH];
    ap_uint<CU_VEC_LANES> masks[MAX_LINEAR_OUT_BLOCKS];
    unsigned int bases[MAX_LINEAR_OUT_BLOCKS];
    unsigned int token_lane = 0;
    #pragma HLS array_partition variable=cache_bank complete dim=1
    #pragma HLS bind_storage variable=cache_bank type=ram_2p impl=bram
    #pragma HLS bind_storage variable=masks type=ram_2p impl=bram
    #pragma HLS bind_storage variable=bases type=ram_2p impl=bram

    const unsigned int requested_packets =
        ceildiv(elem_count, CU_VEC_LANES);
    const unsigned int packet_count =
        requested_packets < MAX_LINEAR_OUT_BLOCKS
        ? requested_packets : MAX_LINEAR_OUT_BLOCKS;
    cu_rms_sum_t sum_sq = 0;
    for (unsigned int packet = 0; packet < packet_count; packet++) {
        cu_vec16_packet_t in_packet = in_stream.read();
        masks[packet] = in_packet.valid_mask;
        bases[packet] = in_packet.elem_base;
        token_lane = in_packet.token_lane;
        cu_rms_tree_accumulate_packet<LANES>(
            cache_bank, in_packet, elem_count, sum_sq);
    }

    const ap_uint<16> divisor = elem_count == 0 ? 1 : elem_count;
    const cu_rms_math_t mean_sq = cu_rms_math_t(sum_sq / divisor);
    // One shared reciprocal square-root is intentionally evaluated per row.
    const cu_rms_math_t inv_rms = hls::rsqrt(
        cu_rms_math_t(mean_sq + cu_rms_math_t(RMS_NORM_EPS)));

    for (unsigned int packet = 0; packet < packet_count; packet++) {
        cu_vec16_packet_t out_packet;
        out_packet.valid_mask = masks[packet];
        out_packet.token_lane = token_lane;
        out_packet.elem_base = bases[packet];
        out_packet.block_id = packet;
        out_packet.last_block = packet + 1 == packet_count;
        out_packet.last_stream = out_packet.last_block;
        cu_rms_tree_emit_packet<LANES>(
            out_packet, cache_bank, weight_bank, elem_count, inv_rms);
        out_stream.write(out_packet);
    }
}

template <unsigned int LANES>
static void run_cu8_rmsnorm_tree_task(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& input_stream,
    hls::stream<cu_vec16_packet_t>& weight_stream,
    const cu8_task_t& task
) {
    #pragma HLS inline off

    wt_norm_t weight_bank[LANES][CU_RMS_BANK_DEPTH];
    #pragma HLS array_partition variable=weight_bank complete dim=1
    #pragma HLS bind_storage variable=weight_bank type=ram_2p impl=bram

    const unsigned int requested_weight_packets =
        ceildiv(task.elem_count, CU_VEC_LANES);
    const unsigned int weight_packets =
        requested_weight_packets < MAX_LINEAR_OUT_BLOCKS
        ? requested_weight_packets : MAX_LINEAR_OUT_BLOCKS;
    // Consume exactly one weight packet set per task, before any row input.
    for (unsigned int packet = 0;
         packet < weight_packets;
         packet++) {
        cu_vec16_packet_t weight_packet = weight_stream.read();
        cu_rms_tree_preload_packet<LANES>(
            weight_bank, weight_packet, task.elem_count);
    }

    const unsigned int active_rows =
        task.token_count < MM_STREAM_8X64_TOKENS
        ? task.token_count : MM_STREAM_8X64_TOKENS;
    for (unsigned int row = 0;
         row < MM_STREAM_8X64_TOKENS;
         row++) {
        if (row < active_rows) {
            cu_rms_tree_row<LANES>(
                out_stream, input_stream, weight_bank, task.elem_count);
        }
    }
}

#endif
