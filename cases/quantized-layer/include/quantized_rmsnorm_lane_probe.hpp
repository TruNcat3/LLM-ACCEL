#ifndef LLM_FPGA_QUANTIZED_RMSNORM_LANE_PROBE_HPP
#define LLM_FPGA_QUANTIZED_RMSNORM_LANE_PROBE_HPP

#include "compute_stream.hpp"
#include <hls_math.h>

// This probe deliberately has its own compile-time knob.  The production
// CU_NL_LANES_CONFIG switch controls SiLU and must not silently change this
// RMSNorm experiment.
#ifndef Q_RMS_LANES
#define Q_RMS_LANES 2
#endif

#if Q_RMS_LANES != 1 && Q_RMS_LANES != 2 && Q_RMS_LANES != 4
#error "Q_RMS_LANES must be 1, 2, or 4"
#endif

// Keep the historical implementations available while the static-bank
// candidate is evaluated.  The Tcl wrapper accepts the corresponding names
// (legacy, banked, static) and maps them to these compile-time values.
#ifndef Q_RMS_VARIANT
#define Q_RMS_VARIANT 2
#endif

#define Q_RMS_VARIANT_LEGACY 0
#define Q_RMS_VARIANT_BANKED 1
#define Q_RMS_VARIANT_STATIC 2
#define Q_RMS_VARIANT_TREE 3

#if Q_RMS_VARIANT != Q_RMS_VARIANT_LEGACY && \
    Q_RMS_VARIANT != Q_RMS_VARIANT_BANKED && \
    Q_RMS_VARIANT != Q_RMS_VARIANT_STATIC && \
    Q_RMS_VARIANT != Q_RMS_VARIANT_TREE
#error "Q_RMS_VARIANT must be 0 (legacy), 1 (banked), 2 (static), or 3 (tree)"
#endif

#ifndef Q_RMS_SQUARE_LATENCY
#define Q_RMS_SQUARE_LATENCY 2
#endif
#if Q_RMS_SQUARE_LATENCY != 1 && Q_RMS_SQUARE_LATENCY != 2
#error "Q_RMS_SQUARE_LATENCY must be 1 or 2"
#endif

// The fixed-point reduction remains source ordered.  A two-cycle initiation
// interval gives the accumulator chain and the banked BRAM reads a timing
// margin while retaining the exact arithmetic sequence.
#ifndef Q_RMS_STATIC_GROUP_II
#define Q_RMS_STATIC_GROUP_II 2
#endif

#if Q_RMS_STATIC_GROUP_II < 1 || Q_RMS_STATIC_GROUP_II > 4
#error "Q_RMS_STATIC_GROUP_II must be between 1 and 4"
#endif

constexpr unsigned int Q_RMS_MAX_ROWS = 2;
constexpr unsigned int Q_RMS_PACKET_LANES = CU_VEC_LANES;
constexpr unsigned int Q_RMS_BANK_DEPTH =
    ceildiv(MAX_LINEAR_OUT_DIM, static_cast<unsigned int>(Q_RMS_LANES));

using q_rms_sum_t = ap_ufixed<48, 32, AP_RND, AP_SAT>;
using q_rms_math_t = ap_fixed<40, 16>;

// With Q8.8 input, an exact raw square has 16 fractional bits and is at
// most 2^30 (including -128). The Q32.16 accumulator discards no fraction.
// Since every summand is nonnegative, saturating after the group sum is
// bit-equivalent to saturation after every scalar addition, even at overflow.
// Keep explicit type guards so a future datatype change cannot reuse this proof.
static_assert(fm_t::width == 16 && fm_t::iwidth == 8,
              "raw RMS reduction requires signed Q8.8 inputs");
static_assert(fm_accum_t::width == 32 && fm_accum_t::iwidth == 16,
              "RMS reference cast must preserve Q8.8 exactly");
static_assert(q_rms_sum_t::width == 48 && q_rms_sum_t::iwidth == 32,
              "RMS sum must have 16 fractional bits");

static ap_uint<31> q_rms_raw_square(fm_t value) {
    #pragma HLS inline
    ap_int<16> raw;
    raw.range(15, 0) = value.range(15, 0);
    ap_int<32> square = raw * raw;
    #pragma HLS bind_op variable=square op=mul impl=dsp latency=Q_RMS_SQUARE_LATENCY
    return ap_uint<31>(square.range(30, 0));
}

template <unsigned int LANES>
static ap_uint<33> q_rms_raw_group_sum(const ap_uint<31> square[LANES]) {
    #pragma HLS inline
    if (LANES == 1) return square[0];
    const ap_uint<32> pair0 = ap_uint<32>(square[0]) + square[1];
    if (LANES == 2) return pair0;
    const ap_uint<32> pair1 = ap_uint<32>(square[2]) + square[3];
    return ap_uint<33>(pair0) + pair1;
}

static q_rms_sum_t q_rms_add_raw_group(q_rms_sum_t sum, ap_uint<33> group) {
    #pragma HLS inline
    const ap_uint<49> total = ap_uint<49>(sum.range(47, 0)) + group;
    const ap_uint<48> saturated = total[48]
        ? ~ap_uint<48>(0) : ap_uint<48>(total.range(47, 0));
    q_rms_sum_t result;
    result.range(47, 0) = saturated;
    return result;
}

// The legacy preload is intentionally lane-serial.  Its packet and validity
// rules mirror run_cu8_rmsnorm_task in compute_core_vector.hpp.
static void q_rms_preload_legacy(
    wt_norm_t weights[MAX_LINEAR_OUT_DIM],
    hls::stream<cu_vec16_packet_t>& weight_stream,
    unsigned int elem_count
) {
    #pragma HLS inline off

    const unsigned int weight_packets = ceildiv(elem_count, CU_VEC_LANES);
    for (unsigned int packet = 0;
         packet < MAX_LINEAR_OUT_BLOCKS;
         packet++) {
        if (packet < weight_packets) {
            cu_vec16_packet_t weight_packet = weight_stream.read();
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS pipeline II=1
                const unsigned int elem = weight_packet.elem_base + lane;
                if (elem < elem_count && weight_packet.valid_mask[lane]) {
                    weights[elem] = wt_norm_t(weight_packet.data[lane]);
                }
            }
        }
    }
}

// The candidate preload uses the same 16-lane packet ABI, but writes a
// cyclic lane bank.  The group loop is unrolled for Q_RMS_LANES lanes while
// keeping the packet order and validity checks unchanged.
template <unsigned int LANES>
static void q_rms_preload_banked(
    wt_norm_t weight_bank[LANES][Q_RMS_BANK_DEPTH],
    hls::stream<cu_vec16_packet_t>& weight_stream,
    unsigned int elem_count
) {
    #pragma HLS inline off

    const unsigned int weight_packets = ceildiv(elem_count, CU_VEC_LANES);
    for (unsigned int packet = 0;
         packet < MAX_LINEAR_OUT_BLOCKS;
         packet++) {
        if (packet < weight_packets) {
            cu_vec16_packet_t weight_packet = weight_stream.read();
            for (unsigned int group = 0;
                 group < CU_VEC_LANES / LANES;
                 group++) {
                #pragma HLS pipeline II=1
                for (unsigned int lane = 0; lane < LANES; lane++) {
                    #pragma HLS unroll
                    const unsigned int packet_lane = group * LANES + lane;
                    const unsigned int elem =
                        weight_packet.elem_base + packet_lane;
                    if (elem < elem_count &&
                        weight_packet.valid_mask[packet_lane]) {
                        weight_bank[elem % LANES][elem / LANES] =
                            wt_norm_t(weight_packet.data[packet_lane]);
                    }
                }
            }
        }
    }
}

// Legacy reference: one scalar lane update per cycle and the exact original
// reduction/cast sequence.  No candidate code calls this through a tolerance
// path; the testbench compares every output bit and packet field.
static void q_rms_legacy_row(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    const wt_norm_t weights[MAX_LINEAR_OUT_DIM],
    unsigned int elem_count
) {
    #pragma HLS inline off

    fm_t cache[MAX_LINEAR_OUT_DIM];
    ap_uint<CU_VEC_LANES> masks[MAX_LINEAR_OUT_BLOCKS];
    unsigned int bases[MAX_LINEAR_OUT_BLOCKS];
    unsigned int token_lane = 0;
    #pragma HLS bind_storage variable=cache type=ram_2p impl=bram

    const unsigned int packet_count = ceildiv(elem_count, CU_VEC_LANES);
    q_rms_sum_t sum_sq = 0;

    for (unsigned int packet = 0;
         packet < MAX_LINEAR_OUT_BLOCKS;
         packet++) {
        if (packet < packet_count) {
            cu_vec16_packet_t in_packet = in_stream.read();
            masks[packet] = in_packet.valid_mask;
            bases[packet] = in_packet.elem_base;
            token_lane = in_packet.token_lane;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS pipeline II=1
                const unsigned int elem = in_packet.elem_base + lane;
                const fm_t value = in_packet.data[lane];
                if (elem < elem_count && in_packet.valid_mask[lane]) {
                    cache[elem] = value;
                    sum_sq += fm_accum_t(value) * fm_accum_t(value);
                }
            }
        }
    }

    const ap_uint<16> divisor = elem_count == 0 ? 1 : elem_count;
    const q_rms_math_t mean_sq = q_rms_math_t(sum_sq / divisor);
    const q_rms_math_t inv_rms = hls::rsqrt(
        q_rms_math_t(mean_sq + q_rms_math_t(RMS_NORM_EPS)));

    for (unsigned int packet = 0;
         packet < MAX_LINEAR_OUT_BLOCKS;
         packet++) {
        if (packet < packet_count) {
            cu_vec16_packet_t out_packet;
            out_packet.valid_mask = masks[packet];
            out_packet.token_lane = token_lane;
            out_packet.elem_base = bases[packet];
            out_packet.block_id = packet;
            out_packet.last_block = packet + 1 == packet_count;
            out_packet.last_stream = out_packet.last_block;

            for (unsigned int lane = 0; lane < CU_VEC_LANES; lane++) {
                #pragma HLS pipeline II=1
                const unsigned int elem = bases[packet] + lane;
                fm_t value = fm_t(0);
                if (elem < elem_count && masks[packet][lane]) {
                    value = cache[elem] * inv_rms * weights[elem];
                }
                out_packet.data[lane] = value;
            }
            out_stream.write(out_packet);
        }
    }
}

template <unsigned int LANES>
static void q_rms_banked_row(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    const wt_norm_t weight_bank[LANES][Q_RMS_BANK_DEPTH],
    unsigned int elem_count
) {
    #pragma HLS inline off

    fm_t cache_bank[LANES][Q_RMS_BANK_DEPTH];
    ap_uint<CU_VEC_LANES> masks[MAX_LINEAR_OUT_BLOCKS];
    unsigned int bases[MAX_LINEAR_OUT_BLOCKS];
    unsigned int token_lane = 0;
    #pragma HLS array_partition variable=cache_bank complete dim=1
    #pragma HLS bind_storage variable=cache_bank type=ram_2p impl=bram

    const unsigned int packet_count = ceildiv(elem_count, CU_VEC_LANES);
    q_rms_sum_t sum_sq = 0;

    for (unsigned int packet = 0;
         packet < MAX_LINEAR_OUT_BLOCKS;
         packet++) {
        if (packet < packet_count) {
            cu_vec16_packet_t in_packet = in_stream.read();
            masks[packet] = in_packet.valid_mask;
            bases[packet] = in_packet.elem_base;
            token_lane = in_packet.token_lane;

            // Products are generated in parallel per group, but the explicit
            // source-order chain below preserves every sum_sq cast/rounding
            // point.  This is intentionally not a tree reduction.
            for (unsigned int group = 0;
                 group < CU_VEC_LANES / LANES;
                 group++) {
                #pragma HLS pipeline II=1
                q_rms_sum_t group_sum = sum_sq;
                for (unsigned int lane = 0; lane < LANES; lane++) {
                    #pragma HLS unroll
                    const unsigned int packet_lane = group * LANES + lane;
                    const unsigned int elem =
                        in_packet.elem_base + packet_lane;
                    const fm_t value = in_packet.data[packet_lane];
                    if (elem < elem_count &&
                        in_packet.valid_mask[packet_lane]) {
                        cache_bank[elem % LANES][elem / LANES] = value;
                        group_sum +=
                            fm_accum_t(value) * fm_accum_t(value);
                    }
                }
                sum_sq = group_sum;
            }
        }
    }

    const ap_uint<16> divisor = elem_count == 0 ? 1 : elem_count;
    const q_rms_math_t mean_sq = q_rms_math_t(sum_sq / divisor);
    // Keep exactly one shared rsqrt call per row.  It is outside the lane
    // groups, so the candidate does not replicate this expensive unit.
    const q_rms_math_t inv_rms = hls::rsqrt(
        q_rms_math_t(mean_sq + q_rms_math_t(RMS_NORM_EPS)));

    for (unsigned int packet = 0;
         packet < MAX_LINEAR_OUT_BLOCKS;
         packet++) {
        if (packet < packet_count) {
            cu_vec16_packet_t out_packet;
            out_packet.valid_mask = masks[packet];
            out_packet.token_lane = token_lane;
            out_packet.elem_base = bases[packet];
            out_packet.block_id = packet;
            out_packet.last_block = packet + 1 == packet_count;
            out_packet.last_stream = out_packet.last_block;

            for (unsigned int group = 0;
                 group < CU_VEC_LANES / LANES;
                 group++) {
                #pragma HLS pipeline II=1
                for (unsigned int lane = 0; lane < LANES; lane++) {
                    #pragma HLS unroll
                    const unsigned int packet_lane = group * LANES + lane;
                    const unsigned int elem = bases[packet] + packet_lane;
                    fm_t value = fm_t(0);
                    if (elem < elem_count && masks[packet][packet_lane]) {
                        const unsigned int bank = elem % LANES;
                        const unsigned int bank_elem = elem / LANES;
                        value = cache_bank[bank][bank_elem] * inv_rms *
                            weight_bank[bank][bank_elem];
                    }
                    out_packet.data[packet_lane] = value;
                }
            }
            out_stream.write(out_packet);
        }
    }
}

// Static-bank candidate uses fixed bank writers/readers and a small offset mux.
template <unsigned int LANES, unsigned int BANK>
static void q_rms_fixed_store_weight_bank_at(
    wt_norm_t weight_bank[LANES][Q_RMS_BANK_DEPTH],
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
struct q_rms_fixed_weight_banks {
    static void run(
        wt_norm_t weight_bank[LANES][Q_RMS_BANK_DEPTH],
        const cu_vec16_packet_t& packet,
        unsigned int group,
        unsigned int elem_count,
        unsigned int offset
    ) {
        #pragma HLS inline
        q_rms_fixed_store_weight_bank_at<LANES, BANK>(
            weight_bank, packet, group, elem_count, offset);
        q_rms_fixed_weight_banks<LANES, BANK + 1>::run(
            weight_bank, packet, group, elem_count, offset);
    }
};

template <unsigned int LANES>
struct q_rms_fixed_weight_banks<LANES, LANES> {
    static void run(
        wt_norm_t[LANES][Q_RMS_BANK_DEPTH],
        const cu_vec16_packet_t&,
        unsigned int,
        unsigned int,
        unsigned int
    ) {
        #pragma HLS inline
    }
};

template <unsigned int LANES>
static void q_rms_fixed_preload_packet(
    wt_norm_t weight_bank[LANES][Q_RMS_BANK_DEPTH],
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
        q_rms_fixed_weight_banks<LANES, 0>::run(
            weight_bank, packet, group, elem_count, offset);
    }
}

template <unsigned int LANES, unsigned int BANK>
static void q_rms_fixed_store_cache_bank_at(
    fm_t cache_bank[LANES][Q_RMS_BANK_DEPTH],
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
struct q_rms_fixed_cache_banks {
    static void run(
        fm_t cache_bank[LANES][Q_RMS_BANK_DEPTH],
        const cu_vec16_packet_t& packet,
        unsigned int group,
        unsigned int elem_count,
        unsigned int offset
    ) {
        #pragma HLS inline
        q_rms_fixed_store_cache_bank_at<LANES, BANK>(
            cache_bank, packet, group, elem_count, offset);
        q_rms_fixed_cache_banks<LANES, BANK + 1>::run(
            cache_bank, packet, group, elem_count, offset);
    }
};

template <unsigned int LANES>
struct q_rms_fixed_cache_banks<LANES, LANES> {
    static void run(
        fm_t[LANES][Q_RMS_BANK_DEPTH],
        const cu_vec16_packet_t&,
        unsigned int,
        unsigned int,
        unsigned int
    ) {
        #pragma HLS inline
    }
};

template <unsigned int LANES>
static void q_rms_fixed_accumulate_packet(
    fm_t cache_bank[LANES][Q_RMS_BANK_DEPTH],
    const cu_vec16_packet_t& packet,
    unsigned int elem_count,
    q_rms_sum_t& sum_sq
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=packet.data complete dim=1

    const unsigned int offset = packet.elem_base & (LANES - 1);
    for (unsigned int group = 0;
         group < CU_VEC_LANES / LANES;
         group++) {
        #pragma HLS pipeline II=Q_RMS_STATIC_GROUP_II
        q_rms_fixed_cache_banks<LANES, 0>::run(
            cache_bank, packet, group, elem_count, offset);
#if Q_RMS_VARIANT == Q_RMS_VARIANT_TREE
        ap_uint<31> square[LANES];
        #pragma HLS array_partition variable=square complete dim=1
        for (unsigned int lane = 0; lane < LANES; lane++) {
            #pragma HLS unroll
            const unsigned int packet_lane = group * LANES + lane;
            const unsigned int elem = packet.elem_base + packet_lane;
            const ap_uint<31> product = q_rms_raw_square(packet.data[packet_lane]);
            square[lane] = (elem < elem_count && packet.valid_mask[packet_lane])
                ? product : ap_uint<31>(0);
        }
        sum_sq = q_rms_add_raw_group(sum_sq, q_rms_raw_group_sum<LANES>(square));
#else
        q_rms_sum_t group_sum = sum_sq;
        for (unsigned int lane = 0; lane < LANES; lane++) {
            #pragma HLS unroll
            const unsigned int packet_lane = group * LANES + lane;
            const unsigned int elem = packet.elem_base + packet_lane;
            const fm_t value = packet.data[packet_lane];
            if (elem < elem_count && packet.valid_mask[packet_lane]) {
                // Keep the original source-order cast and accumulation.
                group_sum += fm_accum_t(value) * fm_accum_t(value);
            }
        }
        sum_sq = group_sum;
#endif
    }
}

template <unsigned int LANES, unsigned int BANK>
static void q_rms_fixed_read_bank_at(
    fm_t cache_values[LANES],
    wt_norm_t weight_values[LANES],
    const fm_t cache_bank[LANES][Q_RMS_BANK_DEPTH],
    const wt_norm_t weight_bank[LANES][Q_RMS_BANK_DEPTH],
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
struct q_rms_fixed_read_banks {
    static void run(
        fm_t cache_values[LANES],
        wt_norm_t weight_values[LANES],
        const fm_t cache_bank[LANES][Q_RMS_BANK_DEPTH],
        const wt_norm_t weight_bank[LANES][Q_RMS_BANK_DEPTH],
        const cu_vec16_packet_t& packet,
        unsigned int group,
        unsigned int elem_count,
        unsigned int offset
    ) {
        #pragma HLS inline
        q_rms_fixed_read_bank_at<LANES, BANK>(
            cache_values, weight_values, cache_bank, weight_bank,
            packet, group, elem_count, offset);
        q_rms_fixed_read_banks<LANES, BANK + 1>::run(
            cache_values, weight_values, cache_bank, weight_bank,
            packet, group, elem_count, offset);
    }
};

template <unsigned int LANES>
struct q_rms_fixed_read_banks<LANES, LANES> {
    static void run(
        fm_t[LANES],
        wt_norm_t[LANES],
        const fm_t[LANES][Q_RMS_BANK_DEPTH],
        const wt_norm_t[LANES][Q_RMS_BANK_DEPTH],
        const cu_vec16_packet_t&,
        unsigned int,
        unsigned int,
        unsigned int
    ) {
        #pragma HLS inline
    }
};

template <unsigned int LANES>
static void q_rms_fixed_emit_packet(
    cu_vec16_packet_t& out_packet,
    const fm_t cache_bank[LANES][Q_RMS_BANK_DEPTH],
    const wt_norm_t weight_bank[LANES][Q_RMS_BANK_DEPTH],
    unsigned int elem_count,
    const q_rms_math_t& inv_rms
) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=out_packet.data complete dim=1

    const unsigned int offset = out_packet.elem_base & (LANES - 1);
    for (unsigned int group = 0;
         group < CU_VEC_LANES / LANES;
         group++) {
        #pragma HLS pipeline II=Q_RMS_STATIC_GROUP_II
        fm_t cache_values[LANES];
        wt_norm_t weight_values[LANES];
        #pragma HLS array_partition variable=cache_values complete dim=1
        #pragma HLS array_partition variable=weight_values complete dim=1
        q_rms_fixed_read_banks<LANES, 0>::run(
            cache_values, weight_values, cache_bank, weight_bank,
            out_packet, group, elem_count, offset);
        for (unsigned int lane = 0; lane < LANES; lane++) {
            #pragma HLS unroll
            const unsigned int packet_lane = group * LANES + lane;
            const unsigned int elem = out_packet.elem_base + packet_lane;
            const unsigned int bank = (offset + lane) & (LANES - 1);
            fm_t value = fm_t(0);
            if (elem < elem_count && out_packet.valid_mask[packet_lane]) {
                value = cache_values[bank] * inv_rms *
                    weight_values[bank];
            }
            out_packet.data[packet_lane] = value;
        }
    }
}

template <unsigned int LANES>
static void q_rms_static_row(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    const wt_norm_t weight_bank[LANES][Q_RMS_BANK_DEPTH],
    unsigned int elem_count
) {
    #pragma HLS inline off

    fm_t cache_bank[LANES][Q_RMS_BANK_DEPTH];
    ap_uint<CU_VEC_LANES> masks[MAX_LINEAR_OUT_BLOCKS];
    unsigned int bases[MAX_LINEAR_OUT_BLOCKS];
    unsigned int token_lane = 0;
    #pragma HLS array_partition variable=cache_bank complete dim=1
    #pragma HLS bind_storage variable=cache_bank type=ram_2p impl=bram
    #pragma HLS bind_storage variable=masks type=ram_2p impl=bram
    #pragma HLS bind_storage variable=bases type=ram_2p impl=bram

    const unsigned int packet_count = ceildiv(elem_count, CU_VEC_LANES);
    q_rms_sum_t sum_sq = 0;

    // Every loop is bounded by the transaction's effective packet count.  A
    // packet's metadata is retained verbatim for the apply phase.
    for (unsigned int packet = 0; packet < packet_count; packet++) {
        cu_vec16_packet_t in_packet = in_stream.read();
        masks[packet] = in_packet.valid_mask;
        bases[packet] = in_packet.elem_base;
        token_lane = in_packet.token_lane;
        #pragma HLS array_partition variable=in_packet.data complete dim=1
        q_rms_fixed_accumulate_packet<LANES>(
            cache_bank, in_packet, elem_count, sum_sq);
    }

    const ap_uint<16> divisor = elem_count == 0 ? 1 : elem_count;
    const q_rms_math_t mean_sq = q_rms_math_t(sum_sq / divisor);
    const q_rms_math_t inv_rms = hls::rsqrt(
        q_rms_math_t(mean_sq + q_rms_math_t(RMS_NORM_EPS)));

    for (unsigned int packet = 0; packet < packet_count; packet++) {
        cu_vec16_packet_t out_packet;
        out_packet.valid_mask = masks[packet];
        out_packet.token_lane = token_lane;
        out_packet.elem_base = bases[packet];
        out_packet.block_id = packet;
        out_packet.last_block = packet + 1 == packet_count;
        out_packet.last_stream = out_packet.last_block;
        q_rms_fixed_emit_packet<LANES>(
            out_packet, cache_bank, weight_bank, elem_count, inv_rms);
        out_stream.write(out_packet);
    }
}

static void q_rms_legacy_pipeline(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    hls::stream<cu_vec16_packet_t>& weight_stream,
    unsigned int elem_count,
    unsigned int row_count
) {
    #pragma HLS inline off

    wt_norm_t weights[MAX_LINEAR_OUT_DIM];
    #pragma HLS bind_storage variable=weights type=ram_2p impl=bram
    q_rms_preload_legacy(weights, weight_stream, elem_count);

    for (unsigned int row = 0; row < Q_RMS_MAX_ROWS; row++) {
        if (row < row_count) {
            q_rms_legacy_row(out_stream, in_stream, weights, elem_count);
        }
    }
}

template <unsigned int LANES>
static void q_rms_banked_pipeline(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    hls::stream<cu_vec16_packet_t>& weight_stream,
    unsigned int elem_count,
    unsigned int row_count
) {
    #pragma HLS inline off

    wt_norm_t weight_bank[LANES][Q_RMS_BANK_DEPTH];
    #pragma HLS array_partition variable=weight_bank complete dim=1
    #pragma HLS bind_storage variable=weight_bank type=ram_2p impl=bram
    q_rms_preload_banked<LANES>(weight_bank, weight_stream, elem_count);

    for (unsigned int row = 0; row < Q_RMS_MAX_ROWS; row++) {
        if (row < row_count) {
            q_rms_banked_row<LANES>(
                out_stream, in_stream, weight_bank, elem_count);
        }
    }
}

template <unsigned int LANES>
static void q_rms_static_pipeline(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    hls::stream<cu_vec16_packet_t>& weight_stream,
    unsigned int elem_count,
    unsigned int row_count
) {
    #pragma HLS inline off

    wt_norm_t weight_bank[LANES][Q_RMS_BANK_DEPTH];
    #pragma HLS array_partition variable=weight_bank complete dim=1
    #pragma HLS bind_storage variable=weight_bank type=ram_2p impl=bram

    const unsigned int weight_packets =
        ceildiv(elem_count, CU_VEC_LANES);
    for (unsigned int packet = 0;
         packet < weight_packets;
         packet++) {
        cu_vec16_packet_t weight_packet = weight_stream.read();
        #pragma HLS array_partition variable=weight_packet.data complete dim=1
        q_rms_fixed_preload_packet<LANES>(
            weight_bank, weight_packet, elem_count);
    }

    const unsigned int active_rows =
        row_count < Q_RMS_MAX_ROWS ? row_count : Q_RMS_MAX_ROWS;
    for (unsigned int row = 0; row < active_rows; row++) {
        q_rms_static_row<LANES>(
            out_stream, in_stream, weight_bank, elem_count);
    }
}

// The top keeps the same cu_vec16_packet_t streams as production vector
// exchange.  Only the internal lane/cache organization changes with the
// compile-time Q_RMS_LANES setting.  Its definition lives in the design TU;
// the CSim/CoSim TB TU sees only this declaration.
void quantized_rmsnorm_lane_probe(
    hls::stream<cu_vec16_packet_t>& out_stream,
    hls::stream<cu_vec16_packet_t>& in_stream,
    hls::stream<cu_vec16_packet_t>& weight_stream,
    unsigned int elem_count,
    unsigned int row_count
);

#endif
