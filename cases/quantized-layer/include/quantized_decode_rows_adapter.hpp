#ifndef LLM_FPGA_QUANTIZED_DECODE_ROWS_ADAPTER_HPP
#define LLM_FPGA_QUANTIZED_DECODE_ROWS_ADAPTER_HPP

// Opt-in bridge from the production matrix stream ABI to the canonical qdr
// kernel. Production weights remain two W4 or four W8 256-bit words. The feed
// stage assembles only valid external DR stripes into one 4096-bit qdr tile.
#include "quantized_decode_rows.hpp"
#include "mm_stream_8x128_int4x4_block.hpp"
#include "mm_stream_4x128_int8x8_block.hpp"

#include <hls_stream.h>

#ifndef QUANTIZED_DECODE_ROWS_MERGED
#define QUANTIZED_DECODE_ROWS_MERGED 0
#endif

#ifndef QUANTIZED_DECODE_ROWS_ADAPTER_FIFO_DEPTH
#define QUANTIZED_DECODE_ROWS_ADAPTER_FIFO_DEPTH 2
#endif

// Activation is emitted before the corresponding assembled weight tile.
// Tune this narrow channel independently to absorb the registered BRAM
// weight-path latency without enlarging a 4096-bit buffer or task metadata.
// Keep the previous depth unless an evaluation explicitly selects another.
#ifndef QUANTIZED_DECODE_ROWS_ADAPTER_ACTIVATION_FIFO_DEPTH
#define QUANTIZED_DECODE_ROWS_ADAPTER_ACTIVATION_FIFO_DEPTH QUANTIZED_DECODE_ROWS_ADAPTER_FIFO_DEPTH
#endif

// A depth-2 HLS BRAM FIFO has only one RAM slot plus its output register.
// Registered full/empty flags then limit continuous transfers to one word
// every two clocks. Four entries keep these wide data channels at II=1;
// task/metadata and activation channels retain their small default depth.
#ifndef QUANTIZED_DECODE_ROWS_ADAPTER_WEIGHT_FIFO_DEPTH
#define QUANTIZED_DECODE_ROWS_ADAPTER_WEIGHT_FIFO_DEPTH 4
#endif
#ifndef QUANTIZED_DECODE_ROWS_ADAPTER_RESULT_FIFO_DEPTH
#define QUANTIZED_DECODE_ROWS_ADAPTER_RESULT_FIFO_DEPTH 4
#endif

constexpr unsigned int QDRA_MODE_BIT = 126;
constexpr unsigned int QDRA_PACKET_LANES = 16;
constexpr unsigned int QDRA_OLD_STRIPE_COLS = 128;
constexpr unsigned int QDRA_OLD_STRIPE_BITS = 512;

#ifdef QUANTIZED_ALIGNMENT_W8
constexpr unsigned int QDRA_WEIGHT_PORTS = 4;
constexpr unsigned int QDRA_INGRESS_BITS = 1024;
using qdra_activation_word_t =
    mm_stream_4x128_int8x8_activation_word_t;
using qdra_weight_word_t = mm_stream_4x128_int8x8_weight_word_t;
using qdra_output_word_t = mm_stream_4x128_int8x8_output_word_t;
constexpr unsigned int QDRA_DECODE_STRIPES = 4;
#else
constexpr unsigned int QDRA_WEIGHT_PORTS = 2;
constexpr unsigned int QDRA_INGRESS_BITS = 512;
using qdra_activation_word_t =
    mm_stream_8x128_int4x4_activation_word_t;
using qdra_weight_word_t = mm_stream_8x128_int4x4_weight_word_t;
using qdra_output_word_t = mm_stream_8x128_int4x4_output_word_t;
constexpr unsigned int QDRA_DECODE_STRIPES = 8;
#endif

// The merged build intentionally uses the qdr kernel's wide internal ABI.
// Macro-off builds never include this header and retain all legacy widths.
static_assert(QDR_INGRESS_BITS == QDR_WEIGHT_BITS,
              "merged adapter requires a 4096-bit qdr internal ingress");
static_assert(qdra_weight_word_t::width == 256,
              "merged adapter requires the production 128-column stripe ABI");
static_assert(QDR_WEIGHT_BITS % QDRA_INGRESS_BITS == 0,
              "qdr internal ingress must contain the full weight tile");
static_assert(QUANTIZED_DECODE_ROWS_ADAPTER_FIFO_DEPTH > 0,
              "merged adapter FIFOs must be finite and non-zero");
static_assert(QUANTIZED_DECODE_ROWS_ADAPTER_ACTIVATION_FIFO_DEPTH >= 2,
              "merged adapter activation FIFO requires at least two entries");
static_assert(QUANTIZED_DECODE_ROWS_ADAPTER_WEIGHT_FIFO_DEPTH >= 2 &&
              QUANTIZED_DECODE_ROWS_ADAPTER_RESULT_FIFO_DEPTH >= 2,
              "merged adapter BRAM FIFOs require at least two entries");

constexpr unsigned int QDRA_PACKET_COUNT =
    QDR_DECODE_COLS / QDRA_PACKET_LANES;
constexpr unsigned int QDRA_META_BASE = QDR_ACCUM_BITS * QDRA_PACKET_LANES;

// The qdr header aliases this to ap_uint<QDR_INGRESS_BITS>. Keep the adapter
// spelling explicit so changing external stream width cannot change it.
using qdra_qdr_weight_word_t = qdr_weight_word_t;

#ifndef QUANTIZED_ALIGNMENT_W8
// Keep the external stream identities explicit at the HLS boundary. Passing
// these streams through a pointer array can make Vitis lose the consumer
// relation for the second W4 port during dataflow analysis.
static ap_uint<512> qdra_read_weight_beat_w4(
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight0,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight1) {
    #pragma HLS inline
    ap_uint<512> word = 0;
    word.range(255, 0) = weight0.read();
    word.range(511, 256) = weight1.read();
    return word;
}
#endif

#ifdef QUANTIZED_ALIGNMENT_W8
static ap_uint<1024> qdra_read_weight_beat_w8(
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight0,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight1,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight2,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight3) {
    #pragma HLS inline
    ap_uint<1024> word = 0;
    word.range(255, 0) = weight0.read();
    word.range(511, 256) = weight1.read();
    word.range(767, 512) = weight2.read();
    word.range(1023, 768) = weight3.read();
    return word;
}
#endif

// Store each external stripe in an independent fixed-width bank. Updating a
// subrange of one 4096-bit ap_uint creates a wide feedback mux in the pipelined
// feeder; separate banks keep each beat's write local and pack only on emit.
#ifndef QUANTIZED_ALIGNMENT_W8
struct qdra_weight_tile_segments_t {
    ap_uint<512> segment0;
    ap_uint<512> segment1;
    ap_uint<512> segment2;
    ap_uint<512> segment3;
    ap_uint<512> segment4;
    ap_uint<512> segment5;
    ap_uint<512> segment6;
    ap_uint<512> segment7;
};
#else
struct qdra_weight_tile_segments_t {
    ap_uint<1024> segment0;
    ap_uint<1024> segment1;
    ap_uint<1024> segment2;
    ap_uint<1024> segment3;
};
#endif

static void qdra_clear_weight_tile(qdra_weight_tile_segments_t& tile) {
    #pragma HLS inline
    tile.segment0 = 0;
    tile.segment1 = 0;
    tile.segment2 = 0;
    tile.segment3 = 0;
#ifndef QUANTIZED_ALIGNMENT_W8
    tile.segment4 = 0;
    tile.segment5 = 0;
    tile.segment6 = 0;
    tile.segment7 = 0;
#endif
}

// The selector now chooses one narrow register bank rather than a slice of a
// 4096-bit feedback register. Invalid tail banks remain zero from clear above.
static void qdra_store_weight_segment(
    qdra_weight_tile_segments_t& tile,
    const ap_uint<QDRA_INGRESS_BITS>& narrow, unsigned int segment) {
    #pragma HLS inline
    switch (segment) {
    case 0:
        tile.segment0 = narrow;
        break;
    case 1:
        tile.segment1 = narrow;
        break;
    case 2:
        tile.segment2 = narrow;
        break;
    case 3:
        tile.segment3 = narrow;
        break;
#ifndef QUANTIZED_ALIGNMENT_W8
    case 4:
        tile.segment4 = narrow;
        break;
    case 5:
        tile.segment5 = narrow;
        break;
    case 6:
        tile.segment6 = narrow;
        break;
    case 7:
        tile.segment7 = narrow;
        break;
#endif
    default:
        break;
    }
}

static qdra_qdr_weight_word_t qdra_pack_weight_tile(
    const qdra_weight_tile_segments_t& tile) {
    #pragma HLS inline
    qdra_qdr_weight_word_t packed = 0;
    packed.range(QDRA_INGRESS_BITS - 1, 0) = tile.segment0;
    packed.range(2 * QDRA_INGRESS_BITS - 1, QDRA_INGRESS_BITS) =
        tile.segment1;
    packed.range(3 * QDRA_INGRESS_BITS - 1, 2 * QDRA_INGRESS_BITS) =
        tile.segment2;
    packed.range(4 * QDRA_INGRESS_BITS - 1, 3 * QDRA_INGRESS_BITS) =
        tile.segment3;
#ifndef QUANTIZED_ALIGNMENT_W8
    packed.range(5 * QDRA_INGRESS_BITS - 1, 4 * QDRA_INGRESS_BITS) =
        tile.segment4;
    packed.range(6 * QDRA_INGRESS_BITS - 1, 5 * QDRA_INGRESS_BITS) =
        tile.segment5;
    packed.range(7 * QDRA_INGRESS_BITS - 1, 6 * QDRA_INGRESS_BITS) =
        tile.segment6;
    packed.range(8 * QDRA_INGRESS_BITS - 1, 7 * QDRA_INGRESS_BITS) =
        tile.segment7;
#endif
    return packed;
}

#ifndef QUANTIZED_ALIGNMENT_W8
static void quantized_decode_rows_adapter_feed_w4(
    hls::stream<mm_stream_quantized_task_word_t>& tasks,
    hls::stream<qdra_activation_word_t>& activations,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight0,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight1,
    hls::stream<mm_stream_quantized_task_word_t>& task_fifo,
    hls::stream<qdra_activation_word_t>& activation_fifo,
    hls::stream<qdra_qdr_weight_word_t>& weight_fifo,
    hls::stream<mm_stream_quantized_task_word_t>& metadata_fifo,
    unsigned int task_count) {
    #pragma HLS inline off
    for (unsigned int task_index = 0; task_index < task_count; ++task_index) {
        #pragma HLS loop_tripcount min=1 max=4096 avg=64
        const mm_stream_quantized_task_word_t packed_task = tasks.read();
        const mm_stream_quantized_task_t task =
            unpack_mm_stream_quantized_task(packed_task);
        const bool decode = packed_task[QDRA_MODE_BIT];
        const unsigned int valid_stripes = decode ?
            (task.valid_tokens > QDRA_DECODE_STRIPES ?
                QDRA_DECODE_STRIPES : task.valid_tokens) : 1;

        // One metadata word covers all qdr packets for this task. This keeps
        // the output sideband finite and avoids carrying the 128-bit task on
        // every result packet.
        task_fifo.write(packed_task);
        metadata_fifo.write(packed_task);
        if (valid_stripes == 0) {
            // An inactive DR task should not consume external HBM words, but
            // qdr still needs one zero tile per K to make its task progress.
            for (unsigned int k = 0; k < task.k_count; ++k) {
                #pragma HLS pipeline II=1
                activation_fifo.write(activations.read());
                weight_fifo.write(qdra_qdr_weight_word_t(0));
            }
        } else {
            // K-major flattening consumes one valid external stripe per
            // iteration and one activation per K. beat_in_k is a counter rather
            // than a division/modulo expression, so the segment selector has
            // a fixed-width switch and tails do not burn padding cycles.
            qdra_weight_tile_segments_t tile;
            unsigned int beat_in_k = 0;
            const unsigned int total_beats = task.k_count * valid_stripes;
            for (unsigned int beat_index = 0; beat_index < total_beats;
                 ++beat_index) {
                #pragma HLS pipeline II=1
                #pragma HLS loop_tripcount min=1 max=88064 avg=2048
                if (beat_in_k == 0) {
                    qdra_clear_weight_tile(tile);
                    activation_fifo.write(activations.read());
                }
                const ap_uint<512> narrow =
                    qdra_read_weight_beat_w4(weight0, weight1);
                qdra_store_weight_segment(tile, narrow, beat_in_k);
                if (beat_in_k + 1 == valid_stripes) {
                    weight_fifo.write(qdra_pack_weight_tile(tile));
                    beat_in_k = 0;
                } else {
                    ++beat_in_k;
                }
            }
        }
    }
}
#endif

#ifdef QUANTIZED_ALIGNMENT_W8
static void quantized_decode_rows_adapter_feed_w8(
    hls::stream<mm_stream_quantized_task_word_t>& tasks,
    hls::stream<qdra_activation_word_t>& activations,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight0,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight1,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight2,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight3,
    hls::stream<mm_stream_quantized_task_word_t>& task_fifo,
    hls::stream<qdra_activation_word_t>& activation_fifo,
    hls::stream<qdra_qdr_weight_word_t>& weight_fifo,
    hls::stream<mm_stream_quantized_task_word_t>& metadata_fifo,
    unsigned int task_count) {
    #pragma HLS inline off
    for (unsigned int task_index = 0; task_index < task_count; ++task_index) {
        #pragma HLS loop_tripcount min=1 max=4096 avg=64
        const mm_stream_quantized_task_word_t packed_task = tasks.read();
        const mm_stream_quantized_task_t task =
            unpack_mm_stream_quantized_task(packed_task);
        const bool decode = packed_task[QDRA_MODE_BIT];
        const unsigned int valid_stripes = decode ?
            (task.valid_tokens > QDRA_DECODE_STRIPES ?
                QDRA_DECODE_STRIPES : task.valid_tokens) : 1;

        task_fifo.write(packed_task);
        metadata_fifo.write(packed_task);
        if (valid_stripes == 0) {
            for (unsigned int k = 0; k < task.k_count; ++k) {
                #pragma HLS pipeline II=1
                activation_fifo.write(activations.read());
                weight_fifo.write(qdra_qdr_weight_word_t(0));
            }
        } else {
            qdra_weight_tile_segments_t tile;
            unsigned int beat_in_k = 0;
            const unsigned int total_beats = task.k_count * valid_stripes;
            for (unsigned int beat_index = 0; beat_index < total_beats;
                 ++beat_index) {
                #pragma HLS pipeline II=1
                #pragma HLS loop_tripcount min=1 max=88064 avg=2048
                if (beat_in_k == 0) {
                    qdra_clear_weight_tile(tile);
                    activation_fifo.write(activations.read());
                }
                const ap_uint<1024> narrow =
                    qdra_read_weight_beat_w8(
                        weight0, weight1, weight2, weight3);
                qdra_store_weight_segment(tile, narrow, beat_in_k);
                if (beat_in_k + 1 == valid_stripes) {
                    weight_fifo.write(qdra_pack_weight_tile(tile));
                    beat_in_k = 0;
                } else {
                    ++beat_in_k;
                }
            }
        }
    }
}
#endif

static void quantized_decode_rows_adapter_output(
    hls::stream<qdra_output_word_t>& output,
    hls::stream<qdr_output_word_t>& qdr_output,
    hls::stream<mm_stream_quantized_task_word_t>& metadata_fifo,
    unsigned int task_count) {
    #pragma HLS inline off
    for (unsigned int task_index = 0; task_index < task_count; ++task_index) {
        #pragma HLS loop_tripcount min=1 max=4096 avg=64
        const mm_stream_quantized_task_word_t packed_task =
            metadata_fifo.read();
        const mm_stream_quantized_task_t task =
            unpack_mm_stream_quantized_task(packed_task);
        const bool decode = packed_task[QDRA_MODE_BIT];
        for (unsigned int packet = 0; packet < QDRA_PACKET_COUNT; ++packet) {
            #pragma HLS pipeline II=1
            const qdr_output_word_t raw = qdr_output.read();
            if (!decode) {
                output.write(raw);
                continue;
            }

            const unsigned int local_base = raw.range(
                QDRA_META_BASE + 39, QDRA_META_BASE + 24).to_uint() -
                task.elem_base;
            const unsigned int cu = (task.elem_base / QDRA_OLD_STRIPE_COLS) & 3u;
            const unsigned int superwave_base =
                task.elem_base - cu * QDRA_OLD_STRIPE_COLS;
            const unsigned int stripe = local_base / QDRA_OLD_STRIPE_COLS;
            const unsigned int mapped_base = superwave_base +
                stripe * QDRA_OLD_STRIPE_BITS + cu * QDRA_OLD_STRIPE_COLS +
                local_base % QDRA_OLD_STRIPE_COLS;
            const unsigned int valid_stripes = task.valid_tokens >
                    QDRA_DECODE_STRIPES ? QDRA_DECODE_STRIPES :
                    task.valid_tokens;
            const unsigned int valid_columns = valid_stripes *
                QDRA_OLD_STRIPE_COLS;
            qdra_output_word_t mapped = 0;
            unsigned int valid_mask = 0;
            for (unsigned int lane = 0; lane < QDRA_PACKET_LANES; ++lane) {
                #pragma HLS unroll
                const unsigned int local_column = local_base + lane;
                const bool valid = local_column < valid_columns;
                if (valid)
                    valid_mask |= 1u << lane;
                mapped.range((lane + 1) * QDR_ACCUM_BITS - 1,
                             lane * QDR_ACCUM_BITS) = valid ? raw.range(
                        (lane + 1) * QDR_ACCUM_BITS - 1,
                        lane * QDR_ACCUM_BITS) : 0;
            }
            mapped.range(QDRA_META_BASE + 15, QDRA_META_BASE) = valid_mask;
            mapped.range(QDRA_META_BASE + 23, QDRA_META_BASE + 16) = 0;
            mapped.range(QDRA_META_BASE + 39, QDRA_META_BASE + 24) =
                mapped_base;
            mapped.range(QDRA_META_BASE + 55, QDRA_META_BASE + 40) =
                raw.range(QDRA_META_BASE + 55, QDRA_META_BASE + 40);
            mapped[QDRA_META_BASE + 56] = raw[QDRA_META_BASE + 56];
            mapped[QDRA_META_BASE + 57] = raw[QDRA_META_BASE + 57];
            output.write(mapped);
        }
    }
}

#ifndef QUANTIZED_ALIGNMENT_W8
static void run_quantized_decode_rows_adapter_w4(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& output,
    hls::stream<mm_stream_quantized_task_word_t>& tasks,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activations,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight0,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight1,
    unsigned int task_count) {
    #pragma HLS inline off
    hls::stream<mm_stream_quantized_task_word_t> task_fifo;
    hls::stream<mm_stream_8x128_int4x4_activation_word_t> activation_fifo;
    hls::stream<qdr_weight_word_t> weight_fifo;
    hls::stream<mm_stream_quantized_task_word_t> metadata_fifo;
    hls::stream<qdr_output_word_t> qdr_output;
    #pragma HLS stream variable=task_fifo depth=QUANTIZED_DECODE_ROWS_ADAPTER_FIFO_DEPTH
    #pragma HLS stream variable=activation_fifo depth=QUANTIZED_DECODE_ROWS_ADAPTER_ACTIVATION_FIFO_DEPTH
    #pragma HLS stream variable=weight_fifo depth=QUANTIZED_DECODE_ROWS_ADAPTER_WEIGHT_FIFO_DEPTH
    #pragma HLS stream variable=metadata_fifo depth=QUANTIZED_DECODE_ROWS_ADAPTER_FIFO_DEPTH
    #pragma HLS stream variable=qdr_output depth=QUANTIZED_DECODE_ROWS_ADAPTER_RESULT_FIFO_DEPTH
    #pragma HLS bind_storage variable=weight_fifo type=fifo impl=bram
    #pragma HLS bind_storage variable=qdr_output type=fifo impl=bram
    #pragma HLS dataflow
    quantized_decode_rows_adapter_feed_w4(
        tasks, activations, weight0, weight1, task_fifo, activation_fifo,
        weight_fifo, metadata_fifo, task_count);
    compute_quantized_decode_rows(
        qdr_output, task_fifo, activation_fifo, weight_fifo, task_count);
    quantized_decode_rows_adapter_output(
        output, qdr_output, metadata_fifo, task_count);
}
#endif

#ifdef QUANTIZED_ALIGNMENT_W8
static void run_quantized_decode_rows_adapter_w8(
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& output,
    hls::stream<mm_stream_quantized_task_word_t>& tasks,
    hls::stream<mm_stream_4x128_int8x8_activation_word_t>& activations,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight0,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight1,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight2,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight3,
    unsigned int task_count) {
    #pragma HLS inline off
    hls::stream<mm_stream_quantized_task_word_t> task_fifo;
    hls::stream<mm_stream_4x128_int8x8_activation_word_t> activation_fifo;
    hls::stream<qdr_weight_word_t> weight_fifo;
    hls::stream<mm_stream_quantized_task_word_t> metadata_fifo;
    hls::stream<qdr_output_word_t> qdr_output;
    #pragma HLS stream variable=task_fifo depth=QUANTIZED_DECODE_ROWS_ADAPTER_FIFO_DEPTH
    #pragma HLS stream variable=activation_fifo depth=QUANTIZED_DECODE_ROWS_ADAPTER_ACTIVATION_FIFO_DEPTH
    #pragma HLS stream variable=weight_fifo depth=QUANTIZED_DECODE_ROWS_ADAPTER_WEIGHT_FIFO_DEPTH
    #pragma HLS stream variable=metadata_fifo depth=QUANTIZED_DECODE_ROWS_ADAPTER_FIFO_DEPTH
    #pragma HLS stream variable=qdr_output depth=QUANTIZED_DECODE_ROWS_ADAPTER_RESULT_FIFO_DEPTH
    #pragma HLS bind_storage variable=weight_fifo type=fifo impl=bram
    #pragma HLS bind_storage variable=qdr_output type=fifo impl=bram
    #pragma HLS dataflow
    quantized_decode_rows_adapter_feed_w8(
        tasks, activations, weight0, weight1, weight2, weight3, task_fifo,
        activation_fifo, weight_fifo, metadata_fifo, task_count);
    compute_quantized_decode_rows(
        qdr_output, task_fifo, activation_fifo, weight_fifo, task_count);
    quantized_decode_rows_adapter_output(
        output, qdr_output, metadata_fifo, task_count);
}
#endif

#endif  // LLM_FPGA_QUANTIZED_DECODE_ROWS_ADAPTER_HPP
