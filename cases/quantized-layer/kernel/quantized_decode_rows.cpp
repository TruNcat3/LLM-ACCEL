#include "quantized_decode_rows.hpp"

#if QDR_COMPACT
namespace {

constexpr unsigned int QDR_PACKET_LANES = 16;
constexpr unsigned int QDR_PACKET_COUNT = QDR_DECODE_COLS / QDR_PACKET_LANES;
constexpr unsigned int QDR_PREFILL_BITS = QDR_PREFILL_COLS * QDR_BITS;
constexpr unsigned int QDR_DECODE_BEATS = QDR_WEIGHT_BITS / QDR_INGRESS_BITS;
constexpr unsigned int QDR_PREFILL_BEATS =
    (QDR_PREFILL_BITS + QDR_INGRESS_BITS - 1) / QDR_INGRESS_BITS;
constexpr unsigned int QDR_COMPACT_FIFO_DEPTH = 2;

using qdr_weight_tile_t = ap_uint<QDR_WEIGHT_BITS>;
using qdr_accum_t = ap_int<QDR_ACCUM_BITS>;

// Reading a narrow lane through ap_uint::range() asks the C model to construct
// a value from a wide range reference. That is functionally correct, but in
// the block path it repeats a wide shift for every DSP lane. Build only the
// requested bits; synthesis sees the same fixed lane wiring and the C model
// avoids the wide conversion.
template <int Bits, int Width>
static ap_uint<Bits> qdr_read_bits(const ap_uint<Width>& word,
                                   unsigned int base) {
    #pragma HLS inline
    ap_uint<Bits> value = 0;
    for (unsigned int bit = 0; bit < Bits; ++bit) {
        #pragma HLS unroll
        value[bit] = word[base + bit];
    }
    return value;
}

template <int Width>
static ap_int<4> qdr_read_int4(const ap_uint<Width>& word,
                               unsigned int base) {
    #pragma HLS inline
    return ap_int<4>(qdr_read_bits<4>(word, base));
}

#if QDR_SHARED_STATE
// Sixteen physical banks serve sixteen output lanes in either mode. In W4,
// xor physical bits 1 and 4 so both a contiguous D packet and a strided P
// packet read exactly one entry per bank. All MAC destinations stay fixed.
static unsigned int qdr_accumulator_bank(unsigned int physical) {
    #pragma HLS inline
#ifdef QUANTIZED_ALIGNMENT_W8
    return physical & 15;
#else
    return (physical & 1) | ((physical >> 1) & 6) |
           ((((physical >> 1) ^ (physical >> 4)) & 1) << 3);
#endif
}

static void qdr_emit_shared_accumulator(
    hls::stream<qdr_output_word_t>& output,
    const qdr_accum_t accumulator[QDR_PACKET_LANES][QDR_PACKET_COUNT],
    const mm_stream_quantized_task_t& task, bool decode,
    unsigned int valid_rows, unsigned int packet) {
    #pragma HLS inline
    const unsigned int row = decode ? 0 : packet / (QDR_PREFILL_COLS / 16);
    const unsigned int column_base = decode ? packet * 16 : (packet % 8) * 16;
    const bool valid = decode || row < valid_rows;
    qdr_accum_t bank_values[QDR_PACKET_LANES];
    #pragma HLS array_partition variable=bank_values complete
    for (unsigned int bank = 0; bank < QDR_PACKET_LANES; ++bank) {
        #pragma HLS unroll
#ifdef QUANTIZED_ALIGNMENT_W8
        const unsigned int slot = packet;
#else
        const unsigned int slot = decode ? packet :
            (row / 2) * 16 + (packet % 8) * 2 + ((bank / 8) ^ (row & 1));
#endif
        // A zero-K task must not read uninitialized state from a prior task.
        bank_values[bank] = task.k_count == 0 ? qdr_accum_t(0) :
            accumulator[bank][slot];
    }
    qdr_output_word_t word = 0;
    for (unsigned int lane = 0; lane < QDR_PACKET_LANES; ++lane) {
        #pragma HLS unroll
#ifdef QUANTIZED_ALIGNMENT_W8
        const qdr_accum_t value = bank_values[lane];
#else
        const unsigned int d_bank = qdr_accumulator_bank(lane);
        const qdr_accum_t d_value = (packet & 1) ?
            bank_values[d_bank ^ 8] : bank_values[d_bank];
        const qdr_accum_t p_value = (row & 1) ?
            bank_values[lane ^ 8] : bank_values[lane];
        const qdr_accum_t value = decode ? d_value : p_value;
#endif
        word.range((lane + 1) * QDR_ACCUM_BITS - 1, lane * QDR_ACCUM_BITS) =
            valid ? value : qdr_accum_t(0);
    }
    constexpr unsigned int meta = QDR_ACCUM_BITS * QDR_PACKET_LANES;
    word.range(meta + 15, meta) = valid ? 0xffff : 0;
    word.range(meta + 23, meta + 16) = row;
    word.range(meta + 39, meta + 24) = task.elem_base + column_base;
    word.range(meta + 55, meta + 40) = task.block_id;
    const bool last = packet + 1 == QDR_PACKET_COUNT;
    word[meta + 56] = last;
    word[meta + 57] = last && task.last_stream;
    output.write(word);
}
#endif

// The feeder owns beat assembly.  The compute stage therefore sees one
// complete 4096-bit tile per K, independent of the selected ingress width.
// Flattening the K/beat loops keeps the next tile in flight while the
// consumer performs the previous tile's MAC.  BeatCount is compile-time so
// the modulo/range expressions collapse to small counters in each ingress
// configuration.
template <unsigned int BeatCount>
static void qdr_compact_feed_task(
    hls::stream<qdr_activation_word_t>& activations,
    hls::stream<qdr_weight_word_t>& weights,
    hls::stream<qdr_activation_word_t>& activation_fifo,
    hls::stream<qdr_weight_tile_t>& weight_fifo,
    unsigned int k_count) {
    #pragma HLS inline off
    qdr_weight_tile_t tile = 0;
    for (unsigned int beat_index = 0; beat_index < k_count * BeatCount;
         ++beat_index) {
        #pragma HLS pipeline II=1
        const unsigned int beat = beat_index % BeatCount;
        if (beat == 0) {
            tile = 0;
            activation_fifo.write(activations.read());
        }
#if QDR_BLOCK_OPS
        // The first input beat becomes the low segment after BeatCount
        // shifts. Both ranges are constants, avoiding a 4096-bit variable
        // write mask/multiplexer for narrow ingress.
        ap_uint<BeatCount * QDR_INGRESS_BITS> assembled = tile;
        assembled >>= QDR_INGRESS_BITS;
        assembled.range(BeatCount * QDR_INGRESS_BITS - 1,
                        (BeatCount - 1) * QDR_INGRESS_BITS) = weights.read();
        tile = assembled;
#else
        tile.range((beat + 1) * QDR_INGRESS_BITS - 1,
                   beat * QDR_INGRESS_BITS) = weights.read();
#endif
        if (beat + 1 == BeatCount) {
            weight_fifo.write(tile);
        }
    }
}

static void qdr_compact_feed(
    hls::stream<mm_stream_quantized_task_word_t>& tasks,
    hls::stream<qdr_activation_word_t>& activations,
    hls::stream<qdr_weight_word_t>& weights,
    hls::stream<mm_stream_quantized_task_word_t>& task_fifo,
    hls::stream<qdr_activation_word_t>& activation_fifo,
    hls::stream<qdr_weight_tile_t>& weight_fifo,
    unsigned int task_count) {
    #pragma HLS inline off
    for (unsigned int task_index = 0; task_index < task_count; ++task_index) {
        #pragma HLS loop_tripcount min=1 max=64
        const mm_stream_quantized_task_word_t packed_task = tasks.read();
        const mm_stream_quantized_task_t task =
            unpack_mm_stream_quantized_task(packed_task);
        const bool decode = packed_task[QDR_MODE_BIT];
        task_fifo.write(packed_task);
        if (decode) {
            qdr_compact_feed_task<QDR_DECODE_BEATS>(
                activations, weights, activation_fifo, weight_fifo,
                task.k_count);
        } else {
            qdr_compact_feed_task<QDR_PREFILL_BEATS>(
                activations, weights, activation_fifo, weight_fifo,
                task.k_count);
        }
    }
}

static void qdr_compact_emit_packet(
    hls::stream<qdr_output_word_t>& output,
    qdr_accum_t packet_window[QDR_PACKET_COUNT][QDR_PACKET_LANES],
    const mm_stream_quantized_task_t& task, bool decode,
    unsigned int valid_rows, unsigned int packet) {
    #pragma HLS inline
    qdr_output_word_t word = 0;
    const unsigned int row = decode ? 0 : packet / (QDR_PREFILL_COLS / 16);
    const unsigned int column_base =
        decode ? packet * QDR_PACKET_LANES :
                 (packet % (QDR_PREFILL_COLS / 16)) * QDR_PACKET_LANES;
    const bool valid = decode || row < valid_rows;
    for (unsigned int lane = 0; lane < QDR_PACKET_LANES; ++lane) {
        #pragma HLS unroll
        const qdr_accum_t value = valid ? packet_window[0][lane] :
                                                qdr_accum_t(0);
        word.range((lane + 1) * QDR_ACCUM_BITS - 1,
                   lane * QDR_ACCUM_BITS) = value;
    }
    constexpr unsigned int meta = QDR_ACCUM_BITS * QDR_PACKET_LANES;
    word.range(meta + 15, meta) = valid ? 0xffff : 0;
    word.range(meta + 23, meta + 16) = row;
    word.range(meta + 39, meta + 24) = task.elem_base + column_base;
    word.range(meta + 55, meta + 40) = task.block_id;
    const bool last = packet + 1 == QDR_PACKET_COUNT;
    word[meta + 56] = last;
    word[meta + 57] = last && task.last_stream;
    output.write(word);
}

static void qdr_compact_shift_packets(
    qdr_accum_t packet_window[QDR_PACKET_COUNT][QDR_PACKET_LANES]) {
    #pragma HLS inline
    // The complete unroll makes every source/destination a fixed register
    // connection.  Packet zero is the sole output read port; no run-time
    // selection of one of the full accumulator banks is synthesized.
    for (unsigned int packet = 0; packet + 1 < QDR_PACKET_COUNT; ++packet) {
        #pragma HLS unroll
        for (unsigned int lane = 0; lane < QDR_PACKET_LANES; ++lane) {
            #pragma HLS unroll
            packet_window[packet][lane] = packet_window[packet + 1][lane];
        }
    }
}

static void qdr_compact_execute(
    hls::stream<qdr_output_word_t>& output,
    hls::stream<mm_stream_quantized_task_word_t>& task_fifo,
    hls::stream<qdr_activation_word_t>& activation_fifo,
    hls::stream<qdr_weight_tile_t>& weight_fifo,
    unsigned int task_count) {
    #pragma HLS inline off
#if QDR_SHARED_STATE
    qdr_accum_t accumulator[QDR_PACKET_LANES][QDR_PACKET_COUNT];
    #pragma HLS array_partition variable=accumulator complete dim=0
#else
    // Physical MAC storage retains the old fixed DSP-to-accumulator mapping.
    // packet_window is output-only staging; it is populated once per task so
    // the MAC loop never writes through a mode-selected accumulator index.
    qdr_accum_t accumulator[QDR_DECODE_COLS];
    #pragma HLS array_partition variable=accumulator complete
    qdr_accum_t packet_window[QDR_PACKET_COUNT][QDR_PACKET_LANES];
    #pragma HLS array_partition variable=packet_window complete
#endif

    for (unsigned int task_index = 0; task_index < task_count; ++task_index) {
        #pragma HLS loop_tripcount min=1 max=64
        const mm_stream_quantized_task_word_t packed_task = task_fifo.read();
        const mm_stream_quantized_task_t task =
            unpack_mm_stream_quantized_task(packed_task);
        const bool decode = packed_task[QDR_MODE_BIT];
        const unsigned int valid_rows = task.valid_tokens == 0 ||
                task.valid_tokens > QDR_ROWS ? QDR_ROWS : task.valid_tokens;

#if QDR_SHARED_STATE
        // One scheduled loop owns the accumulator throughout MAC and output.
        // Separate pipelined loops caused HLS to capture full arrays across
        // module boundaries. The first K initializes; zero-K emits zeros.
        for (unsigned int step = 0; step < task.k_count + QDR_PACKET_COUNT;
             ++step) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=32 max=11072 avg=2112
            if (step < task.k_count) {
                const unsigned int k = step;
#else
        for (unsigned int index = 0; index < QDR_DECODE_COLS; ++index) {
            #pragma HLS unroll
            accumulator[index] = 0;
        }

        for (unsigned int k = 0; k < task.k_count; ++k) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=1 max=11008 avg=2048
#endif
            const qdr_activation_word_t activation_word = activation_fifo.read();
            const qdr_weight_tile_t weight_word = weight_fifo.read();
#ifdef QUANTIZED_ALIGNMENT_W8
            for (unsigned int dsp = 0; dsp < QDR_DSPS; ++dsp) {
                #pragma HLS unroll
                const unsigned int row = dsp / QDR_PREFILL_COLS;
                const unsigned int column = dsp % QDR_PREFILL_COLS;
                const ap_int<8> activation = decode ?
                    ap_int<8>(activation_word.range(7, 0)) :
                    ap_int<8>(activation_word.range(row * 8 + 7, row * 8));
                const ap_int<8> weight = decode ?
                    ap_int<8>(weight_word.range(dsp * 8 + 7, dsp * 8)) :
                    ap_int<8>(weight_word.range(column * 8 + 7, column * 8));
                ap_int<16> product;
                #pragma HLS bind_op variable=product op=mul impl=dsp
                product = activation * weight;
#if QDR_SHARED_STATE
                qdr_accum_t& acc = accumulator[qdr_accumulator_bank(dsp)][dsp / 16];
#else
                qdr_accum_t& acc = accumulator[dsp];
#endif
                acc = k == 0 ? qdr_accum_t(product) : qdr_accum_t(acc + product);
            }
#else
#if QDR_BLOCK_OPS
            // Decode signs/magnitudes into the small fixed words consumed by
            // each DSP. The previous implementation built 4096/1024-bit
            // temporaries for every K, even though prefill uses only the
            // first 128 weight columns. Keeping one packed word per DSP (or
            // per prefill column pair) preserves the hardware lane mapping
            // without making the C model shift a wide ap_uint for each lane.
            ap_uint<8> activation_magnitudes[QDR_ROWS / 2];
            ap_uint<2> activation_signs[QDR_ROWS / 2];
            ap_uint<8> prefill_weight_magnitudes[QDR_PREFILL_COLS / 2];
            ap_uint<2> prefill_weight_signs[QDR_PREFILL_COLS / 2];
            ap_uint<16> decode_weight_magnitudes[QDR_DSPS];
            ap_uint<4> decode_weight_signs[QDR_DSPS];
            #pragma HLS array_partition variable=activation_magnitudes complete
            #pragma HLS array_partition variable=activation_signs complete
            #pragma HLS array_partition variable=prefill_weight_magnitudes complete
            #pragma HLS array_partition variable=prefill_weight_signs complete
            #pragma HLS array_partition variable=decode_weight_magnitudes complete
            #pragma HLS array_partition variable=decode_weight_signs complete

            for (unsigned int row_pair = 0; row_pair < QDR_ROWS / 2;
                 ++row_pair) {
                #pragma HLS unroll
                const ap_int<4> a0 = qdr_read_int4(
                    activation_word, row_pair * 8);
                const ap_int<4> a1 = qdr_read_int4(
                    activation_word, row_pair * 8 + 4);
                activation_magnitudes[row_pair] = 0;
                activation_magnitudes[row_pair].range(3, 0) = qdr_abs4(a0);
                activation_magnitudes[row_pair].range(7, 4) = qdr_abs4(a1);
                activation_signs[row_pair] = 0;
                activation_signs[row_pair][0] = a0[3];
                activation_signs[row_pair][1] = a1[3];
            }

            for (unsigned int column_pair = 0;
                 column_pair < QDR_PREFILL_COLS / 2; ++column_pair) {
                #pragma HLS unroll
                const ap_int<4> w0 = qdr_read_int4(
                    weight_word, column_pair * 8);
                const ap_int<4> w1 = qdr_read_int4(
                    weight_word, column_pair * 8 + 4);
                prefill_weight_magnitudes[column_pair] = 0;
                prefill_weight_magnitudes[column_pair].range(3, 0) =
                    qdr_abs4(w0);
                prefill_weight_magnitudes[column_pair].range(7, 4) =
                    qdr_abs4(w1);
                prefill_weight_signs[column_pair] = 0;
                prefill_weight_signs[column_pair][0] = w0[3];
                prefill_weight_signs[column_pair][1] = w1[3];
            }

            for (unsigned int dsp = 0; dsp < QDR_DSPS; ++dsp) {
                #pragma HLS unroll
                // The unrolled hardware lanes remain present in both modes;
                // prefill only enables the 128-column path and therefore
                // must not unpack the unused decode columns in C simulation.
                if (decode) {
                    ap_uint<16> magnitudes = 0;
                    ap_uint<4> signs = 0;
                    for (unsigned int lane = 0; lane < 4; ++lane) {
                        #pragma HLS unroll
                        const ap_int<4> value = qdr_read_int4(
                            weight_word, dsp * 16 + lane * 4);
                        magnitudes.range(lane * 4 + 3, lane * 4) =
                            qdr_abs4(value);
                        signs[lane] = value[3];
                    }
                    decode_weight_magnitudes[dsp] = magnitudes;
                    decode_weight_signs[dsp] = signs;
                }
            }
#endif
            for (unsigned int dsp = 0; dsp < QDR_DSPS; ++dsp) {
                #pragma HLS unroll
                const unsigned int row_pair = dsp / (QDR_PREFILL_COLS / 2);
                const unsigned int column_pair = dsp % (QDR_PREFILL_COLS / 2);
#if !QDR_BLOCK_OPS
                const ap_int<4> a0 = decode ?
                    qdr_read_int4(activation_word, 0) :
                    qdr_read_int4(activation_word, row_pair * 8);
                const ap_int<4> a1 =
                    qdr_read_int4(activation_word, row_pair * 8 + 4);
                const ap_int<4> w0 = decode ?
                    qdr_read_int4(weight_word, dsp * 16) :
                    qdr_read_int4(weight_word, column_pair * 8);
                const ap_int<4> w1 = decode ?
                    qdr_read_int4(weight_word, dsp * 16 + 4) :
                    qdr_read_int4(weight_word, column_pair * 8 + 4);
                const ap_int<4> w2 =
                    qdr_read_int4(weight_word, dsp * 16 + 8);
                const ap_int<4> w3 =
                    qdr_read_int4(weight_word, dsp * 16 + 12);
#endif
#if QDR_BLOCK_OPS
                const ap_uint<8> am = decode ?
                    activation_magnitudes[0] : activation_magnitudes[row_pair];
                const ap_uint<2> as = decode ?
                    activation_signs[0] : activation_signs[row_pair];
                const ap_uint<16> wm = decode ?
                    decode_weight_magnitudes[dsp] :
                    ap_uint<16>(prefill_weight_magnitudes[column_pair]);
                const ap_uint<4> ws = decode ?
                    decode_weight_signs[dsp] :
                    ap_uint<4>(prefill_weight_signs[column_pair]);
                const ap_uint<32> products = qdr_w4_products_from_magnitudes(
                    decode, am, as, wm, ws);
                for (unsigned int p = 0; p < 4; ++p) {
                    #pragma HLS unroll
                    const ap_int<8> product = products.range(p * 8 + 7, p * 8);
#if QDR_SHARED_STATE
                    const unsigned int physical = dsp * 4 + p;
                    qdr_accum_t& acc = accumulator[qdr_accumulator_bank(physical)][physical / 16];
#else
                    qdr_accum_t& acc = accumulator[dsp * 4 + p];
#endif
                    acc = k == 0 ? qdr_accum_t(product) : qdr_accum_t(acc + product);
                }
#else
                const qdr_w4_quad_t product = qdr_w4_products(
                    decode, a0, a1, w0, w1, w2, w3);
                for (unsigned int p = 0; p < 4; ++p) {
                    #pragma HLS unroll
#if QDR_SHARED_STATE
                    const unsigned int physical = dsp * 4 + p;
                    qdr_accum_t& acc = accumulator[qdr_accumulator_bank(physical)][physical / 16];
#else
                    qdr_accum_t& acc = accumulator[dsp * 4 + p];
#endif
                    acc = k == 0 ? qdr_accum_t(product.value[p]) :
                        qdr_accum_t(acc + product.value[p]);
                }
#endif
            }
#endif
        }

#if QDR_SHARED_STATE
            else {
                qdr_emit_shared_accumulator(output, accumulator, task, decode,
                    valid_rows, step - task.k_count);
            }
        }
#else

        // Convert the fixed physical MAC layout into packet-major output
        // order.  Both loops are fully unrolled, so each read is a fixed
        // register connection for the selected task mode; the run-time mode
        // no longer selects a write destination in the II=1 MAC loop.
        if (decode) {
            for (unsigned int packet = 0; packet < QDR_PACKET_COUNT;
                 ++packet) {
                #pragma HLS unroll
                for (unsigned int lane = 0; lane < QDR_PACKET_LANES; ++lane) {
                    #pragma HLS unroll
                    packet_window[packet][lane] =
                        accumulator[packet * QDR_PACKET_LANES + lane];
                }
            }
        } else {
            for (unsigned int packet = 0; packet < QDR_PACKET_COUNT;
                 ++packet) {
                #pragma HLS unroll
                const unsigned int row = packet / (QDR_PREFILL_COLS / 16);
                const unsigned int column_base =
                    (packet % (QDR_PREFILL_COLS / 16)) * QDR_PACKET_LANES;
                for (unsigned int lane = 0; lane < QDR_PACKET_LANES; ++lane) {
                    #pragma HLS unroll
                    const unsigned int column = column_base + lane;
#ifdef QUANTIZED_ALIGNMENT_W8
                    const unsigned int physical_index =
                        row * QDR_PREFILL_COLS + column;
#else
                    const unsigned int physical_index =
                        ((row / 2) * (QDR_PREFILL_COLS / 2) + column / 2) * 4 +
                        (row % 2) * 2 + column % 2;
#endif
                    packet_window[packet][lane] = accumulator[physical_index];
                }
            }
        }

        for (unsigned int packet = 0; packet < QDR_PACKET_COUNT; ++packet) {
            #pragma HLS pipeline II=1
            qdr_compact_emit_packet(output, packet_window, task, decode,
                                    valid_rows, packet);
            if (packet + 1 < QDR_PACKET_COUNT) {
                qdr_compact_shift_packets(packet_window);
            }
        }
#endif
    }
}

}  // namespace

void compute_quantized_decode_rows(
    hls::stream<qdr_output_word_t>& output,
    hls::stream<mm_stream_quantized_task_word_t>& tasks,
    hls::stream<qdr_activation_word_t>& activations,
    hls::stream<qdr_weight_word_t>& weights,
    unsigned int task_count) {
    #pragma HLS interface axis port=output
    #pragma HLS interface axis port=tasks
    #pragma HLS interface axis port=activations
    #pragma HLS interface axis port=weights
    #pragma HLS interface s_axilite port=task_count bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
    #pragma HLS inline off

#if QDR_DIRECT_WIDE && QDR_INGRESS_BITS == 4096
    // One wide input word already is one complete tile in both P and D.
    // The merged adapter owns stripe assembly and input buffering, so a
    // second tile feeder only copies the same word through another FIFO.
    // Execute keeps the exact task/K/packet loops and accumulator lifetime.
    // Narrow standalone ingress still needs the existing assembly path.
    qdr_compact_execute(output, tasks, activations, weights, task_count);
#else
    hls::stream<mm_stream_quantized_task_word_t> task_fifo;
    hls::stream<qdr_activation_word_t> activation_fifo;
    hls::stream<qdr_weight_tile_t> weight_fifo;
    #pragma HLS stream variable=task_fifo depth=QDR_COMPACT_FIFO_DEPTH
    #pragma HLS stream variable=activation_fifo depth=QDR_COMPACT_FIFO_DEPTH
    #pragma HLS stream variable=weight_fifo depth=QDR_COMPACT_FIFO_DEPTH
    #pragma HLS dataflow
    qdr_compact_feed(tasks, activations, weights, task_fifo, activation_fifo,
                     weight_fifo, task_count);
    qdr_compact_execute(output, task_fifo, activation_fifo, weight_fifo,
                        task_count);
#endif
}

#else

void compute_quantized_decode_rows(
    hls::stream<qdr_output_word_t>& output,
    hls::stream<mm_stream_quantized_task_word_t>& tasks,
    hls::stream<qdr_activation_word_t>& activations,
    hls::stream<qdr_weight_word_t>& weights,
    unsigned int task_count) {
    #pragma HLS interface axis port=output
    #pragma HLS interface axis port=tasks
    #pragma HLS interface axis port=activations
    #pragma HLS interface axis port=weights
    #pragma HLS interface s_axilite port=task_count bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
    #pragma HLS inline off

    // Flat output-major storage in Decode; the same registers represent the
    // existing token/output tile in Prefill. There is no second MAC array.
    ap_int<QDR_ACCUM_BITS> accumulator[QDR_DECODE_COLS];
    #pragma HLS array_partition variable=accumulator complete
    for (unsigned int task_index = 0; task_index < task_count; ++task_index) {
        #pragma HLS loop_tripcount min=1 max=64
        const auto packed_task = tasks.read();
        const auto task = unpack_mm_stream_quantized_task(packed_task);
        const bool decode = packed_task[QDR_MODE_BIT];
        const unsigned int valid_rows = task.valid_tokens == 0 ||
                task.valid_tokens > QDR_ROWS ? QDR_ROWS : task.valid_tokens;
        if (task.k_count == 0) {
            for (unsigned int column = 0; column < QDR_DECODE_COLS; ++column) {
                #pragma HLS unroll
                accumulator[column] = 0;
            }
        }
        MAC_K:
        for (unsigned int k = 0; k < task.k_count; ++k) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=1 max=11008 avg=2048
            const qdr_activation_word_t activation_word = activations.read();
            ap_uint<QDR_WEIGHT_BITS> weight_word = 0;
#if QDR_INGRESS_BITS == 4096
            weight_word = weights.read();
#else
            const unsigned int beats = decode ? QDR_WEIGHT_BITS / QDR_INGRESS_BITS :
                (QDR_PREFILL_COLS * QDR_BITS + QDR_INGRESS_BITS - 1) / QDR_INGRESS_BITS;
            for (unsigned int beat = 0; beat < beats; ++beat) {
                #pragma HLS pipeline II=1
                weight_word.range((beat + 1) * QDR_INGRESS_BITS - 1, beat * QDR_INGRESS_BITS) = weights.read();
            }
#endif
#ifdef QUANTIZED_ALIGNMENT_W8
            for (unsigned int dsp = 0; dsp < QDR_DSPS; ++dsp) {
                #pragma HLS unroll
                const unsigned int row = dsp / QDR_PREFILL_COLS;
                const unsigned int column = dsp % QDR_PREFILL_COLS;
                const ap_int<8> activation = decode ?
                    ap_int<8>(activation_word.range(7, 0)) :
                    ap_int<8>(activation_word.range(row * 8 + 7, row * 8));
                const ap_int<8> weight = decode ?
                    ap_int<8>(weight_word.range(dsp * 8 + 7, dsp * 8)) :
                    ap_int<8>(weight_word.range(column * 8 + 7, column * 8));
                ap_int<16> product;
                #pragma HLS bind_op variable=product op=mul impl=dsp
                product = activation * weight;
                accumulator[dsp] = k == 0 ? ap_int<QDR_ACCUM_BITS>(product) :
                    ap_int<QDR_ACCUM_BITS>(accumulator[dsp] + product);
            }
#else
            for (unsigned int dsp = 0; dsp < QDR_DSPS; ++dsp) {
                #pragma HLS unroll
                const unsigned int row_pair = dsp / (QDR_PREFILL_COLS / 2);
                const unsigned int column_pair = dsp % (QDR_PREFILL_COLS / 2);
                const ap_int<4> a0 = decode ?
                    ap_int<4>(activation_word.range(3, 0)) :
                    ap_int<4>(activation_word.range(row_pair * 8 + 3, row_pair * 8));
                const ap_int<4> a1 =
                    ap_int<4>(activation_word.range(row_pair * 8 + 7, row_pair * 8 + 4));
                const ap_int<4> w0 = decode ?
                    ap_int<4>(weight_word.range(dsp * 16 + 3, dsp * 16)) :
                    ap_int<4>(weight_word.range(column_pair * 8 + 3, column_pair * 8));
                const ap_int<4> w1 = decode ?
                    ap_int<4>(weight_word.range(dsp * 16 + 7, dsp * 16 + 4)) :
                    ap_int<4>(weight_word.range(column_pair * 8 + 7, column_pair * 8 + 4));
                const ap_int<4> w2 =
                    ap_int<4>(weight_word.range(dsp * 16 + 11, dsp * 16 + 8));
                const ap_int<4> w3 =
                    ap_int<4>(weight_word.range(dsp * 16 + 15, dsp * 16 + 12));
                const qdr_w4_quad_t product = qdr_w4_products(decode, a0, a1, w0, w1, w2, w3);
                for (unsigned int p = 0; p < 4; ++p) {
                    #pragma HLS unroll
                    accumulator[dsp * 4 + p] = k == 0 ?
                        ap_int<QDR_ACCUM_BITS>(product.value[p]) :
                        ap_int<QDR_ACCUM_BITS>(accumulator[dsp * 4 + p] + product.value[p]);
                }
            }
#endif
        }

        // The packet count is unchanged by mode: 64 (W4) or 32 (W8).
        EMIT_PACKETS:
        for (unsigned int packet = 0; packet < QDR_DECODE_COLS / 16; ++packet) {
            #pragma HLS pipeline II=1
            const unsigned int row = decode ? 0 : packet / 8;
            const unsigned int column_base = decode ? packet * 16 : (packet % 8) * 16;
            const bool valid = decode || row < valid_rows;
            qdr_output_word_t word = 0;
            for (unsigned int lane = 0; lane < 16; ++lane) {
                #pragma HLS unroll
                const unsigned int column = column_base + lane;
#ifdef QUANTIZED_ALIGNMENT_W8
                const unsigned int index = decode ? column : row * QDR_PREFILL_COLS + column;
#else
                const unsigned int index = decode ? column :
                    ((row / 2) * (QDR_PREFILL_COLS / 2) + column / 2) * 4 +
                    (row % 2) * 2 + column % 2;
#endif
                const ap_int<QDR_ACCUM_BITS> value = valid ? accumulator[index] : ap_int<QDR_ACCUM_BITS>(0);
                word.range((lane + 1) * QDR_ACCUM_BITS - 1, lane * QDR_ACCUM_BITS) = value;
            }
            constexpr unsigned int meta = QDR_ACCUM_BITS * 16;
            word.range(meta + 15, meta) = valid ? 0xffff : 0;
            word.range(meta + 23, meta + 16) = row;
            word.range(meta + 39, meta + 24) = task.elem_base + column_base;
            word.range(meta + 55, meta + 40) = task.block_id;
            const bool last = packet + 1 == QDR_DECODE_COLS / 16;
            word[meta + 56] = last;
            word[meta + 57] = last && task.last_stream;
            output.write(word);
        }
    }
}

#endif  // QDR_COMPACT
