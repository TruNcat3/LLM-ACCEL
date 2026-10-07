#include "mm_stream_8x128_int4x4_block.hpp"

#ifdef MM_STREAM_QUANTIZED_NARROW_ACCUM
static constexpr unsigned int MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS = 20;
#else
static constexpr unsigned int MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS =
    MM_STREAM_8X128_INT4X4_ACCUM_BITS;
#endif

struct mm_stream_8x128_int4x4_accum_bank_t {
#if MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE_CONFIG == 64
    // Both 64-output wave paths emit one fixed half at a time.  Output-major
    // storage leaves only the token index dynamic in that packet loop.  The
    // two-wave path keeps both halves in one continuous tile so HLS does not
    // have to bind two formal references to the same disaggregated array.
#if MM_STREAM_8X128_INT4X4_REUSE_BANK_CONFIG
    ap_int<MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS>
        value[MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE]
             [MM_STREAM_8X128_INT4X4_TOKENS];
#else
    ap_int<MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS>
        value[MM_STREAM_8X128_INT4X4_OUTPUTS]
             [MM_STREAM_8X128_INT4X4_TOKENS];
#endif
#else
    // The full-width path uses the same output-major layout as the reused
    // 64-output wave path.  The packet ABI remains token-major, but keeping
    // storage output-major lets the emitter use a compile-time lane slice.
    ap_int<MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS>
        value[MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE]
             [MM_STREAM_8X128_INT4X4_TOKENS];
#endif
};

using mm_stream_8x128_int4x4_internal_accum_t =
    ap_int<MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS>;
using mm_stream_8x128_int4x4_combined_weight_word_t =
    ap_uint<MM_STREAM_8X128_INT4X4_WEIGHTS_PER_STREAM * 8>;

#define MM_STREAM_8X128_INT4X4_ACCUM_AT(bank, token, output) \
    bank.value[output][token]

#if MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE_CONFIG == 64 && \
    !MM_STREAM_8X128_INT4X4_REUSE_BANK_CONFIG
#define MM_STREAM_8X128_INT4X4_ACCUM_INDEX(output_base, local_output) \
    ((output_base) + (local_output))
#else
#define MM_STREAM_8X128_INT4X4_ACCUM_INDEX(output_base, local_output) \
    (local_output)
#endif

static void update_accum(
    mm_stream_8x128_int4x4_internal_accum_t& accum,
    ap_int<8> product,
    bool initialize) {
    #pragma HLS inline
#ifdef MM_STREAM_QUANTIZED_USE_DSP_ACCUM
    #pragma HLS bind_op variable=accum op=add impl=dsp
#endif
    if (initialize) {
        accum = product;
    } else {
        accum += product;
    }
}

static ap_uint<4> abs_int4x4(ap_int<4> value) {
    #pragma HLS inline
    const int widened = value.to_int();
    return ap_uint<4>(widened < 0 ? -widened : widened);
}

static ap_int<4> unpack_activation(
    const mm_stream_8x128_int4x4_activation_word_t& word,
    unsigned int token) {
    #pragma HLS inline
    return ap_int<4>(word.range(token * 4 + 3, token * 4));
}

static ap_int<4> unpack_weight(
    const mm_stream_8x128_int4x4_combined_weight_word_t& word,
    unsigned int output) {
    #pragma HLS inline
    const unsigned int low = output * 4;
    return ap_int<4>(word.range(low + 3, low));
}

static void packed_int4x4_outer_product(
    ap_int<4> activation0,
    ap_int<4> activation1,
    ap_int<4> weight0,
    ap_int<4> weight1,
    ap_int<8>& product00,
    ap_int<8>& product01,
    ap_int<8>& product10,
    ap_int<8>& product11) {
    #pragma HLS inline

    ap_uint<20> packed_activations = 0;
    packed_activations.range(3, 0) = abs_int4x4(activation0);
    packed_activations.range(19, 16) = abs_int4x4(activation1);

    ap_uint<12> packed_weights = 0;
    packed_weights.range(3, 0) = abs_int4x4(weight0);
    packed_weights.range(11, 8) = abs_int4x4(weight1);

    ap_uint<32> packed_product = 0;
    #pragma HLS bind_op variable=packed_product op=mul impl=dsp
    packed_product = packed_activations * packed_weights;

    ap_int<8> magnitude00 = ap_int<8>(packed_product.range(7, 0));
    ap_int<8> magnitude01 = ap_int<8>(packed_product.range(15, 8));
    ap_int<8> magnitude10 = ap_int<8>(packed_product.range(23, 16));
    ap_int<8> magnitude11 = ap_int<8>(packed_product.range(31, 24));
    if ((activation0 < 0) != (weight0 < 0)) magnitude00 = -magnitude00;
    if ((activation0 < 0) != (weight1 < 0)) magnitude01 = -magnitude01;
    if ((activation1 < 0) != (weight0 < 0)) magnitude10 = -magnitude10;
    if ((activation1 < 0) != (weight1 < 0)) magnitude11 = -magnitude11;
    product00 = magnitude00;
    product01 = magnitude01;
    product10 = magnitude10;
    product11 = magnitude11;
}

static void compute_and_update_bank(
    mm_stream_8x128_int4x4_accum_bank_t& bank,
    const mm_stream_8x128_int4x4_activation_word_t& activation_word,
    const mm_stream_8x128_int4x4_combined_weight_word_t& weight_word,
    unsigned int output_base,
    bool initialize) {
    #pragma HLS inline
    #pragma HLS array_partition variable=bank.value complete dim=0

    for (unsigned int token_pair = 0;
         token_pair < MM_STREAM_8X128_INT4X4_TOKEN_PAIRS; token_pair++) {
        #pragma HLS unroll
        const unsigned int token0 = 2 * token_pair;
        const unsigned int token1 = token0 + 1;
        const ap_int<4> activation0 =
            unpack_activation(activation_word, token0);
        const ap_int<4> activation1 =
            unpack_activation(activation_word, token1);
        for (unsigned int output_pair = 0;
             output_pair < MM_STREAM_8X128_INT4X4_OUTPUT_PAIRS;
             output_pair++) {
            #pragma HLS unroll
            const unsigned int local_output0 = 2 * output_pair;
            const unsigned int local_output1 = local_output0 + 1;
            const ap_int<4> packed_weight0 =
                unpack_weight(weight_word, local_output0);
            const ap_int<4> packed_weight1 =
                unpack_weight(weight_word, local_output1);
            ap_int<8> product00;
            ap_int<8> product01;
            ap_int<8> product10;
            ap_int<8> product11;
            packed_int4x4_outer_product(
                activation0, activation1, packed_weight0, packed_weight1,
                product00, product01, product10, product11);
            if (initialize) {
                MM_STREAM_8X128_INT4X4_ACCUM_AT(
                    bank, token0,
                    MM_STREAM_8X128_INT4X4_ACCUM_INDEX(
                        output_base, local_output0)) = product00;
                MM_STREAM_8X128_INT4X4_ACCUM_AT(
                    bank, token0,
                    MM_STREAM_8X128_INT4X4_ACCUM_INDEX(
                        output_base, local_output1)) = product01;
                MM_STREAM_8X128_INT4X4_ACCUM_AT(
                    bank, token1,
                    MM_STREAM_8X128_INT4X4_ACCUM_INDEX(
                        output_base, local_output0)) = product10;
                MM_STREAM_8X128_INT4X4_ACCUM_AT(
                    bank, token1,
                    MM_STREAM_8X128_INT4X4_ACCUM_INDEX(
                        output_base, local_output1)) = product11;
            } else {
                update_accum(
                    MM_STREAM_8X128_INT4X4_ACCUM_AT(
                        bank, token0,
                        MM_STREAM_8X128_INT4X4_ACCUM_INDEX(
                            output_base, local_output0)), product00,
                    false);
                update_accum(
                    MM_STREAM_8X128_INT4X4_ACCUM_AT(
                        bank, token0,
                        MM_STREAM_8X128_INT4X4_ACCUM_INDEX(
                            output_base, local_output1)), product01,
                    false);
                update_accum(
                    MM_STREAM_8X128_INT4X4_ACCUM_AT(
                        bank, token1,
                        MM_STREAM_8X128_INT4X4_ACCUM_INDEX(
                            output_base, local_output0)), product10,
                    false);
                update_accum(
                    MM_STREAM_8X128_INT4X4_ACCUM_AT(
                        bank, token1,
                    MM_STREAM_8X128_INT4X4_ACCUM_INDEX(
                        output_base, local_output1)), product11,
                    false);
            }
        }
    }
}

#if MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE_CONFIG != 64
#if MM_STREAM_8X128_INT4X4_PACKED_ACCUM_CONFIG
using mm_stream_8x128_int4x4_packed_accum_word_t = ap_uint<
    MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS *
    MM_STREAM_8X128_INT4X4_TOKENS>;

struct mm_stream_8x128_int4x4_half_accum_bank_t {
    mm_stream_8x128_int4x4_packed_accum_word_t
        value[MM_STREAM_8X128_INT4X4_OUTPUTS / 2];
};
#else
struct mm_stream_8x128_int4x4_half_accum_bank_t {
    ap_int<MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS>
        value[MM_STREAM_8X128_INT4X4_OUTPUTS / 2]
             [MM_STREAM_8X128_INT4X4_TOKENS];
};
#endif

using mm_stream_8x128_int4x4_half_weight_word_t =
    ap_uint<(MM_STREAM_8X128_INT4X4_OUTPUTS / 2) * 4>;

static ap_uint<4> int4_magnitude(ap_uint<4> bits) {
    #pragma HLS inline
    return bits[3] ? ap_uint<4>((~bits) + 1) : bits;
}

static void predecode_activation_pairs(
    const mm_stream_8x128_int4x4_activation_word_t& activation_word,
    ap_uint<20> packed_magnitudes[MM_STREAM_8X128_INT4X4_TOKEN_PAIRS],
    ap_uint<2> signs[MM_STREAM_8X128_INT4X4_TOKEN_PAIRS]) {
    #pragma HLS inline
    #pragma HLS array_partition variable=packed_magnitudes complete dim=1
    #pragma HLS array_partition variable=signs complete dim=1

    for (unsigned int token_pair = 0;
         token_pair < MM_STREAM_8X128_INT4X4_TOKEN_PAIRS; token_pair++) {
        #pragma HLS unroll
        const unsigned int token0 = 2 * token_pair;
        const unsigned int token1 = token0 + 1;
        const ap_uint<4> activation0 = activation_word.range(
            token0 * 4 + 3, token0 * 4);
        const ap_uint<4> activation1 = activation_word.range(
            token1 * 4 + 3, token1 * 4);
        ap_uint<20> packed = 0;
        packed.range(3, 0) = int4_magnitude(activation0);
        packed.range(19, 16) = int4_magnitude(activation1);
        packed_magnitudes[token_pair] = packed;
        signs[token_pair][0] = activation0[3];
        signs[token_pair][1] = activation1[3];
    }
}

static void predecode_weight_pairs(
    const mm_stream_8x128_int4x4_half_weight_word_t& weight_word,
    ap_uint<12> packed_magnitudes[MM_STREAM_8X128_INT4X4_OUTPUTS / 4],
    ap_uint<2> signs[MM_STREAM_8X128_INT4X4_OUTPUTS / 4]) {
    #pragma HLS inline
    #pragma HLS array_partition variable=packed_magnitudes complete dim=1
    #pragma HLS array_partition variable=signs complete dim=1

    constexpr unsigned int half_output_pairs =
        (MM_STREAM_8X128_INT4X4_OUTPUTS / 2) / 2;
    for (unsigned int output_pair = 0;
         output_pair < half_output_pairs; output_pair++) {
        #pragma HLS unroll
        const unsigned int output0 = 2 * output_pair;
        const unsigned int output1 = output0 + 1;
        const unsigned int low0 = output0 * 4;
        const unsigned int low1 = output1 * 4;
        const ap_uint<4> weight0 =
            weight_word.range(low0 + 3, low0);
        const ap_uint<4> weight1 =
            weight_word.range(low1 + 3, low1);
        ap_uint<12> packed = 0;
        packed.range(3, 0) = int4_magnitude(weight0);
        packed.range(11, 8) = int4_magnitude(weight1);
        packed_magnitudes[output_pair] = packed;
        signs[output_pair][0] = weight0[3];
        signs[output_pair][1] = weight1[3];
    }
}

static void packed_int4x4_outer_product_predecoded(
    ap_uint<20> packed_activations,
    ap_uint<2> activation_signs,
    ap_uint<12> packed_weights,
    ap_uint<2> weight_signs,
    ap_int<8>& product00,
    ap_int<8>& product01,
    ap_int<8>& product10,
    ap_int<8>& product11) {
    #pragma HLS inline

    ap_uint<32> packed_product = 0;
    #pragma HLS bind_op variable=packed_product op=mul impl=dsp
    packed_product = packed_activations * packed_weights;

    ap_int<8> magnitude00 = ap_int<8>(packed_product.range(7, 0));
    ap_int<8> magnitude01 = ap_int<8>(packed_product.range(15, 8));
    ap_int<8> magnitude10 = ap_int<8>(packed_product.range(23, 16));
    ap_int<8> magnitude11 = ap_int<8>(packed_product.range(31, 24));
    if (activation_signs[0] ^ weight_signs[0]) magnitude00 = -magnitude00;
    if (activation_signs[0] ^ weight_signs[1]) magnitude01 = -magnitude01;
    if (activation_signs[1] ^ weight_signs[0]) magnitude10 = -magnitude10;
    if (activation_signs[1] ^ weight_signs[1]) magnitude11 = -magnitude11;
    product00 = magnitude00;
    product01 = magnitude01;
    product10 = magnitude10;
    product11 = magnitude11;
}

static void compute_and_update_half_bank(
    mm_stream_8x128_int4x4_half_accum_bank_t& bank,
    const ap_uint<20>
        packed_activations[MM_STREAM_8X128_INT4X4_TOKEN_PAIRS],
    const ap_uint<2>
        activation_signs[MM_STREAM_8X128_INT4X4_TOKEN_PAIRS],
    const ap_uint<12>
        packed_weights[MM_STREAM_8X128_INT4X4_OUTPUTS / 4],
    const ap_uint<2>
        weight_signs[MM_STREAM_8X128_INT4X4_OUTPUTS / 4],
    bool initialize) {
    #pragma HLS inline
    #pragma HLS array_partition variable=bank.value complete dim=0
    #pragma HLS array_partition variable=packed_activations complete dim=1
    #pragma HLS array_partition variable=activation_signs complete dim=1
    #pragma HLS array_partition variable=packed_weights complete dim=1
    #pragma HLS array_partition variable=weight_signs complete dim=1

    constexpr unsigned int half_output_pairs =
        (MM_STREAM_8X128_INT4X4_OUTPUTS / 2) / 2;
#if MM_STREAM_8X128_INT4X4_PACKED_ACCUM_CONFIG
    #pragma HLS array_partition variable=bank.value complete dim=1
    for (unsigned int output_pair = 0;
         output_pair < half_output_pairs; output_pair++) {
        #pragma HLS unroll
        const unsigned int output0 = 2 * output_pair;
        const unsigned int output1 = output0 + 1;
        mm_stream_8x128_int4x4_packed_accum_word_t accum0 =
            bank.value[output0];
        mm_stream_8x128_int4x4_packed_accum_word_t accum1 =
            bank.value[output1];
        for (unsigned int token_pair = 0;
             token_pair < MM_STREAM_8X128_INT4X4_TOKEN_PAIRS; token_pair++) {
            #pragma HLS unroll
            const unsigned int token0 = 2 * token_pair;
            const unsigned int token1 = token0 + 1;
            ap_int<8> product00;
            ap_int<8> product01;
            ap_int<8> product10;
            ap_int<8> product11;
            packed_int4x4_outer_product_predecoded(
                packed_activations[token_pair],
                activation_signs[token_pair],
                packed_weights[output_pair], weight_signs[output_pair],
                product00, product01, product10, product11);
            const unsigned int low0 =
                token0 * MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS;
            const unsigned int low1 =
                token1 * MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS;
            if (initialize) {
                accum0.range(
                    low0 + MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS - 1,
                    low0) = product00;
                accum1.range(
                    low0 + MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS - 1,
                    low0) = product01;
                accum0.range(
                    low1 + MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS - 1,
                    low1) = product10;
                accum1.range(
                    low1 + MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS - 1,
                    low1) = product11;
            } else {
                ap_int<MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS> value00 =
                    accum0.range(
                        low0 + MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS - 1,
                        low0);
                ap_int<MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS> value01 =
                    accum1.range(
                        low0 + MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS - 1,
                        low0);
                ap_int<MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS> value10 =
                    accum0.range(
                        low1 + MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS - 1,
                        low1);
                ap_int<MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS> value11 =
                    accum1.range(
                        low1 + MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS - 1,
                        low1);
                value00 += product00;
                value01 += product01;
                value10 += product10;
                value11 += product11;
                accum0.range(
                    low0 + MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS - 1,
                    low0) = value00;
                accum1.range(
                    low0 + MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS - 1,
                    low0) = value01;
                accum0.range(
                    low1 + MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS - 1,
                    low1) = value10;
                accum1.range(
                    low1 + MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS - 1,
                    low1) = value11;
            }
        }
        bank.value[output0] = accum0;
        bank.value[output1] = accum1;
    }
#else
    for (unsigned int output_pair = 0;
         output_pair < half_output_pairs; output_pair++) {
        #pragma HLS unroll
        const unsigned int output0 = 2 * output_pair;
        const unsigned int output1 = output0 + 1;
        for (unsigned int token_pair = 0;
             token_pair < MM_STREAM_8X128_INT4X4_TOKEN_PAIRS; token_pair++) {
            #pragma HLS unroll
            const unsigned int token0 = 2 * token_pair;
            const unsigned int token1 = token0 + 1;
            ap_int<8> product00;
            ap_int<8> product01;
            ap_int<8> product10;
            ap_int<8> product11;
            packed_int4x4_outer_product_predecoded(
                packed_activations[token_pair],
                activation_signs[token_pair],
                packed_weights[output_pair], weight_signs[output_pair],
                product00, product01, product10, product11);
            if (initialize) {
                bank.value[output0][token0] = product00;
                bank.value[output1][token0] = product01;
                bank.value[output0][token1] = product10;
                bank.value[output1][token1] = product11;
            } else {
                update_accum(bank.value[output0][token0], product00, false);
                update_accum(bank.value[output1][token0], product01, false);
                update_accum(bank.value[output0][token1], product10, false);
                update_accum(bank.value[output1][token1], product11, false);
            }
        }
    }
#endif
}
#endif

#if MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE_CONFIG == 64
#if MM_STREAM_8X128_INT4X4_REUSE_BANK_CONFIG
static void emit_output_wave(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream,
    const mm_stream_8x128_int4x4_accum_bank_t& bank,
    const mm_stream_quantized_task_t& task,
    unsigned int wave) {
    #pragma HLS inline off
    for (unsigned int packet = 0;
         packet < MM_STREAM_8X128_INT4X4_TOKENS * 4;
         packet++) {
        #pragma HLS pipeline II=1
        const unsigned int token = packet / 4;
        const unsigned int local_group = packet % 4;
        mm_stream_8x128_int4x4_output_word_t output = 0;
        for (unsigned int lane = 0;
             lane < MM_STREAM_8X128_INT4X4_LANES_PER_GROUP; lane++) {
            #pragma HLS unroll
            const unsigned int out =
                local_group * MM_STREAM_8X128_INT4X4_LANES_PER_GROUP + lane;
            const unsigned int low =
                lane * MM_STREAM_8X128_INT4X4_ACCUM_BITS;
            output.range(
                low + MM_STREAM_8X128_INT4X4_ACCUM_BITS - 1, low) =
                MM_STREAM_8X128_INT4X4_ACCUM_AT(bank, token, out);
        }
        const unsigned int group = wave * 4 + local_group;
        const bool last_block =
            token == MM_STREAM_8X128_INT4X4_TOKENS - 1 &&
            group == MM_STREAM_8X128_INT4X4_GROUPS - 1;
        output.range(399, 384) = 0xffff;
        output.range(407, 400) = token;
        output.range(423, 408) =
            task.elem_base + group * MM_STREAM_8X128_INT4X4_LANES_PER_GROUP;
        output.range(439, 424) = task.block_id;
        output[440] = last_block;
        output[441] = task.last_stream && last_block;
        out_stream.write(output);
    }
}

static void process_two_output_waves_and_emit(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream,
    const mm_stream_quantized_task_t& task,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activation_stream,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream0,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream1) {
    #pragma HLS inline off
    mm_stream_8x128_int4x4_accum_bank_t bank;
    #pragma HLS array_partition variable=bank.value complete dim=0

    // Emit each completed wave before reading the next one.  Keep this loop
    // in the caller so HLS binds one physical accumulator bank rather than
    // retaining a second copy for a by-reference helper invocation.
    for (unsigned int wave = 0; wave < 2; wave++) {
        #pragma HLS loop_tripcount min=2 max=2 avg=2
        for (unsigned int k = 0; k < task.k_count; k++) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=16 max=4096 avg=2048
            const mm_stream_8x128_int4x4_activation_word_t activation_word =
                activation_stream.read();
            const mm_stream_8x128_int4x4_weight_word_t weight0 =
                weight_stream0.read();
            const mm_stream_8x128_int4x4_weight_word_t weight1 =
                weight_stream1.read();
            mm_stream_8x128_int4x4_combined_weight_word_t weight_word = 0;
            constexpr unsigned int weight_bits =
                MM_STREAM_8X128_INT4X4_WEIGHTS_PER_STREAM * 4;
            weight_word.range(weight_bits - 1, 0) = weight0;
            weight_word.range(2 * weight_bits - 1, weight_bits) = weight1;
            compute_and_update_bank(
                bank, activation_word, weight_word, 0, k == 0);
        }
        emit_output_wave(out_stream, bank, task, wave);
    }
}
#else
static void emit_token_major_output_wave(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream,
    const mm_stream_8x128_int4x4_accum_bank_t& bank,
    const mm_stream_quantized_task_t& task,
    unsigned int token,
    unsigned int wave) {
    #pragma HLS inline off
    for (unsigned int local_group = 0; local_group < 4; local_group++) {
        #pragma HLS pipeline II=1
        mm_stream_8x128_int4x4_output_word_t output = 0;
        for (unsigned int lane = 0;
             lane < MM_STREAM_8X128_INT4X4_LANES_PER_GROUP; lane++) {
            #pragma HLS unroll
            const unsigned int out =
                local_group * MM_STREAM_8X128_INT4X4_LANES_PER_GROUP + lane;
            const unsigned int low =
                lane * MM_STREAM_8X128_INT4X4_ACCUM_BITS;
            output.range(
                low + MM_STREAM_8X128_INT4X4_ACCUM_BITS - 1, low) =
                MM_STREAM_8X128_INT4X4_ACCUM_AT(
                    bank, token,
                    wave * MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE + out);
        }
        const unsigned int group = wave * 4 + local_group;
        const bool last_block =
            token == MM_STREAM_8X128_INT4X4_TOKENS - 1 &&
            group == MM_STREAM_8X128_INT4X4_GROUPS - 1;
        output.range(399, 384) = 0xffff;
        output.range(407, 400) = token;
        output.range(423, 408) =
            task.elem_base + group * MM_STREAM_8X128_INT4X4_LANES_PER_GROUP;
        output.range(439, 424) = task.block_id;
        output[440] = last_block;
        output[441] = task.last_stream && last_block;
        out_stream.write(output);
    }
}

static void process_two_output_waves_and_emit(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream,
    const mm_stream_quantized_task_t& task,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activation_stream,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream0,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream1) {
    #pragma HLS inline off

    // Keep both output halves in one continuous tile.  This preserves the
    // token-major packet ABI without a dynamic bank select in the emitter.
    mm_stream_8x128_int4x4_accum_bank_t bank;
    #pragma HLS array_partition variable=bank.value complete dim=0

    for (unsigned int wave = 0; wave < 2; wave++) {
        #pragma HLS loop_tripcount min=2 max=2 avg=2
        for (unsigned int k = 0; k < task.k_count; k++) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=16 max=4096 avg=2048
            const mm_stream_8x128_int4x4_activation_word_t activation_word =
                activation_stream.read();
            const mm_stream_8x128_int4x4_weight_word_t weight0 =
                weight_stream0.read();
            const mm_stream_8x128_int4x4_weight_word_t weight1 =
                weight_stream1.read();
            mm_stream_8x128_int4x4_combined_weight_word_t weight_word = 0;
            constexpr unsigned int weight_bits =
                MM_STREAM_8X128_INT4X4_WEIGHTS_PER_STREAM * 4;
            weight_word.range(weight_bits - 1, 0) = weight0;
            weight_word.range(2 * weight_bits - 1, weight_bits) = weight1;
            compute_and_update_bank(
                bank, activation_word, weight_word,
                wave * MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE, k == 0);
        }
    }

    for (unsigned int token = 0;
         token < MM_STREAM_8X128_INT4X4_TOKENS; token++) {
        #pragma HLS loop_tripcount min=8 max=8 avg=8
        emit_token_major_output_wave(out_stream, bank, task, token, 0);
        emit_token_major_output_wave(out_stream, bank, task, token, 1);
    }
}
#endif
#endif

#if MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE_CONFIG != 64
static void emit_half_output_token(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream,
    const mm_stream_8x128_int4x4_half_accum_bank_t& bank,
    const mm_stream_quantized_task_t& task,
    unsigned int token,
    unsigned int wave) {
    #pragma HLS inline
    for (unsigned int local_group = 0; local_group < 4; local_group++) {
        #pragma HLS pipeline II=1
        mm_stream_8x128_int4x4_output_word_t output = 0;
        for (unsigned int lane = 0;
             lane < MM_STREAM_8X128_INT4X4_LANES_PER_GROUP; lane++) {
            #pragma HLS unroll
            const unsigned int local_output = local_group *
                MM_STREAM_8X128_INT4X4_LANES_PER_GROUP + lane;
            const unsigned int low =
                lane * MM_STREAM_8X128_INT4X4_ACCUM_BITS;
#if MM_STREAM_8X128_INT4X4_PACKED_ACCUM_CONFIG
            const unsigned int accum_low =
                token * MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS;
            const ap_int<MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS> value =
                bank.value[local_output].range(
                    accum_low +
                        MM_STREAM_8X128_INT4X4_INTERNAL_ACCUM_BITS - 1,
                    accum_low);
            output.range(low + MM_STREAM_8X128_INT4X4_ACCUM_BITS - 1, low) =
                value;
#else
            output.range(low + MM_STREAM_8X128_INT4X4_ACCUM_BITS - 1, low) =
                bank.value[local_output][token];
#endif
        }
        const unsigned int group = wave * 4 + local_group;
        const bool last_block =
            token == MM_STREAM_8X128_INT4X4_TOKENS - 1 &&
            group == MM_STREAM_8X128_INT4X4_GROUPS - 1;
        output.range(399, 384) = 0xffff;
        output.range(407, 400) = token;
        output.range(423, 408) =
            task.elem_base + group * MM_STREAM_8X128_INT4X4_LANES_PER_GROUP;
        output.range(439, 424) = task.block_id;
        output[440] = last_block;
        output[441] = task.last_stream && last_block;
        out_stream.write(output);
    }
}

static void process_full_output_wave_and_emit(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream,
    const mm_stream_quantized_task_t& task,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activation_stream,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream0,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream1) {
    #pragma HLS inline off
    mm_stream_8x128_int4x4_half_accum_bank_t bank0;
    mm_stream_8x128_int4x4_half_accum_bank_t bank1;
    #pragma HLS array_partition variable=bank0.value complete dim=0
    #pragma HLS array_partition variable=bank1.value complete dim=0

    for (unsigned int k = 0; k < task.k_count; k++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=16 max=4096 avg=2048
        const mm_stream_8x128_int4x4_activation_word_t activation_word =
            activation_stream.read();
        const mm_stream_8x128_int4x4_weight_word_t weight0 =
            weight_stream0.read();
        const mm_stream_8x128_int4x4_weight_word_t weight1 =
            weight_stream1.read();
        // Each input weight stream already owns one contiguous 64-output
        // half in the 128-output ABI, so no cross-stream repacking is needed.
        const mm_stream_8x128_int4x4_half_weight_word_t weight_word0 = weight0;
        const mm_stream_8x128_int4x4_half_weight_word_t weight_word1 = weight1;
        ap_uint<20>
            packed_activations[MM_STREAM_8X128_INT4X4_TOKEN_PAIRS];
        ap_uint<2>
            activation_signs[MM_STREAM_8X128_INT4X4_TOKEN_PAIRS];
        ap_uint<12>
            packed_weights0[MM_STREAM_8X128_INT4X4_OUTPUTS / 4];
        ap_uint<12>
            packed_weights1[MM_STREAM_8X128_INT4X4_OUTPUTS / 4];
        ap_uint<2>
            weight_signs0[MM_STREAM_8X128_INT4X4_OUTPUTS / 4];
        ap_uint<2>
            weight_signs1[MM_STREAM_8X128_INT4X4_OUTPUTS / 4];
        #pragma HLS array_partition variable=packed_activations complete dim=1
        #pragma HLS array_partition variable=activation_signs complete dim=1
        #pragma HLS array_partition variable=packed_weights0 complete dim=1
        #pragma HLS array_partition variable=packed_weights1 complete dim=1
        #pragma HLS array_partition variable=weight_signs0 complete dim=1
        #pragma HLS array_partition variable=weight_signs1 complete dim=1
        predecode_activation_pairs(
            activation_word, packed_activations, activation_signs);
        predecode_weight_pairs(weight_word0, packed_weights0, weight_signs0);
        predecode_weight_pairs(weight_word1, packed_weights1, weight_signs1);
        compute_and_update_half_bank(
            bank0, packed_activations, activation_signs,
            packed_weights0, weight_signs0, k == 0);
        compute_and_update_half_bank(
            bank1, packed_activations, activation_signs,
            packed_weights1, weight_signs1, k == 0);
    }
    for (unsigned int token = 0;
         token < MM_STREAM_8X128_INT4X4_TOKENS; token++) {
        #pragma HLS loop_tripcount min=8 max=8 avg=8
        emit_half_output_token(out_stream, bank0, task, token, 0);
        emit_half_output_token(out_stream, bank1, task, token, 1);
    }
}
#endif

void compute_mm_stream_8x128_int4x4_block_nk(
    hls::stream<mm_stream_8x128_int4x4_output_word_t>& out_stream,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream,
    hls::stream<mm_stream_8x128_int4x4_activation_word_t>& activation_stream,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream0,
    hls::stream<mm_stream_8x128_int4x4_weight_word_t>& weight_stream1,
    unsigned int task_count) {
    #pragma HLS interface axis port=out_stream
    #pragma HLS interface axis port=task_stream
    #pragma HLS interface axis port=activation_stream
    #pragma HLS interface axis port=weight_stream0
    #pragma HLS interface axis port=weight_stream1
    #pragma HLS interface s_axilite port=task_count bundle=control
    #pragma HLS interface s_axilite port=return bundle=control
    #pragma HLS inline off

    for (unsigned int task_index = 0; task_index < task_count; task_index++) {
        #pragma HLS loop_tripcount min=1 max=256 avg=32
        const mm_stream_quantized_task_t task =
            unpack_mm_stream_quantized_task(task_stream.read());

#if MM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE_CONFIG == 64
        process_two_output_waves_and_emit(
            out_stream, task, activation_stream, weight_stream0, weight_stream1);
        continue;
#else
        process_full_output_wave_and_emit(
            out_stream, task, activation_stream, weight_stream0, weight_stream1);
        continue;
#endif

    }
}
