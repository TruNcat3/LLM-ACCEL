#include "mm_stream_8x64_int4x4_block.hpp"

#ifdef MM_STREAM_QUANTIZED_NARROW_ACCUM
static constexpr unsigned int MM_STREAM_8X64_INT4X4_INTERNAL_ACCUM_BITS = 20;
#else
static constexpr unsigned int MM_STREAM_8X64_INT4X4_INTERNAL_ACCUM_BITS =
    MM_STREAM_8X64_INT4X4_ACCUM_BITS;
#endif

struct mm_stream_8x64_int4x4_accum_bank_t {
    ap_int<MM_STREAM_8X64_INT4X4_INTERNAL_ACCUM_BITS>
        value[MM_STREAM_8X64_INT4X4_TOKENS]
             [MM_STREAM_8X64_INT4X4_OUTPUTS];
};

struct mm_stream_8x64_int4x4_product_tile_t {
    ap_int<8> value[MM_STREAM_8X64_INT4X4_TOKENS]
                    [MM_STREAM_8X64_INT4X4_OUTPUTS];
};

static ap_uint<4> abs_int4x4(ap_int<4> value) {
    #pragma HLS inline
    const int widened = value.to_int();
    return ap_uint<4>(widened < 0 ? -widened : widened);
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

static void update_int4x4_accum(
    ap_int<MM_STREAM_8X64_INT4X4_INTERNAL_ACCUM_BITS>& accum,
    ap_int<8> product,
    bool initialize) {
    #pragma HLS inline
#ifdef MM_STREAM_QUANTIZED_USE_DSP_ACCUM
    // Trade LUT-heavy wide accumulators for otherwise idle DSP adders.
    #pragma HLS bind_op variable=accum op=add impl=dsp
#endif
    if (initialize) {
        accum = product;
    } else {
        accum += product;
    }
}

static void compute_int4x4_products(
    mm_stream_8x64_int4x4_product_tile_t& product,
    const mm_stream_8x64_int4x4_activation_word_t& activation_word,
    const mm_stream_8x64_int4x4_weight_word_t& weight_word) {
    #pragma HLS inline
    #pragma HLS array_partition variable=product.value complete dim=0

    for (unsigned int token_pair = 0;
         token_pair < MM_STREAM_8X64_INT4X4_TOKEN_PAIRS; token_pair++) {
        #pragma HLS unroll
        const unsigned int token0 = 2 * token_pair;
        const unsigned int token1 = token0 + 1;
        const ap_int<4> activation0 =
            unpack_mm_stream_8x64_int4x4_activation(
                activation_word, token0);
        const ap_int<4> activation1 =
            unpack_mm_stream_8x64_int4x4_activation(
                activation_word, token1);
        for (unsigned int output_pair = 0;
             output_pair < MM_STREAM_8X64_INT4X4_OUTPUT_PAIRS;
             output_pair++) {
            #pragma HLS unroll
            const unsigned int output0 = 2 * output_pair;
            const unsigned int output1 = output0 + 1;
            const ap_int<4> weight0 =
                unpack_mm_stream_8x64_int4x4_weight(weight_word, output0);
            const ap_int<4> weight1 =
                unpack_mm_stream_8x64_int4x4_weight(weight_word, output1);
            ap_int<8> product00;
            ap_int<8> product01;
            ap_int<8> product10;
            ap_int<8> product11;
            packed_int4x4_outer_product(
                activation0, activation1, weight0, weight1,
                product00, product01, product10, product11);
            product.value[token0][output0] = product00;
            product.value[token0][output1] = product01;
            product.value[token1][output0] = product10;
            product.value[token1][output1] = product11;
        }
    }
}

static void update_int4x4_bank(
    mm_stream_8x64_int4x4_accum_bank_t& bank,
    const mm_stream_8x64_int4x4_product_tile_t& product,
    bool initialize) {
    #pragma HLS inline
    #pragma HLS array_partition variable=bank.value complete dim=0
    #pragma HLS array_partition variable=product.value complete dim=0
    for (unsigned int token = 0;
         token < MM_STREAM_8X64_INT4X4_TOKENS; token++) {
        #pragma HLS unroll
        for (unsigned int output = 0;
             output < MM_STREAM_8X64_INT4X4_OUTPUTS; output++) {
            #pragma HLS unroll
            update_int4x4_accum(
                bank.value[token][output],
                product.value[token][output], initialize);
        }
    }
}

void compute_mm_stream_8x64_int4x4_block_nk(
    hls::stream<mm_stream_8x64_int4x4_output_word_t>& out_stream,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream,
    hls::stream<mm_stream_8x64_int4x4_activation_word_t>& activation_stream,
    hls::stream<mm_stream_8x64_int4x4_weight_word_t>& weight_stream) {
    #pragma HLS interface axis port=out_stream
    #pragma HLS interface axis port=task_stream
    #pragma HLS interface axis port=activation_stream
    #pragma HLS interface axis port=weight_stream
    #pragma HLS interface ap_ctrl_hs port=return
    #pragma HLS inline off

    const mm_stream_quantized_task_t task =
        unpack_mm_stream_quantized_task(task_stream.read());
    mm_stream_8x64_int4x4_accum_bank_t bank0;
    mm_stream_8x64_int4x4_accum_bank_t bank1;
    mm_stream_8x64_int4x4_accum_bank_t bank2;
    mm_stream_8x64_int4x4_accum_bank_t bank3;
    #pragma HLS array_partition variable=bank0.value complete dim=0
    #pragma HLS array_partition variable=bank1.value complete dim=0
    #pragma HLS array_partition variable=bank2.value complete dim=0
    #pragma HLS array_partition variable=bank3.value complete dim=0

    for (unsigned int k = 0; k < task.k_count; k++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=16 max=4096 avg=2048
        const mm_stream_8x64_int4x4_activation_word_t activation_word =
            activation_stream.read();
        const mm_stream_8x64_int4x4_weight_word_t weight_word =
            weight_stream.read();
        const bool initialize = k < 4;
        mm_stream_8x64_int4x4_product_tile_t product;
        #pragma HLS array_partition variable=product.value complete dim=0
        compute_int4x4_products(product, activation_word, weight_word);
        switch (k & 3) {
        case 0:
            update_int4x4_bank(bank0, product, initialize);
            break;
        case 1:
            update_int4x4_bank(bank1, product, initialize);
            break;
        case 2:
            update_int4x4_bank(bank2, product, initialize);
            break;
        default:
            update_int4x4_bank(bank3, product, initialize);
            break;
        }
    }

    for (unsigned int packet = 0;
         packet < MM_STREAM_8X64_INT4X4_TOKENS *
                      MM_STREAM_8X64_INT4X4_GROUPS;
         packet++) {
        #pragma HLS pipeline II=1
        const unsigned int token = packet / MM_STREAM_8X64_INT4X4_GROUPS;
        const unsigned int group = packet % MM_STREAM_8X64_INT4X4_GROUPS;
        mm_stream_8x64_int4x4_output_word_t output = 0;
        for (unsigned int lane = 0;
             lane < MM_STREAM_8X64_INT4X4_LANES_PER_GROUP; lane++) {
            #pragma HLS unroll
            const unsigned int out =
                group * MM_STREAM_8X64_INT4X4_LANES_PER_GROUP + lane;
            const ap_int<MM_STREAM_8X64_INT4X4_ACCUM_BITS> value =
                bank0.value[token][out] + bank1.value[token][out] +
                bank2.value[token][out] + bank3.value[token][out];
            const unsigned int low =
                lane * MM_STREAM_8X64_INT4X4_ACCUM_BITS;
            output.range(
                low + MM_STREAM_8X64_INT4X4_ACCUM_BITS - 1, low) = value;
        }
        const bool last_block =
            token == MM_STREAM_8X64_INT4X4_TOKENS - 1 &&
            group == MM_STREAM_8X64_INT4X4_GROUPS - 1;
        output.range(399, 384) = 0xffff;
        output.range(407, 400) = token;
        output.range(423, 408) =
            task.elem_base +
            group * MM_STREAM_8X64_INT4X4_LANES_PER_GROUP;
        output.range(439, 424) = task.block_id;
        output[440] = last_block;
        output[441] = task.last_stream && last_block;
        out_stream.write(output);
    }
}
