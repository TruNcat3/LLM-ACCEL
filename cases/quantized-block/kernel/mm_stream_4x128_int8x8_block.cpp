#include "mm_stream_4x128_int8x8_block.hpp"

struct mm_stream_4x128_int8x8_accum_bank_t {
    ap_int<MM_STREAM_4X128_INT8X8_ACCUM_BITS>
        value[MM_STREAM_4X128_INT8X8_TOKENS]
             [MM_STREAM_4X128_INT8X8_OUTPUTS];
};

struct mm_stream_4x128_int8x8_product_tile_t {
    ap_int<16> value[MM_STREAM_4X128_INT8X8_TOKENS]
                      [MM_STREAM_4X128_INT8X8_OUTPUTS];
};

static ap_int<8> unpack_int8x8_weight(
    const mm_stream_4x128_int8x8_weight_word_t& weight0,
    const mm_stream_4x128_int8x8_weight_word_t& weight1,
    const mm_stream_4x128_int8x8_weight_word_t& weight2,
    const mm_stream_4x128_int8x8_weight_word_t& weight3,
    unsigned int output) {
    #pragma HLS inline
    const unsigned int local =
        output & (MM_STREAM_4X128_INT8X8_WEIGHTS_PER_STREAM - 1);
    const unsigned int low = local * 8;
    if (output < 32) {
        return ap_int<8>(weight0.range(low + 7, low));
    }
    if (output < 64) {
        return ap_int<8>(weight1.range(low + 7, low));
    }
    if (output < 96) {
        return ap_int<8>(weight2.range(low + 7, low));
    }
    return ap_int<8>(weight3.range(low + 7, low));
}

static ap_int<16> multiply_int8x8(ap_int<8> activation, ap_int<8> weight) {
    #pragma HLS inline
    ap_int<16> product = 0;
    #pragma HLS bind_op variable=product op=mul impl=dsp
    product = activation * weight;
    return product;
}

static void compute_int8x8_products(
    mm_stream_4x128_int8x8_product_tile_t& product,
    const mm_stream_4x128_int8x8_activation_word_t& activation_word,
    const mm_stream_4x128_int8x8_weight_word_t& weight0,
    const mm_stream_4x128_int8x8_weight_word_t& weight1,
    const mm_stream_4x128_int8x8_weight_word_t& weight2,
    const mm_stream_4x128_int8x8_weight_word_t& weight3) {
    #pragma HLS inline
    #pragma HLS array_partition variable=product.value complete dim=0

    for (unsigned int token = 0;
         token < MM_STREAM_4X128_INT8X8_TOKENS; token++) {
        #pragma HLS unroll
        const ap_int<8> activation =
            unpack_mm_stream_4x128_int8x8_activation(
                activation_word, token);
        for (unsigned int output = 0;
             output < MM_STREAM_4X128_INT8X8_OUTPUTS; output++) {
            #pragma HLS unroll
            const ap_int<8> weight = unpack_int8x8_weight(
                weight0, weight1, weight2, weight3, output);
            product.value[token][output] = multiply_int8x8(activation, weight);
        }
    }
}

static void update_int8x8_bank(
    mm_stream_4x128_int8x8_accum_bank_t& bank,
    const mm_stream_4x128_int8x8_product_tile_t& product,
    bool initialize) {
    #pragma HLS inline
    #pragma HLS array_partition variable=bank.value complete dim=0
    #pragma HLS array_partition variable=product.value complete dim=0
    for (unsigned int token = 0;
         token < MM_STREAM_4X128_INT8X8_TOKENS; token++) {
        #pragma HLS unroll
        for (unsigned int output = 0;
             output < MM_STREAM_4X128_INT8X8_OUTPUTS; output++) {
            #pragma HLS unroll
            if (initialize) {
                bank.value[token][output] = product.value[token][output];
            } else {
                bank.value[token][output] += product.value[token][output];
            }
        }
    }
}

void compute_mm_stream_4x128_int8x8_block_nk(
    hls::stream<mm_stream_4x128_int8x8_output_word_t>& out_stream,
    hls::stream<mm_stream_quantized_task_word_t>& task_stream,
    hls::stream<mm_stream_4x128_int8x8_activation_word_t>& activation_stream,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream0,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream1,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream2,
    hls::stream<mm_stream_4x128_int8x8_weight_word_t>& weight_stream3) {
    #pragma HLS interface axis port=out_stream
    #pragma HLS interface axis port=task_stream
    #pragma HLS interface axis port=activation_stream
    #pragma HLS interface axis port=weight_stream0
    #pragma HLS interface axis port=weight_stream1
    #pragma HLS interface axis port=weight_stream2
    #pragma HLS interface axis port=weight_stream3
    #pragma HLS interface ap_ctrl_hs port=return
    #pragma HLS inline off

    const mm_stream_quantized_task_t task =
        unpack_mm_stream_quantized_task(task_stream.read());
    mm_stream_4x128_int8x8_accum_bank_t bank0;
    mm_stream_4x128_int8x8_accum_bank_t bank1;
    mm_stream_4x128_int8x8_accum_bank_t bank2;
    mm_stream_4x128_int8x8_accum_bank_t bank3;
    #pragma HLS array_partition variable=bank0.value complete dim=0
    #pragma HLS array_partition variable=bank1.value complete dim=0
    #pragma HLS array_partition variable=bank2.value complete dim=0
    #pragma HLS array_partition variable=bank3.value complete dim=0

    for (unsigned int k = 0; k < task.k_count; k++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=16 max=4096 avg=2048
        const mm_stream_4x128_int8x8_activation_word_t activation_word =
            activation_stream.read();
        const mm_stream_4x128_int8x8_weight_word_t weight0 =
            weight_stream0.read();
        const mm_stream_4x128_int8x8_weight_word_t weight1 =
            weight_stream1.read();
        const mm_stream_4x128_int8x8_weight_word_t weight2 =
            weight_stream2.read();
        const mm_stream_4x128_int8x8_weight_word_t weight3 =
            weight_stream3.read();
        const bool initialize = k < 4;
        mm_stream_4x128_int8x8_product_tile_t product;
        #pragma HLS array_partition variable=product.value complete dim=0
        compute_int8x8_products(
            product, activation_word, weight0, weight1, weight2, weight3);
        switch (k & 3) {
        case 0:
            update_int8x8_bank(bank0, product, initialize);
            break;
        case 1:
            update_int8x8_bank(bank1, product, initialize);
            break;
        case 2:
            update_int8x8_bank(bank2, product, initialize);
            break;
        default:
            update_int8x8_bank(bank3, product, initialize);
            break;
        }
    }

    for (unsigned int packet = 0;
         packet < MM_STREAM_4X128_INT8X8_TOKENS *
                      MM_STREAM_4X128_INT8X8_OUTPUT_GROUPS;
         packet++) {
        #pragma HLS pipeline II=1
        const unsigned int token =
            packet / MM_STREAM_4X128_INT8X8_OUTPUT_GROUPS;
        const unsigned int group =
            packet % MM_STREAM_4X128_INT8X8_OUTPUT_GROUPS;
        mm_stream_4x128_int8x8_output_word_t output = 0;
        for (unsigned int lane = 0;
             lane < MM_STREAM_4X128_INT8X8_LANES_PER_GROUP; lane++) {
            #pragma HLS unroll
            const unsigned int out =
                group * MM_STREAM_4X128_INT8X8_LANES_PER_GROUP + lane;
            const ap_int<MM_STREAM_4X128_INT8X8_ACCUM_BITS> value =
                bank0.value[token][out] + bank1.value[token][out] +
                bank2.value[token][out] + bank3.value[token][out];
            const unsigned int low =
                lane * MM_STREAM_4X128_INT8X8_ACCUM_BITS;
            output.range(
                low + MM_STREAM_4X128_INT8X8_ACCUM_BITS - 1, low) = value;
        }
        const bool last_block =
            token == MM_STREAM_4X128_INT8X8_TOKENS - 1 &&
            group == MM_STREAM_4X128_INT8X8_OUTPUT_GROUPS - 1;
        output.range(527, 512) = 0xffff;
        output.range(535, 528) = token;
        output.range(551, 536) =
            task.elem_base +
            group * MM_STREAM_4X128_INT8X8_LANES_PER_GROUP;
        output.range(567, 552) = task.block_id;
        output[568] = last_block;
        output[569] = task.last_stream && last_block;
        out_stream.write(output);
    }
}
