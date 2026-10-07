#include "quantized_layer_task.hpp"

#include <cassert>
#include <iostream>

namespace {

void check_prefill_shape(
    unsigned int sequence_length,
    unsigned int physical_block_size,
    unsigned int expected_blocks,
    unsigned int expected_tail) {
    const quantized_activation_format_t activations[] = {
        QUANTIZED_ACTIVATION_FP16,
        QUANTIZED_ACTIVATION_INT4,
        QUANTIZED_ACTIVATION_INT8
    };
    const quantized_weight_format_t weights[] = {
        QUANTIZED_WEIGHT_INT4,
        QUANTIZED_WEIGHT_INT8
    };

    for (quantized_activation_format_t activation : activations) {
        for (quantized_weight_format_t weight : weights) {
            const unsigned int blocks =
                quantized_layer_block_count(
                    sequence_length, physical_block_size);
            assert(blocks == expected_blocks);
            for (unsigned int block = 0; block < blocks; block++) {
                const quantized_layer_task_t task =
                    make_quantized_prefill_block_task(
                        0, 0, sequence_length, block, block & 1,
                        (block + 1) & 1, activation, weight, 7, 11,
                        MAX_SEQ_LEN, physical_block_size);
                assert(task.query_tokens ==
                       (block + 1 == blocks ? expected_tail :
                                                physical_block_size));
                assert(task.position == block * physical_block_size);
                assert(quantized_layer_source_token_begin(task) ==
                       block * physical_block_size);
                assert(task.kv_context_length == task.position);
                assert(task.physical_block_size == physical_block_size);
                assert(task.last_block == (block + 1 == blocks));
                assert(task.last_task == task.last_block);
                assert(quantized_layer_task_shape_valid(task));

                const quantized_layer_task_t round_trip =
                    unpack_quantized_layer_task(pack_quantized_layer_task(task));
                assert(round_trip.op == task.op);
                assert(round_trip.phase == task.phase);
                assert(round_trip.layer == task.layer);
                assert(round_trip.position == task.position);
                assert(round_trip.query_tokens == task.query_tokens);
                assert(round_trip.sequence_length == task.sequence_length);
                assert(round_trip.kv_context_length == task.kv_context_length);
                assert(round_trip.block_index == task.block_index);
                assert(round_trip.block_count == task.block_count);
                assert(round_trip.input_pair == task.input_pair);
                assert(round_trip.output_pair == task.output_pair);
                assert(round_trip.activation_format == task.activation_format);
                assert(round_trip.weight_format == task.weight_format);
                assert(round_trip.activation_scale_id == 7);
                assert(round_trip.weight_scale_id == 11);
                assert(round_trip.physical_block_size == physical_block_size);
                assert(round_trip.last_block == task.last_block);
                assert(round_trip.last_task == task.last_task);
            }
        }
    }
}

}  // namespace

int main() {
    check_prefill_shape(66, QUANTIZED_LAYER_W4_TOKEN_BLOCK, 9, 2);
    check_prefill_shape(66, QUANTIZED_LAYER_W8_TOKEN_BLOCK, 17, 2);
    check_prefill_shape(2048, QUANTIZED_LAYER_W4_TOKEN_BLOCK, 256, 8);
    check_prefill_shape(2048, QUANTIZED_LAYER_W8_TOKEN_BLOCK, 512, 4);

    for (unsigned int context : {0u, 8u, 64u, 128u, 2047u}) {
        const quantized_layer_task_t task = make_quantized_decode_task(
            0, context, context, 0, 1, QUANTIZED_ACTIVATION_FP16,
            QUANTIZED_WEIGHT_INT4, 0, 0,
            QUANTIZED_LAYER_W4_TOKEN_BLOCK);
        assert(quantized_layer_task_shape_valid(task));
        assert(task.query_tokens == 1);
        assert(task.block_count == 1);
    }

    quantized_layer_task_t invalid = make_quantized_decode_task(
        0, 0, 0, 0, 1, QUANTIZED_ACTIVATION_INT8,
        QUANTIZED_WEIGHT_INT8);
    invalid.query_tokens = 2;
    assert(!quantized_layer_task_shape_valid(invalid));

    quantized_layer_task_t wrong_backend = make_quantized_prefill_block_task(
        0, 0, 66, 16, 0, 1, QUANTIZED_ACTIVATION_INT8,
        QUANTIZED_WEIGHT_INT8, 0, 0, MAX_SEQ_LEN,
        QUANTIZED_LAYER_W8_TOKEN_BLOCK);
    wrong_backend.physical_block_size = QUANTIZED_LAYER_W4_TOKEN_BLOCK;
    assert(!quantized_layer_task_shape_valid(wrong_backend));

    quantized_layer_task_word_t legacy_word = 0;
    legacy_word.range(3, 0) = QUANTIZED_LAYER_OP_DECODE;
    legacy_word.range(5, 4) = QUANTIZED_LAYER_PHASE_DECODE;
    legacy_word.range(35, 32) = 1;
    legacy_word.range(51, 36) = 1;
    legacy_word.range(99, 84) = 1;
    const quantized_layer_task_t legacy =
        unpack_quantized_layer_task(legacy_word);
    assert(legacy.physical_block_size ==
           QUANTIZED_LAYER_DEFAULT_TOKEN_BLOCK);

    std::cout << "QUANTIZED LAYER TASK PASS "
              << "w4_p66_blocks=9 w8_p66_blocks=17 "
              << "w4_p2048_blocks=256 w8_p2048_blocks=512 "
              << "decode_contexts=5"
              << std::endl;
    return 0;
}
