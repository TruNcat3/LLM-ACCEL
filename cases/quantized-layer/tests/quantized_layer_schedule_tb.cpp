#include "quantized_layer_schedule.hpp"

#include <cassert>
#include <iostream>

int main() {
    const unsigned int expected_in[QUANTIZED_PROJECTION_COUNT] = {
        2048, 2048, 2048, 2048, 2048, 2048, 11008
    };
    const unsigned int expected_out[QUANTIZED_PROJECTION_COUNT] = {
        2048, 256, 256, 2048, 11008, 11008, 2048
    };
    const unsigned int expected_waves[QUANTIZED_PROJECTION_COUNT] = {
        4, 1, 1, 4, 22, 22, 4
    };
    const unsigned int expected_last_cus[QUANTIZED_PROJECTION_COUNT] = {
        4, 2, 2, 4, 2, 2, 4
    };
    const unsigned int expected_wave_prefix[QUANTIZED_PROJECTION_COUNT] = {
        0, 4, 5, 6, 10, 32, 54
    };
    const std::size_t expected_shard_word_offset[
        QUANTIZED_PROJECTION_COUNT] = {
        0, 8192, 10240, 12288, 20480, 65536, 110592
    };

    std::size_t expected_offset = 0;
    for (unsigned int index = 0; index < QUANTIZED_PROJECTION_COUNT; ++index) {
        const quantized_projection_plan_t plan =
            get_quantized_projection_plan(
                static_cast<quantized_projection_t>(index));
        assert(plan.input_dim == expected_in[index]);
        assert(plan.output_dim == expected_out[index]);
        assert(plan.wave_count == expected_waves[index]);
        assert(plan.active_cus_in_last_wave == expected_last_cus[index]);
        assert(plan.logical_weight_offset == expected_offset);
        assert(plan.shard_word_offset == expected_shard_word_offset[index]);
        assert(quantized_projection_wave_prefix(plan.projection) ==
               expected_wave_prefix[index]);
        expected_offset += std::size_t(plan.input_dim) * plan.output_dim;
    }
    assert(quantized_layer_weight_words_per_shard() == 154624);
    assert(quantized_projection_waves_per_block() == 58);
    assert(quantized_projection_tasks_per_block_for_cu(0) == 58);
    assert(quantized_projection_tasks_per_block_for_cu(1) == 58);
    assert(quantized_projection_tasks_per_block_for_cu(2) == 54);
    assert(quantized_projection_tasks_per_block_for_cu(3) == 54);
    assert(quantized_projection_tasks_for_cu(0, 512) == 29696);
    assert(quantized_projection_tasks_for_cu(2, 512) == 27648);
    assert(quantized_projection_tasks_per_block_for_cu(4) == 0);
    assert(quantized_w8_weight_shard_words(NUM_LAYERS) == 5566464);
    assert(quantized_w8_weight_shard_bytes(NUM_LAYERS) == 178126848);
    assert(quantized_w8_weight_shard_bytes(NUM_LAYERS) < (256ULL << 20));

    const quantized_w8_weight_location_t q0 =
        get_quantized_w8_weight_location(0, QUANTIZED_PROJECTION_Q, 0, 7);
    assert(q0.valid && q0.wave == 0 && q0.cu == 0 && q0.stream == 0 &&
           q0.lane == 0 && q0.shard_word_offset == 7);
    const quantized_w8_weight_location_t q127 =
        get_quantized_w8_weight_location(0, QUANTIZED_PROJECTION_Q, 127, 9);
    assert(q127.valid && q127.cu == 0 && q127.stream == 3 &&
           q127.lane == 31 && q127.shard_word_offset == 9);
    const quantized_w8_weight_location_t q512 =
        get_quantized_w8_weight_location(0, QUANTIZED_PROJECTION_Q, 512, 11);
    assert(q512.valid && q512.wave == 1 && q512.cu == 0 &&
           q512.stream == 0 && q512.lane == 0 &&
           q512.shard_word_offset == 2048 + 11);
    const quantized_w8_weight_location_t key255 =
        get_quantized_w8_weight_location(0, QUANTIZED_PROJECTION_K, 255, 13);
    assert(key255.valid && key255.wave == 0 && key255.cu == 1 &&
           key255.stream == 3 && key255.lane == 31 &&
           key255.shard_word_offset == 8192 + 13);
    const quantized_w8_weight_location_t gate_last =
        get_quantized_w8_weight_location(
            0, QUANTIZED_PROJECTION_GATE, 11007, 17);
    assert(gate_last.valid && gate_last.wave == 21 && gate_last.cu == 1 &&
           gate_last.stream == 3 && gate_last.lane == 31 &&
           gate_last.shard_word_offset == 63488 + 17);
    const quantized_w8_weight_location_t down_last =
        get_quantized_w8_weight_location(
            0, QUANTIZED_PROJECTION_DOWN, 2047, 11007);
    assert(down_last.valid && down_last.wave == 3 && down_last.cu == 3 &&
           down_last.stream == 3 && down_last.lane == 31 &&
           down_last.shard_word_offset == 154623);
    const quantized_w8_weight_location_t layer35_down_last =
        get_quantized_w8_weight_location(
            35, QUANTIZED_PROJECTION_DOWN, 2047, 11007);
    assert(layer35_down_last.valid &&
           layer35_down_last.shard_word_offset == 5566463);
    assert(!get_quantized_w8_weight_location(
                36, QUANTIZED_PROJECTION_Q, 0, 0).valid);
    assert(!get_quantized_w8_weight_location(
                0, QUANTIZED_PROJECTION_K, 256, 0).valid);

    unsigned int enumerated_slots = 0;
    unsigned int enumerated_useful_tasks = 0;
    unsigned int enumerated_valid_outputs = 0;
    for (unsigned int slot = 0;
         slot < quantized_projection_waves_per_block(); ++slot) {
        quantized_projection_t projection = QUANTIZED_PROJECTION_Q;
        unsigned int wave = 0;
        assert(decode_quantized_projection_slot(slot, projection, wave));
        assert(slot == quantized_projection_wave_prefix(projection) + wave);
        for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
            const quantized_projection_dispatch_t dispatch =
                make_quantized_projection_dispatch_slot(
                    slot, cu, false, 7);
            assert(dispatch.local_task_id == slot * 4 + cu);
            assert(dispatch.shard_weight_word_offset >=
                   7 * quantized_layer_weight_words_per_shard());
            enumerated_useful_tasks += dispatch.active ? 1 : 0;
            enumerated_valid_outputs += dispatch.valid_outputs;
        }
        ++enumerated_slots;
    }
    assert(enumerated_slots == 58);
    assert(enumerated_useful_tasks == 224);
    assert(enumerated_valid_outputs == 28672);
    quantized_projection_t invalid_projection = QUANTIZED_PROJECTION_DOWN;
    unsigned int invalid_wave = 99;
    assert(!decode_quantized_projection_slot(
        58, invalid_projection, invalid_wave));

    const quantized_projection_plan_t key =
        get_quantized_projection_plan(QUANTIZED_PROJECTION_K);
    const quantized_projection_plan_t gate =
        get_quantized_projection_plan(QUANTIZED_PROJECTION_GATE);
    assert(quantized_projection_active_cu_mask(key, 0) == 0x3);
    assert(quantized_projection_active_cu_mask(gate, 20) == 0xf);
    assert(quantized_projection_active_cu_mask(gate, 21) == 0x3);

    const quantized_projection_dispatch_t gate_last_cu1 =
        make_quantized_projection_dispatch(
            QUANTIZED_PROJECTION_GATE, 21, 1, false);
    assert(gate_last_cu1.local_task_id == 125);
    assert(gate_last_cu1.elem_base == 10880);
    assert(gate_last_cu1.valid_outputs == 128);
    assert(gate_last_cu1.valid_output_groups == 8);
    assert(gate_last_cu1.shard_weight_word_offset == 63488);
    assert(gate_last_cu1.active);
    const quantized_projection_dispatch_t gate_last_cu2 =
        make_quantized_projection_dispatch(
            QUANTIZED_PROJECTION_GATE, 21, 2, false);
    assert(!gate_last_cu2.active);
    assert(gate_last_cu2.valid_outputs == 0);
    assert(gate_last_cu2.valid_output_groups == 0);

    const quantized_layer_task_t tail = make_quantized_prefill_block_task(
        0, 0, 66, 16, 0, 1, QUANTIZED_ACTIVATION_INT8,
        QUANTIZED_WEIGHT_INT8, 3, 4, MAX_SEQ_LEN,
        QUANTIZED_LAYER_W8_TOKEN_BLOCK);
    const quantized_projection_dispatch_t final_dispatch =
        make_quantized_projection_dispatch(
            QUANTIZED_PROJECTION_DOWN, 3, 3, true);
    const mm_stream_quantized_task_t final_task =
        make_quantized_projection_task(tail, final_dispatch,
                                       quant_scale_t(0.5),
                                       quant_scale_t(0.25));
    assert(final_task.k_count == 11008);
    assert(final_task.elem_base == 1920);
    assert(final_task.block_id == 231);
    assert(final_task.last_stream);
    assert(final_task.request_position == 64);
    assert(final_task.valid_tokens == 2);
    assert(final_task.kv_context_length == 64);
    assert(final_task.projection == QUANTIZED_PROJECTION_DOWN);
    for (unsigned int cu = 0; cu < QUANTIZED_LAYER_COMPUTE_CUS; ++cu) {
        assert(quantized_projection_is_last_task_for_cu(
            QUANTIZED_PROJECTION_DOWN, 3, cu, true));
    }
    const mm_stream_quantized_task_t roundtrip =
        unpack_mm_stream_quantized_task(
            pack_mm_stream_quantized_task(final_task));
    assert(roundtrip.k_count == final_task.k_count);
    assert(roundtrip.elem_base == final_task.elem_base);
    assert(roundtrip.block_id == final_task.block_id);
    assert(roundtrip.last_stream == final_task.last_stream);
    assert(roundtrip.request_position == final_task.request_position);
    assert(roundtrip.valid_tokens == final_task.valid_tokens);
    assert(roundtrip.kv_context_length == final_task.kv_context_length);
    assert(roundtrip.projection == final_task.projection);

    const quantized_layer_plan_t prefill = make_quantized_layer_plan(
        66, 66, QUANTIZED_LAYER_W8_TOKEN_BLOCK);
    assert(quantized_layer_plan_valid(prefill));
    assert(prefill.block_count == 17);
    assert(prefill.projection_waves_per_block == 58);
    assert(prefill.issued_projection_tasks_per_block == 232);
    assert(prefill.useful_projection_tasks_per_block == 224);
    assert(prefill.useful_projection_macs == 5086642176ULL);
    assert(prefill.issued_projection_macs == 5383389184ULL);

    const quantized_layer_plan_t full_prefill = make_quantized_layer_plan(
        2048, 2048, QUANTIZED_LAYER_W8_TOKEN_BLOCK);
    assert(quantized_layer_plan_valid(full_prefill));
    assert(full_prefill.block_count == 512);
    assert(full_prefill.useful_projection_macs == 157840048128ULL);

    const quantized_layer_plan_t decode = make_quantized_layer_plan(
        1, 2048, QUANTIZED_LAYER_W8_TOKEN_BLOCK);
    assert(quantized_layer_plan_valid(decode));
    assert(decode.block_count == 1);
    assert(decode.useful_projection_macs == 77070336ULL);
    assert(decode.issued_projection_macs == 316669952ULL);

    std::cout << "QUANTIZED LAYER SCHEDULE PASS "
              << "waves_per_block=" << prefill.projection_waves_per_block
              << " useful_tasks_per_block="
              << prefill.useful_projection_tasks_per_block
              << " issued_tasks_per_block="
              << prefill.issued_projection_tasks_per_block
              << " weight_words_per_shard="
              << quantized_layer_weight_words_per_shard()
              << " full_model_shard_bytes="
              << quantized_w8_weight_shard_bytes(NUM_LAYERS)
              << " p66_blocks=" << prefill.block_count
              << " p2048_blocks=" << full_prefill.block_count << '\n';
    return 0;
}
