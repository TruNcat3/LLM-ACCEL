#ifndef QUANTIZED_LAYER_PROTOCOL_FIXTURE_HPP
#define QUANTIZED_LAYER_PROTOCOL_FIXTURE_HPP

#include "quantized_vector_engine.hpp"
#include "quantized_attention_mapping.hpp"

// Controller protocol unit fixture only. Outputs are deliberately seeded;
// this is not numerical or closed-loop evidence. Real feedback is exercised
// by the projection/vector fixtures and by integrated HW Emu.
template <typename Policy>
void preload_quantized_layer_protocol_outputs(
    hls::stream<typename Policy::output_word_t> outputs[4],
    const quantized_layer_task_t& task, bool residual_row_plus_one = false) {
    auto matrix = [&](unsigned int cu, unsigned int base) {
        constexpr unsigned int payload = Policy::output_word_t::width == 576 ? 512 : 384;
        for (unsigned int row = 0; row < Policy::token_rows; ++row) {
            for (unsigned int block = 0; block < 8; ++block) {
                typename Policy::output_word_t word = 0;
                word.range(payload + 15, payload) = 0xffff;
                word.range(payload + 23, payload + 16) = row;
                word.range(payload + 39, payload + 24) = base + block * 16;
                outputs[cu].write(word);
            }
        }
    };
    auto projection = [&](quantized_projection_t kind) {
        const auto plan = get_quantized_projection_plan(kind);
        for (unsigned int wave = 0; wave < plan.wave_count; ++wave) {
            for (unsigned int cu = 0; cu < 4; ++cu) {
                if (quantized_projection_active_cu_mask(plan, wave) & (1u << cu))
                    matrix(cu, wave * QUANTIZED_LAYER_OUTPUTS_PER_WAVE + cu * 128);
            }
        }
    };
    auto vector = [&](unsigned int elements,
                      mm_stream_quantized_compute_mode_t mode,
                      bool residual) {
        const unsigned int blocks = ceildiv(elements, CU_VEC_LANES);
        const bool split_columns = quantized_vector_split_columns(
            task.query_tokens, mode);
        for (unsigned int cu = 0; cu < 4; ++cu) {
            const unsigned int rows = quantized_vector_task_rows(
                Policy::token_rows, task.query_tokens, cu, mode, elements);
            const unsigned int packets_per_row = split_columns ?
                quantized_vector_blocks_for_cu(elements, cu) : blocks;
            for (unsigned int slot = 0; slot < rows; ++slot) {
                const unsigned int row = split_columns ? 0 : slot * 4 + cu;
                for (unsigned int local_block = 0;
                     local_block < packets_per_row; ++local_block) {
                    const unsigned int block = split_columns ?
                        cu + local_block * 4 : local_block;
                    cu_vec16_packet_t packet{};
                    packet.token_lane = row;
                    packet.elem_base = block * CU_VEC_LANES;
                    packet.block_id = block;
                    packet.last_block = split_columns ?
                        block + 4 >= blocks :
                        packet.elem_base + CU_VEC_LANES >= elements;
                    packet.last_stream = packet.last_block;
                    for (unsigned int lane = 0; lane < CU_VEC_LANES; ++lane) {
                        packet.valid_mask[lane] =
                            packet.elem_base + lane < elements;
                        packet.data[lane] = residual && residual_row_plus_one ?
                            fm_t(row + 1) : fm_t(0);
                    }
                    outputs[cu].write(typename Policy::output_word_t(
                        pack_cu8_nk_vector(packet)));
                }
            }
        }
    };
    vector(HIDDEN_SIZE, MM_STREAM_QUANTIZED_MODE_RMSNORM, false);
    projection(QUANTIZED_PROJECTION_Q);
    projection(QUANTIZED_PROJECTION_K);
    projection(QUANTIZED_PROJECTION_V);
    const unsigned int tiles = ceildiv(task.kv_context_length + task.query_tokens, 128u);
    for (unsigned int tile = 0; tile < tiles; ++tile) {
        for (unsigned int wave = 0; wave < quantized_attention_waves<Policy::token_rows>(task); ++wave) {
            for (unsigned int cu = 0; cu < 4; ++cu)
                if (quantized_attention_unit<Policy::token_rows>(task, wave, cu).active)
                    matrix(cu, tile * 128);
            for (unsigned int cu = 0; cu < 4; ++cu)
                if (quantized_attention_unit<Policy::token_rows>(task, wave, cu).active)
                    matrix(cu, 0);
        }
    }
    projection(QUANTIZED_PROJECTION_O);
    vector(HIDDEN_SIZE, MM_STREAM_QUANTIZED_MODE_RESIDUAL_ADD, true);
    vector(HIDDEN_SIZE, MM_STREAM_QUANTIZED_MODE_RMSNORM, false);
    projection(QUANTIZED_PROJECTION_GATE);
#if QUANTIZED_PREFILL_FFN_OVERLAP
    if (task.op == QUANTIZED_LAYER_OP_PREFILL)
        vector(INTERMEDIATE_SIZE, MM_STREAM_QUANTIZED_MODE_SILU_ONLY, false);
#endif
    projection(QUANTIZED_PROJECTION_UP);
#if QUANTIZED_PREFILL_FFN_OVERLAP
    if (task.op == QUANTIZED_LAYER_OP_PREFILL)
        vector(INTERMEDIATE_SIZE, MM_STREAM_QUANTIZED_MODE_MUL, false);
    else
#endif
    vector(INTERMEDIATE_SIZE, MM_STREAM_QUANTIZED_MODE_SILU_MUL, false);
    projection(QUANTIZED_PROJECTION_DOWN);
    vector(HIDDEN_SIZE, MM_STREAM_QUANTIZED_MODE_RESIDUAL_ADD, true);
}

#endif
