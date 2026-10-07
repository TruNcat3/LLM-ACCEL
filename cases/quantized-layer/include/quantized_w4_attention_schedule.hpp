#ifndef LLM_FPGA_QUANTIZED_W4_ATTENTION_SCHEDULE_HPP
#define LLM_FPGA_QUANTIZED_W4_ATTENTION_SCHEDULE_HPP

#include "mm_stream_8x128_int4x4_block.hpp"
#include "quantized_layer_schedule.hpp"
#include "quantized_vector_schedule.hpp"
#include "quantized_attention_mapping.hpp"

constexpr unsigned int QUANTIZED_W4_ATTENTION_POSITION_TILE =
    MM_STREAM_8X128_INT4X4_OUTPUTS;
constexpr unsigned int QUANTIZED_W4_ATTENTION_HEADS_PER_CU =
    quantized_ceildiv(NUM_ATTENTION_HEADS, QUANTIZED_LAYER_COMPUTE_CUS);
constexpr unsigned int QUANTIZED_W4_ATTENTION_TASKS_PER_TILE_PER_CU =
    2 * QUANTIZED_W4_ATTENTION_HEADS_PER_CU;
inline unsigned int quantized_w4_attention_local_task_base() {
    return quantized_projection_waves_per_block() *
        QUANTIZED_LAYER_COMPUTE_CUS;
}

struct quantized_w4_attention_dispatch_t {
    mm_stream_quantized_compute_mode_t mode;
    unsigned int tile;
    unsigned int tile_begin;
    unsigned int tile_length;
    unsigned int query_head;
    unsigned int kv_head;
    unsigned int cu;
    unsigned int local_task_id;
    bool active;
    bool last_attention_task;
};

inline unsigned int quantized_w4_attention_tile_count(
    unsigned int context_length) {
    return quantized_ceildiv(
        context_length, QUANTIZED_W4_ATTENTION_POSITION_TILE);
}

inline unsigned int quantized_w4_attention_context_length(
    const quantized_layer_task_t& layer_task) {
    // kv_context_length excludes the rows produced by the current task.
    return layer_task.kv_context_length + layer_task.query_tokens;
}

inline quantized_w4_attention_dispatch_t
make_quantized_w4_attention_dispatch(
    const quantized_layer_task_t& layer_task,
    mm_stream_quantized_compute_mode_t mode,
    unsigned int tile,
    unsigned int query_head) {
    quantized_w4_attention_dispatch_t dispatch{};
    dispatch.mode = mode;
    dispatch.tile = tile;
    dispatch.tile_begin = tile * QUANTIZED_W4_ATTENTION_POSITION_TILE;
    const unsigned int context_length =
        quantized_w4_attention_context_length(layer_task);
    dispatch.tile_length = dispatch.tile_begin < context_length ?
        context_length - dispatch.tile_begin : 0;
    if (dispatch.tile_length > QUANTIZED_W4_ATTENTION_POSITION_TILE) {
        dispatch.tile_length = QUANTIZED_W4_ATTENTION_POSITION_TILE;
    }
    dispatch.query_head = query_head;
    dispatch.kv_head = query_head / GQA_GROUP_SIZE;
    dispatch.cu = query_head % QUANTIZED_LAYER_COMPUTE_CUS;
    const unsigned int head_slot =
        query_head / QUANTIZED_LAYER_COMPUTE_CUS;
    const unsigned int stage =
        mode == MM_STREAM_QUANTIZED_MODE_ATTENTION_PV ? 1 : 0;
    dispatch.local_task_id = quantized_w4_attention_local_task_base() +
        tile * (2 * NUM_ATTENTION_HEADS) +
        stage * NUM_ATTENTION_HEADS + query_head;
    dispatch.active =
        (mode == MM_STREAM_QUANTIZED_MODE_ATTENTION_QK ||
         mode == MM_STREAM_QUANTIZED_MODE_ATTENTION_PV) &&
        dispatch.tile_length != 0 && query_head < NUM_ATTENTION_HEADS;
    dispatch.last_attention_task = dispatch.active &&
        tile + 1 == quantized_w4_attention_tile_count(context_length) &&
        mode == MM_STREAM_QUANTIZED_MODE_ATTENTION_PV &&
        query_head + 1 == NUM_ATTENTION_HEADS;
    (void)head_slot;
    return dispatch;
}

inline mm_stream_quantized_task_t make_quantized_w4_attention_task(
    const quantized_layer_task_t& layer_task,
    const quantized_w4_attention_dispatch_t& dispatch,
    quant_scale_t activation_scale,
    quant_scale_t weight_scale) {
    mm_stream_quantized_task_t task{};
    task.k_count = dispatch.mode == MM_STREAM_QUANTIZED_MODE_ATTENTION_QK ?
        HEAD_DIM : dispatch.tile_length;
    task.elem_base = dispatch.mode == MM_STREAM_QUANTIZED_MODE_ATTENTION_QK ?
        dispatch.tile_begin : dispatch.query_head * HEAD_DIM;
    task.block_id = dispatch.local_task_id;
    task.last_stream = false;
    task.activation_scale = activation_scale;
    task.weight_scale = weight_scale;
    task.request_position = layer_task.position;
    task.valid_tokens = layer_task.query_tokens;
    task.phase = static_cast<unsigned int>(layer_task.phase);
    task.kv_context_length = layer_task.kv_context_length;
    task.projection = 0;
    task.compute_mode = dispatch.mode;
    return task;
}

inline unsigned int quantized_w4_attention_tasks_for_block_per_cu(
    const quantized_layer_task_t& layer_task,
    unsigned int cu) {
    return quantized_attention_tasks_for_cu<8>(layer_task, cu,
        QUANTIZED_W4_ATTENTION_POSITION_TILE);
}

inline unsigned int quantized_w4_layer_tasks_for_cu(
    unsigned int sequence_length,
    unsigned int request_position,
    unsigned int cu,
    bool decode,
    unsigned int block_size = QUANTIZED_LAYER_W4_TOKEN_BLOCK) {
    if (cu >= QUANTIZED_LAYER_COMPUTE_CUS || sequence_length == 0) return 0;
    if (block_size == 0 ||
        block_size > MM_STREAM_8X128_INT4X4_TOKENS) {
        block_size = QUANTIZED_LAYER_W4_TOKEN_BLOCK;
    }
#if QUANTIZED_LAYER_INTEGRATED_DECODE
    if (decode) {
        const quantized_layer_task_t layer_task = make_quantized_decode_task(
            0, request_position, request_position, 0, 1,
            QUANTIZED_ACTIVATION_INT4, QUANTIZED_WEIGHT_INT4, 0, 0,
            block_size);
        unsigned int tasks =
            quantized_layer_integrated_decode_projection_tasks_for_cu(cu);
        tasks += quantized_vector_layer_tasks_for_cu(
            MM_STREAM_8X128_INT4X4_TOKENS, 1, cu);
#if QUANTIZED_DECODE_FFN_OVERLAP
        tasks += quantized_vector_task_rows(
            MM_STREAM_8X128_INT4X4_TOKENS, 1, cu,
            MM_STREAM_QUANTIZED_MODE_MUL, INTERMEDIATE_SIZE) != 0;
#endif
        return tasks + quantized_w4_attention_tasks_for_block_per_cu(
            layer_task, cu);
    }
#endif
    const unsigned int blocks = decode ? 1 : quantized_layer_block_count(
        sequence_length, block_size);
    unsigned int tasks = quantized_projection_tasks_for_cu(cu, blocks);
    for (unsigned int block = 0; block < blocks; ++block) {
        const quantized_layer_task_t layer_task = decode ?
            make_quantized_decode_task(
                0, request_position, request_position, 0, 1,
                QUANTIZED_ACTIVATION_INT4, QUANTIZED_WEIGHT_INT4, 0, 0,
                block_size) :
            make_quantized_prefill_block_task(
                0, request_position, sequence_length, block, 0, 1,
                QUANTIZED_ACTIVATION_INT4, QUANTIZED_WEIGHT_INT4, 0, 0,
                MAX_SEQ_LEN, block_size);
        tasks += quantized_w4_attention_tasks_for_block_per_cu(
            layer_task, cu);
        // Whole-row RMSNorm and striped single-row elementwise operations
        // use different active-CU masks.
        tasks += quantized_vector_layer_tasks_for_cu(8, layer_task.query_tokens, cu);
#if QUANTIZED_PREFILL_FFN_OVERLAP
        // Prefill splits the fused SiLU/multiply into two ordered commands.
        // A one-row tail uses all four column-striped vector workers.
        if (!decode)
            tasks += quantized_vector_task_rows(8, layer_task.query_tokens, cu,
                MM_STREAM_QUANTIZED_MODE_SILU_ONLY, INTERMEDIATE_SIZE) != 0;
#endif
    }
    return tasks;
}

#endif
