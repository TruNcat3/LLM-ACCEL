#ifndef LLM_FPGA_QUANTIZED_ATTENTION_MAPPING_HPP
#define LLM_FPGA_QUANTIZED_ATTENTION_MAPPING_HPP

#include "quantized_layer_schedule.hpp"

// The reference uses query tokens as PE rows in block prefill and GQA heads
// as PE rows in decode. Interleave KV groups across CUs in either phase.
struct quantized_attention_unit_t {
    unsigned int kv_head;
    unsigned int head_begin;
    unsigned int head_count;
    unsigned int rows;
    bool active;
};

// Bank probability storage by an interleaved logical-head number. Adjacent
// nonlinear lanes then access distinct banks in both GQA Decode and Prefill.
inline unsigned int quantized_attention_probability_head(unsigned int head) {
    #pragma HLS inline
#if QUANTIZED_ATTENTION_HEAD_LANES > 1
    return (head % GQA_GROUP_SIZE) * NUM_KEY_VALUE_HEADS + head / GQA_GROUP_SIZE;
#else
    return head;
#endif
}

template <unsigned int Rows>
inline unsigned int quantized_attention_units(const quantized_layer_task_t& task) {
    #pragma HLS inline
    return NUM_KEY_VALUE_HEADS * quantized_ceildiv(GQA_GROUP_SIZE,
        task.phase == QUANTIZED_LAYER_PHASE_DECODE ? Rows : 1u);
}

template <unsigned int Rows>
inline unsigned int quantized_attention_waves(const quantized_layer_task_t& task) {
    #pragma HLS inline
    return quantized_ceildiv(quantized_attention_units<Rows>(task),
                            QUANTIZED_LAYER_COMPUTE_CUS);
}

template <unsigned int Rows>
inline quantized_attention_unit_t quantized_attention_unit(
    const quantized_layer_task_t& task, unsigned int wave, unsigned int cu) {
    #pragma HLS inline
    const bool decode = task.phase == QUANTIZED_LAYER_PHASE_DECODE;
    const unsigned int heads_per_unit = decode ? Rows : 1;
    const unsigned int index = wave * QUANTIZED_LAYER_COMPUTE_CUS + cu;
    quantized_attention_unit_t unit{};
    unit.kv_head = index % NUM_KEY_VALUE_HEADS;
    const unsigned int local_head = index / NUM_KEY_VALUE_HEADS * heads_per_unit;
    unit.head_begin = unit.kv_head * GQA_GROUP_SIZE + local_head;
    unit.active = cu < QUANTIZED_LAYER_COMPUTE_CUS &&
        index < quantized_attention_units<Rows>(task) && task.query_tokens != 0;
    const unsigned int remaining = local_head < GQA_GROUP_SIZE ?
        GQA_GROUP_SIZE - local_head : 0;
    unit.head_count = remaining < heads_per_unit ? remaining : heads_per_unit;
    unit.rows = unit.active ? (decode ? unit.head_count : task.query_tokens) : 0;
    return unit;
}

template <unsigned int Rows>
inline bool quantized_attention_head_in_wave(
    const quantized_layer_task_t& task, unsigned int wave, unsigned int head) {
    #pragma HLS inline
    const unsigned int heads_per_unit =
        task.phase == QUANTIZED_LAYER_PHASE_DECODE ? Rows : 1;
    const unsigned int index = head / GQA_GROUP_SIZE +
        (head % GQA_GROUP_SIZE / heads_per_unit) * NUM_KEY_VALUE_HEADS;
    return index / QUANTIZED_LAYER_COMPUTE_CUS == wave;
}

template <unsigned int Rows>
inline unsigned int quantized_attention_tasks_for_cu(
    const quantized_layer_task_t& task, unsigned int cu,
    unsigned int position_tile = 128) {
    #pragma HLS inline
    if (cu >= QUANTIZED_LAYER_COMPUTE_CUS || task.query_tokens == 0) return 0;
    const unsigned int units = quantized_attention_units<Rows>(task);
    const unsigned int assigned = cu < units ?
        1 + (units - 1 - cu) / QUANTIZED_LAYER_COMPUTE_CUS : 0;
    return 2 * assigned * quantized_ceildiv(
        task.kv_context_length + task.query_tokens, position_tile);
}

#endif
