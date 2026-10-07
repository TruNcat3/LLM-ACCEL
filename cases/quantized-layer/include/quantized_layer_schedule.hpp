#ifndef LLM_FPGA_QUANTIZED_LAYER_SCHEDULE_HPP
#define LLM_FPGA_QUANTIZED_LAYER_SCHEDULE_HPP

#include "model_config.hpp"
#include "quantized_layer_task.hpp"
#include "mm_stream_quantized_nk.hpp"

#include <cstddef>

constexpr unsigned int QUANTIZED_LAYER_COMPUTE_CUS = 4;
constexpr unsigned int QUANTIZED_LAYER_OUTPUTS_PER_CU = 128;
constexpr unsigned int QUANTIZED_LAYER_OUTPUTS_PER_WAVE =
    QUANTIZED_LAYER_COMPUTE_CUS * QUANTIZED_LAYER_OUTPUTS_PER_CU;

enum quantized_projection_t {
    QUANTIZED_PROJECTION_Q = 0,
    QUANTIZED_PROJECTION_K = 1,
    QUANTIZED_PROJECTION_V = 2,
    QUANTIZED_PROJECTION_O = 3,
    QUANTIZED_PROJECTION_GATE = 4,
    QUANTIZED_PROJECTION_UP = 5,
    QUANTIZED_PROJECTION_DOWN = 6,
    QUANTIZED_PROJECTION_COUNT = 7
};

// Logical controller-resident banks.  Several of these may share the same
// physical RAM after lifetime analysis; the names make the producer/consumer
// contract explicit before that storage optimization is applied.
enum quantized_resident_bank_t {
    QUANTIZED_BANK_ATTN_NORM = 0,
    QUANTIZED_BANK_QUERY = 1,
    QUANTIZED_BANK_KEY = 2,
    QUANTIZED_BANK_VALUE = 3,
    QUANTIZED_BANK_ATTN_CONTEXT = 4,
    QUANTIZED_BANK_ATTN_PROJECTED = 5,
    QUANTIZED_BANK_FFN_NORM = 6,
    QUANTIZED_BANK_GATE = 7,
    QUANTIZED_BANK_UP = 8,
    QUANTIZED_BANK_FFN_PRODUCT = 9,
    QUANTIZED_BANK_FFN_DOWN = 10
};

// Weight offsets are expressed in logical scalar elements.  Their byte
// address is selected later from the W4/W8 packing format.
struct quantized_projection_plan_t {
    quantized_projection_t projection;
    unsigned int input_dim;
    unsigned int output_dim;
    std::size_t logical_weight_offset;
    // Offset in 256-bit words within each CU/weight-stream HBM allocation.
    // Each CU and each of its four streams uses the same wave-major layout.
    std::size_t shard_word_offset;
    unsigned int wave_count;
    unsigned int active_cus_in_last_wave;
    quantized_resident_bank_t source_bank;
    quantized_resident_bank_t destination_bank;
};

struct quantized_projection_dispatch_t {
    quantized_projection_t projection;
    unsigned int wave;
    unsigned int cu;
    unsigned int local_task_id;
    unsigned int elem_base;
    unsigned int valid_outputs;
    unsigned int valid_output_groups;
    std::size_t shard_weight_word_offset;
    bool active;
    bool last_layer_task;
};

struct quantized_w8_weight_location_t {
    unsigned int layer;
    quantized_projection_t projection;
    unsigned int output;
    unsigned int input;
    unsigned int wave;
    unsigned int cu;
    unsigned int stream;
    unsigned int lane;
    std::size_t shard_word_offset;
    bool valid;
};

struct quantized_layer_plan_t {
    unsigned int query_tokens;
    unsigned int physical_block_size;
    unsigned int block_count;
    unsigned int attention_context_length;
    unsigned int projection_waves_per_block;
    unsigned int issued_projection_tasks_per_block;
    unsigned int useful_projection_tasks_per_block;
    unsigned long long useful_projection_macs;
    unsigned long long issued_projection_macs;
};

constexpr unsigned int quantized_ceildiv(
    unsigned int value,
    unsigned int divisor) {
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

constexpr std::size_t quantized_projection_weight_elements(
    unsigned int output_dim,
    unsigned int input_dim) {
    return std::size_t(output_dim) * input_dim;
}

constexpr std::size_t quantized_projection_shard_word_offset(
    quantized_projection_t projection) {
    const unsigned int hidden_waves =
        quantized_ceildiv(HIDDEN_SIZE, QUANTIZED_LAYER_OUTPUTS_PER_WAVE);
    const unsigned int kv_waves =
        quantized_ceildiv(KV_CHANNELS, QUANTIZED_LAYER_OUTPUTS_PER_WAVE);
    const unsigned int wide_waves =
        quantized_ceildiv(INTERMEDIATE_SIZE,
                          QUANTIZED_LAYER_OUTPUTS_PER_WAVE);
    return projection == QUANTIZED_PROJECTION_Q ? 0 :
        projection == QUANTIZED_PROJECTION_K ?
            std::size_t(hidden_waves) * HIDDEN_SIZE :
        projection == QUANTIZED_PROJECTION_V ?
            std::size_t(hidden_waves + kv_waves) * HIDDEN_SIZE :
        projection == QUANTIZED_PROJECTION_O ?
            std::size_t(hidden_waves + 2 * kv_waves) * HIDDEN_SIZE :
        projection == QUANTIZED_PROJECTION_GATE ?
            std::size_t(2 * hidden_waves + 2 * kv_waves) * HIDDEN_SIZE :
        projection == QUANTIZED_PROJECTION_UP ?
            std::size_t(2 * hidden_waves + 2 * kv_waves + wide_waves) *
                HIDDEN_SIZE :
            std::size_t(2 * hidden_waves + 2 * kv_waves + 2 * wide_waves) *
                HIDDEN_SIZE;
}

inline quantized_projection_plan_t get_quantized_projection_plan(
    quantized_projection_t projection) {
    quantized_projection_plan_t plan{};
    plan.projection = projection;

    const std::size_t q_size =
        quantized_projection_weight_elements(HIDDEN_SIZE, HIDDEN_SIZE);
    const std::size_t k_size =
        quantized_projection_weight_elements(KV_CHANNELS, HIDDEN_SIZE);
    const std::size_t v_size = k_size;
    const std::size_t o_size =
        quantized_projection_weight_elements(HIDDEN_SIZE, HIDDEN_SIZE);
    const std::size_t gate_size =
        quantized_projection_weight_elements(INTERMEDIATE_SIZE, HIDDEN_SIZE);
    const std::size_t up_size = gate_size;

    switch (projection) {
    case QUANTIZED_PROJECTION_Q:
        plan.input_dim = HIDDEN_SIZE;
        plan.output_dim = HIDDEN_SIZE;
        plan.logical_weight_offset = 0;
        plan.source_bank = QUANTIZED_BANK_ATTN_NORM;
        plan.destination_bank = QUANTIZED_BANK_QUERY;
        break;
    case QUANTIZED_PROJECTION_K:
        plan.input_dim = HIDDEN_SIZE;
        plan.output_dim = KV_CHANNELS;
        plan.logical_weight_offset = q_size;
        plan.source_bank = QUANTIZED_BANK_ATTN_NORM;
        plan.destination_bank = QUANTIZED_BANK_KEY;
        break;
    case QUANTIZED_PROJECTION_V:
        plan.input_dim = HIDDEN_SIZE;
        plan.output_dim = KV_CHANNELS;
        plan.logical_weight_offset = q_size + k_size;
        plan.source_bank = QUANTIZED_BANK_ATTN_NORM;
        plan.destination_bank = QUANTIZED_BANK_VALUE;
        break;
    case QUANTIZED_PROJECTION_O:
        plan.input_dim = HIDDEN_SIZE;
        plan.output_dim = HIDDEN_SIZE;
        plan.logical_weight_offset = q_size + k_size + v_size;
        plan.source_bank = QUANTIZED_BANK_ATTN_CONTEXT;
        plan.destination_bank = QUANTIZED_BANK_ATTN_PROJECTED;
        break;
    case QUANTIZED_PROJECTION_GATE:
        plan.input_dim = HIDDEN_SIZE;
        plan.output_dim = INTERMEDIATE_SIZE;
        plan.logical_weight_offset = q_size + k_size + v_size + o_size;
        plan.source_bank = QUANTIZED_BANK_FFN_NORM;
        plan.destination_bank = QUANTIZED_BANK_GATE;
        break;
    case QUANTIZED_PROJECTION_UP:
        plan.input_dim = HIDDEN_SIZE;
        plan.output_dim = INTERMEDIATE_SIZE;
        plan.logical_weight_offset =
            q_size + k_size + v_size + o_size + gate_size;
        plan.source_bank = QUANTIZED_BANK_FFN_NORM;
        plan.destination_bank = QUANTIZED_BANK_UP;
        break;
    case QUANTIZED_PROJECTION_DOWN:
    default:
        plan.input_dim = INTERMEDIATE_SIZE;
        plan.output_dim = HIDDEN_SIZE;
        plan.logical_weight_offset =
            q_size + k_size + v_size + o_size + gate_size + up_size;
        plan.source_bank = QUANTIZED_BANK_FFN_PRODUCT;
        plan.destination_bank = QUANTIZED_BANK_FFN_DOWN;
        break;
    }

    plan.shard_word_offset =
        quantized_projection_shard_word_offset(projection);

    plan.wave_count = quantized_ceildiv(
        plan.output_dim, QUANTIZED_LAYER_OUTPUTS_PER_WAVE);
    const unsigned int last_wave_outputs =
        plan.output_dim - (plan.wave_count - 1) *
                              QUANTIZED_LAYER_OUTPUTS_PER_WAVE;
    plan.active_cus_in_last_wave = quantized_ceildiv(
        last_wave_outputs, QUANTIZED_LAYER_OUTPUTS_PER_CU);
    return plan;
}

inline unsigned int quantized_projection_wave_prefix(
    quantized_projection_t projection) {
    unsigned int prefix = 0;
    for (unsigned int preceding = 0;
         preceding < static_cast<unsigned int>(projection); ++preceding) {
        prefix += get_quantized_projection_plan(
            static_cast<quantized_projection_t>(preceding)).wave_count;
    }
    return prefix;
}

inline unsigned int quantized_projection_waves_per_block() {
    unsigned int waves = 0;
    for (unsigned int index = 0;
         index < QUANTIZED_PROJECTION_COUNT; ++index) {
        waves += get_quantized_projection_plan(
            static_cast<quantized_projection_t>(index)).wave_count;
    }
    return waves;
}

inline bool decode_quantized_projection_slot(
    unsigned int slot,
    quantized_projection_t& projection,
    unsigned int& wave) {
    unsigned int cursor = 0;
    for (unsigned int index = 0;
         index < QUANTIZED_PROJECTION_COUNT; ++index) {
        const quantized_projection_plan_t candidate =
            get_quantized_projection_plan(
                static_cast<quantized_projection_t>(index));
        if (slot < cursor + candidate.wave_count) {
            projection = candidate.projection;
            wave = slot - cursor;
            return true;
        }
        cursor += candidate.wave_count;
    }
    projection = QUANTIZED_PROJECTION_Q;
    wave = 0;
    return false;
}

constexpr std::size_t quantized_layer_weight_words_per_shard() {
    // Keep the layer stride a scalar constant expression. Vitis HLS 2022.2
    // dropped the Down plan's shard offset when folding the returned plan
    // aggregate here: the dataflow test used 1280 instead of 2560 words.
    return quantized_projection_shard_word_offset(QUANTIZED_PROJECTION_DOWN) +
        std::size_t(quantized_ceildiv(
            HIDDEN_SIZE, QUANTIZED_LAYER_OUTPUTS_PER_WAVE)) * INTERMEDIATE_SIZE;
}

constexpr unsigned int QUANTIZED_W8_WEIGHT_STREAMS_PER_CU = 4;
constexpr unsigned int QUANTIZED_W8_WEIGHTS_PER_STREAM_WORD = 32;
constexpr unsigned int QUANTIZED_W8_WEIGHT_WORD_BYTES = 32;

inline std::size_t quantized_w8_weight_shard_words(
    unsigned int layer_count) {
    return static_cast<std::size_t>(layer_count) *
        quantized_layer_weight_words_per_shard();
}

inline std::size_t quantized_w8_weight_shard_bytes(
    unsigned int layer_count) {
    return quantized_w8_weight_shard_words(layer_count) *
        QUANTIZED_W8_WEIGHT_WORD_BYTES;
}

inline quantized_w8_weight_location_t get_quantized_w8_weight_location(
    unsigned int layer,
    quantized_projection_t projection_kind,
    unsigned int output,
    unsigned int input) {
    quantized_w8_weight_location_t location{};
    const quantized_projection_plan_t projection =
        get_quantized_projection_plan(projection_kind);
    location.layer = layer;
    location.projection = projection_kind;
    location.output = output;
    location.input = input;
    location.valid = layer < NUM_LAYERS && output < projection.output_dim &&
        input < projection.input_dim;
    if (!location.valid) return location;

    location.wave = output / QUANTIZED_LAYER_OUTPUTS_PER_WAVE;
    const unsigned int within_wave =
        output % QUANTIZED_LAYER_OUTPUTS_PER_WAVE;
    location.cu = within_wave / QUANTIZED_LAYER_OUTPUTS_PER_CU;
    const unsigned int within_cu =
        within_wave % QUANTIZED_LAYER_OUTPUTS_PER_CU;
    location.stream =
        within_cu / QUANTIZED_W8_WEIGHTS_PER_STREAM_WORD;
    location.lane =
        within_cu % QUANTIZED_W8_WEIGHTS_PER_STREAM_WORD;
    location.shard_word_offset =
        static_cast<std::size_t>(layer) *
            quantized_layer_weight_words_per_shard() +
        projection.shard_word_offset +
        static_cast<std::size_t>(location.wave) * projection.input_dim + input;
    return location;
}

inline unsigned int quantized_projection_active_cu_mask(
    const quantized_projection_plan_t& projection,
    unsigned int wave) {
    if (wave >= projection.wave_count) return 0;
    const unsigned int active = wave + 1 == projection.wave_count ?
        projection.active_cus_in_last_wave : QUANTIZED_LAYER_COMPUTE_CUS;
    return (1u << active) - 1u;
}

inline unsigned int quantized_projection_tasks_per_block_for_cu(
    unsigned int cu) {
    if (cu >= QUANTIZED_LAYER_COMPUTE_CUS) return 0;
    unsigned int tasks = 0;
    for (unsigned int index = 0;
         index < QUANTIZED_PROJECTION_COUNT; ++index) {
        const quantized_projection_plan_t projection =
            get_quantized_projection_plan(
                static_cast<quantized_projection_t>(index));
        for (unsigned int wave = 0; wave < projection.wave_count; ++wave) {
            tasks += (quantized_projection_active_cu_mask(projection, wave) &
                      (1u << cu)) != 0;
        }
    }
    return tasks;
}

inline unsigned int quantized_projection_tasks_for_cu(
    unsigned int cu,
    unsigned int block_count) {
    return quantized_projection_tasks_per_block_for_cu(cu) * block_count;
}

inline bool quantized_projection_is_last_task_for_cu(
    quantized_projection_t projection,
    unsigned int wave,
    unsigned int cu,
    bool final_physical_block) {
    if (!final_physical_block || cu >= QUANTIZED_LAYER_COMPUTE_CUS ||
        projection != QUANTIZED_PROJECTION_DOWN) {
        return false;
    }
    const quantized_projection_plan_t plan =
        get_quantized_projection_plan(projection);
    return wave + 1 == plan.wave_count &&
        (quantized_projection_active_cu_mask(plan, wave) & (1u << cu)) != 0;
}

inline quantized_projection_dispatch_t make_quantized_projection_dispatch(
    quantized_projection_t projection_kind,
    unsigned int wave,
    unsigned int cu,
    bool final_physical_block,
    unsigned int layer = 0) {
    quantized_projection_dispatch_t dispatch{};
    const quantized_projection_plan_t projection =
        get_quantized_projection_plan(projection_kind);
    dispatch.projection = projection_kind;
    dispatch.wave = wave;
    dispatch.cu = cu;
    const unsigned int wave_prefix =
        quantized_projection_wave_prefix(projection_kind);
    dispatch.local_task_id =
        (wave_prefix + wave) * QUANTIZED_LAYER_COMPUTE_CUS + cu;
    dispatch.elem_base = wave * QUANTIZED_LAYER_OUTPUTS_PER_WAVE +
        cu * QUANTIZED_LAYER_OUTPUTS_PER_CU;
    const unsigned int remaining = dispatch.elem_base < projection.output_dim ?
        projection.output_dim - dispatch.elem_base : 0;
    dispatch.valid_outputs = remaining < QUANTIZED_LAYER_OUTPUTS_PER_CU ?
        remaining : QUANTIZED_LAYER_OUTPUTS_PER_CU;
    dispatch.valid_output_groups =
        quantized_ceildiv(dispatch.valid_outputs, 16);
    dispatch.shard_weight_word_offset =
        static_cast<std::size_t>(layer) *
            quantized_layer_weight_words_per_shard() +
        projection.shard_word_offset +
        static_cast<std::size_t>(wave) * projection.input_dim;
    dispatch.active = dispatch.valid_outputs != 0;
    dispatch.last_layer_task = quantized_projection_is_last_task_for_cu(
        projection_kind, wave, cu, final_physical_block);
    return dispatch;
}

inline quantized_projection_dispatch_t make_quantized_projection_dispatch_slot(
    unsigned int slot,
    unsigned int cu,
    bool final_physical_block,
    unsigned int layer = 0) {
    quantized_projection_t projection = QUANTIZED_PROJECTION_Q;
    unsigned int wave = 0;
    if (!decode_quantized_projection_slot(slot, projection, wave)) {
        return quantized_projection_dispatch_t{};
    }
    return make_quantized_projection_dispatch(
        projection, wave, cu, final_physical_block, layer);
}

inline mm_stream_quantized_task_t make_quantized_projection_task(
    const quantized_layer_task_t& layer_task,
    const quantized_projection_dispatch_t& dispatch,
    quant_scale_t activation_scale,
    quant_scale_t weight_scale) {
    const quantized_projection_plan_t projection =
        get_quantized_projection_plan(dispatch.projection);
    mm_stream_quantized_task_t task{};
    task.k_count = projection.input_dim;
    task.elem_base = dispatch.elem_base;
    // block_id is deliberately local to a physical token block.  A P2048
    // layer has more than 65,535 issued tasks, while this packed ABI has a
    // 16-bit block_id.  Position and projection identify the outer context.
    task.block_id = dispatch.local_task_id;
    task.last_stream = dispatch.last_layer_task;
    task.activation_scale = activation_scale;
    task.weight_scale = weight_scale;
    task.request_position = layer_task.position;
    task.valid_tokens = layer_task.query_tokens;
    task.phase = static_cast<unsigned int>(layer_task.phase);
    task.kv_context_length = layer_task.kv_context_length;
    task.projection = static_cast<unsigned int>(dispatch.projection);
    task.compute_mode = MM_STREAM_QUANTIZED_MODE_LINEAR;
    return task;
}

inline quantized_layer_plan_t make_quantized_layer_plan(
    unsigned int query_tokens,
    unsigned int attention_context_length,
    unsigned int physical_block_size) {
    quantized_layer_plan_t plan{};
    plan.query_tokens = query_tokens;
    plan.physical_block_size =
        quantized_layer_effective_block_size(physical_block_size);
    plan.block_count = quantized_layer_block_count(
        query_tokens, plan.physical_block_size);
    plan.attention_context_length = attention_context_length;

    unsigned long long useful_macs_per_token = 0;
    unsigned long long issued_macs_per_block = 0;
    for (unsigned int index = 0; index < QUANTIZED_PROJECTION_COUNT; ++index) {
        const quantized_projection_plan_t projection =
            get_quantized_projection_plan(
                static_cast<quantized_projection_t>(index));
        plan.projection_waves_per_block += projection.wave_count;
        plan.issued_projection_tasks_per_block +=
            projection.wave_count * QUANTIZED_LAYER_COMPUTE_CUS;
        plan.useful_projection_tasks_per_block +=
            (projection.wave_count - 1) * QUANTIZED_LAYER_COMPUTE_CUS +
            projection.active_cus_in_last_wave;
        useful_macs_per_token +=
            static_cast<unsigned long long>(projection.input_dim) *
            projection.output_dim;
        issued_macs_per_block +=
            static_cast<unsigned long long>(projection.input_dim) *
            projection.wave_count * QUANTIZED_LAYER_OUTPUTS_PER_WAVE *
            plan.physical_block_size;
    }
    plan.useful_projection_macs =
        useful_macs_per_token * query_tokens;
    plan.issued_projection_macs =
        issued_macs_per_block * plan.block_count;
    return plan;
}

inline bool quantized_layer_plan_valid(const quantized_layer_plan_t& plan) {
    return plan.query_tokens != 0 &&
        plan.query_tokens <= MAX_SEQ_LEN &&
        plan.physical_block_size != 0 &&
        plan.physical_block_size <= QUANTIZED_LAYER_MAX_TOKEN_BLOCK &&
        plan.block_count == quantized_layer_block_count(
            plan.query_tokens, plan.physical_block_size) &&
        plan.attention_context_length <= MAX_SEQ_LEN;
}

// Keep the opt-in full-layer Decode bridge visible to both controller and
// Host task-count translation units after all projection plans are defined.
#include "quantized_layer_integrated_decode.hpp"

#endif
