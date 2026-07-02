#include "mm_controller.hpp"

static mm_global_buffer_id_t get_mm_buffer_for_chunk(unsigned int chunk) {
    #pragma HLS inline
    return (chunk & 1) == 0 ? MM_GLOBAL_BUFFER_0 : MM_GLOBAL_BUFFER_1;
}

static void clear_mm_pipeline_slot(mm_pipeline_slot_t& slot) {
    #pragma HLS inline
    slot.valid = false;
    slot.buffer = MM_GLOBAL_BUFFER_0;
    slot.chunk = MM_INVALID_CHUNK;
}

static void set_mm_pipeline_slot(mm_pipeline_slot_t& slot, unsigned int chunk) {
    #pragma HLS inline
    slot.valid = true;
    slot.buffer = get_mm_buffer_for_chunk(chunk);
    slot.chunk = chunk;
}

unsigned int get_mm_double_buffer_phase_count(unsigned int tile_count) {
    #pragma HLS inline
    return (tile_count == 0) ? 0 : tile_count + 2;
}

unsigned int get_mm_controller_output_wave_count(const mm_controller_task_t& task) {
    #pragma HLS inline
    return ceildiv(task.out_tile_count, MM_CONTROLLER_OUT_TILE_PARALLEL);
}

void init_mm_controller_task(
    mm_controller_task_t& task,
    mm_controller_mode_t mode,
    weight_addr_t weight_base,
    unsigned int token_count,
    unsigned int out_dim,
    unsigned int in_dim
) {
    #pragma HLS inline
    task.mode = mode;
    task.weight_base = weight_base;
    task.token_count = token_count;
    task.out_dim = out_dim;
    task.in_dim = in_dim;
    task.out_tile_begin = 0;
    task.out_tile_count = ceildiv(out_dim, MM_PE_OUT);
    task.in_tile_count = ceildiv(in_dim, MM_PE_IN);
}

void get_mm_projection_spec(
    mm_projection_kind_t projection,
    mm_projection_spec_t& spec
) {
    #pragma HLS inline

    switch (projection) {
    case MM_PROJECTION_Q:
        spec.mode = MM_MODE_QKV;
        spec.weight_base = weight_addr_t(Q_WEIGHT_OFFSET);
        spec.out_dim = HIDDEN_SIZE;
        spec.in_dim = HIDDEN_SIZE;
        break;
    case MM_PROJECTION_K:
        spec.mode = MM_MODE_QKV;
        spec.weight_base = weight_addr_t(K_WEIGHT_OFFSET);
        spec.out_dim = KV_CHANNELS;
        spec.in_dim = HIDDEN_SIZE;
        break;
    case MM_PROJECTION_V:
        spec.mode = MM_MODE_QKV;
        spec.weight_base = weight_addr_t(V_WEIGHT_OFFSET);
        spec.out_dim = KV_CHANNELS;
        spec.in_dim = HIDDEN_SIZE;
        break;
    case MM_PROJECTION_ATTN_O:
        spec.mode = MM_MODE_ATTN_O;
        spec.weight_base = weight_addr_t(O_WEIGHT_OFFSET);
        spec.out_dim = HIDDEN_SIZE;
        spec.in_dim = HIDDEN_SIZE;
        break;
    case MM_PROJECTION_FFN_GATE:
        spec.mode = MM_MODE_FFN_GATE_UP;
        spec.weight_base = weight_addr_t(GATE_WEIGHT_OFFSET);
        spec.out_dim = INTERMEDIATE_SIZE;
        spec.in_dim = HIDDEN_SIZE;
        break;
    case MM_PROJECTION_FFN_UP:
        spec.mode = MM_MODE_FFN_GATE_UP;
        spec.weight_base = weight_addr_t(UP_WEIGHT_OFFSET);
        spec.out_dim = INTERMEDIATE_SIZE;
        spec.in_dim = HIDDEN_SIZE;
        break;
    case MM_PROJECTION_FFN_DOWN:
    default:
        spec.mode = MM_MODE_FFN_DOWN;
        spec.weight_base = weight_addr_t(DOWN_WEIGHT_OFFSET);
        spec.out_dim = HIDDEN_SIZE;
        spec.in_dim = INTERMEDIATE_SIZE;
        break;
    }
}

void init_mm_controller_projection_task(
    mm_controller_task_t& task,
    mm_projection_kind_t projection,
    unsigned int token_count
) {
    #pragma HLS inline

    mm_projection_spec_t spec;
    get_mm_projection_spec(projection, spec);
    init_mm_controller_task(
        task,
        spec.mode,
        spec.weight_base,
        token_count,
        spec.out_dim,
        spec.in_dim
    );
}

void get_mm_double_buffer_step(
    unsigned int phase,
    unsigned int tile_count,
    mm_double_buffer_step_t& step
) {
    #pragma HLS inline
    clear_mm_pipeline_slot(step.load);
    clear_mm_pipeline_slot(step.compute);
    clear_mm_pipeline_slot(step.store);

    if (tile_count == 0 || phase >= get_mm_double_buffer_phase_count(tile_count)) {
        return;
    }

    if (phase < tile_count) {
        set_mm_pipeline_slot(step.load, phase);
    }

    if (phase >= 1) {
        unsigned int compute_chunk = phase - 1;
        if (compute_chunk < tile_count) {
            set_mm_pipeline_slot(step.compute, compute_chunk);
        }
    }

    if (phase >= 2) {
        unsigned int store_chunk = phase - 2;
        if (store_chunk < tile_count) {
            set_mm_pipeline_slot(step.store, store_chunk);
        }
    }
}

mm_double_buffer_phase_t get_mm_double_buffer_phase(
    unsigned int phase,
    unsigned int tile_count
) {
    #pragma HLS inline
    mm_double_buffer_phase_t p = {false, false, false, false, false, false};
    mm_double_buffer_step_t step;
    get_mm_double_buffer_step(phase, tile_count, step);

    p.load_buffer0 = step.load.valid && step.load.buffer == MM_GLOBAL_BUFFER_0;
    p.load_buffer1 = step.load.valid && step.load.buffer == MM_GLOBAL_BUFFER_1;
    p.compute_buffer0 = step.compute.valid && step.compute.buffer == MM_GLOBAL_BUFFER_0;
    p.compute_buffer1 = step.compute.valid && step.compute.buffer == MM_GLOBAL_BUFFER_1;
    p.store_buffer0 = step.store.valid && step.store.buffer == MM_GLOBAL_BUFFER_0;
    p.store_buffer1 = step.store.valid && step.store.buffer == MM_GLOBAL_BUFFER_1;

    return p;
}

void get_mm_controller_core_assignment(
    const mm_controller_task_t& task,
    unsigned int output_wave,
    unsigned int core_id,
    mm_core_assignment_t& assignment
) {
    #pragma HLS inline
    unsigned int local_token_lane = core_id / MM_CONTROLLER_OUT_TILE_PARALLEL;
    unsigned int local_out_offset = core_id % MM_CONTROLLER_OUT_TILE_PARALLEL;
    unsigned int out_tile_offset = output_wave * MM_CONTROLLER_OUT_TILE_PARALLEL + local_out_offset;
    weight_addr_t matrix_tile_base = task.weight_base / weight_addr_t(MM_TILE_WEIGHT_ELEMS);

    assignment.active =
        core_id < MM_CONTROLLER_CORE_COUNT &&
        local_token_lane < task.token_count &&
        local_token_lane < MM_CONTROLLER_TOKEN_LANES &&
        out_tile_offset < task.out_tile_count;
    assignment.core_id = core_id;
    assignment.token_lane = local_token_lane;
    assignment.out_tile = task.out_tile_begin + out_tile_offset;
    assignment.out_tile_offset = out_tile_offset;
    assignment.in_tile_begin = 0;
    assignment.in_tile_count = task.in_tile_count;
    assignment.weight_tile_base =
        matrix_tile_base +
        weight_addr_t(assignment.out_tile) * weight_addr_t(task.in_tile_count);
}
