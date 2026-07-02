#ifndef LLM_FPGA_MM_CONTROLLER_HPP
#define LLM_FPGA_MM_CONTROLLER_HPP

#include "linear.hpp"

constexpr unsigned int MM_GLOBAL_BUFFER_COUNT = 2;
constexpr unsigned int MM_EXTERNAL_PIPE_STAGES = 5;
constexpr unsigned int MM_INTERNAL_PIPE_STAGES = 3;
constexpr unsigned int MM_INVALID_CHUNK = ~0u;

enum mm_global_buffer_id_t {
    MM_GLOBAL_BUFFER_0 = 0,
    MM_GLOBAL_BUFFER_1 = 1
};

enum mm_controller_mode_t {
    MM_MODE_QKV = 0,
    MM_MODE_ATTN_O = 1,
    MM_MODE_FFN_GATE_UP = 2,
    MM_MODE_FFN_DOWN = 3
};

enum mm_projection_kind_t {
    MM_PROJECTION_Q = 0,
    MM_PROJECTION_K = 1,
    MM_PROJECTION_V = 2,
    MM_PROJECTION_ATTN_O = 3,
    MM_PROJECTION_FFN_GATE = 4,
    MM_PROJECTION_FFN_UP = 5,
    MM_PROJECTION_FFN_DOWN = 6
};

struct mm_controller_task_t {
    mm_controller_mode_t mode;
    weight_addr_t weight_base;
    unsigned int token_count;
    unsigned int out_dim;
    unsigned int in_dim;
    unsigned int out_tile_begin;
    unsigned int out_tile_count;
    unsigned int in_tile_count;
};

struct mm_double_buffer_phase_t {
    bool load_buffer0;
    bool load_buffer1;
    bool compute_buffer0;
    bool compute_buffer1;
    bool store_buffer0;
    bool store_buffer1;
};

struct mm_pipeline_slot_t {
    bool valid;
    mm_global_buffer_id_t buffer;
    unsigned int chunk;
};

struct mm_double_buffer_step_t {
    mm_pipeline_slot_t load;
    mm_pipeline_slot_t compute;
    mm_pipeline_slot_t store;
};

struct mm_core_assignment_t {
    bool active;
    unsigned int core_id;
    unsigned int token_lane;
    unsigned int out_tile;
    unsigned int out_tile_offset;
    unsigned int in_tile_begin;
    unsigned int in_tile_count;
    weight_addr_t weight_tile_base;
};

struct mm_projection_spec_t {
    mm_controller_mode_t mode;
    weight_addr_t weight_base;
    unsigned int out_dim;
    unsigned int in_dim;
};

unsigned int get_mm_double_buffer_phase_count(unsigned int tile_count);

unsigned int get_mm_controller_output_wave_count(const mm_controller_task_t& task);

void get_mm_controller_core_assignment(
    const mm_controller_task_t& task,
    unsigned int output_wave,
    unsigned int core_id,
    mm_core_assignment_t& assignment
);

void init_mm_controller_task(
    mm_controller_task_t& task,
    mm_controller_mode_t mode,
    weight_addr_t weight_base,
    unsigned int token_count,
    unsigned int out_dim,
    unsigned int in_dim
);

void get_mm_projection_spec(
    mm_projection_kind_t projection,
    mm_projection_spec_t& spec
);

void init_mm_controller_projection_task(
    mm_controller_task_t& task,
    mm_projection_kind_t projection,
    unsigned int token_count
);

mm_double_buffer_phase_t get_mm_double_buffer_phase(
    unsigned int phase,
    unsigned int tile_count
);

void get_mm_double_buffer_step(
    unsigned int phase,
    unsigned int tile_count,
    mm_double_buffer_step_t& step
);

#endif
