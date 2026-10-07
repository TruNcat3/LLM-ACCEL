#ifndef LLM_ACCEL_HOST_COARSE_TASK_PROGRAM_HPP
#define LLM_ACCEL_HOST_COARSE_TASK_PROGRAM_HPP

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llm_accel {

// Stable Host-visible operator IDs. The Host and HLS controller each retain
// compile-time assertions against these values so the task-program ABI cannot
// drift silently.
constexpr std::uint32_t kCoarseAttentionOp = 18;
constexpr std::uint32_t kCoarseFfnOp = 19;
constexpr std::uint32_t kCoarseFinalNormOp = 20;
constexpr char kCoarseTaskProgramContract[] = "static_descriptor_v1";

// One descriptor is one Host-issued controller invocation. Input/output pairs
// name the two HBM-resident hidden-state ping-pong regions:
//   pair 0 -> data ports 0/1
//   pair 1 -> data ports 2/3
// No descriptor exposes controller-local intermediates or KV-cache addresses.
struct coarse_task_descriptor_t {
    std::uint32_t op = 0;
    unsigned int layer = 0;
    unsigned int position = 0;
    unsigned int query_tokens = 0;
    unsigned int input_pair = 0;
    unsigned int output_pair = 0;
};

struct coarse_task_program_t {
    std::vector<coarse_task_descriptor_t> tasks;
    unsigned int layer_begin = 0;
    unsigned int layer_count = 0;
    unsigned int position = 0;
    unsigned int query_tokens = 0;
    unsigned int final_output_pair = 0;
    bool materialize_output = true;
};

inline const char* coarse_task_phase_name(std::uint32_t op) {
    switch (op) {
    case kCoarseAttentionOp:
        return "attention";
    case kCoarseFfnOp:
        return "ffn";
    case kCoarseFinalNormOp:
        return "final_norm";
    default:
        return "invalid";
    }
}

inline coarse_task_program_t build_coarse_decoder_program(
    unsigned int layer_begin,
    unsigned int layer_count,
    unsigned int position,
    unsigned int query_tokens,
    unsigned int model_layers,
    unsigned int max_sequence_length,
    unsigned int max_query_tokens,
    bool include_final_norm,
    bool materialize_output
) {
    if (
        layer_count == 0 ||
        layer_begin >= model_layers ||
        layer_begin + layer_count > model_layers ||
        query_tokens == 0 ||
        query_tokens > max_query_tokens ||
        position >= max_sequence_length ||
        position + query_tokens > max_sequence_length
    ) {
        throw std::invalid_argument("coarse decoder task-program shape mismatch");
    }

    coarse_task_program_t program;
    program.layer_begin = layer_begin;
    program.layer_count = layer_count;
    program.position = position;
    program.query_tokens = query_tokens;
    program.tasks.reserve(2 * layer_count + (include_final_norm ? 1u : 0u));
    program.materialize_output = materialize_output;

    for (unsigned int offset = 0; offset < layer_count; offset++) {
        const unsigned int layer = layer_begin + offset;
        program.tasks.push_back({
            kCoarseAttentionOp,
            layer,
            position,
            query_tokens,
            1,
            0
        });
        program.tasks.push_back({
            kCoarseFfnOp,
            layer,
            position,
            query_tokens,
            0,
            1
        });
    }

    program.final_output_pair = 1;
    if (include_final_norm) {
        program.tasks.push_back({
            kCoarseFinalNormOp,
            0,
            position,
            query_tokens,
            1,
            0
        });
        program.final_output_pair = 0;
    }
    return program;
}

} // namespace llm_accel

#endif
