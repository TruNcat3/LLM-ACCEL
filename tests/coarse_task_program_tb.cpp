#include "host_coarse_task_program.hpp"

#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_adjacent_hbm_residency(
    const llm_accel::coarse_task_program_t& program
) {
    for (std::size_t index = 1; index < program.tasks.size(); index++) {
        require(
            program.tasks[index - 1].output_pair ==
                program.tasks[index].input_pair,
            "adjacent coarse tasks do not share an HBM-resident boundary"
        );
    }
}

} // namespace

int main() {
    using namespace llm_accel;

    const coarse_task_program_t l2 = build_coarse_decoder_program(
        0, 2, 8, 1, 36, 2048, 8, true, true
    );
    require(l2.tasks.size() == 5, "L2 task count mismatch");
    require(l2.tasks[0].op == kCoarseAttentionOp, "L0 Attention missing");
    require(l2.tasks[1].op == kCoarseFfnOp, "L0 FFN missing");
    require(l2.tasks[2].layer == 1, "L1 Attention layer mismatch");
    require(l2.tasks[3].layer == 1, "L1 FFN layer mismatch");
    require(l2.tasks[4].op == kCoarseFinalNormOp, "final norm missing");
    require(l2.final_output_pair == 0, "normalized output pair mismatch");
    require(l2.materialize_output, "L2 final output must materialize");
    check_adjacent_hbm_residency(l2);

    const coarse_task_program_t released_p8 = build_coarse_decoder_program(
        0, 36, 0, 8, 36, 2048, 8, false, false
    );
    require(released_p8.tasks.size() == 72, "released P8 task count mismatch");
    require(released_p8.final_output_pair == 1, "released P8 pair mismatch");
    require(!released_p8.materialize_output, "released P8 copied to Host");
    check_adjacent_hbm_residency(released_p8);

    const coarse_task_program_t d1 = build_coarse_decoder_program(
        0, 36, 8, 1, 36, 2048, 8, true, true
    );
    require(d1.tasks.size() == 73, "D1 task count mismatch");
    require(d1.tasks.front().position == 8, "D1 position mismatch");
    require(d1.tasks.front().query_tokens == 1, "D1 row count mismatch");
    require(d1.tasks.back().op == kCoarseFinalNormOp, "D1 norm missing");
    check_adjacent_hbm_residency(d1);

    bool rejected = false;
    try {
        (void)build_coarse_decoder_program(
            0, 36, 2047, 8, 36, 2048, 8, true, true
        );
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "sequence overflow was accepted");

    std::cout
        << "COARSE TASK PROGRAM PASS"
        << " L2_tasks=" << l2.tasks.size()
        << " released_P8_tasks=" << released_p8.tasks.size()
        << " D1_tasks=" << d1.tasks.size()
        << " hidden_boundary=HBM_ping_pong"
        << " kv_owner=controller"
        << "\n";
    return 0;
}
