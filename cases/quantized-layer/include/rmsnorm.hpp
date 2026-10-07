#ifndef LLM_FPGA_RMSNORM_HPP
#define LLM_FPGA_RMSNORM_HPP

#include "datatypes.hpp"

void compute_rms_norm(
    fm_t output[HIDDEN_SIZE],
    const fm_t input[HIDDEN_SIZE],
    const wt_norm_t weights[HIDDEN_SIZE]
);

#endif
