#ifndef LLM_FPGA_ROPE_HPP
#define LLM_FPGA_ROPE_HPP

#include "datatypes.hpp"

void apply_qwen_rope(
    fm_t q[HIDDEN_SIZE],
    fm_t k[KV_CHANNELS],
    unsigned int position
);

#endif
