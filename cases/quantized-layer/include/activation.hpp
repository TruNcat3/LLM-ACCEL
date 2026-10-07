#ifndef LLM_FPGA_ACTIVATION_HPP
#define LLM_FPGA_ACTIVATION_HPP

#include "datatypes.hpp"

fm_t clamp_fm(fm_t x, fm_t lo, fm_t hi);
fm_t fast_exp(fm_t x);
fm_t fast_recip(fm_t x);
fm_t fast_rsqrt(fm_accum_t x);
fm_t silu(fm_t x);
fm_t gelu(fm_t x);

#endif
