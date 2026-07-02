#ifndef LLM_FPGA_DATATYPES_HPP
#define LLM_FPGA_DATATYPES_HPP

#include "model_config.hpp"
#include <ap_fixed.h>
#include <ap_int.h>

typedef ap_fixed<16, 8, AP_RND, AP_SAT> fm_t;
typedef ap_fixed<16, 4, AP_RND, AP_SAT> wt_linear_t;
typedef ap_fixed<16, 8, AP_RND, AP_SAT> wt_norm_t;
typedef ap_fixed<32, 16, AP_RND, AP_SAT> fm_accum_t;
typedef unsigned long long weight_addr_t;

#endif
