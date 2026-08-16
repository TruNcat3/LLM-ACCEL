#ifndef LLM_FPGA_DATATYPES_HPP
#define LLM_FPGA_DATATYPES_HPP

#include "model_config.hpp"
#include <ap_fixed.h>
#include <ap_int.h>

typedef ap_fixed<16, 8, AP_RND, AP_SAT> fm_t;
typedef ap_fixed<16, 4, AP_RND, AP_SAT> wt_linear_t;
typedef ap_fixed<16, 8, AP_RND, AP_SAT> wt_norm_t;
typedef ap_fixed<32, 16, AP_RND, AP_SAT> fm_accum_t;
// Online-softmax probabilities and inter-tile rescale factors are confined
// to [0, 1].  Keeping them in a 16-bit Q2.14 format preserves the existing
// stream width while avoiding the long-context precision loss of Q8.8.
typedef ap_fixed<16, 2, AP_RND, AP_SAT> attention_prob_t;
typedef unsigned long long weight_addr_t;

#endif
