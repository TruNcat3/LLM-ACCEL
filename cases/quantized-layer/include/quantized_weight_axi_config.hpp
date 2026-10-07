#ifndef LLM_FPGA_QUANTIZED_WEIGHT_AXI_CONFIG_HPP
#define LLM_FPGA_QUANTIZED_WEIGHT_AXI_CONFIG_HPP

// Shared by full-layer and bounded batch controllers. Profiles must configure
// actual AXI adapters, not just carry unused compiler definitions.
// Read buffering scales with outstanding * burst_length * 32 bytes per port.
#ifndef QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING
#define QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING 16
#endif
#ifndef QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST
#define QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST 16
#endif
static_assert(QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING > 0 &&
              QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING <= 64,
              "weight read outstanding must be in [1,64]");
static_assert(QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST > 0 &&
              QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST <= 256,
              "weight read burst must be in [1,256]");

#endif
