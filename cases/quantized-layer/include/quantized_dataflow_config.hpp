#ifndef LLM_FPGA_QUANTIZED_DATAFLOW_CONFIG_HPP
#define LLM_FPGA_QUANTIZED_DATAFLOW_CONFIG_HPP

// Match the resident Fix16 topology: one DATAFLOW region for all output waves,
// with independently running HBM load, input issue, result drain and commit.
// Packed word widths and CU dimensions belong to the precision-specific policy.
#ifndef QUANTIZED_WEIGHT_FIFO_DEPTH
#define QUANTIZED_WEIGHT_FIFO_DEPTH 16
#endif
#ifndef QUANTIZED_RESULT_FIFO_DEPTH
#define QUANTIZED_RESULT_FIFO_DEPTH 65
#endif
static_assert(QUANTIZED_WEIGHT_FIFO_DEPTH >= 2,
              "Weight buffering must hold at least two packed words");
static_assert(QUANTIZED_RESULT_FIFO_DEPTH >= 2,
              "Result buffering must decouple drain and commit");

#endif
