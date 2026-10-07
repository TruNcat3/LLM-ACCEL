#ifndef LLM_FPGA_QUANTIZED_ASYNC_CONFIG_HPP
#define LLM_FPGA_QUANTIZED_ASYNC_CONFIG_HPP

// Shared by production compute and the closed-loop RTL fixtures. Keep FIFO
// settings independent of compute function definitions so synthesis-only
// wrapper paths see the same finite depths as the actual compute core.
#if defined(QUANTIZED_ASYNC_COMPUTE)
#define LLM_FPGA_QUANTIZED_ASYNC_COMPUTE_ENABLED (QUANTIZED_ASYNC_COMPUTE != 0)
#elif defined(QUANTIZED_BLOCK_PIPELINE)
#define LLM_FPGA_QUANTIZED_ASYNC_COMPUTE_ENABLED (QUANTIZED_BLOCK_PIPELINE != 0)
#else
#define LLM_FPGA_QUANTIZED_ASYNC_COMPUTE_ENABLED 0
#endif

#ifndef QUANTIZED_ASYNC_TASK_FIFO_DEPTH
#define QUANTIZED_ASYNC_TASK_FIFO_DEPTH 2
#endif
#ifndef QUANTIZED_ASYNC_ORDER_FIFO_DEPTH
#define QUANTIZED_ASYNC_ORDER_FIFO_DEPTH 4
#endif
#ifndef QUANTIZED_ASYNC_MATRIX_RESULT_FIFO_DEPTH
#ifdef QUANTIZED_ASYNC_RESULT_FIFO_DEPTH
#define QUANTIZED_ASYNC_MATRIX_RESULT_FIFO_DEPTH QUANTIZED_ASYNC_RESULT_FIFO_DEPTH
#else
#define QUANTIZED_ASYNC_MATRIX_RESULT_FIFO_DEPTH 128
#endif
#endif
#ifndef QUANTIZED_ASYNC_VECTOR_RESULT_FIFO_DEPTH
#ifdef QUANTIZED_ASYNC_RESULT_FIFO_DEPTH
#define QUANTIZED_ASYNC_VECTOR_RESULT_FIFO_DEPTH QUANTIZED_ASYNC_RESULT_FIFO_DEPTH
#else
#define QUANTIZED_ASYNC_VECTOR_RESULT_FIFO_DEPTH 16
#endif
#endif

static_assert(QUANTIZED_ASYNC_TASK_FIFO_DEPTH > 0,
              "async task FIFO must be finite and non-zero");
static_assert(QUANTIZED_ASYNC_ORDER_FIFO_DEPTH > 0,
              "async order FIFO must be finite and non-zero");
static_assert(QUANTIZED_ASYNC_MATRIX_RESULT_FIFO_DEPTH > 0,
              "async matrix result FIFO must be finite and non-zero");
static_assert(QUANTIZED_ASYNC_VECTOR_RESULT_FIFO_DEPTH > 0,
              "async vector result FIFO must be finite and non-zero");

#endif
