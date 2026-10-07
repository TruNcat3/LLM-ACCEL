#ifndef LLM_FPGA_QUANTIZED_VECTOR_SCHEDULE_HPP
#define LLM_FPGA_QUANTIZED_VECTOR_SCHEDULE_HPP

#include "quantized_vector_packets.hpp"

#ifndef QUANTIZED_DECODE_FFN_OVERLAP
#define QUANTIZED_DECODE_FFN_OVERLAP 0
#endif
#ifndef QUANTIZED_PREFILL_FFN_OVERLAP
#define QUANTIZED_PREFILL_FFN_OVERLAP 0
#endif

// Controller scheduling is separate from the shared compute stream ABI.
inline unsigned int quantized_vector_dispatch_rows_for_cu(
    unsigned int physical_rows, unsigned int valid_rows, unsigned int cu) {
    #pragma HLS inline
    const unsigned int rows = valid_rows < physical_rows ? valid_rows : physical_rows;
    if (cu >= 4 || cu >= rows) return 0;
    return (rows - cu + 3) / 4;
}

inline bool quantized_vector_split_columns(
    unsigned int valid_rows, mm_stream_quantized_compute_mode_t mode) {
    #pragma HLS inline
    // Elementwise operations need no cross-column reduction. RMSNorm keeps
    // the complete row and its original arithmetic order on one CU.
    return valid_rows == 1 && (mode == MM_STREAM_QUANTIZED_MODE_SILU_MUL ||
                              mode == MM_STREAM_QUANTIZED_MODE_RESIDUAL_ADD
#if QUANTIZED_DECODE_FFN_OVERLAP || QUANTIZED_PREFILL_FFN_OVERLAP
                              || mode == MM_STREAM_QUANTIZED_MODE_SILU_ONLY ||
                              mode == MM_STREAM_QUANTIZED_MODE_MUL
#endif
                              );
}

inline unsigned int quantized_vector_blocks_for_cu(
    unsigned int elements, unsigned int cu) {
    #pragma HLS inline
    const unsigned int blocks = ceildiv(elements, CU_VEC_LANES);
    return cu < 4 && cu < blocks ? (blocks - cu + 3) / 4 : 0;
}

inline unsigned int quantized_vector_task_rows(
    unsigned int physical_rows, unsigned int valid_rows, unsigned int cu,
    mm_stream_quantized_compute_mode_t mode, unsigned int elements) {
    #pragma HLS inline
    return quantized_vector_split_columns(valid_rows, mode) ?
        (quantized_vector_blocks_for_cu(elements, cu) != 0 ? 1 : 0) :
        quantized_vector_dispatch_rows_for_cu(physical_rows, valid_rows, cu);
}

inline unsigned int quantized_vector_layer_tasks_for_cu(
    unsigned int physical_rows, unsigned int valid_rows, unsigned int cu) {
    #pragma HLS inline
    return 2 * (quantized_vector_task_rows(physical_rows, valid_rows, cu,
                    MM_STREAM_QUANTIZED_MODE_RMSNORM, HIDDEN_SIZE) != 0) +
           2 * (quantized_vector_task_rows(physical_rows, valid_rows, cu,
                    MM_STREAM_QUANTIZED_MODE_RESIDUAL_ADD, HIDDEN_SIZE) != 0) +
               (quantized_vector_task_rows(physical_rows, valid_rows, cu,
                    MM_STREAM_QUANTIZED_MODE_SILU_MUL, INTERMEDIATE_SIZE) != 0);
}

#endif
