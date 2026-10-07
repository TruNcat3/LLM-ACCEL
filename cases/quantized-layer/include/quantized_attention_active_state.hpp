#ifndef LLM_FPGA_QUANTIZED_ATTENTION_ACTIVE_STATE_HPP
#define LLM_FPGA_QUANTIZED_ATTENTION_ACTIVE_STATE_HPP

// This header is included after quantized_attention_dataflow.hpp.  The state
// layout and packed destination helpers are provided by that dependency.

template <typename Policy>
void initialize_quantized_attention_active_state(
    typename Policy::state_t& state, unsigned int rows) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=state.context cyclic factor=16 dim=3
    const unsigned int active_rows = rows < Policy::token_rows ?
        rows : Policy::token_rows;
    for (unsigned int row = 0; row < active_rows; ++row) {
        for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
            state.maximum[row][head] = fm_t(-128);
            state.sum[row][head] = 0;
            for (unsigned int block = 0;
                 block < (HEAD_DIM + 15) / 16; ++block) {
                #pragma HLS pipeline II=1
                for (unsigned int lane = 0; lane < 16; ++lane) {
                    #pragma HLS unroll
                    if (block * 16 + lane < HEAD_DIM)
                        state.context[row][head][block * 16 + lane] = 0;
                }
            }
        }
    }
}

template <typename Policy>
void finalize_quantized_attention_active_state(
    typename Policy::hidden_t& destination,
    const typename Policy::state_t& state, unsigned int rows) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=state.context cyclic factor=16 dim=3
    static_assert(HEAD_DIM % 16 == 0,
                  "resident heads must align to vector words");
    const unsigned int active_rows = rows < Policy::token_rows ?
        rows : Policy::token_rows;

    // Only this loop reads state.  Keeping invalid rows in a separate loop
    // prevents stale state from being consumed when rows is a short tile.
    for (unsigned int row = 0; row < active_rows; ++row) {
        for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
            const fm_t inverse = Policy::reciprocal(state.sum[row][head]);
            for (unsigned int block = 0; block < HEAD_DIM / 16; ++block) {
                #pragma HLS pipeline II=1
                mm_input_block_t word = 0;
                for (unsigned int lane = 0; lane < 16; ++lane) {
                    #pragma HLS unroll
                    const fm_t value = fm_t(
                        state.context[row][head][block * 16 + lane] *
                        fm_accum_t(inverse));
                    set_mm_input_block_lane(word, lane, value);
                }
                destination.block[row][head * (HEAD_DIM / 16) + block] = word;
            }
        }
    }

    // Preserve the reference contract for resident rows that are not active:
    // every packed output word is explicitly zeroed without touching state.
    for (unsigned int row = active_rows; row < Policy::token_rows; ++row) {
        for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
            for (unsigned int block = 0; block < HEAD_DIM / 16; ++block) {
                #pragma HLS pipeline II=1
                destination.block[row][head * (HEAD_DIM / 16) + block] =
                    mm_input_block_t(0);
            }
        }
    }
}

#endif
