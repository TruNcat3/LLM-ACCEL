#ifndef LLM_FPGA_QUANTIZED_ATTENTION_NONLINEAR_HPP
#define LLM_FPGA_QUANTIZED_ATTENTION_NONLINEAR_HPP

// Included after the common result-panel accessors and scalar reference path.
// Heads are independent; columns within a head keep their reference sum order.
template <typename Policy>
void update_quantized_attention_wave_parallel(
    typename Policy::state_t& state, typename Policy::probability_t& probability,
    attention_prob_t old_scale[Policy::token_rows][NUM_ATTENTION_HEADS],
    const quantized_attention_result_panel_t<Policy>& scores,
    const quantized_layer_task_t& task, unsigned int wave,
    unsigned int tile_begin, unsigned int length,
    fm_t probability_maximum[NUM_ATTENTION_HEADS]) {
    #pragma HLS inline off
    constexpr unsigned int lanes = QUANTIZED_ATTENTION_HEAD_LANES;
    #pragma HLS array_partition variable=probability_maximum complete
#if QUANTIZED_ATTENTION_HEAD_LANES > 1
    #pragma HLS array_partition variable=probability.value cyclic factor=QUANTIZED_ATTENTION_HEAD_LANES dim=2
#endif
#if QUANTIZED_ATTENTION_PACK_LANES > 1
    #pragma HLS array_partition variable=probability.value cyclic factor=QUANTIZED_ATTENTION_PACK_LANES dim=3
#endif
    for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
        #pragma HLS pipeline II=1
        probability_maximum[head] = 0;
    }
    const unsigned int columns = length < 128 ? length : 128;
    for (unsigned int row = 0; row < task.query_tokens; ++row) {
        const unsigned int position = task.position + row;
        const unsigned int causal = tile_begin > position ? 0 :
            (position - tile_begin < columns ? position - tile_begin + 1 : columns);
        for (unsigned int base = 0; base < NUM_ATTENTION_HEADS; base += lanes) {
            unsigned int heads[lanes];
            bool active[lanes];
            fm_t maximum[lanes], pmaximum[lanes];
            fm_accum_t sum[lanes];
            attention_prob_t scale[lanes];
            #pragma HLS array_partition variable=heads complete
            #pragma HLS array_partition variable=active complete
            #pragma HLS array_partition variable=maximum complete
            #pragma HLS array_partition variable=pmaximum complete
            #pragma HLS array_partition variable=sum complete
            #pragma HLS array_partition variable=scale complete
            bool any = false;
            for (unsigned int lane = 0; lane < lanes; ++lane) {
                #pragma HLS unroll
                const unsigned int index = base + lane;
#if QUANTIZED_ATTENTION_HEAD_LANES > 1
                const unsigned int head = (index % NUM_KEY_VALUE_HEADS) * GQA_GROUP_SIZE +
                    index / NUM_KEY_VALUE_HEADS;
#else
                const unsigned int head = index;
#endif
                heads[lane] = head;
                active[lane] = index < NUM_ATTENTION_HEADS &&
                    quantized_attention_head_in_wave<Policy::token_rows>(task, wave, head);
                any |= active[lane];
                maximum[lane] = fm_t(-128);
                pmaximum[lane] = active[lane] ? probability_maximum[head] : fm_t(0);
                sum[lane] = 0;
            }
            if (!any) continue;
            for (unsigned int col = 0; col < causal; ++col) {
                #pragma HLS pipeline II=1
                #pragma HLS loop_tripcount min=0 max=128
                for (unsigned int lane = 0; lane < lanes; ++lane) {
                    #pragma HLS unroll
                    if (active[lane]) {
                        const fm_t value = quantized_attention_result_value<Policy>(
                            scores, task, row, heads[lane], col);
                        if (value > maximum[lane]) maximum[lane] = value;
                    }
                }
            }
            for (unsigned int lane = 0; lane < lanes; ++lane) {
                #pragma HLS unroll
                if (active[lane]) {
                    const unsigned int head = heads[lane];
                    const bool previous = state.sum[row][head] != fm_accum_t(0);
                    if (previous && state.maximum[row][head] > maximum[lane])
                        maximum[lane] = state.maximum[row][head];
                    scale[lane] = previous ? Policy::exp(state.maximum[row][head] - maximum[lane]) :
                        attention_prob_t(0);
                }
            }
            for (unsigned int col = 0; col < columns; ++col) {
                #pragma HLS pipeline II=1
                #pragma HLS loop_tripcount min=0 max=128
                for (unsigned int lane = 0; lane < lanes; ++lane) {
                    #pragma HLS unroll
                    if (active[lane]) {
                        const unsigned int head = heads[lane];
                        const attention_prob_t value = col < causal ? Policy::exp(
                            quantized_attention_result_value<Policy>(scores, task, row, head, col) -
                            maximum[lane]) : attention_prob_t(0);
                        probability.value[row][quantized_attention_probability_head(head)][col] = value;
                        sum[lane] += fm_accum_t(value);
#if QUANTIZED_ATTENTION_PROBABILITY_FUSE
                        // Compare after the identical conversion used by the
                        // independent reference probability-scale scan.
                        const fm_t converted = fm_t(value);
                        if (converted > pmaximum[lane]) pmaximum[lane] = converted;
#endif
                    }
                }
            }
            for (unsigned int lane = 0; lane < lanes; ++lane) {
                #pragma HLS unroll
                if (active[lane]) {
                    const unsigned int head = heads[lane];
                    old_scale[row][head] = scale[lane];
                    state.maximum[row][head] = maximum[lane];
                    state.sum[row][head] = state.sum[row][head] * fm_accum_t(scale[lane]) + sum[lane];
                    probability_maximum[head] = pmaximum[lane];
                }
            }
        }
    }
}

template <typename Policy>
void update_and_scale_quantized_attention_wave(
    typename Policy::state_t& state, typename Policy::probability_t& probability,
    attention_prob_t old_scale[Policy::token_rows][NUM_ATTENTION_HEADS],
    const quantized_attention_result_panel_t<Policy>& scores,
    const quantized_layer_task_t& task, unsigned int wave,
    unsigned int tile_begin, unsigned int length,
    quantized_symmetric_scale_t scales[NUM_ATTENTION_HEADS]) {
    #pragma HLS inline
#if QUANTIZED_ATTENTION_PROBABILITY_FUSE || QUANTIZED_ATTENTION_HEAD_LANES > 1
    fm_t maximum[NUM_ATTENTION_HEADS];
    #pragma HLS array_partition variable=maximum complete
    update_quantized_attention_wave_parallel<Policy>(state, probability, old_scale,
        scores, task, wave, tile_begin, length, maximum);
#else
    update_quantized_attention_wave<Policy>(state, probability, old_scale,
        scores, task, wave, tile_begin, length);
#endif
    for (unsigned int head = 0; head < NUM_ATTENTION_HEADS; ++head) {
        if (quantized_attention_head_in_wave<Policy::token_rows>(task, wave, head)) {
#if QUANTIZED_ATTENTION_PROBABILITY_FUSE
            scales[head] = Policy::scale_from_maximum(maximum[head]);
#else
            scales[head] = Policy::probability_scale(probability,
                quantized_attention_probability_head(head), task.query_tokens, length);
#endif
        }
    }
}

#endif
