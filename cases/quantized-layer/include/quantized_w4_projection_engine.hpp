#ifndef LLM_FPGA_QUANTIZED_W4_PROJECTION_ENGINE_HPP
#define LLM_FPGA_QUANTIZED_W4_PROJECTION_ENGINE_HPP

#include "quantized_layer_schedule.hpp"
#include "quantized_vector_packets.hpp"
#include "quantized_w4_resident_buffers.hpp"

#include <hls_stream.h>

using quantized_w4_activation_word_t =
    mm_stream_8x128_int4x4_activation_word_t;
using quantized_w4_weight_word_t = mm_stream_8x128_int4x4_weight_word_t;
using quantized_w4_output_word_t = mm_stream_8x128_int4x4_output_word_t;

constexpr unsigned int QUANTIZED_W4_WEIGHT_STREAMS_PER_CU = 2;
constexpr unsigned int QUANTIZED_W4_WEIGHTS_PER_STREAM_WORD =
    MM_STREAM_8X128_INT4X4_WEIGHTS_PER_STREAM;

// Internal bundles keep the AXIS channels and eight independent HBM weight
// ports manageable. The top-level kernel still exposes every port separately.
struct quantized_w4_projection_streams_t {
#ifdef __SYNTHESIS__
    hls::stream<mm_stream_quantized_task_word_t>* task0;
    hls::stream<mm_stream_quantized_task_word_t>* task1;
    hls::stream<mm_stream_quantized_task_word_t>* task2;
    hls::stream<mm_stream_quantized_task_word_t>* task3;
    hls::stream<quantized_w4_activation_word_t>* activation0;
    hls::stream<quantized_w4_activation_word_t>* activation1;
    hls::stream<quantized_w4_activation_word_t>* activation2;
    hls::stream<quantized_w4_activation_word_t>* activation3;
    hls::stream<quantized_w4_weight_word_t>* weight0a;
    hls::stream<quantized_w4_weight_word_t>* weight0b;
    hls::stream<quantized_w4_weight_word_t>* weight1a;
    hls::stream<quantized_w4_weight_word_t>* weight1b;
    hls::stream<quantized_w4_weight_word_t>* weight2a;
    hls::stream<quantized_w4_weight_word_t>* weight2b;
    hls::stream<quantized_w4_weight_word_t>* weight3a;
    hls::stream<quantized_w4_weight_word_t>* weight3b;
    hls::stream<quantized_w4_output_word_t>* output0;
    hls::stream<quantized_w4_output_word_t>* output1;
    hls::stream<quantized_w4_output_word_t>* output2;
    hls::stream<quantized_w4_output_word_t>* output3;
#else
    hls::stream<mm_stream_quantized_task_word_t>* task[4];
    hls::stream<quantized_w4_activation_word_t>* activation[4];
    hls::stream<quantized_w4_weight_word_t>* weight[4][2];
    hls::stream<quantized_w4_output_word_t>* output[4];
#endif
    quantized_vector_streams_t vector;
};

struct quantized_w4_weight_memories_t {
#ifdef __SYNTHESIS__
    const quantized_w4_weight_word_t* weight0a;
    const quantized_w4_weight_word_t* weight0b;
    const quantized_w4_weight_word_t* weight1a;
    const quantized_w4_weight_word_t* weight1b;
    const quantized_w4_weight_word_t* weight2a;
    const quantized_w4_weight_word_t* weight2b;
    const quantized_w4_weight_word_t* weight3a;
    const quantized_w4_weight_word_t* weight3b;
#else
    const quantized_w4_weight_word_t* weight[4][2];
#endif
};

inline hls::stream<mm_stream_quantized_task_word_t>&
quantized_w4_task_stream(quantized_w4_projection_streams_t& streams,
                         unsigned int cu) {
    #pragma HLS inline
#ifdef __SYNTHESIS__
    switch (cu) {
        case 0: return *streams.task0;
        case 1: return *streams.task1;
        case 2: return *streams.task2;
        default: return *streams.task3;
    }
#else
    return *streams.task[cu];
#endif
}

inline void quantized_w4_write_task(
    quantized_w4_projection_streams_t& streams,
    unsigned int cu,
    const mm_stream_quantized_task_word_t& value) {
    #pragma HLS inline
#ifdef __SYNTHESIS__
    if (cu == 0) streams.task0->write(value);
    else if (cu == 1) streams.task1->write(value);
    else if (cu == 2) streams.task2->write(value);
    else streams.task3->write(value);
#else
    streams.task[cu]->write(value);
#endif
}

inline hls::stream<quantized_w4_activation_word_t>&
quantized_w4_activation_stream(quantized_w4_projection_streams_t& streams,
                               unsigned int cu) {
    #pragma HLS inline
#ifdef __SYNTHESIS__
    switch (cu) {
        case 0: return *streams.activation0;
        case 1: return *streams.activation1;
        case 2: return *streams.activation2;
        default: return *streams.activation3;
    }
#else
    return *streams.activation[cu];
#endif
}

inline void quantized_w4_write_activation(
    quantized_w4_projection_streams_t& streams,
    unsigned int cu,
    const quantized_w4_activation_word_t& value) {
    #pragma HLS inline
#ifdef __SYNTHESIS__
    if (cu == 0) streams.activation0->write(value);
    else if (cu == 1) streams.activation1->write(value);
    else if (cu == 2) streams.activation2->write(value);
    else streams.activation3->write(value);
#else
    streams.activation[cu]->write(value);
#endif
}

inline hls::stream<quantized_w4_weight_word_t>&
quantized_w4_weight_stream(quantized_w4_projection_streams_t& streams,
                           unsigned int cu, unsigned int port) {
    #pragma HLS inline
#ifdef __SYNTHESIS__
    if (cu == 0) {
        return port == 0 ? *streams.weight0a : *streams.weight0b;
    }
    if (cu == 1) {
        return port == 0 ? *streams.weight1a : *streams.weight1b;
    }
    if (cu == 2) {
        return port == 0 ? *streams.weight2a : *streams.weight2b;
    }
    return port == 0 ? *streams.weight3a : *streams.weight3b;
#else
    return *streams.weight[cu][port];
#endif
}

inline void quantized_w4_write_weight(
    quantized_w4_projection_streams_t& streams,
    unsigned int cu,
    unsigned int port,
    const quantized_w4_weight_word_t& value) {
    #pragma HLS inline
#ifdef __SYNTHESIS__
    if (cu == 0) {
        if (port == 0) streams.weight0a->write(value);
        else streams.weight0b->write(value);
    } else if (cu == 1) {
        if (port == 0) streams.weight1a->write(value);
        else streams.weight1b->write(value);
    } else if (cu == 2) {
        if (port == 0) streams.weight2a->write(value);
        else streams.weight2b->write(value);
    } else {
        if (port == 0) streams.weight3a->write(value);
        else streams.weight3b->write(value);
    }
#else
    streams.weight[cu][port]->write(value);
#endif
}

inline hls::stream<quantized_w4_output_word_t>&
quantized_w4_output_stream(quantized_w4_projection_streams_t& streams,
                           unsigned int cu) {
    #pragma HLS inline
#ifdef __SYNTHESIS__
    switch (cu) {
        case 0: return *streams.output0;
        case 1: return *streams.output1;
        case 2: return *streams.output2;
        default: return *streams.output3;
    }
#else
    return *streams.output[cu];
#endif
}

#if defined(QUANTIZED_CSIM_FEEDBACK) && !defined(__SYNTHESIS__)
// Test-only cooperative execution: run a real compute task when C simulation
// reaches its result read. RTL uses independently running finite-FIFO CUs.
void quantized_csim_feedback(quantized_w4_projection_streams_t&, unsigned int);
#endif

inline quantized_w4_output_word_t quantized_w4_read_output(
    quantized_w4_projection_streams_t& streams,
    unsigned int cu) {
    #pragma HLS inline
    quantized_w4_output_word_t value = 0;
#ifdef __SYNTHESIS__
    if (cu == 0) value = streams.output0->read();
    else if (cu == 1) value = streams.output1->read();
    else if (cu == 2) value = streams.output2->read();
    else value = streams.output3->read();
#else
    #ifdef QUANTIZED_CSIM_FEEDBACK
    if (streams.output[cu]->empty()) quantized_csim_feedback(streams, cu);
    #endif
    value = streams.output[cu]->read();
#endif
    return value;
}

inline quantized_w4_weight_word_t read_quantized_w4_weight(
    const quantized_w4_weight_memories_t& memories,
    unsigned int cu,
    unsigned int stream,
    std::size_t index) {
    #pragma HLS inline
#ifdef __SYNTHESIS__
    if (cu == 0) {
        return stream == 0 ? memories.weight0a[index] : memories.weight0b[index];
    }
    if (cu == 1) {
        return stream == 0 ? memories.weight1a[index] : memories.weight1b[index];
    }
    if (cu == 2) {
        return stream == 0 ? memories.weight2a[index] : memories.weight2b[index];
    }
    return stream == 0 ? memories.weight3a[index] : memories.weight3b[index];
#else
    return memories.weight[cu][stream][index];
#endif
}

#include "quantized_projection_dataflow.hpp"

struct quantized_w4_projection_policy {
    static constexpr unsigned int token_rows = 8;
    using weight_word_t = quantized_w4_weight_word_t;
    using activation_word_t = quantized_w4_activation_word_t;
    using output_word_t = quantized_w4_output_word_t;
    using streams_t = quantized_w4_projection_streams_t;
    using memories_t = quantized_w4_weight_memories_t;
    static constexpr unsigned int weight_ports = QUANTIZED_W4_WEIGHT_STREAMS_PER_CU;
    static constexpr unsigned int packets_per_wave =
        MM_STREAM_8X128_INT4X4_TOKENS * MM_STREAM_8X128_INT4X4_OUTPUT_GROUPS;

    static weight_word_t read_weight(const memories_t& memories, unsigned int cu,
                                     unsigned int port, std::size_t index) {
        #pragma HLS inline
        return read_quantized_w4_weight(memories, cu, port, index);
    }
    static void write_task(streams_t& streams, unsigned int cu,
                           mm_stream_quantized_task_word_t task) {
        #pragma HLS inline
        quantized_w4_write_task(streams, cu, task);
    }
    static void write_activation(streams_t& streams, unsigned int cu,
                                 activation_word_t word) {
        #pragma HLS inline
        quantized_w4_write_activation(streams, cu, word);
    }
    static void write_weight(streams_t& streams, unsigned int cu,
                             unsigned int port, weight_word_t word) {
        #pragma HLS inline
        quantized_w4_write_weight(streams, cu, port, word);
    }
    static output_word_t read_output(streams_t& streams, unsigned int cu) {
        #pragma HLS inline
        return quantized_w4_read_output(streams, cu);
    }
    template <typename Source>
    static activation_word_t pack_activation(const Source& source, unsigned int k,
                    unsigned int rows, quant_inverse_scale_t inverse) {
        #pragma HLS inline
        return pack_quantized_w4_activation_word(source, k, rows, inverse);
    }
    template <typename Destination>
    static void store_output(Destination& destination, output_word_t word,
                    unsigned int rows, unsigned int output_dim,
                    quant_scale_t activation_scale, quant_scale_t weight_scale) {
        #pragma HLS inline
        store_quantized_w4_aligned_output_word(destination, word, rows, output_dim,
                                        activation_scale, weight_scale);
    }
};

template <unsigned int SOURCE_BLOCKS, unsigned int DESTINATION_BLOCKS>
void run_quantized_w4_projection_wave(
    const quantized_layer_task_t& layer_task,
    quantized_projection_t projection_kind,
    unsigned int wave, bool final_physical_block,
    const quantized_w4_feature_buffer_t<SOURCE_BLOCKS>& source,
    quantized_w4_feature_buffer_t<DESTINATION_BLOCKS>& destination,
    quant_scale_t activation_scale, quant_inverse_scale_t activation_inverse_scale,
    quant_scale_t weight_scale, quantized_w4_projection_streams_t& streams,
    const quantized_w4_weight_memories_t& memories) {
    #pragma HLS inline off
    run_quantized_projection_wave_range_overlapped<quantized_w4_projection_policy>(
        layer_task, projection_kind, final_physical_block, source, destination,
        activation_scale, activation_inverse_scale, weight_scale, streams,
        memories, wave, wave + 1);
}

template <unsigned int SOURCE_BLOCKS, unsigned int DESTINATION_BLOCKS>
void run_quantized_w4_projection(
    const quantized_layer_task_t& layer_task,
    quantized_projection_t projection_kind, bool final_physical_block,
    const quantized_w4_feature_buffer_t<SOURCE_BLOCKS>& source,
    quantized_w4_feature_buffer_t<DESTINATION_BLOCKS>& destination,
    quant_scale_t activation_scale, quant_inverse_scale_t activation_inverse_scale,
    quant_scale_t weight_scale, quantized_w4_projection_streams_t& streams,
    const quantized_w4_weight_memories_t& memories) {
    #pragma HLS inline off
    const auto plan = get_quantized_projection_plan(projection_kind);
    run_quantized_projection_wave_range_overlapped<quantized_w4_projection_policy>(
        layer_task, projection_kind, final_physical_block, source, destination,
        activation_scale, activation_inverse_scale, weight_scale, streams,
        memories, 0, plan.wave_count);
}

#endif
