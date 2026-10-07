#ifndef LLM_FPGA_QUANTIZED_W8_PROJECTION_ENGINE_HPP
#define LLM_FPGA_QUANTIZED_W8_PROJECTION_ENGINE_HPP

#include "quantized_layer_schedule.hpp"
#include "quantized_vector_packets.hpp"
#include "quantized_resident_buffers.hpp"

#include <hls_stream.h>

using quantized_w8_activation_word_t =
    mm_stream_4x128_int8x8_activation_word_t;
using quantized_w8_weight_word_t = mm_stream_4x128_int8x8_weight_word_t;
using quantized_w8_output_word_t = mm_stream_4x128_int8x8_output_word_t;

// Internal bundles keep the 24 AXIS channels and 16 independent HBM weight
// ports manageable. The top-level kernel still exposes every port separately.
struct quantized_w8_projection_streams_t {
#ifdef __SYNTHESIS__
    hls::stream<mm_stream_quantized_task_word_t>* task0;
    hls::stream<mm_stream_quantized_task_word_t>* task1;
    hls::stream<mm_stream_quantized_task_word_t>* task2;
    hls::stream<mm_stream_quantized_task_word_t>* task3;
    hls::stream<quantized_w8_activation_word_t>* activation0;
    hls::stream<quantized_w8_activation_word_t>* activation1;
    hls::stream<quantized_w8_activation_word_t>* activation2;
    hls::stream<quantized_w8_activation_word_t>* activation3;
    hls::stream<quantized_w8_weight_word_t>* weight0a;
    hls::stream<quantized_w8_weight_word_t>* weight0b;
    hls::stream<quantized_w8_weight_word_t>* weight0c;
    hls::stream<quantized_w8_weight_word_t>* weight0d;
    hls::stream<quantized_w8_weight_word_t>* weight1a;
    hls::stream<quantized_w8_weight_word_t>* weight1b;
    hls::stream<quantized_w8_weight_word_t>* weight1c;
    hls::stream<quantized_w8_weight_word_t>* weight1d;
    hls::stream<quantized_w8_weight_word_t>* weight2a;
    hls::stream<quantized_w8_weight_word_t>* weight2b;
    hls::stream<quantized_w8_weight_word_t>* weight2c;
    hls::stream<quantized_w8_weight_word_t>* weight2d;
    hls::stream<quantized_w8_weight_word_t>* weight3a;
    hls::stream<quantized_w8_weight_word_t>* weight3b;
    hls::stream<quantized_w8_weight_word_t>* weight3c;
    hls::stream<quantized_w8_weight_word_t>* weight3d;
    hls::stream<quantized_w8_output_word_t>* output0;
    hls::stream<quantized_w8_output_word_t>* output1;
    hls::stream<quantized_w8_output_word_t>* output2;
    hls::stream<quantized_w8_output_word_t>* output3;
#else
    hls::stream<mm_stream_quantized_task_word_t>* task[4];
    hls::stream<quantized_w8_activation_word_t>* activation[4];
    hls::stream<quantized_w8_weight_word_t>* weight[4][4];
    hls::stream<quantized_w8_output_word_t>* output[4];
#endif
    quantized_vector_streams_t vector;
};

struct quantized_w8_weight_memories_t {
#ifdef __SYNTHESIS__
    const quantized_w8_weight_word_t* weight0a;
    const quantized_w8_weight_word_t* weight0b;
    const quantized_w8_weight_word_t* weight0c;
    const quantized_w8_weight_word_t* weight0d;
    const quantized_w8_weight_word_t* weight1a;
    const quantized_w8_weight_word_t* weight1b;
    const quantized_w8_weight_word_t* weight1c;
    const quantized_w8_weight_word_t* weight1d;
    const quantized_w8_weight_word_t* weight2a;
    const quantized_w8_weight_word_t* weight2b;
    const quantized_w8_weight_word_t* weight2c;
    const quantized_w8_weight_word_t* weight2d;
    const quantized_w8_weight_word_t* weight3a;
    const quantized_w8_weight_word_t* weight3b;
    const quantized_w8_weight_word_t* weight3c;
    const quantized_w8_weight_word_t* weight3d;
#else
    const quantized_w8_weight_word_t* weight[4][4];
#endif
};

inline hls::stream<mm_stream_quantized_task_word_t>&
quantized_w8_task_stream(quantized_w8_projection_streams_t& streams,
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

inline void quantized_w8_write_task(
    quantized_w8_projection_streams_t& streams,
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

inline hls::stream<quantized_w8_activation_word_t>&
quantized_w8_activation_stream(quantized_w8_projection_streams_t& streams,
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

inline void quantized_w8_write_activation(
    quantized_w8_projection_streams_t& streams,
    unsigned int cu,
    const quantized_w8_activation_word_t& value) {
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

inline hls::stream<quantized_w8_weight_word_t>&
quantized_w8_weight_stream(quantized_w8_projection_streams_t& streams,
                           unsigned int cu, unsigned int port) {
    #pragma HLS inline
#ifdef __SYNTHESIS__
    if (cu == 0) {
        switch (port) {
            case 0: return *streams.weight0a;
            case 1: return *streams.weight0b;
            case 2: return *streams.weight0c;
            default: return *streams.weight0d;
        }
    }
    if (cu == 1) {
        switch (port) {
            case 0: return *streams.weight1a;
            case 1: return *streams.weight1b;
            case 2: return *streams.weight1c;
            default: return *streams.weight1d;
        }
    }
    if (cu == 2) {
        switch (port) {
            case 0: return *streams.weight2a;
            case 1: return *streams.weight2b;
            case 2: return *streams.weight2c;
            default: return *streams.weight2d;
        }
    }
    switch (port) {
        case 0: return *streams.weight3a;
        case 1: return *streams.weight3b;
        case 2: return *streams.weight3c;
        default: return *streams.weight3d;
    }
#else
    return *streams.weight[cu][port];
#endif
}

inline void quantized_w8_write_weight(
    quantized_w8_projection_streams_t& streams,
    unsigned int cu,
    unsigned int port,
    const quantized_w8_weight_word_t& value) {
    #pragma HLS inline
#ifdef __SYNTHESIS__
    if (cu == 0) {
        if (port == 0) streams.weight0a->write(value);
        else if (port == 1) streams.weight0b->write(value);
        else if (port == 2) streams.weight0c->write(value);
        else streams.weight0d->write(value);
    } else if (cu == 1) {
        if (port == 0) streams.weight1a->write(value);
        else if (port == 1) streams.weight1b->write(value);
        else if (port == 2) streams.weight1c->write(value);
        else streams.weight1d->write(value);
    } else if (cu == 2) {
        if (port == 0) streams.weight2a->write(value);
        else if (port == 1) streams.weight2b->write(value);
        else if (port == 2) streams.weight2c->write(value);
        else streams.weight2d->write(value);
    } else {
        if (port == 0) streams.weight3a->write(value);
        else if (port == 1) streams.weight3b->write(value);
        else if (port == 2) streams.weight3c->write(value);
        else streams.weight3d->write(value);
    }
#else
    streams.weight[cu][port]->write(value);
#endif
}

inline hls::stream<quantized_w8_output_word_t>&
quantized_w8_output_stream(quantized_w8_projection_streams_t& streams,
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
void quantized_csim_feedback(quantized_w8_projection_streams_t&, unsigned int);
#endif

inline quantized_w8_output_word_t quantized_w8_read_output(
    quantized_w8_projection_streams_t& streams,
    unsigned int cu) {
    #pragma HLS inline
    quantized_w8_output_word_t value = 0;
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

inline quantized_w8_weight_word_t read_quantized_w8_weight(
    const quantized_w8_weight_memories_t& memories,
    unsigned int cu,
    unsigned int stream,
    std::size_t index) {
    #pragma HLS inline
#ifdef __SYNTHESIS__
    if (cu == 0) {
        switch (stream) {
            case 0: return memories.weight0a[index];
            case 1: return memories.weight0b[index];
            case 2: return memories.weight0c[index];
            default: return memories.weight0d[index];
        }
    }
    if (cu == 1) {
        switch (stream) {
            case 0: return memories.weight1a[index];
            case 1: return memories.weight1b[index];
            case 2: return memories.weight1c[index];
            default: return memories.weight1d[index];
        }
    }
    if (cu == 2) {
        switch (stream) {
            case 0: return memories.weight2a[index];
            case 1: return memories.weight2b[index];
            case 2: return memories.weight2c[index];
            default: return memories.weight2d[index];
        }
    }
    switch (stream) {
        case 0: return memories.weight3a[index];
        case 1: return memories.weight3b[index];
        case 2: return memories.weight3c[index];
        default: return memories.weight3d[index];
    }
#else
    return memories.weight[cu][stream][index];
#endif
}

#include "quantized_projection_dataflow.hpp"

struct quantized_w8_projection_policy {
    static constexpr unsigned int token_rows = 4;
    using weight_word_t = quantized_w8_weight_word_t;
    using activation_word_t = quantized_w8_activation_word_t;
    using output_word_t = quantized_w8_output_word_t;
    using streams_t = quantized_w8_projection_streams_t;
    using memories_t = quantized_w8_weight_memories_t;
    static constexpr unsigned int weight_ports = QUANTIZED_W8_WEIGHT_STREAMS_PER_CU;
    static constexpr unsigned int packets_per_wave =
        MM_STREAM_4X128_INT8X8_TOKENS * MM_STREAM_4X128_INT8X8_OUTPUT_GROUPS;

    static weight_word_t read_weight(const memories_t& memories, unsigned int cu,
                                     unsigned int port, std::size_t index) {
        #pragma HLS inline
        return read_quantized_w8_weight(memories, cu, port, index);
    }
    static void write_task(streams_t& streams, unsigned int cu,
                           mm_stream_quantized_task_word_t task) {
        #pragma HLS inline
        quantized_w8_write_task(streams, cu, task);
    }
    static void write_activation(streams_t& streams, unsigned int cu,
                                 activation_word_t word) {
        #pragma HLS inline
        quantized_w8_write_activation(streams, cu, word);
    }
    static void write_weight(streams_t& streams, unsigned int cu,
                             unsigned int port, weight_word_t word) {
        #pragma HLS inline
        quantized_w8_write_weight(streams, cu, port, word);
    }
    static output_word_t read_output(streams_t& streams, unsigned int cu) {
        #pragma HLS inline
        return quantized_w8_read_output(streams, cu);
    }
    template <typename Source>
    static activation_word_t pack_activation(const Source& source, unsigned int k,
                    unsigned int rows, quant_inverse_scale_t inverse) {
        #pragma HLS inline
        return pack_quantized_w8_activation_word(source, k, rows, inverse);
    }
    template <typename Destination>
    static void store_output(Destination& destination, output_word_t word,
                    unsigned int rows, unsigned int output_dim,
                    quant_scale_t activation_scale, quant_scale_t weight_scale) {
        #pragma HLS inline
        store_quantized_w8_aligned_output_word(destination, word, rows, output_dim,
                                        activation_scale, weight_scale);
    }
};

template <unsigned int SOURCE_BLOCKS, unsigned int DESTINATION_BLOCKS>
void run_quantized_w8_projection_wave(
    const quantized_layer_task_t& layer_task,
    quantized_projection_t projection_kind,
    unsigned int wave, bool final_physical_block,
    const quantized_w8_feature_buffer_t<SOURCE_BLOCKS>& source,
    quantized_w8_feature_buffer_t<DESTINATION_BLOCKS>& destination,
    quant_scale_t activation_scale, quant_inverse_scale_t activation_inverse_scale,
    quant_scale_t weight_scale, quantized_w8_projection_streams_t& streams,
    const quantized_w8_weight_memories_t& memories) {
    #pragma HLS inline off
    run_quantized_projection_wave_range_overlapped<quantized_w8_projection_policy>(
        layer_task, projection_kind, final_physical_block, source, destination,
        activation_scale, activation_inverse_scale, weight_scale, streams,
        memories, wave, wave + 1);
}

template <unsigned int SOURCE_BLOCKS, unsigned int DESTINATION_BLOCKS>
void run_quantized_w8_projection(
    const quantized_layer_task_t& layer_task,
    quantized_projection_t projection_kind, bool final_physical_block,
    const quantized_w8_feature_buffer_t<SOURCE_BLOCKS>& source,
    quantized_w8_feature_buffer_t<DESTINATION_BLOCKS>& destination,
    quant_scale_t activation_scale, quant_inverse_scale_t activation_inverse_scale,
    quant_scale_t weight_scale, quantized_w8_projection_streams_t& streams,
    const quantized_w8_weight_memories_t& memories) {
    #pragma HLS inline off
    const auto plan = get_quantized_projection_plan(projection_kind);
    run_quantized_projection_wave_range_overlapped<quantized_w8_projection_policy>(
        layer_task, projection_kind, final_physical_block, source, destination,
        activation_scale, activation_inverse_scale, weight_scale, streams,
        memories, 0, plan.wave_count);
}

#endif
