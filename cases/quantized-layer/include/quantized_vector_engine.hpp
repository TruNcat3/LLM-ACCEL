#ifndef LLM_FPGA_QUANTIZED_VECTOR_ENGINE_HPP
#define LLM_FPGA_QUANTIZED_VECTOR_ENGINE_HPP

#include "quantized_vector_schedule.hpp"
#include "quantized_layer_schedule.hpp"

template <typename Buffer>
cu_vec16_packet_t quantized_vector_buffer_packet(
    const Buffer& buffer, unsigned int row, unsigned int block,
    unsigned int elements) {
    #pragma HLS inline
    cu_vec16_packet_t packet{};
    const mm_input_block_t word = buffer.block[row][block];
    packet.token_lane = row;
    packet.elem_base = block * CU_VEC_LANES;
    packet.block_id = block;
    packet.last_block = packet.elem_base + CU_VEC_LANES >= elements;
    packet.last_stream = packet.last_block;
    for (unsigned int lane = 0; lane < CU_VEC_LANES; ++lane) {
        #pragma HLS unroll
        packet.valid_mask[lane] = packet.elem_base + lane < elements;
        packet.data[lane].range(fm_t::width - 1, 0) =
            word.range((lane + 1) * fm_t::width - 1, lane * fm_t::width);
    }
    return packet;
}

template <typename Policy>
void emit_quantized_vector_tasks(
    typename Policy::streams_t& streams,
    const quantized_layer_task_t& layer_task,
    mm_stream_quantized_compute_mode_t mode, unsigned int elements) {
    #pragma HLS inline off
    const bool split_columns = quantized_vector_split_columns(layer_task.query_tokens, mode);
    for (unsigned int cu = 0; cu < 4; ++cu) {
        #pragma HLS unroll
        const unsigned int rows = quantized_vector_task_rows(
            Policy::token_rows, layer_task.query_tokens, cu, mode, elements);
        if (rows != 0) {
            mm_stream_quantized_task_t task{};
            task.compute_mode = mode;
            task.k_count = split_columns ?
                quantized_vector_blocks_for_cu(elements, cu) * CU_VEC_LANES : elements;
            task.valid_tokens = rows;
            task.request_position = layer_task.position;
            task.kv_context_length = layer_task.kv_context_length;
            task.phase = layer_task.phase;
            Policy::write_task(streams, cu, pack_mm_stream_quantized_task(task));
        }
    }
}

template <typename Policy, bool IssueTasks = true, typename Source0, typename Source1>
void emit_quantized_vector_operands(
    typename Policy::streams_t& streams,
    const Source0& source0, const Source1& source1,
    const quantized_layer_task_t& layer_task,
    mm_stream_quantized_compute_mode_t mode, unsigned int elements) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=source0.block complete dim=1
    #pragma HLS array_partition variable=source1.block complete dim=1
    const unsigned int slots = (layer_task.query_tokens + 3) / 4;
    const unsigned int blocks = ceildiv(elements, CU_VEC_LANES);
    const bool split_columns = quantized_vector_split_columns(layer_task.query_tokens, mode);
    // Task setup owns no RAM port. Finish its control transaction before
    // entering the packed operand loops, which retain their II=1 pipeline.
    if (IssueTasks)
        emit_quantized_vector_tasks<Policy>(streams, layer_task, mode, elements);
    if (split_columns) {
        // One packed word per cycle fits the resident BRAM port. Four scalar
        // nonlinear engines consume the striped words concurrently; no extra
        // compute hardware or wider inter-kernel stream is needed.
        for (unsigned int block = 0; block < blocks; ++block) {
            #pragma HLS pipeline II=1
            const unsigned int cu = block % 4;
            auto packet0 = quantized_vector_buffer_packet(source0, 0, block, elements);
            auto packet1 = quantized_vector_buffer_packet(source1, 0, block, elements);
            packet0.last_block = packet1.last_block = block + 4 >= blocks;
            packet0.last_stream = packet0.last_block;
            packet1.last_stream = packet1.last_block;
            write_quantized_vector_operand(streams.vector, cu, 0, pack_cu8_nk_vector(packet0));
            write_quantized_vector_operand(streams.vector, cu, 1, pack_cu8_nk_vector(packet1));
        }
        return;
    }
    // The reference RMSNorm CU consumes the shared norm weights before rows.
    if (mode == MM_STREAM_QUANTIZED_MODE_RMSNORM) {
        for (unsigned int block = 0; block < blocks; ++block) {
            #pragma HLS pipeline II=1
            const auto packet = quantized_vector_buffer_packet(
                source1, 0, block, elements);
            for (unsigned int cu = 0; cu < 4; ++cu) {
                #pragma HLS unroll
                if (quantized_vector_dispatch_rows_for_cu(
                        Policy::token_rows, layer_task.query_tokens, cu))
                    write_quantized_vector_operand(streams.vector, cu, 1,
                                                   pack_cu8_nk_vector(packet));
            }
        }
    }
    for (unsigned int slot = 0; slot < slots; ++slot) {
        #pragma HLS loop_flatten off
        for (unsigned int block = 0; block < blocks; ++block) {
            #pragma HLS pipeline II=1
            for (unsigned int cu = 0; cu < 4; ++cu) {
                #pragma HLS unroll
                const unsigned int row = slot * 4 + cu;
                if (row < layer_task.query_tokens) {
                    write_quantized_vector_operand(streams.vector, cu, 0,
                        pack_cu8_nk_vector(quantized_vector_buffer_packet(
                            source0, row, block, elements)));
                    if (mode != MM_STREAM_QUANTIZED_MODE_RMSNORM)
                        write_quantized_vector_operand(streams.vector, cu, 1,
                            pack_cu8_nk_vector(quantized_vector_buffer_packet(
                                source1, row, block, elements)));
                }
            }
        }
    }
}

template <typename Policy, typename Destination>
void collect_quantized_vector_results(
    Destination& destination, typename Policy::streams_t& streams,
    unsigned int valid_rows, unsigned int elements,
    mm_stream_quantized_compute_mode_t mode) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=destination.block complete dim=1
    const unsigned int slots = (valid_rows + 3) / 4;
    const unsigned int blocks = ceildiv(elements, CU_VEC_LANES);
    const bool split_columns = quantized_vector_split_columns(valid_rows, mode);
    if (split_columns) {
        for (unsigned int block = 0; block < blocks; ++block) {
            #pragma HLS pipeline II=1
            const auto packet = unpack_cu8_nk_vector(
                quantized_vector_word_t(Policy::read_output(streams, block % 4)));
            mm_input_block_t word = 0;
            for (unsigned int lane = 0; lane < CU_VEC_LANES; ++lane) {
                #pragma HLS unroll
                if (packet.valid_mask[lane] && block * CU_VEC_LANES + lane < elements)
                    word.range((lane + 1) * fm_t::width - 1, lane * fm_t::width) =
                        packet.data[lane].range(fm_t::width - 1, 0);
            }
            destination.block[0][block] = word;
        }
        return;
    }
    for (unsigned int slot = 0; slot < slots; ++slot) {
        #pragma HLS loop_flatten off
        for (unsigned int block = 0; block < blocks; ++block) {
            #pragma HLS pipeline II=1
            for (unsigned int cu = 0; cu < 4; ++cu) {
                #pragma HLS unroll
                const unsigned int row = slot * 4 + cu;
                if (row < valid_rows) {
                    const auto packet = unpack_cu8_nk_vector(
                        quantized_vector_word_t(Policy::read_output(streams, cu)));
                    mm_input_block_t word = 0;
                    for (unsigned int lane = 0; lane < CU_VEC_LANES; ++lane) {
                        #pragma HLS unroll
                        if (packet.valid_mask[lane] && block * CU_VEC_LANES + lane < elements)
                            word.range((lane + 1) * fm_t::width - 1, lane * fm_t::width) =
                                packet.data[lane].range(fm_t::width - 1, 0);
                    }
                    destination.block[row][block] = word;
                }
            }
        }
    }
}

struct quantized_vector_dispatch_t {
    unsigned int rows;
    unsigned int elements;
    unsigned int position;
    unsigned int context;
    quantized_layer_phase_t phase;
    mm_stream_quantized_compute_mode_t mode;
};

inline void register_quantized_vector_dispatch(
    const quantized_layer_task_t& task,
    mm_stream_quantized_compute_mode_t mode, unsigned int elements,
    hls::stream<quantized_vector_dispatch_t>& emit_control,
    hls::stream<quantized_vector_dispatch_t>& collect_control) {
    #pragma HLS inline off
    const quantized_vector_dispatch_t control{
        task.query_tokens, elements, task.position,
        task.kv_context_length, task.phase, mode};
    emit_control.write(control);
    collect_control.write(control);
}

template <typename Policy, typename Source0, typename Source1>
void emit_quantized_vector_registered(
    hls::stream<quantized_vector_dispatch_t>& control,
    typename Policy::streams_t& streams,
    const Source0& source0, const Source1& source1) {
    #pragma HLS inline off
    const auto command = control.read();
    quantized_layer_task_t task{};
    task.query_tokens = command.rows;
    task.position = command.position;
    task.kv_context_length = command.context;
    task.phase = command.phase;
    emit_quantized_vector_operands<Policy>(
        streams, source0, source1, task, command.mode, command.elements);
}

template <typename Policy, typename Destination>
void collect_quantized_vector_registered(
    hls::stream<quantized_vector_dispatch_t>& control,
    Destination& destination, typename Policy::streams_t& streams) {
    #pragma HLS inline off
    const auto command = control.read();
    collect_quantized_vector_results<Policy>(
        destination, streams, command.rows, command.elements, command.mode);
}

template <typename Policy, typename Source0, typename Source1, typename Destination>
void run_quantized_vector_exchange(
    const quantized_layer_task_t& layer_task,
    mm_stream_quantized_compute_mode_t mode, unsigned int elements,
    const Source0& source0, const Source1& source1, Destination& destination,
    typename Policy::streams_t& streams) {
    #pragma HLS inline off
    #pragma HLS dataflow
    hls::stream<quantized_vector_dispatch_t> emit_control("emit_control");
    hls::stream<quantized_vector_dispatch_t> collect_control("collect_control");
    #pragma HLS stream variable=emit_control depth=2
    #pragma HLS stream variable=collect_control depth=2
    register_quantized_vector_dispatch(
        layer_task, mode, elements, emit_control, collect_control);
    emit_quantized_vector_registered<Policy>(emit_control, streams, source0, source1);
    collect_quantized_vector_registered<Policy>(collect_control, destination, streams);
}

template <typename Policy, typename Buffer>
void copy_quantized_vector_active_rows(
    Buffer& destination, const Buffer& source,
    unsigned int valid_rows, unsigned int elements) {
    #pragma HLS inline off
    #pragma HLS array_partition variable=destination.block complete dim=1
    #pragma HLS array_partition variable=source.block complete dim=1
    // Every token row already owns an independent RAM bank. Keep row indices
    // constant so copy-back needs no runtime bank selector or flattened
    // row-by-column loop bound. Inactive rows retain their previous contents.
    for (unsigned int block = 0; block < ceildiv(elements, CU_VEC_LANES); ++block) {
        #pragma HLS pipeline II=1
        for (unsigned int row = 0; row < Policy::token_rows; ++row) {
            #pragma HLS unroll
            if (row < valid_rows)
                destination.block[row][block] = source.block[row][block];
        }
    }
}

// As in run_cc8_vector_gbuf_inplace_binary, drain the complete stream into a
// distinct bank before modifying an input bank. This prevents an HLS memory
// dependence from serializing the feedback loop or overwriting unsent data.
template <typename Policy, typename Buffer, typename Source1>
void run_quantized_vector_inplace(
    const quantized_layer_task_t& layer_task,
    mm_stream_quantized_compute_mode_t mode, unsigned int elements,
    Buffer& source_destination, const Source1& source1,
    typename Policy::streams_t& streams) {
    #pragma HLS inline off
    Buffer scratch;
    #pragma HLS array_partition variable=scratch.block complete dim=1
    #pragma HLS bind_storage variable=scratch.block type=ram_2p impl=bram
    run_quantized_vector_exchange<Policy>(
        layer_task, mode, elements, source_destination, source1, scratch, streams);
    copy_quantized_vector_active_rows<Policy>(
        source_destination, scratch, layer_task.query_tokens, elements);
}

#endif
