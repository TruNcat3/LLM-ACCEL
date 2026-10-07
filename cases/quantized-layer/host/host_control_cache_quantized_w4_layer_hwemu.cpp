#include "xcl2.hpp"
#include "quantized_layer_test_io.hpp"
#include "quantized_w4_attention_schedule.hpp"
#include "quantized_w4_projection_engine.hpp"
#include "quantized_w4_resident_buffers.hpp"

#include <CL/cl_ext_xilinx.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kCus = QUANTIZED_LAYER_COMPUTE_CUS;
constexpr unsigned int kWeightPortsPerCu = QUANTIZED_W4_WEIGHT_STREAMS_PER_CU;
constexpr unsigned int kDefaultPrefill = 66;
constexpr unsigned int kDefaultBlockSize = QUANTIZED_LAYER_W4_TOKEN_BLOCK;
constexpr unsigned int kRopeBlocks =
    (HEAD_DIM / 2 + QUANTIZED_W4_RESIDENT_LANES_PER_WORD - 1) /
    QUANTIZED_W4_RESIDENT_LANES_PER_WORD;

using weight_word_t = quantized_w4_weight_word_t;
using scale_word_t = ap_uint<128>;

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T> >;

void check_cl(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << operation << " failed with OpenCL error " << err << "\n";
        std::exit(EXIT_FAILURE);
    }
}

unsigned int parse_unsigned(int argc, const char* argv[], const char* name,
                            unsigned int fallback, unsigned int maximum) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) != name) continue;
        char* end = nullptr;
        const unsigned long value = std::strtoul(argv[i + 1], &end, 10);
        if (end == argv[i + 1] || *end != '\0' || value == 0 ||
            value > maximum) {
            std::cerr << name << " must be in 1.." << maximum << "\n";
            std::exit(EXIT_FAILURE);
        }
        return static_cast<unsigned int>(value);
    }
    return fallback;
}

std::string parse_string(int argc, const char* argv[], const char* name,
                         const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == name) return argv[i + 1];
    }
    return fallback;
}

cl::Buffer make_hbm_buffer(const cl::Context& context, void* host_ptr,
                           std::size_t bytes, unsigned int bank,
                           cl_mem_flags access, cl_int* err) {
    cl_mem_ext_ptr_t ext = {};
    ext.obj = host_ptr;
    ext.flags = static_cast<int>(bank) | XCL_MEM_TOPOLOGY;
    return cl::Buffer(context,
                      CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | access,
                      bytes, &ext, err);
}

using quantized_layer_test_io::fill_hidden;
using quantized_layer_test_io::fill_norm;
using quantized_layer_test_io::fill_rope;

void fill_random_weights(
    std::array<std::array<aligned_vector<weight_word_t>, kWeightPortsPerCu>,
               kCus>& weights) {
    quantized_layer_test_io::fill_random_weights<4>(weights);
}

bool equal_hidden(const aligned_vector<mm_input_block_t>& expected,
                  const aligned_vector<mm_input_block_t>& actual,
                  unsigned int tokens, const char* phase) {
    const std::size_t words =
        static_cast<std::size_t>(tokens) * QUANTIZED_W4_HIDDEN_BLOCKS;
    for (std::size_t index = 0; index < words; ++index) {
        if (expected[index] != actual[index]) {
            std::cerr << phase << " residual mismatch at word " << index
                      << "\n";
            return false;
        }
    }
    return true;
}

std::uint64_t raw_checksum(const aligned_vector<mm_input_block_t>& values,
                           std::size_t words) {
    std::uint64_t checksum = 1469598103934665603ULL;
    for (std::size_t index = 0; index < words; ++index) {
        for (unsigned int lane = 0;
             lane < QUANTIZED_W4_RESIDENT_LANES_PER_WORD; ++lane) {
            checksum ^= values[index].range(lane * 16 + 15, lane * 16).to_uint();
            checksum *= 1099511628211ULL;
        }
    }
    return checksum;
}

double event_interval_us(const cl::Event& controller_event,
                         const std::array<cl::Event, kCus>& compute_events) {
    cl_ulong earliest = std::numeric_limits<cl_ulong>::max();
    cl_ulong latest = 0;
    auto add_event = [&](const cl::Event& event) {
        cl_ulong start = 0;
        cl_ulong end = 0;
        check_cl(clGetEventProfilingInfo(
            event(), CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr),
            "read event start");
        check_cl(clGetEventProfilingInfo(
            event(), CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr),
            "read event end");
        earliest = std::min(earliest, start);
        latest = std::max(latest, end);
    };
    add_event(controller_event);
    for (const auto& event : compute_events) add_event(event);
    return static_cast<double>(latest - earliest) * 1.0e-3;
}

}  // namespace

int main(int argc, const char* argv[]) {
    const std::string xclbin = parse_string(argc, argv, "--xclbin", "");
    const std::string dump_path = parse_string(argc, argv, "--dump-output", "");
    const std::string weight_mode =
        parse_string(argc, argv, "--weights", "zero");
    const unsigned int prefill = parse_unsigned(
        argc, argv, "--prefill", kDefaultPrefill, MAX_SEQ_LEN - 1);
    const unsigned int block_size = parse_unsigned(
        argc, argv, "--block-size", kDefaultBlockSize,
        QUANTIZED_W4_RESIDENT_TOKEN_ROWS);
    if (xclbin.empty() || (weight_mode != "zero" && weight_mode != "random")) {
        std::cerr << "usage: " << argv[0]
                  << " --xclbin <image> [--prefill 66] [--block-size 8]"
                     " [--weights zero|random] [--dump-output path]\n";
        return EXIT_FAILURE;
    }

    std::ofstream dump;
    if (!dump_path.empty()) {
        dump.open(dump_path);
        if (!dump) { std::cerr << "Cannot open dump: " << dump_path << "\n"; return EXIT_FAILURE; }
        quantized_layer_test_io::dump_header(dump, 4, prefill, block_size, weight_mode);
    }

    const std::size_t hidden_words =
        static_cast<std::size_t>(prefill) * QUANTIZED_W4_HIDDEN_BLOCKS;
    const std::size_t kv_words =
        static_cast<std::size_t>(prefill + 1) * QUANTIZED_W4_KV_BLOCKS;
    const std::size_t norm_words = 2 * QUANTIZED_W4_HIDDEN_BLOCKS;
    const std::size_t rope_words =
        static_cast<std::size_t>(prefill + 1) * 2 * kRopeBlocks;
    const std::size_t weight_words = quantized_layer_weight_words_per_shard();

    aligned_vector<mm_input_block_t> hidden_input(hidden_words);
    aligned_vector<mm_input_block_t> hidden_output(hidden_words);
    aligned_vector<mm_input_block_t> key_cache(kv_words);
    aligned_vector<mm_input_block_t> value_cache(kv_words);
    aligned_vector<mm_input_block_t> norm(norm_words);
    aligned_vector<mm_input_block_t> rope(rope_words);
    aligned_vector<scale_word_t> scales(1, scale_word_t(0));
    std::array<std::array<aligned_vector<weight_word_t>, kWeightPortsPerCu>,
               kCus> weights;
    for (auto& cu : weights) {
        for (auto& port : cu) port.resize(weight_words, weight_word_t(0));
    }
    fill_hidden(hidden_input, prefill, 3);
    const aligned_vector<mm_input_block_t> prefill_expected(
        hidden_input.begin(), hidden_input.end());
    fill_norm(norm);
    fill_rope(rope, prefill + 1);
    if (weight_mode == "random") fill_random_weights(weights);

    cl_int err = CL_SUCCESS;
    const std::vector<cl::Device> devices = xcl::get_xil_devices();
    if (devices.empty()) {
        std::cerr << "no Xilinx accelerator device found\n";
        return EXIT_FAILURE;
    }
    const std::vector<unsigned char> binary = xcl::read_binary_file(xclbin);
    cl::Program::Binaries binaries{{binary.data(), binary.size()}};
    cl::Context context(devices[0], nullptr, nullptr, nullptr, &err);
    check_cl(err, "create context");
    cl::Program program(context, {devices[0]}, binaries, nullptr, &err);
    check_cl(err, "load full-layer xclbin");
    cl::Kernel controller(
        program, "control_cache_quantized_w4_layer:{q4_layer_ctrl}", &err);
    check_cl(err, "create full-layer controller");
    std::array<cl::Kernel, kCus> computes;
    for (unsigned int cu = 0; cu < kCus; ++cu) {
        const std::string name =
            "compute_core_quantized_w4_unified_nk:{q4_layer_cu" +
            std::to_string(cu) + "}";
        computes[cu] = cl::Kernel(program, name.c_str(), &err);
        check_cl(err, "create W4 compute CU");
    }

    cl::Buffer hidden_input_buffer = make_hbm_buffer(
        context, hidden_input.data(), hidden_input.size() * sizeof(hidden_input[0]),
        0, CL_MEM_READ_WRITE, &err);
    check_cl(err, "create hidden input buffer");
    cl::Buffer hidden_output_buffer = make_hbm_buffer(
        context, hidden_output.data(), hidden_output.size() * sizeof(hidden_output[0]),
        1, CL_MEM_READ_WRITE, &err);
    check_cl(err, "create hidden output buffer");
    cl::Buffer key_buffer = make_hbm_buffer(
        context, key_cache.data(), key_cache.size() * sizeof(key_cache[0]), 2,
        CL_MEM_READ_WRITE, &err);
    check_cl(err, "create key cache buffer");
    cl::Buffer value_buffer = make_hbm_buffer(
        context, value_cache.data(), value_cache.size() * sizeof(value_cache[0]),
        3, CL_MEM_READ_WRITE, &err);
    check_cl(err, "create value cache buffer");
    cl::Buffer norm_buffer = make_hbm_buffer(
        context, norm.data(), norm.size() * sizeof(norm[0]), 4,
        CL_MEM_READ_ONLY, &err);
    check_cl(err, "create norm buffer");
    cl::Buffer rope_buffer = make_hbm_buffer(
        context, rope.data(), rope.size() * sizeof(rope[0]), 5,
        CL_MEM_READ_ONLY, &err);
    check_cl(err, "create RoPE buffer");
    cl::Buffer scale_buffer = make_hbm_buffer(
        context, scales.data(), scales.size() * sizeof(scales[0]), 6,
        CL_MEM_READ_ONLY, &err);
    check_cl(err, "create scale buffer");
    std::array<std::array<cl::Buffer, kWeightPortsPerCu>, kCus>
        weight_buffers;
    for (unsigned int cu = 0; cu < kCus; ++cu) {
        for (unsigned int port = 0; port < kWeightPortsPerCu; ++port) {
            const unsigned int bank = 8 + cu * kWeightPortsPerCu + port;
            weight_buffers[cu][port] = make_hbm_buffer(
                context, weights[cu][port].data(),
                weights[cu][port].size() * sizeof(weight_word_t), bank,
                CL_MEM_READ_ONLY, &err);
            check_cl(err, "create weight buffer");
        }
    }

    for (cl_uint index = 0; index < 28; ++index) {
        check_cl(controller.setArg(index, sizeof(cl_mem), nullptr),
                 "set controller stream placeholder");
    }
    cl_uint arg = 28;
    check_cl(controller.setArg(arg++, hidden_input_buffer), "set hidden input");
    check_cl(controller.setArg(arg++, hidden_output_buffer), "set hidden output");
    check_cl(controller.setArg(arg++, key_buffer), "set key cache");
    check_cl(controller.setArg(arg++, value_buffer), "set value cache");
    check_cl(controller.setArg(arg++, norm_buffer), "set norm memory");
    check_cl(controller.setArg(arg++, rope_buffer), "set RoPE memory");
    check_cl(controller.setArg(arg++, scale_buffer), "set scale memory");
    for (unsigned int cu = 0; cu < kCus; ++cu) {
        for (unsigned int port = 0; port < kWeightPortsPerCu; ++port) {
            check_cl(controller.setArg(arg++, weight_buffers[cu][port]),
                     "set weight memory");
        }
    }
    const cl_uint scalar_arg_base = arg;
    for (unsigned int cu = 0; cu < kCus; ++cu) {
        for (cl_uint index = 0; index < 7; ++index) {
            check_cl(computes[cu].setArg(index, sizeof(cl_mem), nullptr),
                     "set compute stream placeholder");
        }
    }

    cl::CommandQueue transfer_queue(
        context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    check_cl(err, "create transfer queue");
    cl::CommandQueue controller_queue(
        context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    check_cl(err, "create controller queue");
    std::array<cl::CommandQueue, kCus> compute_queues;
    for (auto& queue : compute_queues) {
        queue = cl::CommandQueue(
            context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
        check_cl(err, "create compute queue");
    }

    std::vector<cl::Memory> initial_inputs{
        hidden_input_buffer, key_buffer, value_buffer, norm_buffer,
        rope_buffer, scale_buffer};
    for (auto& cu : weight_buffers) {
        for (auto& port : cu) initial_inputs.push_back(port);
    }
    check_cl(transfer_queue.enqueueMigrateMemObjects(initial_inputs, 0),
             "migrate initial layer state");
    check_cl(transfer_queue.finish(), "finish initial migration");

    auto run_request = [&](unsigned int sequence_length,
                           unsigned int request_position,
                           unsigned int kv_context_length,
                           unsigned int request_op,
                           const char* phase) {
        cl_uint scalar = scalar_arg_base;
        check_cl(controller.setArg(scalar++, 0u), "set layer");
        check_cl(controller.setArg(scalar++, sequence_length),
                 "set sequence length");
        check_cl(controller.setArg(scalar++, request_position),
                 "set request position");
        check_cl(controller.setArg(scalar++, kv_context_length),
                 "set KV context length");
        check_cl(controller.setArg(scalar++, request_op), "set request op");
        check_cl(controller.setArg(scalar++, block_size), "set block size");

        std::array<unsigned int, kCus> task_counts{};
        const bool decode = request_op == QUANTIZED_LAYER_OP_DECODE;
        for (unsigned int cu = 0; cu < kCus; ++cu) {
            task_counts[cu] = quantized_w4_layer_tasks_for_cu(
                sequence_length, request_position, cu, decode, block_size);
            check_cl(computes[cu].setArg(7, task_counts[cu]),
                     "set compute task count");
        }

        cl::Event controller_event;
        std::array<cl::Event, kCus> compute_events;
        check_cl(controller_queue.enqueueTask(
                     controller, nullptr, &controller_event),
                 "enqueue full-layer controller");
        for (unsigned int cu = 0; cu < kCus; ++cu) {
            check_cl(compute_queues[cu].enqueueTask(
                         computes[cu], nullptr, &compute_events[cu]),
                     "enqueue W4 compute CU");
        }
        check_cl(controller_queue.flush(), "flush controller");
        for (auto& queue : compute_queues) {
            check_cl(queue.flush(), "flush compute CU");
        }
        check_cl(controller_queue.finish(), "finish controller");
        for (auto& queue : compute_queues) {
            check_cl(queue.finish(), "finish compute CU");
        }
        std::cout << "phase=" << phase << " sequence_length="
                  << sequence_length << " position=" << request_position
                  << " kv_context_length=" << kv_context_length
                  << " block_size=" << block_size << " blocks="
                  << (decode ? 1 : quantized_layer_block_count(
                                      sequence_length, block_size))
                  << " tasks=" << task_counts[0] << "," << task_counts[1]
                  << "," << task_counts[2] << "," << task_counts[3]
                  << " opencl_event_us_not_device_time="
                  << event_interval_us(controller_event, compute_events)
                  << "\n";
    };

    run_request(prefill, 0, 0, QUANTIZED_LAYER_OP_PREFILL, "prefill");
    check_cl(transfer_queue.enqueueMigrateMemObjects(
                 {hidden_output_buffer}, CL_MIGRATE_MEM_OBJECT_HOST),
             "read prefill output");
    check_cl(transfer_queue.finish(), "finish prefill output read");
    const bool prefill_ok = weight_mode == "zero" ?
        equal_hidden(prefill_expected, hidden_output, prefill, "prefill") : true;
    const std::uint64_t prefill_checksum =
        raw_checksum(hidden_output, hidden_words);

    fill_hidden(hidden_input, 1, 19);
    if (dump.is_open())
        quantized_layer_test_io::dump_words(dump, "prefill_hidden", hidden_output, hidden_words);
    const aligned_vector<mm_input_block_t> decode_expected = hidden_input;
    check_cl(transfer_queue.enqueueMigrateMemObjects({hidden_input_buffer}, 0),
             "migrate decode input");
    check_cl(transfer_queue.finish(), "finish decode input migration");
    run_request(1, prefill, prefill, QUANTIZED_LAYER_OP_DECODE, "decode");
    check_cl(transfer_queue.enqueueMigrateMemObjects(
                 {hidden_output_buffer, key_buffer, value_buffer},
                 CL_MIGRATE_MEM_OBJECT_HOST),
             "read decode output and KV");
    check_cl(transfer_queue.finish(), "finish decode output read");
    const bool decode_ok = weight_mode == "zero" ?
        equal_hidden(decode_expected, hidden_output, 1, "decode") : true;
    bool kv_ok = true;
    if (weight_mode == "zero") {
        for (std::size_t index = 0; index < kv_words; ++index) {
            if (key_cache[index] != 0 || value_cache[index] != 0) {
                std::cerr << "zero-weight KV mismatch at word " << index << "\n";
                kv_ok = false;
                break;
            }
        }
    }
    const std::uint64_t decode_checksum = raw_checksum(
        hidden_output, QUANTIZED_W4_HIDDEN_BLOCKS);
    if (dump.is_open()) {
        quantized_layer_test_io::dump_words(dump, "decode_hidden", hidden_output, QUANTIZED_W4_HIDDEN_BLOCKS);
        quantized_layer_test_io::dump_words(dump, "key_cache", key_cache, kv_words);
        quantized_layer_test_io::dump_words(dump, "value_cache", value_cache, kv_words);
        std::cout << "numerical_dump=" << dump_path << " validation=post_inference_pending\n";
    }
    if (!prefill_ok || !decode_ok || !kv_ok) return EXIT_FAILURE;

    std::cout << "Q4 FULL-LAYER HW EMU EXECUTION PASS prefill=" << prefill
              << " decode=1 block_size=" << block_size
              << " weight_mode=" << weight_mode
              << " numerical_validation=" << (weight_mode == "zero" ? "PASS" : "NOT_RUN")
              << " controller_owned_kv=1 host_intermediate_compute=0"
              << " prefill_checksum=" << prefill_checksum
              << " decode_checksum=" << decode_checksum << "\n";
    return EXIT_SUCCESS;
}
