#include "xcl2.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kWeightShardCount = 16;
constexpr unsigned int kBufferWords = 72;
constexpr unsigned int kTokenCount = 8;
constexpr unsigned int kTokensPerPort = 4;
constexpr unsigned int kHiddenSize = 64;
constexpr unsigned int kIntermediateSize = 128;
constexpr unsigned int kValuesPerWord = 32;
constexpr unsigned int kFeatureWordStride = 4;
constexpr unsigned int kWeightTileSize = 16;
constexpr unsigned int kWeightTileWords = 8;
constexpr unsigned int kGateBaseTile = 48;
constexpr int16_t kWeightOneRaw = 4096;

constexpr unsigned int kControllerOutput0Arg = 19;
constexpr unsigned int kControllerOutput1Arg = 20;
constexpr unsigned int kControllerInput0Arg = 21;
constexpr unsigned int kControllerInput1Arg = 22;
constexpr unsigned int kControllerAux0Arg = 23;
constexpr unsigned int kControllerAux1Arg = 24;
constexpr unsigned int kControllerOperatorArg = 25;
constexpr unsigned int kControllerLayerArg = 26;
constexpr unsigned int kControllerTokenCountArg = 27;
constexpr unsigned int kControllerPositionArg = 28;
constexpr unsigned int kControllerTileLenArg = 29;
constexpr unsigned int kControllerWeight0Arg = 30;
constexpr unsigned int kStatusOutputArg = 1;

constexpr unsigned int kOpNop = 0;
constexpr unsigned int kOpFfnGate = 5;
constexpr unsigned int kOpResidualAdd = 10;

struct alignas(64) word512_t {
    int16_t value[kValuesPerWord];
};

static_assert(sizeof(word512_t) == 64, "host word must match one 512-bit HBM beat");

using aligned_word_vector =
    std::vector<word512_t, aligned_allocator<word512_t> >;

struct decoded_status_t {
    uint32_t op;
    uint32_t code;
    uint32_t token_count;
    uint32_t output_waves;
    uint32_t mm_tasks;
    uint32_t vector_tasks;
    uint32_t completed_packets;
    bool last_task;
};

void check_cl(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << operation << " failed with OpenCL error " << err << "\n";
        std::exit(EXIT_FAILURE);
    }
}

int hbm_bank(unsigned int bank) {
    return int(bank) | XCL_MEM_TOPOLOGY;
}

void clear_words(aligned_word_vector& words) {
    std::memset(words.data(), 0, words.size() * sizeof(word512_t));
}

decoded_status_t decode_status(const word512_t& word) {
    std::array<uint32_t, 16> field{};
    std::memcpy(field.data(), &word, sizeof(word));

    decoded_status_t status;
    status.op = field[0];
    status.code = field[1];
    status.token_count = field[2];
    status.output_waves = field[3];
    status.mm_tasks = field[4];
    status.vector_tasks = field[5];
    status.completed_packets = field[6];
    status.last_task = (field[7] & 1u) != 0;
    return status;
}

double event_milliseconds(const cl::Event& event) {
    cl_ulong start = 0;
    cl_ulong end = 0;
    cl_int err0 = clGetEventProfilingInfo(
        event(),
        CL_PROFILING_COMMAND_START,
        sizeof(start),
        &start,
        nullptr
    );
    cl_int err1 = clGetEventProfilingInfo(
        event(),
        CL_PROFILING_COMMAND_END,
        sizeof(end),
        &end,
        nullptr
    );
    if (err0 != CL_SUCCESS || err1 != CL_SUCCESS || end < start) {
        return -1.0;
    }
    return double(end - start) * 1.0e-6;
}

std::string get_option(
    int argc,
    const char* argv[],
    const std::string& option,
    const std::string& fallback = ""
) {
    for (int i = 1; i + 1 < argc; i++) {
        if (option == argv[i]) {
            return argv[i + 1];
        }
    }
    return fallback;
}

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable
        << " --xclbin <qwen_8x64_dual.xclbin>"
        << " [--case all|nop|residual-add|gate-mm]\n";
}

class smoke_runtime_t {
public:
    explicit smoke_runtime_t(const std::string& xclbin)
        : data_words_{
              aligned_word_vector(kBufferWords),
              aligned_word_vector(kBufferWords),
              aligned_word_vector(kBufferWords),
              aligned_word_vector(kBufferWords),
              aligned_word_vector(kBufferWords),
              aligned_word_vector(kBufferWords)
          },
          status_words_(kBufferWords) {
        for (auto& shard : weight_words_) {
            shard.resize(kBufferWords);
        }
        initialize_gate_weights();
        initialize_opencl(xclbin);
        create_buffers();
        initialize_kernel_args();
        migrate_constant_buffers();
    }

    bool run_nop() {
        clear_case_buffers();
        set_controller_scalars(kOpNop, 0);

        cl::Event controller_event;
        if (!launch(controller_event)) {
            return false;
        }

        decoded_status_t status = decode_status(status_words_[0]);
        bool pass =
            status.op == kOpNop &&
            status.code == 0 &&
            status.token_count == 0 &&
            status.output_waves == 0 &&
            status.mm_tasks == 0 &&
            status.vector_tasks == 0 &&
            status.completed_packets == 0 &&
            status.last_task;
        print_status("NOP", status, event_milliseconds(controller_event), pass);
        return pass;
    }

    bool run_residual_add() {
        clear_case_buffers();
        fill_residual_inputs();
        set_controller_scalars(kOpResidualAdd, kTokenCount);

        cl::Event controller_event;
        if (!launch(controller_event)) {
            return false;
        }

        decoded_status_t status = decode_status(status_words_[0]);
        bool status_pass =
            status.op == kOpResidualAdd &&
            status.code == 0 &&
            status.token_count == kTokenCount &&
            status.output_waves == 0 &&
            status.mm_tasks == 0 &&
            status.vector_tasks == 2 &&
            status.completed_packets == 32 &&
            status.last_task;
        bool data_pass = validate_residual_outputs();
        bool pass = status_pass && data_pass;
        print_status(
            "RESIDUAL_ADD_8x64",
            status,
            event_milliseconds(controller_event),
            pass
        );
        return pass;
    }

    bool run_gate_mm() {
        clear_case_buffers();
        fill_gate_inputs();
        set_controller_scalars(kOpFfnGate, kTokenCount);

        cl::Event controller_event;
        if (!launch(controller_event)) {
            return false;
        }

        decoded_status_t status = decode_status(status_words_[0]);
        bool status_pass =
            status.op == kOpFfnGate &&
            status.code == 0 &&
            status.token_count == kTokenCount &&
            status.output_waves == 1 &&
            status.mm_tasks == 2 &&
            status.vector_tasks == 0 &&
            status.completed_packets == 64 &&
            status.last_task;
        bool data_pass = validate_gate_outputs();
        bool pass = status_pass && data_pass;
        print_status(
            "FFN_GATE_8x64x128",
            status,
            event_milliseconds(controller_event),
            pass
        );
        return pass;
    }

private:
    void initialize_gate_weights() {
        for (auto& shard : weight_words_) {
            clear_words(shard);
        }

        constexpr unsigned int input_tiles =
            kHiddenSize / kWeightTileSize;
        for (unsigned int output = 0;
             output < kIntermediateSize;
             output++) {
            unsigned int input = output % kHiddenSize;
            unsigned int output_tile = output / kWeightTileSize;
            unsigned int output_lane = output % kWeightTileSize;
            unsigned int input_tile = input / kWeightTileSize;
            unsigned int input_lane = input % kWeightTileSize;
            unsigned int global_tile =
                kGateBaseTile +
                output_tile * input_tiles +
                input_tile;
            unsigned int shard_group = global_tile & 1u;
            unsigned int local_block = global_tile >> 1;
            unsigned int lane_in_tile =
                output_lane * kWeightTileSize + input_lane;
            unsigned int tile_word =
                lane_in_tile / kValuesPerWord;
            unsigned int word_lane =
                lane_in_tile % kValuesPerWord;
            unsigned int shard =
                shard_group * kWeightTileWords + tile_word;
            weight_words_[shard][local_block].value[word_lane] =
                kWeightOneRaw;
        }
    }

    void initialize_opencl(const std::string& xclbin) {
        cl_int err = CL_SUCCESS;
        auto devices = xcl::get_xil_devices();
        auto binary = xcl::read_binary_file(xclbin);
        cl::Program::Binaries bins{{binary.data(), binary.size()}};

        bool programmed = false;
        for (const auto& device : devices) {
            cl::Context candidate_context(
                device,
                nullptr,
                nullptr,
                nullptr,
                &err
            );
            if (err != CL_SUCCESS) {
                continue;
            }
            cl::Program candidate_program(
                candidate_context,
                {device},
                bins,
                nullptr,
                &err
            );
            if (err == CL_SUCCESS) {
                device_ = device;
                context_ = candidate_context;
                program_ = candidate_program;
                programmed = true;
                break;
            }
        }
        if (!programmed) {
            std::cerr << "failed to program a Xilinx device with " << xclbin << "\n";
            std::exit(EXIT_FAILURE);
        }

        constexpr cl_command_queue_properties queue_flags =
            CL_QUEUE_PROFILING_ENABLE;
        transfer_queue_ =
            cl::CommandQueue(context_, device_, queue_flags, &err);
        check_cl(err, "create transfer queue");
        compute0_queue_ =
            cl::CommandQueue(context_, device_, queue_flags, &err);
        check_cl(err, "create compute0 queue");
        compute1_queue_ =
            cl::CommandQueue(context_, device_, queue_flags, &err);
        check_cl(err, "create compute1 queue");
        status_queue_ =
            cl::CommandQueue(context_, device_, queue_flags, &err);
        check_cl(err, "create status queue");
        controller_queue_ =
            cl::CommandQueue(context_, device_, queue_flags, &err);
        check_cl(err, "create controller queue");

        compute0_kernel_ = cl::Kernel(
            program_,
            "compute_core_8x64_unified_nk:{cc8_cu0}",
            &err
        );
        check_cl(err, "create cc8_cu0 kernel");
        compute1_kernel_ = cl::Kernel(
            program_,
            "compute_core_8x64_unified_nk:{cc8_cu1}",
            &err
        );
        check_cl(err, "create cc8_cu1 kernel");
        status_kernel_ = cl::Kernel(
            program_,
            "cc8_status_sink_nk:{cc8_status}",
            &err
        );
        check_cl(err, "create cc8_status kernel");
        controller_kernel_ = cl::Kernel(
            program_,
            "control_cache_8x64_dual_core_nk:{cc8_ctrl}",
            &err
        );
        check_cl(err, "create cc8_ctrl kernel");

        std::cout << "programmed "
                  << device_.getInfo<CL_DEVICE_NAME>()
                  << " with four connected CUs\n";
    }

    void create_buffers() {
        static constexpr std::array<unsigned int, 6> data_banks = {
            0, 1, 0, 1, 2, 3
        };
        cl_int err = CL_SUCCESS;

        for (unsigned int i = 0; i < data_words_.size(); i++) {
            data_ext_[i].obj = data_words_[i].data();
            data_ext_[i].param = 0;
            data_ext_[i].flags = hbm_bank(data_banks[i]);
            data_buffers_[i] = cl::Buffer(
                context_,
                CL_MEM_EXT_PTR_XILINX |
                    CL_MEM_USE_HOST_PTR |
                    CL_MEM_READ_WRITE,
                data_words_[i].size() * sizeof(word512_t),
                &data_ext_[i],
                &err
            );
            check_cl(err, "create data HBM buffer");
        }

        for (unsigned int shard = 0; shard < kWeightShardCount; shard++) {
            weight_ext_[shard].obj = weight_words_[shard].data();
            weight_ext_[shard].param = 0;
            weight_ext_[shard].flags = hbm_bank(4 + shard);
            weight_buffers_[shard] = cl::Buffer(
                context_,
                CL_MEM_EXT_PTR_XILINX |
                    CL_MEM_USE_HOST_PTR |
                    CL_MEM_READ_ONLY,
                weight_words_[shard].size() * sizeof(word512_t),
                &weight_ext_[shard],
                &err
            );
            check_cl(err, "create weight HBM buffer");
        }

        status_ext_.obj = status_words_.data();
        status_ext_.param = 0;
        status_ext_.flags = hbm_bank(3);
        status_buffer_ = cl::Buffer(
            context_,
            CL_MEM_EXT_PTR_XILINX |
                CL_MEM_USE_HOST_PTR |
                CL_MEM_READ_WRITE,
            status_words_.size() * sizeof(word512_t),
            &status_ext_,
            &err
        );
        check_cl(err, "create status HBM buffer");
    }

    void initialize_kernel_args() {
        cl_int err = CL_SUCCESS;
        check_cl(
            status_kernel_.setArg(kStatusOutputArg, status_buffer_),
            "set status_output"
        );

        check_cl(
            controller_kernel_.setArg(
                kControllerOutput0Arg,
                data_buffers_[0]
            ),
            "set output_port0"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerOutput1Arg,
                data_buffers_[1]
            ),
            "set output_port1"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerInput0Arg,
                data_buffers_[2]
            ),
            "set input_port0"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerInput1Arg,
                data_buffers_[3]
            ),
            "set input_port1"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerAux0Arg,
                data_buffers_[4]
            ),
            "set aux_port0"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerAux1Arg,
                data_buffers_[5]
            ),
            "set aux_port1"
        );
        for (unsigned int shard = 0; shard < kWeightShardCount; shard++) {
            err = controller_kernel_.setArg(
                kControllerWeight0Arg + shard,
                weight_buffers_[shard]
            );
            check_cl(err, "set weight shard");
        }
    }

    void migrate_constant_buffers() {
        std::vector<cl::Memory> buffers;
        for (const auto& buffer : weight_buffers_) {
            buffers.push_back(buffer);
        }
        cl_int err = transfer_queue_.enqueueMigrateMemObjects(buffers, 0);
        check_cl(err, "migrate weight buffers");
        check_cl(transfer_queue_.finish(), "finish weight migration");
    }

    void clear_case_buffers() {
        for (auto& words : data_words_) {
            clear_words(words);
        }
        clear_words(status_words_);
    }

    void set_controller_scalars(unsigned int op, unsigned int token_count) {
        check_cl(
            controller_kernel_.setArg(kControllerOperatorArg, op),
            "set operator_kind"
        );
        check_cl(
            controller_kernel_.setArg(kControllerLayerArg, 0u),
            "set layer_id"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerTokenCountArg,
                token_count
            ),
            "set token_count"
        );
        check_cl(
            controller_kernel_.setArg(kControllerPositionArg, 0u),
            "set position"
        );
        check_cl(
            controller_kernel_.setArg(kControllerTileLenArg, 0u),
            "set tile_len"
        );
    }

    bool launch(cl::Event& controller_event) {
        std::vector<cl::Memory> to_device;
        for (const auto& buffer : data_buffers_) {
            to_device.push_back(buffer);
        }
        to_device.push_back(status_buffer_);
        cl_int err =
            transfer_queue_.enqueueMigrateMemObjects(to_device, 0);
        check_cl(err, "migrate case buffers to device");
        check_cl(transfer_queue_.finish(), "finish case input migration");

        cl::Event compute0_event;
        cl::Event compute1_event;
        cl::Event status_event;
        check_cl(
            compute0_queue_.enqueueTask(
                compute0_kernel_,
                nullptr,
                &compute0_event
            ),
            "enqueue cc8_cu0"
        );
        check_cl(
            compute1_queue_.enqueueTask(
                compute1_kernel_,
                nullptr,
                &compute1_event
            ),
            "enqueue cc8_cu1"
        );
        check_cl(
            status_queue_.enqueueTask(
                status_kernel_,
                nullptr,
                &status_event
            ),
            "enqueue cc8_status"
        );
        check_cl(
            controller_queue_.enqueueTask(
                controller_kernel_,
                nullptr,
                &controller_event
            ),
            "enqueue cc8_ctrl"
        );

        check_cl(compute0_queue_.flush(), "flush cc8_cu0");
        check_cl(compute1_queue_.flush(), "flush cc8_cu1");
        check_cl(status_queue_.flush(), "flush cc8_status");
        check_cl(controller_queue_.flush(), "flush cc8_ctrl");

        controller_event.wait();
        status_event.wait();
        compute0_event.wait();
        compute1_event.wait();

        std::vector<cl::Memory> from_device = {
            data_buffers_[0],
            data_buffers_[1],
            status_buffer_
        };
        err = transfer_queue_.enqueueMigrateMemObjects(
            from_device,
            CL_MIGRATE_MEM_OBJECT_HOST
        );
        check_cl(err, "migrate case results to host");
        check_cl(transfer_queue_.finish(), "finish result migration");
        return true;
    }

    void fill_residual_inputs() {
        for (unsigned int token = 0; token < kTokenCount; token++) {
            unsigned int port = token / kTokensPerPort;
            unsigned int token_in_port = token % kTokensPerPort;
            aligned_word_vector& lhs = data_words_[2 + port];
            aligned_word_vector& rhs = data_words_[4 + port];

            for (unsigned int elem = 0; elem < kHiddenSize; elem++) {
                unsigned int word_idx =
                    token_in_port * kFeatureWordStride +
                    elem / kValuesPerWord;
                unsigned int lane = elem % kValuesPerWord;
                lhs[word_idx].value[lane] =
                    int16_t(64 + 3 * token + elem);
                rhs[word_idx].value[lane] =
                    int16_t(32 + token + 2 * elem);
            }
        }
    }

    void fill_gate_inputs() {
        for (unsigned int token = 0; token < kTokenCount; token++) {
            unsigned int port = token / kTokensPerPort;
            unsigned int token_in_port = token % kTokensPerPort;
            aligned_word_vector& input = data_words_[2 + port];

            for (unsigned int elem = 0; elem < kHiddenSize; elem++) {
                unsigned int word_idx =
                    token_in_port * kFeatureWordStride +
                    elem / kValuesPerWord;
                unsigned int lane = elem % kValuesPerWord;
                input[word_idx].value[lane] =
                    int16_t(32 + 7 * token + 2 * elem);
            }
        }
    }

    bool validate_residual_outputs() const {
        for (unsigned int token = 0; token < kTokenCount; token++) {
            unsigned int port = token / kTokensPerPort;
            unsigned int token_in_port = token % kTokensPerPort;
            const aligned_word_vector& output = data_words_[port];

            for (unsigned int elem = 0; elem < kHiddenSize; elem++) {
                unsigned int word_idx =
                    token_in_port * kFeatureWordStride +
                    elem / kValuesPerWord;
                unsigned int lane = elem % kValuesPerWord;
                int16_t expected =
                    int16_t(96 + 4 * token + 3 * elem);
                int16_t actual = output[word_idx].value[lane];
                if (actual != expected) {
                    std::cerr
                        << "residual mismatch token=" << token
                        << " elem=" << elem
                        << " expected_raw=" << expected
                        << " actual_raw=" << actual << "\n";
                    return false;
                }
            }
        }
        return true;
    }

    bool validate_gate_outputs() const {
        for (unsigned int token = 0; token < kTokenCount; token++) {
            unsigned int port = token / kTokensPerPort;
            unsigned int token_in_port = token % kTokensPerPort;
            const aligned_word_vector& output = data_words_[port];

            for (unsigned int elem = 0;
                 elem < kIntermediateSize;
                 elem++) {
                unsigned int word_idx =
                    token_in_port * kFeatureWordStride +
                    elem / kValuesPerWord;
                unsigned int lane = elem % kValuesPerWord;
                int16_t expected =
                    int16_t(32 + 7 * token + 2 * (elem % kHiddenSize));
                int16_t actual = output[word_idx].value[lane];
                if (actual != expected) {
                    std::cerr
                        << "gate mismatch token=" << token
                        << " elem=" << elem
                        << " expected_raw=" << expected
                        << " actual_raw=" << actual << "\n";
                    return false;
                }
            }
        }
        return true;
    }

    static void print_status(
        const char* case_name,
        const decoded_status_t& status,
        double elapsed_ms,
        bool pass
    ) {
        std::cout
            << case_name
            << " status={op:" << status.op
            << ", code:" << status.code
            << ", tokens:" << status.token_count
            << ", waves:" << status.output_waves
            << ", mm_tasks:" << status.mm_tasks
            << ", vector_tasks:" << status.vector_tasks
            << ", packets:" << status.completed_packets
            << ", last:" << status.last_task
            << "}";
        if (elapsed_ms >= 0.0) {
            std::cout << " controller_ms=" << elapsed_ms;
        }
        std::cout << " " << (pass ? "PASS" : "FAIL") << "\n";
    }

    cl::Device device_;
    cl::Context context_;
    cl::Program program_;
    cl::CommandQueue transfer_queue_;
    cl::CommandQueue compute0_queue_;
    cl::CommandQueue compute1_queue_;
    cl::CommandQueue status_queue_;
    cl::CommandQueue controller_queue_;
    cl::Kernel compute0_kernel_;
    cl::Kernel compute1_kernel_;
    cl::Kernel status_kernel_;
    cl::Kernel controller_kernel_;

    std::array<aligned_word_vector, 6> data_words_;
    std::array<aligned_word_vector, kWeightShardCount> weight_words_;
    aligned_word_vector status_words_;
    std::array<cl_mem_ext_ptr_t, 6> data_ext_{};
    std::array<cl_mem_ext_ptr_t, kWeightShardCount> weight_ext_{};
    cl_mem_ext_ptr_t status_ext_{};
    std::array<cl::Buffer, 6> data_buffers_;
    std::array<cl::Buffer, kWeightShardCount> weight_buffers_;
    cl::Buffer status_buffer_;
};

}  // namespace

int main(int argc, const char* argv[]) {
    std::string xclbin = get_option(argc, argv, "--xclbin");
    std::string test_case = get_option(argc, argv, "--case", "all");
    if (xclbin.empty() || (
        test_case != "all" &&
        test_case != "nop" &&
        test_case != "residual-add" &&
        test_case != "gate-mm"
    )) {
        print_usage(argv[0]);
        return 1;
    }

    auto begin = std::chrono::steady_clock::now();
    smoke_runtime_t runtime(xclbin);
    bool pass = true;
    if (test_case == "all" || test_case == "nop") {
        pass = runtime.run_nop() && pass;
    }
    if (test_case == "all" || test_case == "residual-add") {
        pass = runtime.run_residual_add() && pass;
    }
    if (test_case == "all" || test_case == "gate-mm") {
        pass = runtime.run_gate_mm() && pass;
    }
    auto end = std::chrono::steady_clock::now();
    double total_seconds =
        std::chrono::duration<double>(end - begin).count();
    std::cout << "HW_EMU 8X64 " << (pass ? "PASS" : "FAIL")
              << " total_seconds=" << total_seconds << "\n";
    return pass ? 0 : 1;
}
