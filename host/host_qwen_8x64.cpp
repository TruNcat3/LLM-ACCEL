#include "xcl2.hpp"

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

#ifndef LLM_FPGA_DEVICE_PROFILE
#define LLM_FPGA_DEVICE_PROFILE "small"
#endif

constexpr const char* kDeviceProfile = LLM_FPGA_DEVICE_PROFILE;

constexpr unsigned int kWeightShardCount = 16;
constexpr unsigned int kValuesPerWord = 32;
constexpr unsigned int kTokensPerPort = 4;
constexpr unsigned int kMaxTokensPerLaunch = 8;
constexpr unsigned int kMmCoreCount = 2;
constexpr unsigned int kMmOutputsPerCore = 64;
constexpr unsigned int kMmOutputsPerWave = kMmCoreCount * kMmOutputsPerCore;
constexpr unsigned int kMmInputBlock = 64;
constexpr unsigned int kMmPacketsPerBlock = 32;
constexpr unsigned int kAttentionTile = 64;
constexpr unsigned int kFmFractionBits = 8;
constexpr unsigned int kWeightFractionBits = 12;
constexpr double kFmScale = double(1u << kFmFractionBits);
constexpr double kRopeTheta = 1000000.0;
constexpr std::size_t kHbmPseudoChannelBytes = 256ull * 1024ull * 1024ull;

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
constexpr unsigned int kControllerKvCacheKArg =
    kControllerWeight0Arg + kWeightShardCount;
constexpr unsigned int kControllerKvCacheVArg =
    kControllerKvCacheKArg + 1;
constexpr unsigned int kStatusOutputArg = 1;

enum operator_kind_t {
    kOpNop = 0,
    kOpQProjection = 1,
    kOpKProjection = 2,
    kOpVProjection = 3,
    kOpOProjection = 4,
    kOpFfnGate = 5,
    kOpFfnUp = 6,
    kOpFfnDown = 7,
    kOpRmsNorm = 8,
    kOpSiluMul = 9,
    kOpResidualAdd = 10,
    kOpAttentionQk = 11,
    kOpAttentionPv = 12,
    kOpSoftmax = 13,
    kOpAttentionFlash = 14,
    kOpDecodeSmoke = 15,
    kOpDecoderLayer = 16,
    kOpAttnPrefillBlock = 17,
    kOpAttentionSublayer = 18,
    kOpFfnSublayer = 19,
    kOpFinalNorm = 20
};

struct alignas(64) word512_t {
    int16_t value[kValuesPerWord];
};

static_assert(sizeof(word512_t) == 64, "one host word must be one 512-bit beat");

using aligned_word_vector =
    std::vector<word512_t, aligned_allocator<word512_t> >;
using aligned_fix16_vector =
    std::vector<int16_t, aligned_allocator<int16_t> >;
using golden_fm_t = ap_fixed<16, 8, AP_RND, AP_SAT>;
using golden_weight_t = ap_fixed<16, 4, AP_RND, AP_SAT>;
using golden_accum_t = ap_fixed<32, 16, AP_RND, AP_SAT>;
using golden_scale_t = ap_fixed<24, 4, AP_RND, AP_SAT>;
using golden_probability_t = ap_fixed<16, 2, AP_RND, AP_SAT>;
using golden_fused_raw_accum_t = ap_int<48>;
using golden_fused_fixed_accum_t = ap_fixed<48, 28, AP_TRN, AP_WRAP>;

std::size_t ceildiv(std::size_t value, std::size_t divisor) {
    return (value + divisor - 1) / divisor;
}

struct model_shape_t {
    std::string name;
    unsigned int vocab_size;
    unsigned int hidden_size;
    unsigned int intermediate_size;
    unsigned int num_layers;
    unsigned int num_heads;
    unsigned int num_kv_heads;
    unsigned int max_seq_len;

    unsigned int head_dim() const {
        return hidden_size / num_heads;
    }

    unsigned int kv_channels() const {
        return num_kv_heads * head_dim();
    }

    unsigned int gqa_group_size() const {
        return num_heads / num_kv_heads;
    }

    unsigned int max_linear_dim() const {
        return std::max(hidden_size, intermediate_size);
    }

    std::size_t feature_words_per_token() const {
        return ceildiv(max_linear_dim(), kValuesPerWord);
    }

    std::size_t feature_words_per_port() const {
        return kTokensPerPort * feature_words_per_token();
    }

    std::size_t data_port_words() const {
        const std::size_t mv_cache_words =
            ceildiv(std::size_t(max_seq_len) * head_dim(), kValuesPerWord);
        return std::max(feature_words_per_port(), mv_cache_words);
    }

    std::size_t packed_linear_tiles(
        std::size_t out_dim,
        std::size_t in_dim
    ) const {
        return ceildiv(out_dim, 16) * ceildiv(in_dim, 16);
    }

    std::size_t layer_weight_tiles() const {
        return
            packed_linear_tiles(hidden_size, hidden_size) +
            packed_linear_tiles(kv_channels(), hidden_size) +
            packed_linear_tiles(kv_channels(), hidden_size) +
            packed_linear_tiles(hidden_size, hidden_size) +
            packed_linear_tiles(intermediate_size, hidden_size) +
            packed_linear_tiles(intermediate_size, hidden_size) +
            packed_linear_tiles(hidden_size, intermediate_size);
    }

    std::size_t weight_shard_words() const {
        const std::size_t total_tiles =
            std::size_t(num_layers) * layer_weight_tiles();
        return ceildiv(total_tiles, 2);
    }

    std::size_t embedding_values() const {
        return std::size_t(vocab_size) * hidden_size;
    }

    std::size_t norm_values() const {
        return (std::size_t(num_layers) * 2 + 1) * hidden_size;
    }

    std::size_t kv_cache_row_major_words() const {
        return
            std::size_t(num_layers) *
            max_seq_len *
            num_kv_heads *
            ceildiv(head_dim(), kValuesPerWord);
    }

    std::size_t k_cache_transposed_words() const {
        return
            std::size_t(num_layers) *
            num_kv_heads *
            head_dim() *
            ceildiv(max_seq_len, kValuesPerWord);
    }

    std::size_t kv_cache_words() const {
        // K uses the appended transposed region; V keeps the same allocation
        // size for a uniform host/kernel buffer contract and leaves that
        // suffix unused.
        return kv_cache_row_major_words() + k_cache_transposed_words();
    }

    void validate() const {
        if (
            hidden_size == 0 ||
            intermediate_size == 0 ||
            num_layers == 0 ||
            num_heads == 0 ||
            num_kv_heads != 2 ||
            hidden_size % num_heads != 0 ||
            num_heads % num_kv_heads != 0 ||
            head_dim() % 2 != 0 ||
            gqa_group_size() > 8
        ) {
            throw std::runtime_error("profile is incompatible with the 8x64 controller");
        }
    }
};

bool parse_mm_profile_op(
    const std::string& name,
    operator_kind_t& op
);
std::string mm_profile_op_name(operator_kind_t op);
unsigned int mm_profile_input_dim(
    const model_shape_t& shape,
    operator_kind_t op
);
unsigned int mm_profile_output_dim(
    const model_shape_t& shape,
    operator_kind_t op
);

struct projection_spec_t {
    unsigned int in_dim;
    unsigned int out_dim;
    std::size_t base_tile;
};

projection_spec_t projection_spec(
    const model_shape_t& shape,
    operator_kind_t op
) {
    const std::size_t q_tiles =
        shape.packed_linear_tiles(shape.hidden_size, shape.hidden_size);
    const std::size_t k_tiles =
        shape.packed_linear_tiles(shape.kv_channels(), shape.hidden_size);
    const std::size_t v_tiles = k_tiles;
    const std::size_t o_tiles = q_tiles;
    const std::size_t gate_tiles =
        shape.packed_linear_tiles(
            shape.intermediate_size,
            shape.hidden_size
        );
    const std::size_t up_tiles = gate_tiles;

    switch (op) {
    case kOpQProjection:
        return {shape.hidden_size, shape.hidden_size, 0};
    case kOpKProjection:
        return {shape.hidden_size, shape.kv_channels(), q_tiles};
    case kOpVProjection:
        return {
            shape.hidden_size,
            shape.kv_channels(),
            q_tiles + k_tiles
        };
    case kOpOProjection:
        return {
            shape.hidden_size,
            shape.hidden_size,
            q_tiles + k_tiles + v_tiles
        };
    case kOpFfnGate:
        return {
            shape.hidden_size,
            shape.intermediate_size,
            q_tiles + k_tiles + v_tiles + o_tiles
        };
    case kOpFfnUp:
        return {
            shape.hidden_size,
            shape.intermediate_size,
            q_tiles + k_tiles + v_tiles + o_tiles + gate_tiles
        };
    case kOpFfnDown:
        return {
            shape.intermediate_size,
            shape.hidden_size,
            q_tiles + k_tiles + v_tiles + o_tiles +
                gate_tiles + up_tiles
        };
    default:
        throw std::runtime_error("operator has no packed projection weights");
    }
}

bool parse_profile(const std::string& name, model_shape_t& shape) {
    if (name == "small") {
        shape = {"small", 64, 64, 128, 2, 4, 2, 32};
        return true;
    }
    if (name == "medium") {
        shape = {"medium", 128, 128, 256, 2, 4, 2, 64};
        return true;
    }
    if (name == "qwen-layer") {
        shape = {"qwen-layer", 128, 2048, 11008, 1, 16, 2, 96};
        return true;
    }
    if (name == "qwen-layer-long") {
        shape = {"qwen-layer-long", 128, 2048, 11008, 1, 16, 2, 2048};
        return true;
    }
    if (name == "qwen2.5-3b") {
        shape = {
            "qwen2.5-3b",
            151936,
            2048,
            11008,
            36,
            16,
            2,
            2048
        };
        return true;
    }
    return false;
}

struct tensor_t {
    unsigned int rows = 0;
    unsigned int cols = 0;
    std::vector<int16_t> values;

    tensor_t() = default;

    tensor_t(unsigned int row_count, unsigned int col_count)
        : rows(row_count),
          cols(col_count),
          values(std::size_t(row_count) * col_count, 0) {}

    int16_t& at(unsigned int row, unsigned int col) {
        return values[std::size_t(row) * cols + col];
    }

    int16_t at(unsigned int row, unsigned int col) const {
        return values[std::size_t(row) * cols + col];
    }
};

struct decoded_status_t {
    uint32_t op = 0;
    uint32_t code = 0;
    uint32_t token_count = 0;
    uint32_t output_waves = 0;
    uint32_t mm_tasks = 0;
    uint32_t vector_tasks = 0;
    uint32_t completed_packets = 0;
    bool last_task = false;
};

struct operator_result_t {
    tensor_t port0;
    tensor_t port1;
    decoded_status_t status;
    double controller_ms = -1.0;
};

struct composed_layer_result_t {
    tensor_t output;
    decoded_status_t attention_status;
    decoded_status_t ffn_status;
    decoded_status_t final_norm_status;
    double attention_controller_ms = -1.0;
    double ffn_controller_ms = -1.0;
    double final_norm_controller_ms = 0.0;
    double kernel_active_ms = -1.0;
    double input_migration_ms = 0.0;
    double auxiliary_migration_ms = 0.0;
    double status_kernel_ms = 0.0;
    double status_migration_ms = 0.0;
    double output_migration_ms = 0.0;
    double profiled_sequence_ms = -1.0;
    double host_elapsed_ms = -1.0;
    unsigned int layer_count = 0;
    unsigned int task_count = 0;
};

struct command_line_t {
    std::string mode = "plan";
    std::string profile = "small";
    std::string data_dir = "data";
    std::string xclbin;
    std::string profile_op = "q";
    std::vector<unsigned int> tokens{0};
    unsigned int layer_count = 0;
    unsigned int max_new_tokens = 0;
    unsigned int profile_wave = 0;
    unsigned int profile_wave_count = 1;
    unsigned int profile_k_limit = 0;
    unsigned int profile_debug_stage = 0;
    unsigned int profile_core_mask = 3;
    unsigned int profile_token_count = 1;
    unsigned int attention_position = 0;
    unsigned int attention_prefill_len = 0;
    unsigned int attention_prefill_start = 0;
    std::string attention_phase = "pd";
    bool profile_zero_weight_stream = false;
    bool profile_single_launch = false;
    uint32_t random_seed = 20260701u;
    bool zero_model = false;
    bool random_model = false;
    bool verbose_ops = false;
    bool hardware_softmax = false;
    bool resident_layer = false;
    bool coarse_tasks = false;
    bool tie_embeddings = false;
    bool skip_weight_preload = false;
    bool load_only = false;
};

void check_cl(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::ostringstream message;
        message
            << operation
            << " failed with OpenCL error "
            << err;
        throw std::runtime_error(message.str());
    }
}

int hbm_bank(unsigned int bank) {
    return int(bank) | XCL_MEM_TOPOLOGY;
}

void clear_words(aligned_word_vector& words) {
    std::memset(words.data(), 0, words.size() * sizeof(word512_t));
}

int16_t quantize_fix16(double value) {
    const double scaled = std::round(value * kFmScale);
    if (scaled > std::numeric_limits<int16_t>::max()) {
        return std::numeric_limits<int16_t>::max();
    }
    if (scaled < std::numeric_limits<int16_t>::min()) {
        return std::numeric_limits<int16_t>::min();
    }
    return int16_t(scaled);
}

double dequantize_fix16(int16_t value) {
    return double(value) / kFmScale;
}

template <typename Fixed>
Fixed fixed_from_raw(int16_t raw) {
    Fixed value;
    value.range(15, 0) = ap_uint<16>(uint16_t(raw));
    return value;
}

template <typename Fixed>
int16_t fixed_to_raw(const Fixed& value) {
    return int16_t(uint16_t(value.range(15, 0).to_uint()));
}

golden_fm_t golden_clamp(
    golden_fm_t value,
    golden_fm_t lower,
    golden_fm_t upper
) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

golden_fm_t golden_recip(golden_fm_t value) {
    golden_fm_t safe =
        value < golden_fm_t(0.000244140625) ?
        golden_fm_t(0.000244140625) :
        value;
    golden_fm_t estimate;
    if (safe >= golden_fm_t(64)) {
        estimate = golden_fm_t(0.015625);
    } else if (safe >= golden_fm_t(32)) {
        estimate = golden_fm_t(0.03125);
    } else if (safe >= golden_fm_t(16)) {
        estimate = golden_fm_t(0.0625);
    } else if (safe >= golden_fm_t(8)) {
        estimate = golden_fm_t(0.125);
    } else if (safe >= golden_fm_t(4)) {
        estimate = golden_fm_t(0.25);
    } else if (safe >= golden_fm_t(2)) {
        estimate = golden_fm_t(0.5);
    } else if (safe >= golden_fm_t(1)) {
        estimate = golden_fm_t(1);
    } else if (safe >= golden_fm_t(0.5)) {
        estimate = golden_fm_t(2);
    } else if (safe >= golden_fm_t(0.25)) {
        estimate = golden_fm_t(4);
    } else if (safe >= golden_fm_t(0.125)) {
        estimate = golden_fm_t(8);
    } else {
        estimate = golden_fm_t(16);
    }
    for (unsigned int i = 0; i < 3; i++) {
        estimate =
            estimate *
            (golden_fm_t(2) - safe * estimate);
    }
    return golden_clamp(
        estimate,
        golden_fm_t(0),
        golden_fm_t(64)
    );
}

golden_fm_t golden_exp(golden_fm_t value) {
    golden_fm_t clamped = golden_clamp(
        value,
        golden_fm_t(-8),
        golden_fm_t(8)
    );
    ap_fixed<16, 8> exp_input = ap_fixed<16, 8>(clamped);
    return golden_fm_t(hls::exp(exp_input));
}

golden_probability_t golden_attention_probability(golden_fm_t value) {
    const golden_fm_t clamped = golden_clamp(
        value,
        golden_fm_t(-8),
        golden_fm_t(0)
    );
    const ap_fixed<18, 4> exp_input = ap_fixed<18, 4>(clamped);
    return golden_probability_t(hls::exp(exp_input));
}

golden_fm_t golden_rescale_exp(
    golden_fm_t value,
    golden_fm_t*
) {
    return golden_exp(value);
}

golden_scale_t golden_rescale_exp(
    golden_fm_t value,
    golden_scale_t*
) {
    const golden_fm_t clamped = golden_clamp(
        value,
        golden_fm_t(-8),
        golden_fm_t(0)
    );
    // This is a CPU-only numerical probe: use a high-precision exponential
    // and quantize only its result.  It tells us whether widening the
    // rescale arithmetic is beneficial independently of a particular HLS
    // exp fixed-point instantiation.
    return golden_scale_t(std::exp(double(clamped)));
}

golden_fm_t golden_rsqrt(golden_accum_t value) {
    const golden_accum_t minimum(0.000244140625);
    golden_accum_t safe = value < minimum ? minimum : value;
    golden_fm_t estimate;
    if (safe >= golden_accum_t(64)) {
        estimate = golden_fm_t(0.125);
    } else if (safe >= golden_accum_t(16)) {
        estimate = golden_fm_t(0.25);
    } else if (safe >= golden_accum_t(4)) {
        estimate = golden_fm_t(0.5);
    } else if (safe >= golden_accum_t(1)) {
        estimate = golden_fm_t(1);
    } else if (safe >= golden_accum_t(0.25)) {
        estimate = golden_fm_t(2);
    } else if (safe >= golden_accum_t(0.0625)) {
        estimate = golden_fm_t(4);
    } else {
        estimate = golden_fm_t(8);
    }
    for (unsigned int i = 0; i < 3; i++) {
        golden_accum_t square =
            golden_accum_t(estimate) *
            golden_accum_t(estimate);
        golden_accum_t refine =
            golden_accum_t(1.5) -
            golden_accum_t(0.5) * safe * square;
        estimate = golden_fm_t(
            golden_accum_t(estimate) * refine
        );
    }
    return golden_clamp(
        estimate,
        golden_fm_t(0),
        golden_fm_t(32)
    );
}

golden_fm_t golden_silu(golden_fm_t value) {
    if (value >= golden_fm_t(0)) {
        golden_fm_t exponential = golden_exp(golden_fm_t(-value));
        return value * golden_recip(golden_fm_t(1) + exponential);
    }
    golden_fm_t exponential = golden_exp(value);
    return
        value *
        exponential *
        golden_recip(golden_fm_t(1) + exponential);
}

decoded_status_t decode_status(const word512_t& word) {
    std::array<uint32_t, 16> fields{};
    std::memcpy(fields.data(), &word, sizeof(word));

    decoded_status_t status;
    status.op = fields[0];
    status.code = fields[1];
    status.token_count = fields[2];
    status.output_waves = fields[3];
    status.mm_tasks = fields[4];
    status.vector_tasks = fields[5];
    status.completed_packets = fields[6];
    status.last_task = (fields[7] & 1u) != 0;
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

const char* event_timing_domain() {
    const char* mode = std::getenv("XCL_EMULATION_MODE");
    return mode != nullptr && std::strcmp(mode, "hw_emu") == 0 ?
        "hw_emu_host_wall_proxy" :
        "device_event";
}

double add_profiled_milliseconds(double total, double sample) {
    if (total < 0.0 || sample < 0.0) {
        return -1.0;
    }
    return total + sample;
}

bool file_exists(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    return input.good();
}

std::size_t file_size(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input.good()) {
        return 0;
    }
    return std::size_t(input.tellg());
}

std::string shard_path(const std::string& data_dir, unsigned int shard) {
    std::ostringstream path;
    path
        << data_dir
        << "/layer_weights_shard"
        << std::setw(2)
        << std::setfill('0')
        << shard
        << ".fix16.bin";
    return path.str();
}

template <typename Vector>
void load_exact_file(
    const std::string& path,
    Vector& destination,
    const char* description
) {
    const std::size_t expected = destination.size() * sizeof(destination[0]);
    const std::size_t actual = file_size(path);
    if (actual != expected) {
        std::ostringstream message;
        message
            << description
            << " size mismatch: "
            << path
            << " expected="
            << expected
            << " actual="
            << actual;
        throw std::runtime_error(message.str());
    }

    std::ifstream input(path.c_str(), std::ios::binary);
    input.read(
        reinterpret_cast<char*>(destination.data()),
        std::streamsize(expected)
    );
    if (!input) {
        throw std::runtime_error("failed to read " + path);
    }
}

class model_data_t {
public:
    explicit model_data_t(
        const model_shape_t& shape,
        bool thin_weight_shards = false
    )
        : shape_(shape),
          thin_weight_shards_(thin_weight_shards),
          embedding_(shape.embedding_values()),
          norm_weights_(shape.norm_values()) {
        const std::size_t shard_words =
            thin_weight_shards_ ? 1 : shape.weight_shard_words();
        for (auto& shard : weight_shards_) {
            shard.resize(shard_words);
        }
    }

    void load(
        const std::string& data_dir,
        bool zero_model,
        bool tie_embeddings,
        bool require_lm_head
    ) {
        if (zero_model) {
            initialize_zero_model();
            use_tied_lm_head_ = true;
            std::cout << "using deterministic zero-weight model\n";
            return;
        }

        load_exact_file(
            data_dir + "/token_embedding.fix16.bin",
            embedding_,
            "embedding"
        );
        load_exact_file(
            data_dir + "/norm_weights.fix16.bin",
            norm_weights_,
            "norm weights"
        );
        for (unsigned int shard = 0; shard < kWeightShardCount; shard++) {
            load_exact_file(
                shard_path(data_dir, shard),
                weight_shards_[shard],
                "weight shard"
            );
        }
        const std::string lm_head_path =
            data_dir + "/lm_head.fix16.bin";
        if (tie_embeddings) {
            use_tied_lm_head_ = true;
        } else if (file_exists(lm_head_path)) {
            lm_head_.resize(shape_.embedding_values());
            load_exact_file(lm_head_path, lm_head_, "LM head");
        } else if (require_lm_head) {
            throw std::runtime_error(
                "generate mode requires lm_head.fix16.bin or "
                "--tie-embeddings"
            );
        }
        std::cout << "loaded packed Fix16 model from " << data_dir << "\n";
    }

    void initialize_random_model(uint32_t seed) {
        std::mt19937 generator(seed);
        std::uniform_int_distribution<int> feature_distribution(-128, 128);
        std::uniform_int_distribution<int> norm_distribution(224, 288);
        std::uniform_int_distribution<int> weight_distribution(-64, 64);

        for (auto& value : embedding_) {
            value = int16_t(feature_distribution(generator));
        }
        for (auto& value : norm_weights_) {
            value = int16_t(norm_distribution(generator));
        }
        for (auto& shard : weight_shards_) {
            for (auto& word : shard) {
                for (unsigned int lane = 0;
                     lane < kValuesPerWord;
                     lane++) {
                    word.value[lane] =
                        int16_t(weight_distribution(generator));
                }
            }
        }
        use_tied_lm_head_ = true;
        std::cout
            << "using reproducible random Fix16 model seed="
            << seed
            << "\n";
    }

    tensor_t embedding(unsigned int token_id) const {
        if (token_id >= shape_.vocab_size) {
            throw std::runtime_error("token id is outside the selected profile");
        }
        tensor_t result(1, shape_.hidden_size);
        const std::size_t offset =
            std::size_t(token_id) * shape_.hidden_size;
        std::copy(
            embedding_.begin() + offset,
            embedding_.begin() + offset + shape_.hidden_size,
            result.values.begin()
        );
        return result;
    }

    tensor_t norm_row(unsigned int layer, bool post_attention) const {
        if (layer >= shape_.num_layers) {
            throw std::runtime_error("norm layer is outside the selected profile");
        }
        const std::size_t row = std::size_t(layer) * 2 +
            (post_attention ? 1 : 0);
        return copy_norm_row(row);
    }

    tensor_t final_norm_row() const {
        return copy_norm_row(std::size_t(shape_.num_layers) * 2);
    }

    int16_t projection_weight_raw(
        unsigned int layer,
        operator_kind_t op,
        unsigned int output,
        unsigned int input
    ) const {
        if (thin_weight_shards_) {
            throw std::runtime_error(
                "projection weights are unavailable with thin smoke buffers"
            );
        }
        const projection_spec_t spec = projection_spec(shape_, op);
        if (
            layer >= shape_.num_layers ||
            output >= spec.out_dim ||
            input >= spec.in_dim
        ) {
            throw std::runtime_error("projection weight index is out of range");
        }

        const std::size_t input_tiles = ceildiv(spec.in_dim, 16);
        const std::size_t global_tile =
            std::size_t(layer) * shape_.layer_weight_tiles() +
            spec.base_tile +
            std::size_t(output / 16) * input_tiles +
            input / 16;
        const unsigned int group = unsigned(global_tile & 1u);
        const std::size_t local_block = global_tile >> 1;
        const unsigned int lane_in_tile =
            (output % 16) * 16 + input % 16;
        const unsigned int tile_word =
            lane_in_tile / kValuesPerWord;
        const unsigned int word_lane =
            lane_in_tile % kValuesPerWord;
        const unsigned int shard = group * 8 + tile_word;
        return weight_shards_[shard][local_block].value[word_lane];
    }

    unsigned int lm_head_argmax(const tensor_t& hidden) const {
        if (hidden.rows != 1 || hidden.cols != shape_.hidden_size) {
            throw std::runtime_error("LM head input shape mismatch");
        }

        int64_t best_score = std::numeric_limits<int64_t>::min();
        unsigned int best_token = 0;
        const aligned_fix16_vector& output_weights =
            use_tied_lm_head_ ? embedding_ : lm_head_;
        if (output_weights.empty()) {
            throw std::runtime_error(
                "LM head is unavailable; use generate with "
                "lm_head.fix16.bin or --tie-embeddings"
            );
        }
        for (unsigned int token = 0; token < shape_.vocab_size; token++) {
            const std::size_t offset =
                std::size_t(token) * shape_.hidden_size;
            int64_t score = 0;
            for (unsigned int elem = 0;
                 elem < shape_.hidden_size;
                 elem++) {
                score +=
                    int64_t(hidden.values[elem]) *
                    int64_t(output_weights[offset + elem]);
            }
            if (score > best_score) {
                best_score = score;
                best_token = token;
            }
        }
        return best_token;
    }

    const std::array<aligned_word_vector, kWeightShardCount>&
    weight_shards() const {
        return weight_shards_;
    }

private:
    tensor_t copy_norm_row(std::size_t row) const {
        tensor_t result(1, shape_.hidden_size);
        const std::size_t offset = row * shape_.hidden_size;
        std::copy(
            norm_weights_.begin() + offset,
            norm_weights_.begin() + offset + shape_.hidden_size,
            result.values.begin()
        );
        return result;
    }

    void initialize_zero_model() {
        std::fill(embedding_.begin(), embedding_.end(), int16_t(0));
        for (unsigned int token = 0; token < shape_.vocab_size; token++) {
            for (unsigned int elem = 0;
                 elem < shape_.hidden_size;
                 elem++) {
                const int value =
                    int((token * 17 + elem * 5) % 65) - 32;
                embedding_[
                    std::size_t(token) * shape_.hidden_size + elem
                ] = int16_t(value);
            }
        }
        std::fill(
            norm_weights_.begin(),
            norm_weights_.end(),
            int16_t(1u << kFmFractionBits)
        );
        for (auto& shard : weight_shards_) {
            clear_words(shard);
        }
    }

    const model_shape_t& shape_;
    bool thin_weight_shards_ = false;
    aligned_fix16_vector embedding_;
    aligned_fix16_vector norm_weights_;
    aligned_fix16_vector lm_head_;
    bool use_tied_lm_head_ = false;
    std::array<aligned_word_vector, kWeightShardCount> weight_shards_;
};

class accelerator_t {
public:
    accelerator_t(
        const model_shape_t& shape,
        const model_data_t& model,
        const std::string& xclbin,
        bool verbose_ops,
        bool skip_weight_preload
    )
        : shape_(shape),
          model_(model),
          verbose_ops_(verbose_ops),
          skip_weight_preload_(skip_weight_preload),
          data_words_{
              aligned_word_vector(shape.data_port_words()),
              aligned_word_vector(shape.data_port_words()),
              aligned_word_vector(shape.data_port_words()),
              aligned_word_vector(shape.data_port_words()),
              aligned_word_vector(shape.data_port_words()),
              aligned_word_vector(shape.data_port_words())
          },
          kv_cache_k_words_(shape.kv_cache_words()),
          kv_cache_v_words_(shape.kv_cache_words()),
          status_words_(1) {
        const std::size_t shard_bytes =
            shape.weight_shard_words() * sizeof(word512_t);
        if (shard_bytes > kHbmPseudoChannelBytes) {
            std::ostringstream message;
            message
                << "one weight shard needs "
                << shard_bytes
                << " bytes, but conn_u50_8x64_dual.cfg assigns one "
                << "256 MiB HBM pseudo-channel per shard";
            throw std::runtime_error(message.str());
        }
        initialize_opencl(xclbin);
        create_buffers();
        initialize_kernel_args();
        if (skip_weight_preload_) {
            std::cout
                << "[qwen-host] skip weight preload; use this only for "
                << "load-only or diagnostic zero-model runs\n";
        } else {
            migrate_weights();
        }
    }

    tensor_t run_feature(
        operator_kind_t op,
        const tensor_t& lhs,
        const tensor_t* rhs,
        unsigned int layer,
        unsigned int position
    ) {
        const unsigned int in_dim = feature_input_dim(op);
        const unsigned int out_dim = feature_output_dim(op);
        if (
            lhs.rows == 0 ||
            lhs.rows > kMaxTokensPerLaunch ||
            lhs.cols != in_dim
        ) {
            throw std::runtime_error("feature operator input shape mismatch");
        }
        if (needs_rhs(op)) {
            if (rhs == nullptr || rhs->cols != in_dim) {
                throw std::runtime_error("feature operator RHS shape mismatch");
            }
            const unsigned int expected_rows =
                op == kOpRmsNorm ? 1 : lhs.rows;
            if (rhs->rows != expected_rows) {
                throw std::runtime_error("feature operator RHS row mismatch");
            }
        }

        clear_data();
        pack_feature(lhs, data_words_[2], data_words_[3]);
        if (rhs != nullptr) {
            pack_feature(*rhs, data_words_[4], data_words_[5]);
        }

        const decoded_status_t status = launch(
            op,
            layer,
            lhs.rows,
            position,
            0
        );
        check_status(op, status);
        return unpack_feature(
            data_words_[0],
            data_words_[1],
            lhs.rows,
            out_dim
        );
    }

    tensor_t run_decoder_layer(
        const tensor_t& hidden,
        const tensor_t& attention_norm,
        const tensor_t& ffn_norm,
        unsigned int layer,
        unsigned int position
    ) {
        if (
            hidden.rows != 1 ||
            hidden.cols != shape_.hidden_size ||
            attention_norm.rows != 1 ||
            attention_norm.cols != shape_.hidden_size ||
            ffn_norm.rows != 1 ||
            ffn_norm.cols != shape_.hidden_size ||
            layer >= shape_.num_layers ||
            position >= shape_.max_seq_len
        ) {
            throw std::runtime_error("resident decoder layer input shape mismatch");
        }

        clear_data();
        pack_feature(hidden, data_words_[2], data_words_[3]);
        pack_tight(attention_norm, data_words_[4]);
        pack_tight(ffn_norm, data_words_[5]);

        const std::size_t norm_words =
            ceildiv(shape_.hidden_size, kValuesPerWord);
        const std::size_t rope_words =
            ceildiv(shape_.head_dim() / 2, kValuesPerWord);
        if (norm_words + 2 * rope_words > data_words_[4].size()) {
            throw std::runtime_error("resident layer aux0 is too small for RoPE");
        }
        for (unsigned int i = 0; i < shape_.head_dim() / 2; i++) {
            const double exponent = double(2 * i) / shape_.head_dim();
            const double inverse_frequency =
                1.0 / std::pow(kRopeTheta, exponent);
            const double angle = position * inverse_frequency;
            const std::size_t word = i / kValuesPerWord;
            const unsigned int lane = i % kValuesPerWord;
            data_words_[4][norm_words + word].value[lane] =
                quantize_fix16(std::cos(angle));
            data_words_[4][norm_words + rope_words + word].value[lane] =
                quantize_fix16(std::sin(angle));
        }

        const decoded_status_t status = launch(
            kOpDecoderLayer,
            layer,
            1,
            position,
            0
        );
        check_status(kOpDecoderLayer, status);
        return unpack_feature(
            data_words_[0],
            data_words_[1],
            1,
            shape_.hidden_size
        );
    }

    composed_layer_result_t run_composed_decoder_stack(
        const tensor_t& hidden,
        unsigned int layer_begin,
        unsigned int layer_count,
        unsigned int position,
        bool include_final_norm
    ) {
        if (
            hidden.rows != 1 ||
            hidden.cols != shape_.hidden_size ||
            layer_count == 0 ||
            layer_begin >= shape_.num_layers ||
            layer_begin + layer_count > shape_.num_layers ||
            position >= shape_.max_seq_len
        ) {
            throw std::runtime_error(
                "composed decoder stack input shape mismatch"
            );
        }

        clear_data();
        pack_feature(hidden, data_words_[2], data_words_[3]);

        const std::size_t norm_words =
            ceildiv(shape_.hidden_size, kValuesPerWord);
        const std::size_t rope_words =
            ceildiv(shape_.head_dim() / 2, kValuesPerWord);
        if (norm_words + 2 * rope_words > data_words_[4].size()) {
            throw std::runtime_error(
                "composed layer aux0 is too small for RoPE"
            );
        }
        composed_layer_result_t result;
        result.attention_controller_ms = 0.0;
        result.ffn_controller_ms = 0.0;
        result.layer_count = layer_count;
        result.task_count =
            2 * layer_count + (include_final_norm ? 1u : 0u);
        const auto host_begin = std::chrono::steady_clock::now();
        std::vector<cl::Memory> initial_inputs = {
            data_buffers_[2],
            data_buffers_[3],
            status_buffer_
        };
        cl::Event initial_input_event;
        check_cl(
            transfer_queue_.enqueueMigrateMemObjects(
                initial_inputs,
                0,
                nullptr,
                &initial_input_event
            ),
            "migrate composed layer inputs"
        );
        check_cl(
            transfer_queue_.finish(),
            "finish composed layer input migration"
        );
        result.input_migration_ms =
            event_milliseconds(initial_input_event);

        try {
            for (unsigned int layer_offset = 0;
                 layer_offset < layer_count;
                 layer_offset++) {
                const unsigned int layer = layer_begin + layer_offset;
                clear_words(data_words_[4]);
                clear_words(data_words_[5]);
                pack_tight(model_.norm_row(layer, false), data_words_[4]);
                pack_tight(model_.norm_row(layer, true), data_words_[5]);
                for (unsigned int i = 0;
                     i < shape_.head_dim() / 2;
                     i++) {
                    const double exponent =
                        double(2 * i) / shape_.head_dim();
                    const double inverse_frequency =
                        1.0 / std::pow(kRopeTheta, exponent);
                    const double angle = position * inverse_frequency;
                    const std::size_t word = i / kValuesPerWord;
                    const unsigned int lane = i % kValuesPerWord;
                    data_words_[4][norm_words + word].value[lane] =
                        quantize_fix16(std::cos(angle));
                    data_words_[4][norm_words + rope_words + word]
                        .value[lane] =
                        quantize_fix16(std::sin(angle));
                }
                std::vector<cl::Memory> layer_aux = {
                    data_buffers_[4],
                    data_buffers_[5]
                };
                cl::Event layer_aux_event;
                check_cl(
                    transfer_queue_.enqueueMigrateMemObjects(
                        layer_aux,
                        0,
                        nullptr,
                        &layer_aux_event
                    ),
                    "migrate composed layer auxiliary inputs"
                );
                check_cl(
                    transfer_queue_.finish(),
                    "finish composed layer auxiliary migration"
                );
                result.auxiliary_migration_ms =
                    add_profiled_milliseconds(
                        result.auxiliary_migration_ms,
                        event_milliseconds(layer_aux_event)
                    );

                // Task A: input pair 2/3 -> attention residual pair 0/1.
                bind_controller_data_ports(0, 1, 2, 3);
                result.attention_status = execute_bound_resident_task(
                    kOpAttentionSublayer,
                    layer,
                    position
                );
                result.attention_controller_ms += last_controller_ms_;
                result.status_kernel_ms = add_profiled_milliseconds(
                    result.status_kernel_ms,
                    last_status_kernel_ms_
                );
                result.status_migration_ms = add_profiled_milliseconds(
                    result.status_migration_ms,
                    last_status_migration_ms_
                );
                check_status(
                    kOpAttentionSublayer,
                    result.attention_status
                );

                // Task B consumes pair 0/1 directly and writes pair 2/3.
                // The completed hidden state therefore remains in the same
                // device pair for the next layer's Task A.
                bind_controller_data_ports(2, 3, 0, 1);
                result.ffn_status = execute_bound_resident_task(
                    kOpFfnSublayer,
                    layer,
                    position
                );
                result.ffn_controller_ms += last_controller_ms_;
                result.status_kernel_ms = add_profiled_milliseconds(
                    result.status_kernel_ms,
                    last_status_kernel_ms_
                );
                result.status_migration_ms = add_profiled_milliseconds(
                    result.status_migration_ms,
                    last_status_migration_ms_
                );
                check_status(kOpFfnSublayer, result.ffn_status);
            }

            unsigned int final_output0 = 2;
            unsigned int final_output1 = 3;
            if (include_final_norm) {
                clear_words(data_words_[4]);
                clear_words(data_words_[5]);
                pack_tight(model_.final_norm_row(), data_words_[4]);
                std::vector<cl::Memory> final_aux = {data_buffers_[4]};
                cl::Event final_aux_event;
                check_cl(
                    transfer_queue_.enqueueMigrateMemObjects(
                        final_aux,
                        0,
                        nullptr,
                        &final_aux_event
                    ),
                    "migrate final norm weights"
                );
                check_cl(
                    transfer_queue_.finish(),
                    "finish final norm weight migration"
                );
                result.auxiliary_migration_ms =
                    add_profiled_milliseconds(
                        result.auxiliary_migration_ms,
                        event_milliseconds(final_aux_event)
                    );
                bind_controller_data_ports(0, 1, 2, 3);
                result.final_norm_status = execute_bound_resident_task(
                    kOpFinalNorm,
                    0,
                    position
                );
                result.final_norm_controller_ms = last_controller_ms_;
                result.status_kernel_ms = add_profiled_milliseconds(
                    result.status_kernel_ms,
                    last_status_kernel_ms_
                );
                result.status_migration_ms = add_profiled_milliseconds(
                    result.status_migration_ms,
                    last_status_migration_ms_
                );
                check_status(kOpFinalNorm, result.final_norm_status);
                final_output0 = 0;
                final_output1 = 1;
            }

            std::vector<cl::Memory> final_outputs = {
                data_buffers_[final_output0],
                data_buffers_[final_output1]
            };
            cl::Event final_output_event;
            check_cl(
                transfer_queue_.enqueueMigrateMemObjects(
                    final_outputs,
                    CL_MIGRATE_MEM_OBJECT_HOST,
                    nullptr,
                    &final_output_event
                ),
                "migrate composed stack outputs"
            );
            check_cl(
                transfer_queue_.finish(),
                "finish composed stack output migration"
            );
            result.output_migration_ms =
                event_milliseconds(final_output_event);
        } catch (...) {
            bind_controller_data_ports(0, 1, 2, 3);
            throw;
        }
        bind_controller_data_ports(0, 1, 2, 3);

        const auto host_end = std::chrono::steady_clock::now();
        result.kernel_active_ms =
            result.attention_controller_ms +
            result.ffn_controller_ms +
            result.final_norm_controller_ms;
        result.profiled_sequence_ms = result.kernel_active_ms;
        result.profiled_sequence_ms = add_profiled_milliseconds(
            result.profiled_sequence_ms,
            result.input_migration_ms
        );
        result.profiled_sequence_ms = add_profiled_milliseconds(
            result.profiled_sequence_ms,
            result.auxiliary_migration_ms
        );
        result.profiled_sequence_ms = add_profiled_milliseconds(
            result.profiled_sequence_ms,
            result.status_kernel_ms
        );
        result.profiled_sequence_ms = add_profiled_milliseconds(
            result.profiled_sequence_ms,
            result.status_migration_ms
        );
        result.profiled_sequence_ms = add_profiled_milliseconds(
            result.profiled_sequence_ms,
            result.output_migration_ms
        );
        result.host_elapsed_ms =
            std::chrono::duration<double, std::milli>(
                host_end - host_begin
            ).count();
        const unsigned int host_output0 = include_final_norm ? 0 : 2;
        const unsigned int host_output1 = include_final_norm ? 1 : 3;
        result.output = unpack_feature(
            data_words_[host_output0],
            data_words_[host_output1],
            1,
            shape_.hidden_size
        );
        return result;
    }

    decoded_status_t run_mm_wave_profile(
        operator_kind_t op,
        const tensor_t& lhs,
        unsigned int layer,
        unsigned int wave_begin,
        unsigned int wave_count,
        unsigned int k_limit,
        unsigned int debug_stage,
        unsigned int core_mask,
        bool zero_weight_stream
    ) {
        const unsigned int in_dim = feature_input_dim(op);
        if (lhs.rows == 0 || lhs.rows > kMaxTokensPerLaunch ||
            lhs.cols != in_dim) {
            throw std::runtime_error("MM wave profile input shape mismatch");
        }
        switch (op) {
        case kOpQProjection:
        case kOpKProjection:
        case kOpVProjection:
        case kOpOProjection:
        case kOpFfnGate:
        case kOpFfnUp:
        case kOpFfnDown:
            break;
        default:
            throw std::runtime_error("operator is not an MM profile operator");
        }

        clear_data();
        pack_feature(lhs, data_words_[2], data_words_[3]);
        if (wave_count > 0xffffu) {
            throw std::runtime_error("MM wave profile wave_count exceeds 16-bit encoding");
        }
        if (k_limit > 0xffffu) {
            throw std::runtime_error("MM wave profile k_limit exceeds 16-bit encoding");
        }
        if (debug_stage > 15u) {
            throw std::runtime_error("MM wave profile debug stage exceeds 4-bit encoding");
        }
        if (core_mask == 0 || core_mask > 3u) {
            throw std::runtime_error("MM wave profile core mask must be 1, 2, or 3");
        }
        const unsigned int encoded_tile_len =
            (k_limit << 16) | (wave_count & 0xffffu);
        const unsigned int encoded_token_count =
            lhs.rows |
            (zero_weight_stream ? (1u << 16) : 0u) |
            ((debug_stage & 0xfu) << 17) |
            ((core_mask & 0x3u) << 21);
        const decoded_status_t status = launch(
            op,
            layer,
            encoded_token_count,
            wave_begin,
            encoded_tile_len
        );
        check_status(op, status);
        return status;
    }

    operator_result_t run_attention(
        operator_kind_t op,
        const tensor_t& input0,
        const tensor_t& input1,
        const tensor_t* aux0,
        const tensor_t* aux1,
        unsigned int position,
        unsigned int tile_len,
        unsigned int layer = 0
    ) {
        const unsigned int rows = shape_.gqa_group_size();
        const unsigned int source_cols =
            (op == kOpAttentionQk ||
             op == kOpAttentionFlash ||
             op == kOpDecodeSmoke) ?
            shape_.head_dim() :
            kAttentionTile;
        if (
            input0.rows != rows ||
            input1.rows != rows ||
            input0.cols != source_cols ||
            input1.cols != source_cols ||
            tile_len == 0 ||
            tile_len > kAttentionTile
        ) {
            throw std::runtime_error("attention source shape mismatch");
        }
        if (op == kOpDecodeSmoke) {
            if (
                aux0 == nullptr ||
                aux1 == nullptr ||
                aux0->rows != shape_.num_kv_heads ||
                aux1->rows != shape_.num_kv_heads ||
                aux0->cols != shape_.head_dim() ||
                aux1->cols != shape_.head_dim()
            ) {
                throw std::runtime_error("decode smoke KV shape mismatch");
            }
        } else if (op != kOpSoftmax && op != kOpAttentionFlash) {
            if (
                aux0 == nullptr ||
                aux1 == nullptr ||
                aux0->rows != tile_len ||
                aux1->rows != tile_len ||
                aux0->cols != shape_.head_dim() ||
                aux1->cols != shape_.head_dim()
            ) {
                throw std::runtime_error("attention panel shape mismatch");
            }
        }

        clear_data();
        pack_tight(input0, data_words_[2]);
        pack_tight(input1, data_words_[3]);
        if (aux0 != nullptr) {
            pack_tight(*aux0, data_words_[4]);
            pack_tight(*aux1, data_words_[5]);
        }

        const decoded_status_t status = launch(
            op,
            layer,
            rows,
            position,
            tile_len
        );
        check_status(op, status);

        unsigned int output_cols = 0;
        if (op == kOpAttentionQk) {
            output_cols = kAttentionTile;
        } else if (
            op == kOpAttentionPv ||
            op == kOpAttentionFlash ||
            op == kOpDecodeSmoke
        ) {
            output_cols = shape_.head_dim();
        } else if (op == kOpSoftmax) {
            output_cols = tile_len;
        } else {
            throw std::runtime_error("invalid attention operator");
        }

        operator_result_t result;
        result.port0 = unpack_tight(data_words_[0], rows, output_cols);
        result.port1 = unpack_tight(data_words_[1], rows, output_cols);
        result.status = status;
        result.controller_ms = last_controller_ms_;
        return result;
    }

    operator_result_t run_prefill_attention_block(
        const tensor_t& query0,
        const tensor_t& query1,
        unsigned int position,
        unsigned int token_count,
        unsigned int layer = 0
    ) {
        const unsigned int query_rows =
            token_count * shape_.gqa_group_size();
        if (
            token_count == 0 ||
            token_count > kMaxTokensPerLaunch ||
            position + token_count > shape_.max_seq_len ||
            query0.rows != query_rows ||
            query1.rows != query_rows ||
            query0.cols != shape_.head_dim() ||
            query1.cols != shape_.head_dim()
        ) {
            throw std::runtime_error("prefill attention block shape mismatch");
        }

        clear_data();
        // The block-attention ABI is token-major, then local GQA-head-major.
        pack_tight(query0, data_words_[2]);
        pack_tight(query1, data_words_[3]);
        const decoded_status_t status = launch(
            kOpAttnPrefillBlock,
            layer,
            token_count,
            position,
            0
        );
        check_status(kOpAttnPrefillBlock, status);

        operator_result_t result;
        result.port0 = unpack_tight(
            data_words_[0], query_rows, shape_.head_dim()
        );
        result.port1 = unpack_tight(
            data_words_[1], query_rows, shape_.head_dim()
        );
        result.status = status;
        result.controller_ms = last_controller_ms_;
        return result;
    }

    decoded_status_t run_nop() {
        clear_data();
        const decoded_status_t status = launch(kOpNop, 0, 0, 0, 0);
        check_status(kOpNop, status);
        return status;
    }

    decoded_status_t run_nop_controller_only() {
        clear_data();
        const decoded_status_t status = launch_controller_status_only(
            kOpNop,
            0,
            0,
            0,
            0
        );
        check_status(kOpNop, status);
        return status;
    }

    decoded_status_t run_nop_controller_enqueue_only() {
        clear_data();
        launch_controller_only_no_status(
            kOpNop,
            0,
            0,
            0,
            0
        );
        decoded_status_t status;
        status.op = unsigned(kOpNop);
        status.code = 0;
        status.last_task = true;
        return status;
    }

    void initialize_kv_cache_heads(
        unsigned int layer,
        const tensor_t& k0,
        const tensor_t& k1,
        const tensor_t& v0,
        const tensor_t& v1
    ) {
        const unsigned int head_dim = shape_.head_dim();
        if (
            layer >= shape_.num_layers ||
            k0.rows != k1.rows ||
            k0.rows != v0.rows ||
            k0.rows != v1.rows ||
            k0.rows > shape_.max_seq_len ||
            k0.cols != head_dim ||
            k1.cols != head_dim ||
            v0.cols != head_dim ||
            v1.cols != head_dim
        ) {
            throw std::runtime_error("KV cache preload shape mismatch");
        }

        clear_words(kv_cache_k_words_);
        clear_words(kv_cache_v_words_);
        const std::size_t head_words = ceildiv(head_dim, kValuesPerWord);
        const auto cache_index = [this, head_words](
            unsigned int cache_layer,
            unsigned int position,
            unsigned int kv_head,
            unsigned int word_idx
        ) {
            return
                (
                    (
                        std::size_t(cache_layer) * shape_.max_seq_len +
                        position
                    ) *
                    shape_.num_kv_heads +
                    kv_head
                ) *
                head_words +
                word_idx;
        };
        const std::size_t transposed_base =
            shape_.kv_cache_row_major_words();
        const std::size_t position_words =
            ceildiv(shape_.max_seq_len, kValuesPerWord);
        const auto transposed_k_index = [
            this,
            transposed_base,
            position_words
        ](
            unsigned int cache_layer,
            unsigned int kv_head,
            unsigned int elem,
            unsigned int position_word
        ) {
            return
                transposed_base +
                (
                    (
                        std::size_t(cache_layer) * shape_.num_kv_heads +
                        kv_head
                    ) *
                    shape_.head_dim() +
                    elem
                ) *
                position_words +
                position_word;
        };

        for (unsigned int position = 0; position < k0.rows; position++) {
            for (unsigned int elem = 0; elem < head_dim; elem++) {
                const unsigned int word_idx = elem / kValuesPerWord;
                const unsigned int lane = elem % kValuesPerWord;
                kv_cache_k_words_[cache_index(layer, position, 0, word_idx)]
                    .value[lane] = k0.at(position, elem);
                kv_cache_k_words_[cache_index(layer, position, 1, word_idx)]
                    .value[lane] = k1.at(position, elem);
                const unsigned int position_word =
                    position / kValuesPerWord;
                const unsigned int position_lane =
                    position % kValuesPerWord;
                kv_cache_k_words_[transposed_k_index(
                    layer,
                    0,
                    elem,
                    position_word
                )].value[position_lane] = k0.at(position, elem);
                kv_cache_k_words_[transposed_k_index(
                    layer,
                    1,
                    elem,
                    position_word
                )].value[position_lane] = k1.at(position, elem);
                kv_cache_v_words_[cache_index(layer, position, 0, word_idx)]
                    .value[lane] = v0.at(position, elem);
                kv_cache_v_words_[cache_index(layer, position, 1, word_idx)]
                    .value[lane] = v1.at(position, elem);
            }
        }

        std::vector<cl::Memory> buffers = {
            kv_cache_k_buffer_,
            kv_cache_v_buffer_
        };
        check_cl(
            transfer_queue_.enqueueMigrateMemObjects(buffers, 0),
            "migrate preloaded KV cache buffers"
        );
        check_cl(
            transfer_queue_.finish(),
            "finish preloaded KV cache migration"
        );
    }

private:
    void bind_controller_data_ports(
        unsigned int output0,
        unsigned int output1,
        unsigned int input0,
        unsigned int input1
    ) {
        if (
            output0 >= data_buffers_.size() ||
            output1 >= data_buffers_.size() ||
            input0 >= data_buffers_.size() ||
            input1 >= data_buffers_.size()
        ) {
            throw std::runtime_error("controller data-port index out of range");
        }
        check_cl(
            controller_kernel_.setArg(
                kControllerOutput0Arg,
                data_buffers_[output0]
            ),
            "bind controller output port 0"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerOutput1Arg,
                data_buffers_[output1]
            ),
            "bind controller output port 1"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerInput0Arg,
                data_buffers_[input0]
            ),
            "bind controller input port 0"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerInput1Arg,
                data_buffers_[input1]
            ),
            "bind controller input port 1"
        );
    }

    decoded_status_t execute_bound_resident_task(
        operator_kind_t op,
        unsigned int layer,
        unsigned int position
    ) {
        check_cl(
            controller_kernel_.setArg(kControllerOperatorArg, unsigned(op)),
            "set composed task operator"
        );
        check_cl(
            controller_kernel_.setArg(kControllerLayerArg, layer),
            "set composed task layer"
        );
        check_cl(
            controller_kernel_.setArg(kControllerTokenCountArg, 1u),
            "set composed task token count"
        );
        check_cl(
            controller_kernel_.setArg(kControllerPositionArg, position),
            "set composed task position"
        );
        check_cl(
            controller_kernel_.setArg(kControllerTileLenArg, 0u),
            "set composed task tile length"
        );

        cl::Event controller_event;
        cl::Event compute0_event;
        cl::Event compute1_event;
        cl::Event status_event;
        check_cl(
            controller_queue_.enqueueTask(
                controller_kernel_,
                nullptr,
                &controller_event
            ),
            "enqueue composed controller task"
        );
        check_cl(
            compute0_queue_.enqueueTask(
                compute0_kernel_,
                nullptr,
                &compute0_event
            ),
            "enqueue composed compute0 task"
        );
        check_cl(
            compute1_queue_.enqueueTask(
                compute1_kernel_,
                nullptr,
                &compute1_event
            ),
            "enqueue composed compute1 task"
        );

        std::array<cl_int, 3> flush_status = {{
            CL_SUCCESS,
            CL_SUCCESS,
            CL_SUCCESS
        }};
        const auto flush_queue = [](
            cl::CommandQueue& queue,
            cl_int& status
        ) {
            status = queue.flush();
        };
        std::thread flush_compute0(
            flush_queue,
            std::ref(compute0_queue_),
            std::ref(flush_status[0])
        );
        std::thread flush_compute1(
            flush_queue,
            std::ref(compute1_queue_),
            std::ref(flush_status[1])
        );
        std::thread flush_controller(
            flush_queue,
            std::ref(controller_queue_),
            std::ref(flush_status[2])
        );
        flush_compute0.join();
        flush_compute1.join();
        flush_controller.join();
        check_cl(flush_status[0], "flush composed compute0");
        check_cl(flush_status[1], "flush composed compute1");
        check_cl(flush_status[2], "flush composed controller");

        controller_event.wait();
        compute0_event.wait();
        compute1_event.wait();
        last_controller_ms_ = event_milliseconds(controller_event);

        check_cl(
            status_queue_.enqueueTask(
                status_kernel_,
                nullptr,
                &status_event
            ),
            "enqueue composed status task"
        );
        check_cl(status_queue_.flush(), "flush composed status task");
        status_event.wait();
        last_status_kernel_ms_ = event_milliseconds(status_event);
        std::vector<cl::Memory> status_only = {status_buffer_};
        cl::Event status_migration_event;
        check_cl(
            transfer_queue_.enqueueMigrateMemObjects(
                status_only,
                CL_MIGRATE_MEM_OBJECT_HOST,
                nullptr,
                &status_migration_event
            ),
            "migrate composed task status"
        );
        check_cl(
            transfer_queue_.finish(),
            "finish composed task status migration"
        );
        last_status_migration_ms_ =
            event_milliseconds(status_migration_event);
        return decode_status(status_words_[0]);
    }

    unsigned int feature_input_dim(operator_kind_t op) const {
        switch (op) {
        case kOpQProjection:
        case kOpKProjection:
        case kOpVProjection:
        case kOpFfnGate:
        case kOpFfnUp:
        case kOpRmsNorm:
        case kOpResidualAdd:
            return shape_.hidden_size;
        case kOpOProjection:
            return shape_.hidden_size;
        case kOpFfnDown:
        case kOpSiluMul:
            return shape_.intermediate_size;
        default:
            throw std::runtime_error("operator is not a feature operator");
        }
    }

    unsigned int feature_output_dim(operator_kind_t op) const {
        switch (op) {
        case kOpQProjection:
        case kOpOProjection:
        case kOpFfnDown:
        case kOpRmsNorm:
        case kOpResidualAdd:
            return shape_.hidden_size;
        case kOpKProjection:
        case kOpVProjection:
            return shape_.kv_channels();
        case kOpFfnGate:
        case kOpFfnUp:
        case kOpSiluMul:
            return shape_.intermediate_size;
        default:
            throw std::runtime_error("operator is not a feature operator");
        }
    }

    bool needs_rhs(operator_kind_t op) const {
        return
            op == kOpRmsNorm ||
            op == kOpSiluMul ||
            op == kOpResidualAdd;
    }

    void pack_feature(
        const tensor_t& tensor,
        aligned_word_vector& port0,
        aligned_word_vector& port1
    ) {
        const std::size_t stride = shape_.feature_words_per_token();
        for (unsigned int token = 0; token < tensor.rows; token++) {
            const unsigned int port = token / kTokensPerPort;
            const unsigned int token_in_port = token % kTokensPerPort;
            aligned_word_vector& destination = port == 0 ? port0 : port1;
            for (unsigned int elem = 0; elem < tensor.cols; elem++) {
                const std::size_t word =
                    std::size_t(token_in_port) * stride +
                    elem / kValuesPerWord;
                destination[word].value[elem % kValuesPerWord] =
                    tensor.at(token, elem);
            }
        }
    }

    tensor_t unpack_feature(
        const aligned_word_vector& port0,
        const aligned_word_vector& port1,
        unsigned int rows,
        unsigned int cols
    ) const {
        tensor_t tensor(rows, cols);
        const std::size_t stride = shape_.feature_words_per_token();
        for (unsigned int token = 0; token < rows; token++) {
            const unsigned int port = token / kTokensPerPort;
            const unsigned int token_in_port = token % kTokensPerPort;
            const aligned_word_vector& source = port == 0 ? port0 : port1;
            for (unsigned int elem = 0; elem < cols; elem++) {
                const std::size_t word =
                    std::size_t(token_in_port) * stride +
                    elem / kValuesPerWord;
                tensor.at(token, elem) =
                    source[word].value[elem % kValuesPerWord];
            }
        }
        return tensor;
    }

    void pack_tight(
        const tensor_t& tensor,
        aligned_word_vector& destination
    ) {
        const std::size_t words_per_row =
            ceildiv(tensor.cols, kValuesPerWord);
        if (std::size_t(tensor.rows) * words_per_row > destination.size()) {
            throw std::runtime_error("tight tensor exceeds controller port");
        }
        for (unsigned int row = 0; row < tensor.rows; row++) {
            for (unsigned int elem = 0; elem < tensor.cols; elem++) {
                const std::size_t word =
                    std::size_t(row) * words_per_row +
                    elem / kValuesPerWord;
                destination[word].value[elem % kValuesPerWord] =
                    tensor.at(row, elem);
            }
        }
    }

    tensor_t unpack_tight(
        const aligned_word_vector& source,
        unsigned int rows,
        unsigned int cols
    ) const {
        tensor_t tensor(rows, cols);
        const std::size_t words_per_row = ceildiv(cols, kValuesPerWord);
        for (unsigned int row = 0; row < rows; row++) {
            for (unsigned int elem = 0; elem < cols; elem++) {
                const std::size_t word =
                    std::size_t(row) * words_per_row +
                    elem / kValuesPerWord;
                tensor.at(row, elem) =
                    source[word].value[elem % kValuesPerWord];
            }
        }
        return tensor;
    }

    void clear_data() {
        for (auto& words : data_words_) {
            clear_words(words);
        }
        clear_words(status_words_);
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
            throw std::runtime_error(
                "failed to program a Xilinx device with " + xclbin
            );
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

        std::cout
            << "programmed "
            << device_.getInfo<CL_DEVICE_NAME>()
            << " for profile "
            << shape_.name
            << "\n";
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
            weight_ext_[shard].obj =
                const_cast<word512_t*>(
                    model_.weight_shards()[shard].data()
                );
            weight_ext_[shard].param = 0;
            weight_ext_[shard].flags = hbm_bank(4 + shard);
            weight_buffers_[shard] = cl::Buffer(
                context_,
                CL_MEM_EXT_PTR_XILINX |
                    CL_MEM_USE_HOST_PTR |
                    CL_MEM_READ_ONLY,
                model_.weight_shards()[shard].size() * sizeof(word512_t),
                &weight_ext_[shard],
                &err
            );
            check_cl(err, "create weight HBM buffer");
        }

        kv_cache_k_ext_.obj = kv_cache_k_words_.data();
        kv_cache_k_ext_.param = 0;
        kv_cache_k_ext_.flags = hbm_bank(20);
        kv_cache_k_buffer_ = cl::Buffer(
            context_,
            CL_MEM_EXT_PTR_XILINX |
                CL_MEM_USE_HOST_PTR |
                CL_MEM_READ_WRITE,
            kv_cache_k_words_.size() * sizeof(word512_t),
            &kv_cache_k_ext_,
            &err
        );
        check_cl(err, "create K cache HBM buffer");

        kv_cache_v_ext_.obj = kv_cache_v_words_.data();
        kv_cache_v_ext_.param = 0;
        kv_cache_v_ext_.flags = hbm_bank(21);
        kv_cache_v_buffer_ = cl::Buffer(
            context_,
            CL_MEM_EXT_PTR_XILINX |
                CL_MEM_USE_HOST_PTR |
                CL_MEM_READ_WRITE,
            kv_cache_v_words_.size() * sizeof(word512_t),
            &kv_cache_v_ext_,
            &err
        );
        check_cl(err, "create V cache HBM buffer");

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
        check_cl(
            status_kernel_.setArg(kStatusOutputArg, status_buffer_),
            "set status output"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerOutput0Arg,
                data_buffers_[0]
            ),
            "set output port 0"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerOutput1Arg,
                data_buffers_[1]
            ),
            "set output port 1"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerInput0Arg,
                data_buffers_[2]
            ),
            "set input port 0"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerInput1Arg,
                data_buffers_[3]
            ),
            "set input port 1"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerAux0Arg,
                data_buffers_[4]
            ),
            "set aux port 0"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerAux1Arg,
                data_buffers_[5]
            ),
            "set aux port 1"
        );
        for (unsigned int shard = 0; shard < kWeightShardCount; shard++) {
            check_cl(
                controller_kernel_.setArg(
                    kControllerWeight0Arg + shard,
                    weight_buffers_[shard]
                ),
                "set weight shard"
            );
        }
        check_cl(
            controller_kernel_.setArg(
                kControllerKvCacheKArg,
                kv_cache_k_buffer_
            ),
            "set K cache buffer"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerKvCacheVArg,
                kv_cache_v_buffer_
            ),
            "set V cache buffer"
        );
    }

    void migrate_weights() {
        std::vector<cl::Memory> buffers;
        for (const auto& buffer : weight_buffers_) {
            buffers.push_back(buffer);
        }
        buffers.push_back(kv_cache_k_buffer_);
        buffers.push_back(kv_cache_v_buffer_);
        check_cl(
            transfer_queue_.enqueueMigrateMemObjects(buffers, 0),
            "migrate weight and KV cache buffers"
        );
        check_cl(
            transfer_queue_.finish(),
            "finish weight and KV cache migration"
        );
    }

    decoded_status_t launch(
        operator_kind_t op,
        unsigned int layer,
        unsigned int token_count,
        unsigned int position,
        unsigned int tile_len
    ) {
        check_cl(
            controller_kernel_.setArg(
                kControllerOperatorArg,
                unsigned(op)
            ),
            "set operator"
        );
        check_cl(
            controller_kernel_.setArg(kControllerLayerArg, layer),
            "set layer"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerTokenCountArg,
                token_count
            ),
            "set token count"
        );
        check_cl(
            controller_kernel_.setArg(kControllerPositionArg, position),
            "set position"
        );
        check_cl(
            controller_kernel_.setArg(kControllerTileLenArg, tile_len),
            "set tile length"
        );

        std::vector<cl::Memory> to_device;
        for (const auto& buffer : data_buffers_) {
            to_device.push_back(buffer);
        }
        to_device.push_back(status_buffer_);
        if (verbose_ops_) {
            std::cout << "migrating operator inputs\n" << std::flush;
        }
        check_cl(
            transfer_queue_.enqueueMigrateMemObjects(to_device, 0),
            "migrate operator inputs"
        );
        if (verbose_ops_) {
            std::cout << "operator input migrate enqueued\n" << std::flush;
        }
        check_cl(transfer_queue_.finish(), "finish operator input migration");
        if (verbose_ops_) {
            std::cout << "operator input migration finished\n" << std::flush;
        }

        cl::Event compute0_event;
        cl::Event compute1_event;
        cl::Event status_event;
        cl::Event controller_event;
        if (verbose_ops_) {
            std::cout << "enqueue cc8_ctrl\n" << std::flush;
        }
        check_cl(
            controller_queue_.enqueueTask(
                controller_kernel_,
                nullptr,
                &controller_event
            ),
            "enqueue cc8_ctrl"
        );
        if (verbose_ops_) {
            std::cout << "enqueue cc8_ctrl returned\n" << std::flush;
        }
        if (verbose_ops_) {
            std::cout << "enqueue cc8_cu0\n" << std::flush;
        }
        check_cl(
            compute0_queue_.enqueueTask(
                compute0_kernel_,
                nullptr,
                &compute0_event
            ),
            "enqueue cc8_cu0"
        );
        if (verbose_ops_) {
            std::cout << "enqueue cc8_cu0 returned\n" << std::flush;
        }
        if (verbose_ops_) {
            std::cout << "enqueue cc8_cu1\n" << std::flush;
        }
        check_cl(
            compute1_queue_.enqueueTask(
                compute1_kernel_,
                nullptr,
                &compute1_event
            ),
            "enqueue cc8_cu1"
        );
        if (verbose_ops_) {
            std::cout << "enqueue cc8_cu1 returned\n" << std::flush;
        }

        const auto flush_queue = [this](
            const char* name,
            cl::CommandQueue& queue,
            cl_int& status
        ) {
            if (verbose_ops_) {
                std::cout << "flushing " << name << " queue\n" << std::flush;
            }
            status = queue.flush();
            if (verbose_ops_) {
                std::cout
                    << name << " queue flush returned " << status << "\n"
                    << std::flush;
            }
        };
        std::array<cl_int, 3> flush_status = {{
            CL_SUCCESS,
            CL_SUCCESS,
            CL_SUCCESS,
        }};
        std::thread flush_compute0(
            flush_queue,
            "cc8_cu0",
            std::ref(compute0_queue_),
            std::ref(flush_status[0])
        );
        std::thread flush_compute1(
            flush_queue,
            "cc8_cu1",
            std::ref(compute1_queue_),
            std::ref(flush_status[1])
        );
        std::thread flush_controller(
            flush_queue,
            "cc8_ctrl",
            std::ref(controller_queue_),
            std::ref(flush_status[2])
        );
        flush_compute0.join();
        flush_compute1.join();
        flush_controller.join();
        check_cl(flush_status[0], "flush cc8_cu0");
        check_cl(flush_status[1], "flush cc8_cu1");
        check_cl(flush_status[2], "flush cc8_ctrl");

        if (verbose_ops_) {
            struct tracked_event_t {
                const char* name;
                const cl::Event* event;
                cl_int last_status;
            };
            std::array<tracked_event_t, 3> events = {{
                {"controller", &controller_event, std::numeric_limits<cl_int>::min()},
                {"compute0", &compute0_event, std::numeric_limits<cl_int>::min()},
                {"compute1", &compute1_event, std::numeric_limits<cl_int>::min()}
            }};
            const auto status_name = [](cl_int status) {
                switch (status) {
                case CL_QUEUED: return "QUEUED";
                case CL_SUBMITTED: return "SUBMITTED";
                case CL_RUNNING: return "RUNNING";
                case CL_COMPLETE: return "COMPLETE";
                default: return "ERROR";
                }
            };
            const auto wait_start = std::chrono::steady_clock::now();
            bool all_complete = false;
            while (!all_complete) {
                all_complete = true;
                for (auto& tracked : events) {
                    cl_int event_status = CL_QUEUED;
                    const std::string query_operation =
                        std::string("query ") + tracked.name + " event";
                    check_cl(
                        tracked.event->getInfo(
                            CL_EVENT_COMMAND_EXECUTION_STATUS,
                            &event_status
                        ),
                        query_operation.c_str()
                    );
                    if (event_status != tracked.last_status) {
                        const double elapsed_seconds =
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - wait_start
                            ).count();
                        std::cout
                            << tracked.name << " event "
                            << status_name(event_status)
                            << " (" << event_status << ")"
                            << " host_elapsed_s=" << elapsed_seconds
                            << "\n" << std::flush;
                        tracked.last_status = event_status;
                    }
                    all_complete = all_complete && event_status == CL_COMPLETE;
                }
                if (!all_complete) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
        } else {
            controller_event.wait();
            compute0_event.wait();
            compute1_event.wait();
        }
        last_controller_ms_ = event_milliseconds(controller_event);

        if (verbose_ops_) {
            std::cout
                << "enqueue cc8_status after main kernels complete\n"
                << std::flush;
        }
        check_cl(
            status_queue_.enqueueTask(
                status_kernel_,
                nullptr,
                &status_event
            ),
            "enqueue cc8_status after main kernels"
        );
        if (verbose_ops_) {
            std::cout
                << "enqueue cc8_status after main kernels returned\n"
                << "flush cc8_status after main kernels\n"
                << std::flush;
        }
        check_cl(status_queue_.flush(), "flush cc8_status after main kernels");
        status_event.wait();

        std::vector<cl::Memory> from_device = {
            data_buffers_[0],
            data_buffers_[1],
            status_buffer_
        };
        check_cl(
            transfer_queue_.enqueueMigrateMemObjects(
                from_device,
                CL_MIGRATE_MEM_OBJECT_HOST
            ),
            "migrate operator outputs"
        );
        check_cl(transfer_queue_.finish(), "finish operator output migration");

        const decoded_status_t status = decode_status(status_words_[0]);
        if (verbose_ops_) {
            std::cout
                << "op=" << unsigned(op)
                << " layer=" << layer
                << " position=" << position
                << " tile_len=" << tile_len
                << " status=" << status.code
                << " waves=" << status.output_waves
                << " controller_ms=" << last_controller_ms_
                << "\n";
        }
        return status;
    }

    decoded_status_t launch_controller_status_only(
        operator_kind_t op,
        unsigned int layer,
        unsigned int token_count,
        unsigned int position,
        unsigned int tile_len
    ) {
        {
            launch_controller_only_no_status(
                op,
                layer,
                token_count,
                position,
                tile_len
            );

            cl::Event status_event;
            if (verbose_ops_) {
                std::cout
                    << "enqueue cc8_status after controller completion\n"
                    << std::flush;
            }
            check_cl(
                status_queue_.enqueueTask(
                    status_kernel_,
                    nullptr,
                    &status_event
                ),
                "enqueue cc8_status after controller"
            );
            if (verbose_ops_) {
                std::cout
                    << "enqueue cc8_status after controller returned\n"
                    << "flush cc8_status after controller\n"
                    << std::flush;
            }
            check_cl(status_queue_.flush(), "flush cc8_status after controller");
            if (verbose_ops_) {
                std::cout
                    << "waiting cc8_status after controller event\n"
                    << std::flush;
            }
            status_event.wait();

            std::vector<cl::Memory> from_device = {
                status_buffer_
            };
            check_cl(
                transfer_queue_.enqueueMigrateMemObjects(
                    from_device,
                    CL_MIGRATE_MEM_OBJECT_HOST
                ),
                "migrate operator outputs"
            );
            check_cl(
                transfer_queue_.finish(),
                "finish operator output migration"
            );

            const decoded_status_t status = decode_status(status_words_[0]);
            if (verbose_ops_) {
                std::cout
                    << "op=" << unsigned(op)
                    << " status=" << status.code
                    << " waves=" << status.output_waves
                    << " packets=" << status.completed_packets
                    << "\n";
            }
            return status;
        }

        check_cl(
            controller_kernel_.setArg(
                kControllerOperatorArg,
                unsigned(op)
            ),
            "set operator"
        );
        check_cl(
            controller_kernel_.setArg(kControllerLayerArg, layer),
            "set layer"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerTokenCountArg,
                token_count
            ),
            "set token count"
        );
        check_cl(
            controller_kernel_.setArg(kControllerPositionArg, position),
            "set position"
        );
        check_cl(
            controller_kernel_.setArg(kControllerTileLenArg, tile_len),
            "set tile length"
        );

        std::vector<cl::Memory> to_device;
        for (const auto& buffer : data_buffers_) {
            to_device.push_back(buffer);
        }
        to_device.push_back(status_buffer_);
        if (verbose_ops_) {
            std::cout
                << "migrating operator inputs (controller/status only)\n"
                << std::flush;
        }
        check_cl(
            transfer_queue_.enqueueMigrateMemObjects(to_device, 0),
            "migrate operator inputs"
        );
        check_cl(transfer_queue_.finish(), "finish operator input migration");
        if (verbose_ops_) {
            std::cout
                << "operator input migration finished\n"
                << std::flush;
        }

        cl::Event status_event;
        cl::Event controller_event;
        if (verbose_ops_) {
            std::cout << "enqueue cc8_status\n" << std::flush;
        }
        check_cl(
            status_queue_.enqueueTask(
                status_kernel_,
                nullptr,
                &status_event
            ),
            "enqueue cc8_status"
        );
        if (verbose_ops_) {
            std::cout << "enqueue cc8_status returned\n" << std::flush;
            std::cout << "pre-flush cc8_status before controller enqueue\n" << std::flush;
        }
        check_cl(status_queue_.flush(), "pre-flush cc8_status");
        if (verbose_ops_) {
            std::cout << "enqueue cc8_ctrl\n" << std::flush;
        }
        check_cl(
            controller_queue_.enqueueTask(
                controller_kernel_,
                nullptr,
                &controller_event
            ),
            "enqueue cc8_ctrl"
        );
        if (verbose_ops_) {
            std::cout << "enqueue cc8_ctrl returned\n" << std::flush;
        }

        check_cl(status_queue_.flush(), "flush cc8_status");
        check_cl(controller_queue_.flush(), "flush cc8_ctrl");
        if (verbose_ops_) {
            std::cout << "waiting controller/status events\n" << std::flush;
        }
        controller_event.wait();
        status_event.wait();
        last_controller_ms_ = event_milliseconds(controller_event);

        std::vector<cl::Memory> from_device = {
            status_buffer_
        };
        check_cl(
            transfer_queue_.enqueueMigrateMemObjects(
                from_device,
                CL_MIGRATE_MEM_OBJECT_HOST
            ),
            "migrate operator outputs"
        );
        check_cl(transfer_queue_.finish(), "finish operator output migration");

        const decoded_status_t status = decode_status(status_words_[0]);
        if (verbose_ops_) {
            std::cout
                << "op=" << unsigned(op)
                << " status=" << status.code
                << " waves=" << status.output_waves
                << " packets=" << status.completed_packets
                << "\n";
        }
        return status;
    }

    void launch_controller_only_no_status(
        operator_kind_t op,
        unsigned int layer,
        unsigned int token_count,
        unsigned int position,
        unsigned int tile_len
    ) {
        check_cl(
            controller_kernel_.setArg(
                kControllerOperatorArg,
                unsigned(op)
            ),
            "set operator"
        );
        check_cl(
            controller_kernel_.setArg(kControllerLayerArg, layer),
            "set layer"
        );
        check_cl(
            controller_kernel_.setArg(
                kControllerTokenCountArg,
                token_count
            ),
            "set token count"
        );
        check_cl(
            controller_kernel_.setArg(kControllerPositionArg, position),
            "set position"
        );
        check_cl(
            controller_kernel_.setArg(kControllerTileLenArg, tile_len),
            "set tile length"
        );

        std::vector<cl::Memory> to_device;
        for (const auto& buffer : data_buffers_) {
            to_device.push_back(buffer);
        }
        if (verbose_ops_) {
            std::cout
                << "migrating operator inputs (controller only)\n"
                << std::flush;
        }
        check_cl(
            transfer_queue_.enqueueMigrateMemObjects(to_device, 0),
            "migrate operator inputs"
        );
        check_cl(transfer_queue_.finish(), "finish operator input migration");
        if (verbose_ops_) {
            std::cout
                << "operator input migration finished\n"
                << std::flush;
        }

        cl::Event controller_event;
        if (verbose_ops_) {
            std::cout << "enqueue cc8_ctrl only\n" << std::flush;
        }
        check_cl(
            controller_queue_.enqueueTask(
                controller_kernel_,
                nullptr,
                &controller_event
            ),
            "enqueue cc8_ctrl only"
        );
        if (verbose_ops_) {
            std::cout << "enqueue cc8_ctrl only returned\n" << std::flush;
            std::cout << "flush cc8_ctrl only\n" << std::flush;
        }
        check_cl(controller_queue_.flush(), "flush cc8_ctrl only");
        if (verbose_ops_) {
            std::cout << "waiting cc8_ctrl only event\n" << std::flush;
        }
        controller_event.wait();
        last_controller_ms_ = event_milliseconds(controller_event);
        if (verbose_ops_) {
            std::cout
                << "cc8_ctrl only completed controller_ms="
                << last_controller_ms_
                << "\n" << std::flush;
        }
    }

    void check_status(
        operator_kind_t expected_op,
        const decoded_status_t& status
    ) const {
        if (
            status.op != unsigned(expected_op) ||
            status.code != 0 ||
            !status.last_task
        ) {
            std::ostringstream message;
            message
                << "controller status failure: expected_op="
                << unsigned(expected_op)
                << " actual_op="
                << status.op
                << " code="
                << status.code
                << " last="
                << status.last_task;
            throw std::runtime_error(message.str());
        }
    }

    const model_shape_t& shape_;
    const model_data_t& model_;
    bool verbose_ops_;
    bool skip_weight_preload_;
    double last_controller_ms_ = -1.0;
    double last_status_kernel_ms_ = -1.0;
    double last_status_migration_ms_ = -1.0;

    std::array<aligned_word_vector, 6> data_words_;
    aligned_word_vector kv_cache_k_words_;
    aligned_word_vector kv_cache_v_words_;
    aligned_word_vector status_words_;
    std::array<cl_mem_ext_ptr_t, 6> data_ext_{};
    std::array<cl_mem_ext_ptr_t, kWeightShardCount> weight_ext_{};
    cl_mem_ext_ptr_t kv_cache_k_ext_{};
    cl_mem_ext_ptr_t kv_cache_v_ext_{};
    cl_mem_ext_ptr_t status_ext_{};
    std::array<cl::Buffer, 6> data_buffers_;
    std::array<cl::Buffer, kWeightShardCount> weight_buffers_;
    cl::Buffer kv_cache_k_buffer_;
    cl::Buffer kv_cache_v_buffer_;
    cl::Buffer status_buffer_;

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
};

tensor_t make_random_tensor(
    unsigned int rows,
    unsigned int cols,
    uint32_t seed,
    int minimum = -128,
    int maximum = 128
) {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(minimum, maximum);
    tensor_t tensor(rows, cols);
    for (auto& value : tensor.values) {
        value = int16_t(distribution(generator));
    }
    return tensor;
}

tensor_t golden_projection(
    const model_shape_t& shape,
    const model_data_t& model,
    operator_kind_t op,
    const tensor_t& input,
    unsigned int layer
) {
    const projection_spec_t spec = projection_spec(shape, op);
    if (input.cols != spec.in_dim) {
        throw std::runtime_error("golden projection input shape mismatch");
    }

    tensor_t output(input.rows, spec.out_dim);
    for (unsigned int token = 0; token < input.rows; token++) {
        for (unsigned int out = 0; out < spec.out_dim; out++) {
            // The deployed 8x64 fused MAC accumulates the signed Q8.8 x
            // Q4.12 products as raw Q28.20 values in a 48-bit integer and
            // rounds only after the complete dot product.  Narrowing every
            // product to fm_accum_t here looks harmless on small matrices,
            // but creates a large systematic drift across the 2048/11008-K
            // projections in a full decoder layer.
            golden_fused_raw_accum_t raw_total = 0;
            for (unsigned int in = 0; in < spec.in_dim; in++) {
                const ap_int<16> activation_raw = input.at(token, in);
                const ap_int<16> weight_raw =
                    model.projection_weight_raw(layer, op, out, in);
                const ap_int<32> product = activation_raw * weight_raw;
                raw_total += golden_fused_raw_accum_t(product);
            }

            golden_fused_fixed_accum_t fused_total;
            fused_total.range(47, 0) = raw_total.range(47, 0);
            const golden_accum_t narrowed = golden_accum_t(fused_total);
            output.at(token, out) = fixed_to_raw(golden_fm_t(narrowed));
        }
    }
    return output;
}

tensor_t golden_residual(
    const tensor_t& lhs,
    const tensor_t& rhs
) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
        throw std::runtime_error("golden residual shape mismatch");
    }
    tensor_t output(lhs.rows, lhs.cols);
    for (std::size_t i = 0; i < lhs.values.size(); i++) {
        const golden_fm_t left =
            fixed_from_raw<golden_fm_t>(lhs.values[i]);
        const golden_fm_t right =
            fixed_from_raw<golden_fm_t>(rhs.values[i]);
        output.values[i] = fixed_to_raw(golden_fm_t(left + right));
    }
    return output;
}

tensor_t golden_rmsnorm(
    const tensor_t& input,
    const tensor_t& weights
) {
    if (weights.rows != 1 || weights.cols != input.cols) {
        throw std::runtime_error("golden RMSNorm shape mismatch");
    }
    tensor_t output(input.rows, input.cols);
    for (unsigned int token = 0; token < input.rows; token++) {
        golden_accum_t sum_sq(0);
        for (unsigned int elem = 0; elem < input.cols; elem++) {
            const golden_fm_t value =
                fixed_from_raw<golden_fm_t>(input.at(token, elem));
            sum_sq +=
                golden_accum_t(value) *
                golden_accum_t(value);
        }
        const golden_fm_t inv_elems =
            golden_recip(golden_fm_t(input.cols));
        const golden_accum_t mean_sq =
            sum_sq * golden_accum_t(inv_elems);
        const golden_fm_t inv_rms = golden_rsqrt(
            mean_sq + golden_accum_t(0.000001)
        );
        for (unsigned int elem = 0; elem < input.cols; elem++) {
            const golden_fm_t value =
                fixed_from_raw<golden_fm_t>(input.at(token, elem));
            const golden_fm_t weight =
                fixed_from_raw<golden_fm_t>(weights.at(0, elem));
            output.at(token, elem) = fixed_to_raw(
                golden_fm_t(value * inv_rms * weight)
            );
        }
    }
    return output;
}

tensor_t golden_silu_mul(
    const tensor_t& gate,
    const tensor_t& up
) {
    if (gate.rows != up.rows || gate.cols != up.cols) {
        throw std::runtime_error("golden SiLU-Mul shape mismatch");
    }
    tensor_t output(gate.rows, gate.cols);
    for (std::size_t i = 0; i < gate.values.size(); i++) {
        const golden_fm_t gate_value =
            fixed_from_raw<golden_fm_t>(gate.values[i]);
        const golden_fm_t up_value =
            fixed_from_raw<golden_fm_t>(up.values[i]);
        output.values[i] = fixed_to_raw(
            golden_fm_t(golden_silu(gate_value) * up_value)
        );
    }
    return output;
}

tensor_t golden_attention_mm(
    const tensor_t& activation,
    const tensor_t& panel,
    unsigned int output_cols,
    bool scale_qk
) {
    tensor_t output(activation.rows, output_cols);
    const golden_fm_t scale = golden_fm_t(
        1.0 / std::sqrt(double(activation.cols))
    );
    for (unsigned int row = 0; row < activation.rows; row++) {
        for (unsigned int out = 0; out < output_cols; out++) {
            golden_accum_t banks[4] = {
                golden_accum_t(0),
                golden_accum_t(0),
                golden_accum_t(0),
                golden_accum_t(0)
            };
            const unsigned int k_count = activation.cols;
            for (unsigned int k = 0; k < k_count; k++) {
                const golden_fm_t a =
                    fixed_from_raw<golden_fm_t>(
                        activation.at(row, k)
                    );
                golden_fm_t panel_value(0);
                if (scale_qk) {
                    if (out < panel.rows) {
                        panel_value =
                            fixed_from_raw<golden_fm_t>(
                                panel.at(out, k)
                            );
                    }
                } else {
                    panel_value =
                        fixed_from_raw<golden_fm_t>(
                            panel.at(k, out)
                        );
                }
                const golden_weight_t weight =
                    golden_weight_t(panel_value);
                const golden_accum_t product =
                    golden_accum_t(a * weight);
                const unsigned int phase = k & 3u;
                if (k < 4) {
                    banks[phase] = product;
                } else {
                    banks[phase] += product;
                }
            }
            golden_accum_t total =
                banks[0] + banks[1] + banks[2] + banks[3];
            if (scale_qk) {
                total *= golden_accum_t(scale);
            }
            output.at(row, out) =
                fixed_to_raw(golden_fm_t(total));
        }
    }
    return output;
}

tensor_t golden_flash_attention(
    const tensor_t& query,
    const tensor_t& k_cache,
    const tensor_t& v_cache
) {
    if (
        k_cache.rows != v_cache.rows ||
        k_cache.cols != query.cols ||
        v_cache.cols != query.cols
    ) {
        throw std::runtime_error("golden flash attention shape mismatch");
    }

    const unsigned int context_len = k_cache.rows;
    tensor_t scores = golden_attention_mm(
        query,
        k_cache,
        context_len,
        true
    );
    tensor_t unnormalized(query.rows, context_len);

    std::vector<golden_accum_t> sums(
        query.rows,
        golden_accum_t(0)
    );
    for (unsigned int row = 0; row < query.rows; row++) {
        golden_fm_t row_max = golden_fm_t(-128);
        for (unsigned int pos = 0; pos < context_len; pos++) {
            const golden_fm_t score =
                fixed_from_raw<golden_fm_t>(scores.at(row, pos));
            if (score > row_max) {
                row_max = score;
            }
        }
        golden_accum_t sum = golden_accum_t(0);
        for (unsigned int pos = 0; pos < context_len; pos++) {
            const golden_fm_t score =
                fixed_from_raw<golden_fm_t>(scores.at(row, pos));
            const golden_fm_t probability =
                golden_exp(score - row_max);
            unnormalized.at(row, pos) = fixed_to_raw(probability);
            sum += golden_accum_t(probability);
        }
        sums[row] = sum;
    }

    tensor_t weighted = golden_attention_mm(
        unnormalized,
        v_cache,
        query.cols,
        false
    );
    tensor_t output(query.rows, query.cols);
    for (unsigned int row = 0; row < query.rows; row++) {
        const golden_accum_t safe_sum =
            sums[row] < golden_accum_t(0.000244140625) ?
            golden_accum_t(0.000244140625) :
            sums[row];
        const golden_fm_t inv_sum =
            golden_fm_t(golden_accum_t(1) / safe_sum);
        for (unsigned int elem = 0; elem < query.cols; elem++) {
            const golden_fm_t value =
                fixed_from_raw<golden_fm_t>(weighted.at(row, elem));
            output.at(row, elem) = fixed_to_raw(
                golden_fm_t(golden_accum_t(value) * golden_accum_t(inv_sum))
            );
        }
    }
    return output;
}

// Mathematical reference for long-context accuracy.  Unlike the historical
// global fixed-point golden above, this does not quantize/saturate the full
// unnormalised PV sum to Q8.8 before division by the softmax denominator.
// QK scores and V values still enter through the same hardware-visible fixed
// formats; only the softmax reduction is evaluated in double precision.
tensor_t golden_flash_attention_reference(
    const tensor_t& query,
    const tensor_t& k_cache,
    const tensor_t& v_cache
) {
    if (
        k_cache.rows != v_cache.rows ||
        k_cache.cols != query.cols ||
        v_cache.cols != query.cols
    ) {
        throw std::runtime_error(
            "golden flash attention reference shape mismatch"
        );
    }
    const tensor_t scores = golden_attention_mm(
        query, k_cache, k_cache.rows, true
    );
    tensor_t output(query.rows, query.cols);
    for (unsigned int row = 0; row < query.rows; row++) {
        double row_max = -128.0;
        for (unsigned int pos = 0; pos < k_cache.rows; pos++) {
            row_max = std::max(
                row_max,
                double(fixed_from_raw<golden_fm_t>(
                    scores.at(row, pos)
                ))
            );
        }
        double sum = 0.0;
        std::vector<double> probabilities(k_cache.rows, 0.0);
        for (unsigned int pos = 0; pos < k_cache.rows; pos++) {
            const double score = double(fixed_from_raw<golden_fm_t>(
                scores.at(row, pos)
            ));
            probabilities[pos] = std::exp(score - row_max);
            sum += probabilities[pos];
        }
        const double safe_sum = std::max(sum, 1.0 / 4096.0);
        for (unsigned int elem = 0; elem < query.cols; elem++) {
            double weighted = 0.0;
            for (unsigned int pos = 0; pos < k_cache.rows; pos++) {
                const golden_weight_t value = golden_weight_t(
                    fixed_from_raw<golden_fm_t>(
                        v_cache.at(pos, elem)
                    )
                );
                weighted += probabilities[pos] * double(value);
            }
            output.at(row, elem) = fixed_to_raw(
                golden_fm_t(weighted / safe_sum)
            );
        }
    }
    return output;
}

// Reproduce the controller's tiled online-softmax arithmetic exactly enough
// to distinguish a scheduling/stream error from fixed-point drift relative to
// the mathematically equivalent global-softmax reference above.  In
// particular, each tile quantizes QK scores, probabilities, the PV result and
// the inter-tile rescale factor back to Q8.8, while the running sum and output
// accumulator remain Q16.16.
template <typename Scale, bool WidePv>
tensor_t golden_tiled_flash_attention_impl(
    const tensor_t& query,
    const tensor_t& k_cache,
    const tensor_t& v_cache
) {
    if (
        k_cache.rows != v_cache.rows ||
        k_cache.cols != query.cols ||
        v_cache.cols != query.cols
    ) {
        throw std::runtime_error(
            "golden tiled flash attention shape mismatch"
        );
    }

    std::vector<golden_fm_t> online_max(
        query.rows, golden_fm_t(-128)
    );
    std::vector<golden_accum_t> online_sum(
        query.rows, golden_accum_t(0)
    );
    std::vector<golden_accum_t> accumulator(
        std::size_t(query.rows) * query.cols,
        golden_accum_t(0)
    );

    for (unsigned int tile_begin = 0;
         tile_begin < k_cache.rows;
         tile_begin += kAttentionTile) {
        const unsigned int tile_len = std::min(
            kAttentionTile, k_cache.rows - tile_begin
        );
        tensor_t k_tile(tile_len, query.cols);
        tensor_t v_tile(tile_len, query.cols);
        for (unsigned int pos = 0; pos < tile_len; pos++) {
            for (unsigned int elem = 0; elem < query.cols; elem++) {
                k_tile.at(pos, elem) =
                    k_cache.at(tile_begin + pos, elem);
                v_tile.at(pos, elem) =
                    v_cache.at(tile_begin + pos, elem);
            }
        }

        const tensor_t scores = golden_attention_mm(
            query, k_tile, kAttentionTile, true
        );
        tensor_t probabilities(query.rows, tile_len);
        std::vector<Scale> old_scale(
            query.rows, Scale(0)
        );

        for (unsigned int row = 0; row < query.rows; row++) {
            golden_fm_t row_max = online_max[row];
            for (unsigned int pos = 0; pos < tile_len; pos++) {
                const golden_fm_t score =
                    fixed_from_raw<golden_fm_t>(scores.at(row, pos));
                if (score > row_max) {
                    row_max = score;
                }
            }
            old_scale[row] = golden_rescale_exp(
                online_max[row] - row_max,
                static_cast<Scale*>(nullptr)
            );
            golden_accum_t tile_sum(0);
            for (unsigned int pos = 0; pos < tile_len; pos++) {
                const golden_fm_t score =
                    fixed_from_raw<golden_fm_t>(scores.at(row, pos));
                const golden_fm_t probability =
                    golden_exp(score - row_max);
                probabilities.at(row, pos) = fixed_to_raw(probability);
                tile_sum += golden_accum_t(probability);
            }
            online_sum[row] =
                online_sum[row] * golden_accum_t(old_scale[row]) +
                tile_sum;
            online_max[row] = row_max;
            for (unsigned int elem = 0; elem < query.cols; elem++) {
                accumulator[std::size_t(row) * query.cols + elem] *=
                    golden_accum_t(old_scale[row]);
            }
        }

        if (WidePv) {
            // Diagnostic model for a widened compute-to-controller PV
            // boundary.  The production path below deliberately quantizes
            // each tile's unnormalised PV result back to Q8.8; retaining the
            // Q16.16 accumulator here isolates saturation/rounding at that
            // boundary from inter-tile rescale error.
            for (unsigned int row = 0; row < query.rows; row++) {
                for (unsigned int elem = 0; elem < query.cols; elem++) {
                    golden_accum_t tile_value(0);
                    for (unsigned int pos = 0; pos < tile_len; pos++) {
                        const golden_fm_t probability =
                            fixed_from_raw<golden_fm_t>(
                                probabilities.at(row, pos)
                            );
                        const golden_fm_t value =
                            fixed_from_raw<golden_fm_t>(
                                v_tile.at(pos, elem)
                            );
                        tile_value +=
                            golden_accum_t(probability) *
                            golden_accum_t(value);
                    }
                    accumulator[
                        std::size_t(row) * query.cols + elem
                    ] += tile_value;
                }
            }
        } else {
            const tensor_t tile_weighted = golden_attention_mm(
                probabilities, v_tile, query.cols, false
            );
            for (unsigned int row = 0; row < query.rows; row++) {
                for (unsigned int elem = 0; elem < query.cols; elem++) {
                    accumulator[std::size_t(row) * query.cols + elem] +=
                        golden_accum_t(fixed_from_raw<golden_fm_t>(
                            tile_weighted.at(row, elem)
                        ));
                }
            }
        }
    }

    tensor_t output(query.rows, query.cols);
    for (unsigned int row = 0; row < query.rows; row++) {
        const golden_accum_t safe_sum =
            online_sum[row] < golden_accum_t(0.000244140625) ?
            golden_accum_t(0.000244140625) : online_sum[row];
        const golden_fm_t inv_sum =
            golden_fm_t(golden_accum_t(1) / safe_sum);
        for (unsigned int elem = 0; elem < query.cols; elem++) {
            output.at(row, elem) = fixed_to_raw(golden_fm_t(
                accumulator[std::size_t(row) * query.cols + elem] *
                golden_accum_t(inv_sum)
            ));
        }
    }
    return output;
}

tensor_t golden_tiled_flash_attention(
    const tensor_t& query,
    const tensor_t& k_cache,
    const tensor_t& v_cache
) {
    return golden_tiled_flash_attention_impl<golden_fm_t, false>(
        query, k_cache, v_cache
    );
}

tensor_t golden_tiled_flash_attention_wide_rescale(
    const tensor_t& query,
    const tensor_t& k_cache,
    const tensor_t& v_cache
) {
    return golden_tiled_flash_attention_impl<golden_scale_t, false>(
        query, k_cache, v_cache
    );
}

tensor_t golden_tiled_flash_attention_wide_pv(
    const tensor_t& query,
    const tensor_t& k_cache,
    const tensor_t& v_cache
) {
    return golden_tiled_flash_attention_impl<golden_fm_t, true>(
        query, k_cache, v_cache
    );
}

tensor_t golden_tiled_flash_attention_wide_rescale_pv(
    const tensor_t& query,
    const tensor_t& k_cache,
    const tensor_t& v_cache
) {
    return golden_tiled_flash_attention_impl<golden_scale_t, true>(
        query, k_cache, v_cache
    );
}

// Production Q2.14 online-softmax model.  The existing 16-bit activation
// packet carries these raw probability bits; the compute CU interprets them
// as Q8.8 and applies output_scale=1/64, which is mathematically identical to
// the Q2.14 multiply below without changing the stream ABI.
tensor_t golden_tiled_flash_attention_q2_14(
    const tensor_t& query,
    const tensor_t& k_cache,
    const tensor_t& v_cache
) {
    std::vector<golden_fm_t> online_max(
        query.rows, golden_fm_t(-128)
    );
    std::vector<golden_accum_t> online_sum(
        query.rows, golden_accum_t(0)
    );
    std::vector<golden_accum_t> accumulator(
        std::size_t(query.rows) * query.cols,
        golden_accum_t(0)
    );

    for (unsigned int tile_begin = 0;
         tile_begin < k_cache.rows;
         tile_begin += kAttentionTile) {
        const unsigned int tile_len = std::min(
            kAttentionTile, k_cache.rows - tile_begin
        );
        tensor_t k_tile(tile_len, query.cols);
        tensor_t v_tile(tile_len, query.cols);
        for (unsigned int pos = 0; pos < tile_len; pos++) {
            for (unsigned int elem = 0; elem < query.cols; elem++) {
                k_tile.at(pos, elem) = k_cache.at(tile_begin + pos, elem);
                v_tile.at(pos, elem) = v_cache.at(tile_begin + pos, elem);
            }
        }
        const tensor_t scores = golden_attention_mm(
            query, k_tile, kAttentionTile, true
        );
        std::vector<golden_probability_t> probabilities(
            std::size_t(query.rows) * tile_len,
            golden_probability_t(0)
        );
        for (unsigned int row = 0; row < query.rows; row++) {
            golden_fm_t row_max = online_max[row];
            for (unsigned int pos = 0; pos < tile_len; pos++) {
                const golden_fm_t score =
                    fixed_from_raw<golden_fm_t>(scores.at(row, pos));
                if (score > row_max) {
                    row_max = score;
                }
            }
            const golden_probability_t old_scale =
                golden_attention_probability(online_max[row] - row_max);
            golden_accum_t tile_sum(0);
            for (unsigned int pos = 0; pos < tile_len; pos++) {
                const golden_fm_t score =
                    fixed_from_raw<golden_fm_t>(scores.at(row, pos));
                const golden_probability_t probability =
                    golden_attention_probability(score - row_max);
                probabilities[std::size_t(row) * tile_len + pos] =
                    probability;
                tile_sum += golden_accum_t(probability);
            }
            online_sum[row] =
                online_sum[row] * golden_accum_t(old_scale) + tile_sum;
            online_max[row] = row_max;
            for (unsigned int elem = 0; elem < query.cols; elem++) {
                accumulator[std::size_t(row) * query.cols + elem] *=
                    golden_accum_t(old_scale);
            }
        }

        // Match the four-bank compute accumulation and its Q8.8 result
        // packet after the Q2.14 payload's 64x reinterpretation has been
        // cancelled by the task's 1/64 output scale.
        for (unsigned int row = 0; row < query.rows; row++) {
            for (unsigned int elem = 0; elem < query.cols; elem++) {
                golden_accum_t banks[4] = {
                    golden_accum_t(0), golden_accum_t(0),
                    golden_accum_t(0), golden_accum_t(0)
                };
                for (unsigned int pos = 0; pos < tile_len; pos++) {
                    const golden_probability_t probability = probabilities[
                        std::size_t(row) * tile_len + pos
                    ];
                    const golden_weight_t value = golden_weight_t(
                        fixed_from_raw<golden_fm_t>(v_tile.at(pos, elem))
                    );
                    const golden_accum_t product =
                        golden_accum_t(probability * value);
                    const unsigned int phase = pos & 3u;
                    if (pos < 4) {
                        banks[phase] = product;
                    } else {
                        banks[phase] += product;
                    }
                }
                const golden_fm_t tile_value = golden_fm_t(
                    banks[0] + banks[1] + banks[2] + banks[3]
                );
                accumulator[std::size_t(row) * query.cols + elem] +=
                    golden_accum_t(tile_value);
            }
        }
    }

    tensor_t output(query.rows, query.cols);
    for (unsigned int row = 0; row < query.rows; row++) {
        const golden_accum_t safe_sum =
            online_sum[row] < golden_accum_t(0.000244140625) ?
            golden_accum_t(0.000244140625) : online_sum[row];
        const golden_fm_t inv_sum =
            golden_fm_t(golden_accum_t(1) / safe_sum);
        for (unsigned int elem = 0; elem < query.cols; elem++) {
            output.at(row, elem) = fixed_to_raw(golden_fm_t(
                accumulator[std::size_t(row) * query.cols + elem] *
                golden_accum_t(inv_sum)
            ));
        }
    }
    return output;
}

bool compare_tensors(
    const std::string& name,
    const tensor_t& actual,
    const tensor_t& expected,
    int tolerance
) {
    if (
        actual.rows != expected.rows ||
        actual.cols != expected.cols
    ) {
        std::cout << "RANDOM " << name << " FAIL shape mismatch\n";
        return false;
    }

    unsigned int mismatches = 0;
    int maximum_error = 0;
    for (std::size_t i = 0; i < actual.values.size(); i++) {
        const int error = std::abs(
            int(actual.values[i]) - int(expected.values[i])
        );
        maximum_error = std::max(maximum_error, error);
        if (error > tolerance) {
            if (mismatches < 4) {
                const unsigned int row =
                    unsigned(i / actual.cols);
                const unsigned int col =
                    unsigned(i % actual.cols);
                std::cout
                    << "  mismatch row=" << row
                    << " col=" << col
                    << " actual_raw=" << actual.values[i]
                    << " expected_raw=" << expected.values[i]
                    << " error=" << error
                    << "\n";
            }
            mismatches++;
        }
    }
    const bool pass = mismatches == 0;
    std::cout
        << "RANDOM " << name
        << " " << (pass ? "PASS" : "FAIL")
        << " values=" << actual.values.size()
        << " max_raw_error=" << maximum_error
        << " tolerance=" << tolerance;
    if (!pass) {
        std::cout << " mismatches=" << mismatches;
    }
    std::cout << "\n";
    return pass;
}

bool run_random_verification(
    const model_shape_t& shape,
    const model_data_t& model,
    accelerator_t& accelerator,
    uint32_t seed
) {
    constexpr unsigned int token_count = 5;
    bool pass = true;
    tensor_t hidden = make_random_tensor(
        token_count,
        shape.hidden_size,
        seed ^ 0x13579bdu
    );

    const std::array<operator_kind_t, 3> hidden_projections = {
        kOpQProjection,
        kOpKProjection,
        kOpFfnGate
    };
    for (operator_kind_t op : hidden_projections) {
        const tensor_t actual = accelerator.run_feature(
            op,
            hidden,
            nullptr,
            0,
            0
        );
        const tensor_t expected = golden_projection(
            shape,
            model,
            op,
            hidden,
            0
        );
        pass = compare_tensors(
            "projection_op_" + std::to_string(unsigned(op)),
            actual,
            expected,
            1
        ) && pass;
    }

    tensor_t intermediate = make_random_tensor(
        token_count,
        shape.intermediate_size,
        seed ^ 0x2468aceu
    );
    tensor_t down_actual = accelerator.run_feature(
        kOpFfnDown,
        intermediate,
        nullptr,
        0,
        0
    );
    tensor_t down_expected = golden_projection(
        shape,
        model,
        kOpFfnDown,
        intermediate,
        0
    );
    pass = compare_tensors(
        "projection_down",
        down_actual,
        down_expected,
        1
    ) && pass;

    tensor_t residual_rhs = make_random_tensor(
        token_count,
        shape.hidden_size,
        seed ^ 0x55aa55aau
    );
    tensor_t residual_actual = accelerator.run_feature(
        kOpResidualAdd,
        hidden,
        &residual_rhs,
        0,
        0
    );
    pass = compare_tensors(
        "residual_add",
        residual_actual,
        golden_residual(hidden, residual_rhs),
        0
    ) && pass;

    tensor_t norm_weights = model.norm_row(0, false);
    tensor_t rms_actual = accelerator.run_feature(
        kOpRmsNorm,
        hidden,
        &norm_weights,
        0,
        0
    );
    pass = compare_tensors(
        "rmsnorm",
        rms_actual,
        golden_rmsnorm(hidden, norm_weights),
        1
    ) && pass;

    tensor_t gate = make_random_tensor(
        token_count,
        shape.intermediate_size,
        seed ^ 0x11223344u
    );
    tensor_t up = make_random_tensor(
        token_count,
        shape.intermediate_size,
        seed ^ 0x55667788u
    );
    tensor_t silu_actual = accelerator.run_feature(
        kOpSiluMul,
        gate,
        &up,
        0,
        0
    );
    pass = compare_tensors(
        "silu_mul",
        silu_actual,
        golden_silu_mul(gate, up),
        1
    ) && pass;

    const unsigned int group_size = shape.gqa_group_size();
    const unsigned int head_dim = shape.head_dim();
    const unsigned int tile_len = 13;
    tensor_t query0 = make_random_tensor(
        group_size,
        head_dim,
        seed ^ 0x01020304u,
        -96,
        96
    );
    tensor_t query1 = make_random_tensor(
        group_size,
        head_dim,
        seed ^ 0x05060708u,
        -96,
        96
    );
    tensor_t k0 = make_random_tensor(
        tile_len,
        head_dim,
        seed ^ 0x11121314u,
        -96,
        96
    );
    tensor_t k1 = make_random_tensor(
        tile_len,
        head_dim,
        seed ^ 0x15161718u,
        -96,
        96
    );
    operator_result_t qk_actual = accelerator.run_attention(
        kOpAttentionQk,
        query0,
        query1,
        &k0,
        &k1,
        0,
        tile_len
    );
    pass = compare_tensors(
        "attention_qk_port0",
        qk_actual.port0,
        golden_attention_mm(query0, k0, kAttentionTile, true),
        1
    ) && pass;
    pass = compare_tensors(
        "attention_qk_port1",
        qk_actual.port1,
        golden_attention_mm(query1, k1, kAttentionTile, true),
        1
    ) && pass;

    tensor_t probability0 = make_random_tensor(
        group_size,
        kAttentionTile,
        seed ^ 0x21222324u,
        0,
        24
    );
    tensor_t probability1 = make_random_tensor(
        group_size,
        kAttentionTile,
        seed ^ 0x25262728u,
        0,
        24
    );
    tensor_t v0 = make_random_tensor(
        tile_len,
        head_dim,
        seed ^ 0x31323334u,
        -96,
        96
    );
    tensor_t v1 = make_random_tensor(
        tile_len,
        head_dim,
        seed ^ 0x35363738u,
        -96,
        96
    );
    tensor_t probability0_active(group_size, tile_len);
    tensor_t probability1_active(group_size, tile_len);
    for (unsigned int row = 0; row < group_size; row++) {
        for (unsigned int pos = 0; pos < tile_len; pos++) {
            probability0_active.at(row, pos) =
                probability0.at(row, pos);
            probability1_active.at(row, pos) =
                probability1.at(row, pos);
        }
    }
    operator_result_t pv_actual = accelerator.run_attention(
        kOpAttentionPv,
        probability0,
        probability1,
        &v0,
        &v1,
        0,
        tile_len
    );
    pass = compare_tensors(
        "attention_pv_port0",
        pv_actual.port0,
        golden_attention_mm(
            probability0_active,
            v0,
            head_dim,
            false
        ),
        1
    ) && pass;
    pass = compare_tensors(
        "attention_pv_port1",
        pv_actual.port1,
        golden_attention_mm(
            probability1_active,
            v1,
            head_dim,
            false
        ),
        1
    ) && pass;

    const unsigned int decode_steps = 2;
    tensor_t decode_k0_cache(decode_steps, head_dim);
    tensor_t decode_k1_cache(decode_steps, head_dim);
    tensor_t decode_v0_cache(decode_steps, head_dim);
    tensor_t decode_v1_cache(decode_steps, head_dim);
    for (unsigned int step = 0; step < decode_steps; step++) {
        tensor_t decode_query0 = make_random_tensor(
            group_size,
            head_dim,
            seed ^ (0x41420000u + step),
            -48,
            48
        );
        tensor_t decode_query1 = make_random_tensor(
            group_size,
            head_dim,
            seed ^ (0x43440000u + step),
            -48,
            48
        );
        tensor_t k_payload(shape.num_kv_heads, head_dim);
        tensor_t v_payload(shape.num_kv_heads, head_dim);
        tensor_t k_head0 = make_random_tensor(
            1,
            head_dim,
            seed ^ (0x45460000u + step),
            -48,
            48
        );
        tensor_t k_head1 = make_random_tensor(
            1,
            head_dim,
            seed ^ (0x47480000u + step),
            -48,
            48
        );
        tensor_t v_head0 = make_random_tensor(
            1,
            head_dim,
            seed ^ (0x494a0000u + step),
            -48,
            48
        );
        tensor_t v_head1 = make_random_tensor(
            1,
            head_dim,
            seed ^ (0x4b4c0000u + step),
            -48,
            48
        );
        for (unsigned int elem = 0; elem < head_dim; elem++) {
            k_payload.at(0, elem) = k_head0.at(0, elem);
            k_payload.at(1, elem) = k_head1.at(0, elem);
            v_payload.at(0, elem) = v_head0.at(0, elem);
            v_payload.at(1, elem) = v_head1.at(0, elem);
            decode_k0_cache.at(step, elem) = k_head0.at(0, elem);
            decode_k1_cache.at(step, elem) = k_head1.at(0, elem);
            decode_v0_cache.at(step, elem) = v_head0.at(0, elem);
            decode_v1_cache.at(step, elem) = v_head1.at(0, elem);
        }

        operator_result_t decode_actual = accelerator.run_attention(
            kOpDecodeSmoke,
            decode_query0,
            decode_query1,
            &k_payload,
            &v_payload,
            step,
            1
        );

        tensor_t active_k0(step + 1, head_dim);
        tensor_t active_k1(step + 1, head_dim);
        tensor_t active_v0(step + 1, head_dim);
        tensor_t active_v1(step + 1, head_dim);
        for (unsigned int pos = 0; pos <= step; pos++) {
            for (unsigned int elem = 0; elem < head_dim; elem++) {
                active_k0.at(pos, elem) = decode_k0_cache.at(pos, elem);
                active_k1.at(pos, elem) = decode_k1_cache.at(pos, elem);
                active_v0.at(pos, elem) = decode_v0_cache.at(pos, elem);
                active_v1.at(pos, elem) = decode_v1_cache.at(pos, elem);
            }
        }

        pass = compare_tensors(
            "decode_smoke_port0_pos" + std::to_string(step),
            decode_actual.port0,
            golden_flash_attention(decode_query0, active_k0, active_v0),
            4
        ) && pass;
        pass = compare_tensors(
            "decode_smoke_port1_pos" + std::to_string(step),
            decode_actual.port1,
            golden_flash_attention(decode_query1, active_k1, active_v1),
            4
        ) && pass;
    }

    std::cout
        << "RANDOM_VERIFY seed=" << seed
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool run_decode_smoke_verification(
    const model_shape_t& shape,
    accelerator_t& accelerator,
    uint32_t seed
) {
    const unsigned int group_size = shape.gqa_group_size();
    const unsigned int head_dim = shape.head_dim();
    constexpr unsigned int position = 0;

    tensor_t query0 = make_random_tensor(
        group_size,
        head_dim,
        seed ^ 0x41420000u,
        -48,
        48
    );
    tensor_t query1 = make_random_tensor(
        group_size,
        head_dim,
        seed ^ 0x43440000u,
        -48,
        48
    );
    tensor_t k_payload(shape.num_kv_heads, head_dim);
    tensor_t v_payload(shape.num_kv_heads, head_dim);
    tensor_t k_head0 = make_random_tensor(
        1,
        head_dim,
        seed ^ 0x45460000u,
        -48,
        48
    );
    tensor_t k_head1 = make_random_tensor(
        1,
        head_dim,
        seed ^ 0x47480000u,
        -48,
        48
    );
    tensor_t v_head0 = make_random_tensor(
        1,
        head_dim,
        seed ^ 0x494a0000u,
        -48,
        48
    );
    tensor_t v_head1 = make_random_tensor(
        1,
        head_dim,
        seed ^ 0x4b4c0000u,
        -48,
        48
    );
    for (unsigned int elem = 0; elem < head_dim; elem++) {
        k_payload.at(0, elem) = k_head0.at(0, elem);
        k_payload.at(1, elem) = k_head1.at(0, elem);
        v_payload.at(0, elem) = v_head0.at(0, elem);
        v_payload.at(1, elem) = v_head1.at(0, elem);
    }

    std::cout
        << "DECODE_SMOKE launching position=" << position
        << " group_size=" << group_size
        << " head_dim=" << head_dim
        << "\n";
    operator_result_t actual = accelerator.run_attention(
        kOpDecodeSmoke,
        query0,
        query1,
        &k_payload,
        &v_payload,
        position,
        1
    );
    std::cout
        << "DECODE_SMOKE returned status="
        << actual.status.code
        << " packets="
        << actual.status.completed_packets
        << "\n";

    const bool pass0 = compare_tensors(
        "decode_smoke_only_port0",
        actual.port0,
        golden_flash_attention(query0, k_head0, v_head0),
        4
    );
    const bool pass1 = compare_tensors(
        "decode_smoke_only_port1",
        actual.port1,
        golden_flash_attention(query1, k_head1, v_head1),
        4
    );
    const bool pass = pass0 && pass1;
    std::cout
        << "DECODE_SMOKE_VERIFY seed=" << seed
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

tensor_t golden_single_entry_decoder_layer(
    const model_shape_t& shape,
    const model_data_t& model,
    const tensor_t& hidden,
    const tensor_t& attention_norm,
    const tensor_t& ffn_norm,
    unsigned int layer
) {
    tensor_t normalized = golden_rmsnorm(hidden, attention_norm);
    tensor_t v = golden_projection(
        shape,
        model,
        kOpVProjection,
        normalized,
        layer
    );
    // With one KV entry, online softmax is exactly one.  Each query head
    // therefore receives the V row of its GQA group.
    tensor_t attention(1, shape.hidden_size);
    const unsigned int group_size = shape.gqa_group_size();
    const unsigned int head_dim = shape.head_dim();
    for (unsigned int query_head = 0;
         query_head < shape.num_heads;
         query_head++) {
        const unsigned int kv_head = query_head / group_size;
        for (unsigned int elem = 0; elem < head_dim; elem++) {
            attention.at(0, query_head * head_dim + elem) =
                v.at(0, kv_head * head_dim + elem);
        }
    }
    tensor_t projected = golden_projection(
        shape,
        model,
        kOpOProjection,
        attention,
        layer
    );
    tensor_t post_attention = golden_residual(hidden, projected);
    tensor_t ffn_input = golden_rmsnorm(post_attention, ffn_norm);
    tensor_t gate = golden_projection(
        shape,
        model,
        kOpFfnGate,
        ffn_input,
        layer
    );
    tensor_t up = golden_projection(
        shape,
        model,
        kOpFfnUp,
        ffn_input,
        layer
    );
    tensor_t activated = golden_silu_mul(gate, up);
    tensor_t down = golden_projection(
        shape,
        model,
        kOpFfnDown,
        activated,
        layer
    );
    return golden_residual(post_attention, down);
}

bool run_resident_layer_verification(
    const model_shape_t& shape,
    const model_data_t& model,
    accelerator_t& accelerator,
    uint32_t seed,
    unsigned int position
) {
    if (position != 0) {
        throw std::runtime_error(
            "initial resident-layer golden test currently starts at position 0"
        );
    }
    tensor_t hidden = make_random_tensor(
        1,
        shape.hidden_size,
        seed ^ 0x6c617965u,
        -48,
        48
    );
    tensor_t attention_norm = model.norm_row(0, false);
    tensor_t ffn_norm = model.norm_row(0, true);
    tensor_t actual = accelerator.run_decoder_layer(
        hidden,
        attention_norm,
        ffn_norm,
        0,
        position
    );

    tensor_t expected = golden_single_entry_decoder_layer(
        shape,
        model,
        hidden,
        attention_norm,
        ffn_norm,
        0
    );

    const bool pass = compare_tensors(
        "resident_decoder_layer",
        actual,
        expected,
        16
    );
    std::cout
        << "RESIDENT_LAYER_VERIFY seed=" << seed
        << " position=" << position
        << " hidden=" << shape.hidden_size
        << " intermediate=" << shape.intermediate_size
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool run_composed_layer_verification(
    const model_shape_t& shape,
    const model_data_t& model,
    accelerator_t& accelerator,
    uint32_t seed,
    unsigned int position
) {
    if (position != 0) {
        throw std::runtime_error(
            "initial composed-layer golden test currently starts at position 0"
        );
    }
    tensor_t hidden = make_random_tensor(
        1,
        shape.hidden_size,
        seed ^ 0x7461736bu,
        -48,
        48
    );
    tensor_t attention_norm = model.norm_row(0, false);
    tensor_t ffn_norm = model.norm_row(0, true);
    composed_layer_result_t actual =
        accelerator.run_composed_decoder_stack(
            hidden,
            0,
            1,
            position,
            false
        );
    tensor_t expected = golden_single_entry_decoder_layer(
        shape,
        model,
        hidden,
        attention_norm,
        ffn_norm,
        0
    );
    const bool pass = compare_tensors(
        "task_composed_decoder_layer",
        actual.output,
        expected,
        16
    );
    std::cout
        << "TASK_COMPOSED_LAYER_VERIFY seed=" << seed
        << " position=" << position
        << " attention_op=" << actual.attention_status.op
        << " ffn_op=" << actual.ffn_status.op
        << " layers=" << actual.layer_count
        << " tasks=" << actual.task_count
        << " intermediate_host_copy=0"
        << " event_timing_domain=" << event_timing_domain()
        << " attention_controller_ms="
        << actual.attention_controller_ms
        << " ffn_controller_ms=" << actual.ffn_controller_ms
        << " kernel_active_ms=" << actual.kernel_active_ms
        << " input_migration_ms=" << actual.input_migration_ms
        << " auxiliary_migration_ms="
        << actual.auxiliary_migration_ms
        << " status_kernel_ms=" << actual.status_kernel_ms
        << " status_migration_ms=" << actual.status_migration_ms
        << " output_migration_ms=" << actual.output_migration_ms
        << " profiled_sequence_ms=" << actual.profiled_sequence_ms
        << " host_elapsed_ms=" << actual.host_elapsed_ms
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool run_composed_stack_verification(
    const model_shape_t& shape,
    const model_data_t& model,
    accelerator_t& accelerator,
    uint32_t seed,
    unsigned int position
) {
    if (position != 0) {
        throw std::runtime_error(
            "initial composed-stack golden test currently starts at position 0"
        );
    }
    tensor_t hidden = make_random_tensor(
        1,
        shape.hidden_size,
        seed ^ 0x73746163u,
        -48,
        48
    );
    composed_layer_result_t actual =
        accelerator.run_composed_decoder_stack(
            hidden,
            0,
            shape.num_layers,
            position,
            true
        );

    tensor_t expected = hidden;
    for (unsigned int layer = 0; layer < shape.num_layers; layer++) {
        expected = golden_single_entry_decoder_layer(
            shape,
            model,
            expected,
            model.norm_row(layer, false),
            model.norm_row(layer, true),
            layer
        );
    }
    expected = golden_rmsnorm(expected, model.final_norm_row());

    const unsigned int expected_tasks = 2 * shape.num_layers + 1;
    const bool protocol_pass =
        actual.layer_count == shape.num_layers &&
        actual.task_count == expected_tasks &&
        actual.attention_status.op == unsigned(kOpAttentionSublayer) &&
        actual.ffn_status.op == unsigned(kOpFfnSublayer) &&
        actual.final_norm_status.op == unsigned(kOpFinalNorm);
    const int tolerance = 16 * int(shape.num_layers);
    const bool data_pass = compare_tensors(
        "task_composed_decoder_stack",
        actual.output,
        expected,
        tolerance
    );
    const bool pass = protocol_pass && data_pass;
    std::cout
        << "TASK_COMPOSED_STACK_VERIFY seed=" << seed
        << " position=" << position
        << " attention_op=" << actual.attention_status.op
        << " ffn_op=" << actual.ffn_status.op
        << " final_norm_op=" << actual.final_norm_status.op
        << " layers=" << actual.layer_count
        << " tasks=" << actual.task_count
        << " expected_tasks=" << expected_tasks
        << " intermediate_host_copy=0"
        << " event_timing_domain=" << event_timing_domain()
        << " attention_controller_ms="
        << actual.attention_controller_ms
        << " ffn_controller_ms=" << actual.ffn_controller_ms
        << " final_norm_controller_ms="
        << actual.final_norm_controller_ms
        << " kernel_active_ms=" << actual.kernel_active_ms
        << " input_migration_ms=" << actual.input_migration_ms
        << " auxiliary_migration_ms="
        << actual.auxiliary_migration_ms
        << " status_kernel_ms=" << actual.status_kernel_ms
        << " status_migration_ms=" << actual.status_migration_ms
        << " output_migration_ms=" << actual.output_migration_ms
        << " profiled_sequence_ms=" << actual.profiled_sequence_ms
        << " host_elapsed_ms=" << actual.host_elapsed_ms
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

uint64_t tensor_checksum(const tensor_t& tensor);

unsigned int attention_output_waves(const model_shape_t& shape) {
    return unsigned(ceildiv(shape.head_dim(), kMmOutputsPerCore));
}

unsigned int attention_tile_count(
    unsigned int position
) {
    return unsigned(ceildiv(std::size_t(position) + 1, kAttentionTile));
}

unsigned int attention_expected_mm_tasks(
    const model_shape_t& shape,
    unsigned int position
) {
    const unsigned int output_waves = attention_output_waves(shape);
    return
        kMmCoreCount *
        attention_tile_count(position) *
        (1 + output_waves);
}

tensor_t tensor_prefix_rows(
    const tensor_t& source,
    unsigned int rows
) {
    if (rows > source.rows) {
        throw std::runtime_error("tensor prefix exceeds source rows");
    }
    tensor_t result(rows, source.cols);
    for (unsigned int row = 0; row < rows; row++) {
        for (unsigned int col = 0; col < source.cols; col++) {
            result.at(row, col) = source.at(row, col);
        }
    }
    return result;
}

tensor_t tensor_rows(
    const tensor_t& source,
    unsigned int first_row,
    unsigned int row_count
) {
    if (first_row > source.rows || row_count > source.rows - first_row) {
        throw std::runtime_error("tensor row slice exceeds source rows");
    }
    tensor_t result(row_count, source.cols);
    for (unsigned int row = 0; row < row_count; row++) {
        for (unsigned int col = 0; col < source.cols; col++) {
            result.at(row, col) = source.at(first_row + row, col);
        }
    }
    return result;
}

void tensor_store_rows(
    tensor_t& destination,
    unsigned int first_row,
    const tensor_t& source
) {
    if (
        first_row > destination.rows ||
        source.rows > destination.rows - first_row ||
        source.cols != destination.cols
    ) {
        throw std::runtime_error("tensor row store exceeds destination");
    }
    for (unsigned int row = 0; row < source.rows; row++) {
        for (unsigned int col = 0; col < source.cols; col++) {
            destination.at(first_row + row, col) = source.at(row, col);
        }
    }
}

tensor_t tensor_append_row(
    const tensor_t& prefix,
    const tensor_t& row_tensor
) {
    if (row_tensor.rows != 1 || row_tensor.cols != prefix.cols) {
        throw std::runtime_error("tensor append row shape mismatch");
    }
    tensor_t result(prefix.rows + 1, prefix.cols);
    for (unsigned int row = 0; row < prefix.rows; row++) {
        for (unsigned int col = 0; col < prefix.cols; col++) {
            result.at(row, col) = prefix.at(row, col);
        }
    }
    for (unsigned int col = 0; col < prefix.cols; col++) {
        result.at(prefix.rows, col) = row_tensor.at(0, col);
    }
    return result;
}

void fill_profile_kv_payload(
    const model_shape_t& shape,
    tensor_t& k_payload,
    tensor_t& v_payload,
    const tensor_t& k0,
    const tensor_t& k1,
    const tensor_t& v0,
    const tensor_t& v1,
    unsigned int position
) {
    const unsigned int head_dim = shape.head_dim();
    if (
        k_payload.rows != shape.num_kv_heads ||
        v_payload.rows != shape.num_kv_heads ||
        k_payload.cols != head_dim ||
        v_payload.cols != head_dim ||
        position >= k0.rows ||
        position >= k1.rows ||
        position >= v0.rows ||
        position >= v1.rows
    ) {
        throw std::runtime_error("profile KV payload fill shape mismatch");
    }
    for (unsigned int elem = 0; elem < head_dim; elem++) {
        k_payload.at(0, elem) = k0.at(position, elem);
        k_payload.at(1, elem) = k1.at(position, elem);
        v_payload.at(0, elem) = v0.at(position, elem);
        v_payload.at(1, elem) = v1.at(position, elem);
    }
}

tensor_t profile_build_query_group(
    const model_shape_t& shape,
    const tensor_t& q,
    unsigned int kv_head
) {
    if (q.rows != 1 || q.cols != shape.hidden_size) {
        throw std::runtime_error("profile query shape mismatch");
    }
    if (kv_head >= shape.num_kv_heads) {
        throw std::runtime_error("profile query KV head is out of range");
    }
    tensor_t group(shape.gqa_group_size(), shape.head_dim());
    for (unsigned int row = 0; row < shape.gqa_group_size(); row++) {
        const unsigned int query_head =
            kv_head * shape.gqa_group_size() + row;
        const unsigned int base = query_head * shape.head_dim();
        for (unsigned int elem = 0; elem < shape.head_dim(); elem++) {
            group.at(row, elem) = q.at(0, base + elem);
        }
    }
    return group;
}

tensor_t profile_build_kv_payload(
    const model_shape_t& shape,
    const tensor_t& kv
) {
    if (kv.rows != 1 || kv.cols != shape.kv_channels()) {
        throw std::runtime_error("profile KV payload source shape mismatch");
    }
    tensor_t payload(shape.num_kv_heads, shape.head_dim());
    for (unsigned int head = 0; head < shape.num_kv_heads; head++) {
        const unsigned int source_base = head * shape.head_dim();
        for (unsigned int elem = 0; elem < shape.head_dim(); elem++) {
            payload.at(head, elem) = kv.at(0, source_base + elem);
        }
    }
    return payload;
}

tensor_t profile_payload_head(
    const tensor_t& payload,
    unsigned int head
) {
    if (head >= payload.rows) {
        throw std::runtime_error("profile payload head is out of range");
    }
    tensor_t result(1, payload.cols);
    for (unsigned int elem = 0; elem < payload.cols; elem++) {
        result.at(0, elem) = payload.at(head, elem);
    }
    return result;
}

void profile_scatter_attention_group(
    tensor_t& attention,
    const tensor_t& group,
    unsigned int kv_head,
    unsigned int group_size,
    unsigned int head_dim
) {
    if (group.rows != group_size || group.cols != head_dim) {
        throw std::runtime_error("profile attention group shape mismatch");
    }
    for (unsigned int row = 0; row < group_size; row++) {
        const unsigned int query_head = kv_head * group_size + row;
        const unsigned int destination_base = query_head * head_dim;
        for (unsigned int elem = 0; elem < head_dim; elem++) {
            attention.at(0, destination_base + elem) =
                group.at(row, elem);
        }
    }
}

void profile_scatter_attention_group_row(
    tensor_t& attention,
    unsigned int destination_row,
    const tensor_t& group,
    unsigned int kv_head,
    unsigned int group_size,
    unsigned int head_dim
) {
    if (
        destination_row >= attention.rows ||
        group.rows != group_size ||
        group.cols != head_dim
    ) {
        throw std::runtime_error("profile attention row scatter shape mismatch");
    }
    for (unsigned int row = 0; row < group_size; row++) {
        const unsigned int query_head = kv_head * group_size + row;
        const unsigned int destination_base = query_head * head_dim;
        for (unsigned int elem = 0; elem < head_dim; elem++) {
            attention.at(destination_row, destination_base + elem) =
                group.at(row, elem);
        }
    }
}

void profile_apply_rope(
    const model_shape_t& shape,
    tensor_t& q,
    tensor_t& k,
    unsigned int position
) {
    if (
        q.rows != 1 ||
        q.cols != shape.hidden_size ||
        k.rows != 1 ||
        k.cols != shape.kv_channels()
    ) {
        throw std::runtime_error("profile RoPE input shape mismatch");
    }
    const unsigned int head_dim = shape.head_dim();
    const unsigned int half = head_dim / 2;
    for (unsigned int head = 0; head < shape.num_heads; head++) {
        const unsigned int base = head * head_dim;
        for (unsigned int i = 0; i < half; i++) {
            const double exponent = double(2 * i) / head_dim;
            const double inverse_frequency =
                1.0 / std::pow(kRopeTheta, exponent);
            const double angle = position * inverse_frequency;
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            const double x0 = dequantize_fix16(q.at(0, base + i));
            const double x1 =
                dequantize_fix16(q.at(0, base + half + i));
            q.at(0, base + i) =
                quantize_fix16(x0 * cosine - x1 * sine);
            q.at(0, base + half + i) =
                quantize_fix16(x1 * cosine + x0 * sine);
        }
    }
    for (unsigned int head = 0; head < shape.num_kv_heads; head++) {
        const unsigned int base = head * head_dim;
        for (unsigned int i = 0; i < half; i++) {
            const double exponent = double(2 * i) / head_dim;
            const double inverse_frequency =
                1.0 / std::pow(kRopeTheta, exponent);
            const double angle = position * inverse_frequency;
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            const double x0 = dequantize_fix16(k.at(0, base + i));
            const double x1 =
                dequantize_fix16(k.at(0, base + half + i));
            k.at(0, base + i) =
                quantize_fix16(x0 * cosine - x1 * sine);
            k.at(0, base + half + i) =
                quantize_fix16(x1 * cosine + x0 * sine);
        }
    }
}

bool check_attention_status(
    const std::string& tag,
    const model_shape_t& shape,
    const decoded_status_t& status,
    unsigned int position,
    double controller_ms
) {
    const unsigned int expected_output_waves =
        attention_output_waves(shape);
    const unsigned int expected_tasks =
        attention_expected_mm_tasks(shape, position);
    const unsigned int expected_packets =
        expected_tasks * kMmPacketsPerBlock;
    const bool pass =
        status.op == unsigned(kOpDecodeSmoke) &&
        status.code == 0 &&
        status.output_waves == expected_output_waves &&
        status.mm_tasks == expected_tasks &&
        status.completed_packets == expected_packets &&
        status.last_task;
    std::cout
        << tag
        << " position=" << position
        << " context_len=" << (position + 1)
        << " attention_tile=" << kAttentionTile
        << " tile_count=" << attention_tile_count(position)
        << " head_dim=" << shape.head_dim()
        << " group_size=" << shape.gqa_group_size()
        << " output_waves=" << status.output_waves
        << " expected_output_waves=" << expected_output_waves
        << " status=" << status.code
        << " mm_tasks=" << status.mm_tasks
        << " expected_mm_tasks=" << expected_tasks
        << " packets=" << status.completed_packets
        << " expected_packets=" << expected_packets
        << " controller_ms=" << controller_ms
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool check_prefill_block_attention_status(
    const std::string& tag,
    const model_shape_t& shape,
    const decoded_status_t& status,
    unsigned int position,
    unsigned int token_count,
    double controller_ms
) {
    const unsigned int tile_count =
        unsigned(ceildiv(position + token_count, kAttentionTile));
    const unsigned int tasks_per_core =
        tile_count * shape.gqa_group_size() *
        (1 + attention_output_waves(shape));
    const unsigned int expected_tasks = 2 * tasks_per_core;
    const unsigned int expected_packets =
        expected_tasks * kMmPacketsPerBlock;
    const bool pass =
        status.op == unsigned(kOpAttnPrefillBlock) &&
        status.code == 0 &&
        status.output_waves == attention_output_waves(shape) &&
        status.mm_tasks == expected_tasks &&
        status.completed_packets == expected_packets &&
        status.last_task;
    std::cout
        << tag
        << " position_begin=" << position
        << " token_count=" << token_count
        << " context_len=" << (position + token_count)
        << " tile_count=" << tile_count
        << " attention_calls=1"
        << " mm_tasks=" << status.mm_tasks
        << " expected_mm_tasks=" << expected_tasks
        << " packets=" << status.completed_packets
        << " expected_packets=" << expected_packets
        << " controller_ms=" << controller_ms
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool run_attention_profile(
    const model_shape_t& shape,
    accelerator_t& accelerator,
    unsigned int position,
    uint32_t seed
) {
    if (position >= shape.max_seq_len) {
        throw std::runtime_error("--position exceeds profile max_seq_len");
    }
    if (shape.num_kv_heads != 2) {
        throw std::runtime_error("attention profile currently expects 2 KV heads");
    }

    const unsigned int group_size = shape.gqa_group_size();
    const unsigned int head_dim = shape.head_dim();
    tensor_t query0 = make_random_tensor(
        group_size,
        head_dim,
        seed ^ 0x71000001u,
        -48,
        48
    );
    tensor_t query1 = make_random_tensor(
        group_size,
        head_dim,
        seed ^ 0x71000002u,
        -48,
        48
    );
    tensor_t k0 = make_random_tensor(
        position + 1,
        head_dim,
        seed ^ 0x71000003u,
        -48,
        48
    );
    tensor_t k1 = make_random_tensor(
        position + 1,
        head_dim,
        seed ^ 0x71000004u,
        -48,
        48
    );
    tensor_t v0 = make_random_tensor(
        position + 1,
        head_dim,
        seed ^ 0x71000005u,
        -48,
        48
    );
    tensor_t v1 = make_random_tensor(
        position + 1,
        head_dim,
        seed ^ 0x71000006u,
        -48,
        48
    );

    accelerator.initialize_kv_cache_heads(
        0,
        tensor_prefix_rows(k0, position),
        tensor_prefix_rows(k1, position),
        tensor_prefix_rows(v0, position),
        tensor_prefix_rows(v1, position)
    );

    tensor_t k_payload(shape.num_kv_heads, head_dim);
    tensor_t v_payload(shape.num_kv_heads, head_dim);
    for (unsigned int elem = 0; elem < head_dim; elem++) {
        k_payload.at(0, elem) = k0.at(position, elem);
        k_payload.at(1, elem) = k1.at(position, elem);
        v_payload.at(0, elem) = v0.at(position, elem);
        v_payload.at(1, elem) = v1.at(position, elem);
    }

    operator_result_t actual = accelerator.run_attention(
        kOpDecodeSmoke,
        query0,
        query1,
        &k_payload,
        &v_payload,
        position,
        1
    );

    bool pass = check_attention_status(
        "ATTENTION_PROFILE_STATUS",
        shape,
        actual.status,
        position,
        actual.controller_ms
    );
    pass = compare_tensors(
        "attention_profile_port0",
        actual.port0,
        golden_flash_attention(query0, k0, v0),
        4
    ) && pass;
    pass = compare_tensors(
        "attention_profile_port1",
        actual.port1,
        golden_flash_attention(query1, k1, v1),
        4
    ) && pass;
    std::cout
        << "ATTENTION_PROFILE position=" << position
        << " seed=" << seed
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool run_attention_pd_profile(
    const model_shape_t& shape,
    accelerator_t& accelerator,
    unsigned int requested_prefill_len,
    unsigned int prefill_start,
    const std::string& phase,
    uint32_t seed
) {
    if (shape.num_kv_heads != 2) {
        throw std::runtime_error("attention P/D profile currently expects 2 KV heads");
    }
    const unsigned int prefill_len =
        requested_prefill_len == 0 ?
        unsigned(std::min<std::size_t>(64, shape.max_seq_len - 1)) :
        requested_prefill_len;
    if (prefill_len == 0 || prefill_len >= shape.max_seq_len) {
        throw std::runtime_error("--prefill-len must be in 1..max_seq_len-1");
    }
    if (prefill_start >= prefill_len) {
        throw std::runtime_error("--prefill-start must be smaller than --prefill-len");
    }

    const bool run_prefill =
        phase == "p" || phase == "prefill" || phase == "pd" ||
        phase == "both";
    const bool run_decode =
        phase == "d" || phase == "decode" || phase == "pd" ||
        phase == "both";
    if (!run_prefill && !run_decode) {
        throw std::runtime_error("--phase must be p, d, or pd");
    }

    const unsigned int group_size = shape.gqa_group_size();
    const unsigned int head_dim = shape.head_dim();
    const unsigned int total_positions = prefill_len + 1;
    tensor_t k0 = make_random_tensor(
        total_positions,
        head_dim,
        seed ^ 0x75000001u,
        -48,
        48
    );
    tensor_t k1 = make_random_tensor(
        total_positions,
        head_dim,
        seed ^ 0x75000002u,
        -48,
        48
    );
    tensor_t v0 = make_random_tensor(
        total_positions,
        head_dim,
        seed ^ 0x75000003u,
        -48,
        48
    );
    tensor_t v1 = make_random_tensor(
        total_positions,
        head_dim,
        seed ^ 0x75000004u,
        -48,
        48
    );
    tensor_t k_payload(shape.num_kv_heads, head_dim);
    tensor_t v_payload(shape.num_kv_heads, head_dim);

    bool pass = true;
    unsigned int prefill_checks = 0;
    unsigned int decode_checks = 0;
    double prefill_controller_ms = 0.0;
    double decode_controller_ms = 0.0;

    if (run_prefill) {
        accelerator.initialize_kv_cache_heads(
            0,
            tensor_prefix_rows(k0, prefill_start),
            tensor_prefix_rows(k1, prefill_start),
            tensor_prefix_rows(v0, prefill_start),
            tensor_prefix_rows(v1, prefill_start)
        );
        for (unsigned int position = prefill_start;
             position < prefill_len;
             position++) {
            tensor_t query0 = make_random_tensor(
                group_size,
                head_dim,
                seed ^ (0x75100000u + position * 2u),
                -48,
                48
            );
            tensor_t query1 = make_random_tensor(
                group_size,
                head_dim,
                seed ^ (0x75100001u + position * 2u),
                -48,
                48
            );
            fill_profile_kv_payload(
                shape,
                k_payload,
                v_payload,
                k0,
                k1,
                v0,
                v1,
                position
            );
            operator_result_t actual = accelerator.run_attention(
                kOpDecodeSmoke,
                query0,
                query1,
                &k_payload,
                &v_payload,
                position,
                1
            );
            prefill_controller_ms += actual.controller_ms;
            const std::string prefix =
                "attention_pd_prefill_pos" + std::to_string(position);
            bool position_pass = check_attention_status(
                "ATTENTION_PD_PREFILL_STATUS",
                shape,
                actual.status,
                position,
                actual.controller_ms
            );
            position_pass = compare_tensors(
                prefix + "_port0",
                actual.port0,
                golden_flash_attention(
                    query0,
                    tensor_prefix_rows(k0, position + 1),
                    tensor_prefix_rows(v0, position + 1)
                ),
                4
            ) && position_pass;
            position_pass = compare_tensors(
                prefix + "_port1",
                actual.port1,
                golden_flash_attention(
                    query1,
                    tensor_prefix_rows(k1, position + 1),
                    tensor_prefix_rows(v1, position + 1)
                ),
                4
            ) && position_pass;
            prefill_checks++;
            pass = position_pass && pass;
        }
    }

    if (run_decode) {
        if (!run_prefill) {
            accelerator.initialize_kv_cache_heads(
                0,
                tensor_prefix_rows(k0, prefill_len),
                tensor_prefix_rows(k1, prefill_len),
                tensor_prefix_rows(v0, prefill_len),
                tensor_prefix_rows(v1, prefill_len)
            );
        }
        const unsigned int position = prefill_len;
        tensor_t query0 = make_random_tensor(
            group_size,
            head_dim,
            seed ^ (0x75200000u + position * 2u),
            -48,
            48
        );
        tensor_t query1 = make_random_tensor(
            group_size,
            head_dim,
            seed ^ (0x75200001u + position * 2u),
            -48,
            48
        );
        fill_profile_kv_payload(
            shape,
            k_payload,
            v_payload,
            k0,
            k1,
            v0,
            v1,
            position
        );
        operator_result_t actual = accelerator.run_attention(
            kOpDecodeSmoke,
            query0,
            query1,
            &k_payload,
            &v_payload,
            position,
            1
        );
        decode_controller_ms += actual.controller_ms;
        bool decode_pass = check_attention_status(
            "ATTENTION_PD_DECODE_STATUS",
            shape,
            actual.status,
            position,
            actual.controller_ms
        );
        decode_pass = compare_tensors(
            "attention_pd_decode_port0",
            actual.port0,
            golden_flash_attention(
                query0,
                tensor_prefix_rows(k0, position + 1),
                tensor_prefix_rows(v0, position + 1)
            ),
            4
        ) && decode_pass;
        decode_pass = compare_tensors(
            "attention_pd_decode_port1",
            actual.port1,
            golden_flash_attention(
                query1,
                tensor_prefix_rows(k1, position + 1),
                tensor_prefix_rows(v1, position + 1)
            ),
            4
        ) && decode_pass;
        decode_checks++;
        pass = decode_pass && pass;
    }

    std::cout
        << "ATTENTION_PD_PROFILE phase=" << phase
        << " prefill_len=" << prefill_len
        << " prefill_start=" << prefill_start
        << " prefill_checks=" << prefill_checks
        << " decode_checks=" << decode_checks
        << " prefill_controller_ms=" << prefill_controller_ms
        << " decode_controller_ms=" << decode_controller_ms
        << " seed=" << seed
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool run_attention_block_profile(
    const model_shape_t& shape,
    const model_data_t& model,
    accelerator_t& accelerator,
    unsigned int position,
    uint32_t seed
) {
    if (position >= shape.max_seq_len) {
        throw std::runtime_error("--position exceeds profile max_seq_len");
    }
    if (shape.num_kv_heads != 2) {
        throw std::runtime_error("attention block profile currently expects 2 KV heads");
    }

    const unsigned int layer = 0;
    const unsigned int head_dim = shape.head_dim();
    tensor_t hidden = make_random_tensor(
        1,
        shape.hidden_size,
        seed ^ 0x72000001u,
        -64,
        64
    );
    tensor_t q = accelerator.run_feature(
        kOpQProjection,
        hidden,
        nullptr,
        layer,
        position
    );
    tensor_t k = accelerator.run_feature(
        kOpKProjection,
        hidden,
        nullptr,
        layer,
        position
    );
    tensor_t v = accelerator.run_feature(
        kOpVProjection,
        hidden,
        nullptr,
        layer,
        position
    );

    bool pass = true;
    pass = compare_tensors(
        "attention_block_q_projection",
        q,
        golden_projection(shape, model, kOpQProjection, hidden, layer),
        1
    ) && pass;
    pass = compare_tensors(
        "attention_block_k_projection",
        k,
        golden_projection(shape, model, kOpKProjection, hidden, layer),
        1
    ) && pass;
    pass = compare_tensors(
        "attention_block_v_projection",
        v,
        golden_projection(shape, model, kOpVProjection, hidden, layer),
        1
    ) && pass;

    profile_apply_rope(shape, q, k, position);
    tensor_t query0 = profile_build_query_group(shape, q, 0);
    tensor_t query1 = profile_build_query_group(shape, q, 1);
    tensor_t k_payload = profile_build_kv_payload(shape, k);
    tensor_t v_payload = profile_build_kv_payload(shape, v);

    tensor_t previous_k0 = make_random_tensor(
        position,
        head_dim,
        seed ^ 0x72000002u,
        -48,
        48
    );
    tensor_t previous_k1 = make_random_tensor(
        position,
        head_dim,
        seed ^ 0x72000003u,
        -48,
        48
    );
    tensor_t previous_v0 = make_random_tensor(
        position,
        head_dim,
        seed ^ 0x72000004u,
        -48,
        48
    );
    tensor_t previous_v1 = make_random_tensor(
        position,
        head_dim,
        seed ^ 0x72000005u,
        -48,
        48
    );
    accelerator.initialize_kv_cache_heads(
        layer,
        previous_k0,
        previous_k1,
        previous_v0,
        previous_v1
    );

    operator_result_t decoded = accelerator.run_attention(
        kOpDecodeSmoke,
        query0,
        query1,
        &k_payload,
        &v_payload,
        position,
        1,
        layer
    );
    pass = check_attention_status(
        "ATTENTION_BLOCK_STATUS",
        shape,
        decoded.status,
        position,
        decoded.controller_ms
    ) && pass;

    const tensor_t active_k0 = tensor_append_row(
        previous_k0,
        profile_payload_head(k_payload, 0)
    );
    const tensor_t active_k1 = tensor_append_row(
        previous_k1,
        profile_payload_head(k_payload, 1)
    );
    const tensor_t active_v0 = tensor_append_row(
        previous_v0,
        profile_payload_head(v_payload, 0)
    );
    const tensor_t active_v1 = tensor_append_row(
        previous_v1,
        profile_payload_head(v_payload, 1)
    );
    pass = compare_tensors(
        "attention_block_flash_port0",
        decoded.port0,
        golden_flash_attention(query0, active_k0, active_v0),
        4
    ) && pass;
    pass = compare_tensors(
        "attention_block_flash_port1",
        decoded.port1,
        golden_flash_attention(query1, active_k1, active_v1),
        4
    ) && pass;

    tensor_t attention(1, shape.hidden_size);
    profile_scatter_attention_group(
        attention,
        decoded.port0,
        0,
        shape.gqa_group_size(),
        head_dim
    );
    profile_scatter_attention_group(
        attention,
        decoded.port1,
        1,
        shape.gqa_group_size(),
        head_dim
    );
    tensor_t projected = accelerator.run_feature(
        kOpOProjection,
        attention,
        nullptr,
        layer,
        position
    );
    pass = compare_tensors(
        "attention_block_o_projection",
        projected,
        golden_projection(shape, model, kOpOProjection, attention, layer),
        1
    ) && pass;

    std::cout
        << "ATTENTION_BLOCK_PROFILE position=" << position
        << " seed=" << seed
        << " hidden_checksum=0x" << std::hex << tensor_checksum(hidden)
        << " attention_checksum=0x" << tensor_checksum(attention)
        << " projected_checksum=0x" << tensor_checksum(projected)
        << std::dec
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool run_attention_sublayer_profile(
    const model_shape_t& shape,
    const model_data_t& model,
    accelerator_t& accelerator,
    unsigned int position,
    uint32_t seed
) {
    if (position >= shape.max_seq_len) {
        throw std::runtime_error("--position exceeds profile max_seq_len");
    }
    if (shape.num_kv_heads != 2) {
        throw std::runtime_error("attention sublayer profile currently expects 2 KV heads");
    }

    const unsigned int layer = 0;
    const unsigned int head_dim = shape.head_dim();
    tensor_t hidden = make_random_tensor(
        1,
        shape.hidden_size,
        seed ^ 0x73000001u,
        -64,
        64
    );
    const tensor_t norm_weight = model.norm_row(layer, false);
    tensor_t normalized = accelerator.run_feature(
        kOpRmsNorm,
        hidden,
        &norm_weight,
        layer,
        position
    );

    bool pass = true;
    pass = compare_tensors(
        "attention_sublayer_rmsnorm",
        normalized,
        golden_rmsnorm(hidden, norm_weight),
        1
    ) && pass;

    tensor_t q = accelerator.run_feature(
        kOpQProjection,
        normalized,
        nullptr,
        layer,
        position
    );
    tensor_t k = accelerator.run_feature(
        kOpKProjection,
        normalized,
        nullptr,
        layer,
        position
    );
    tensor_t v = accelerator.run_feature(
        kOpVProjection,
        normalized,
        nullptr,
        layer,
        position
    );
    pass = compare_tensors(
        "attention_sublayer_q_projection",
        q,
        golden_projection(shape, model, kOpQProjection, normalized, layer),
        1
    ) && pass;
    pass = compare_tensors(
        "attention_sublayer_k_projection",
        k,
        golden_projection(shape, model, kOpKProjection, normalized, layer),
        1
    ) && pass;
    pass = compare_tensors(
        "attention_sublayer_v_projection",
        v,
        golden_projection(shape, model, kOpVProjection, normalized, layer),
        1
    ) && pass;

    profile_apply_rope(shape, q, k, position);
    tensor_t query0 = profile_build_query_group(shape, q, 0);
    tensor_t query1 = profile_build_query_group(shape, q, 1);
    tensor_t k_payload = profile_build_kv_payload(shape, k);
    tensor_t v_payload = profile_build_kv_payload(shape, v);

    tensor_t previous_k0 = make_random_tensor(
        position,
        head_dim,
        seed ^ 0x73000002u,
        -48,
        48
    );
    tensor_t previous_k1 = make_random_tensor(
        position,
        head_dim,
        seed ^ 0x73000003u,
        -48,
        48
    );
    tensor_t previous_v0 = make_random_tensor(
        position,
        head_dim,
        seed ^ 0x73000004u,
        -48,
        48
    );
    tensor_t previous_v1 = make_random_tensor(
        position,
        head_dim,
        seed ^ 0x73000005u,
        -48,
        48
    );
    accelerator.initialize_kv_cache_heads(
        layer,
        previous_k0,
        previous_k1,
        previous_v0,
        previous_v1
    );

    operator_result_t decoded = accelerator.run_attention(
        kOpDecodeSmoke,
        query0,
        query1,
        &k_payload,
        &v_payload,
        position,
        1,
        layer
    );
    pass = check_attention_status(
        "ATTENTION_SUBLAYER_STATUS",
        shape,
        decoded.status,
        position,
        decoded.controller_ms
    ) && pass;

    const tensor_t active_k0 = tensor_append_row(
        previous_k0,
        profile_payload_head(k_payload, 0)
    );
    const tensor_t active_k1 = tensor_append_row(
        previous_k1,
        profile_payload_head(k_payload, 1)
    );
    const tensor_t active_v0 = tensor_append_row(
        previous_v0,
        profile_payload_head(v_payload, 0)
    );
    const tensor_t active_v1 = tensor_append_row(
        previous_v1,
        profile_payload_head(v_payload, 1)
    );
    pass = compare_tensors(
        "attention_sublayer_flash_port0",
        decoded.port0,
        golden_flash_attention(query0, active_k0, active_v0),
        4
    ) && pass;
    pass = compare_tensors(
        "attention_sublayer_flash_port1",
        decoded.port1,
        golden_flash_attention(query1, active_k1, active_v1),
        4
    ) && pass;

    tensor_t attention(1, shape.hidden_size);
    profile_scatter_attention_group(
        attention,
        decoded.port0,
        0,
        shape.gqa_group_size(),
        head_dim
    );
    profile_scatter_attention_group(
        attention,
        decoded.port1,
        1,
        shape.gqa_group_size(),
        head_dim
    );
    tensor_t projected = accelerator.run_feature(
        kOpOProjection,
        attention,
        nullptr,
        layer,
        position
    );
    pass = compare_tensors(
        "attention_sublayer_o_projection",
        projected,
        golden_projection(shape, model, kOpOProjection, attention, layer),
        1
    ) && pass;

    tensor_t residual = accelerator.run_feature(
        kOpResidualAdd,
        hidden,
        &projected,
        layer,
        position
    );
    pass = compare_tensors(
        "attention_sublayer_residual",
        residual,
        golden_residual(hidden, projected),
        0
    ) && pass;

    std::cout
        << "ATTENTION_SUBLAYER_PROFILE position=" << position
        << " seed=" << seed
        << " hidden_checksum=0x" << std::hex << tensor_checksum(hidden)
        << " normalized_checksum=0x" << tensor_checksum(normalized)
        << " attention_checksum=0x" << tensor_checksum(attention)
        << " projected_checksum=0x" << tensor_checksum(projected)
        << " residual_checksum=0x" << tensor_checksum(residual)
        << std::dec
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool run_ffn_sublayer_profile(
    const model_shape_t& shape,
    const model_data_t& model,
    accelerator_t& accelerator,
    unsigned int position,
    uint32_t seed
) {
    if (position >= shape.max_seq_len) {
        throw std::runtime_error("--position exceeds profile max_seq_len");
    }

    const unsigned int layer = 0;
    tensor_t hidden = make_random_tensor(
        1,
        shape.hidden_size,
        seed ^ 0x74000001u,
        -64,
        64
    );
    const tensor_t norm_weight = model.norm_row(layer, true);
    tensor_t normalized = accelerator.run_feature(
        kOpRmsNorm,
        hidden,
        &norm_weight,
        layer,
        position
    );

    bool pass = true;
    pass = compare_tensors(
        "ffn_sublayer_rmsnorm",
        normalized,
        golden_rmsnorm(hidden, norm_weight),
        1
    ) && pass;

    tensor_t gate = accelerator.run_feature(
        kOpFfnGate,
        normalized,
        nullptr,
        layer,
        position
    );
    tensor_t up = accelerator.run_feature(
        kOpFfnUp,
        normalized,
        nullptr,
        layer,
        position
    );
    pass = compare_tensors(
        "ffn_sublayer_gate_projection",
        gate,
        golden_projection(shape, model, kOpFfnGate, normalized, layer),
        1
    ) && pass;
    pass = compare_tensors(
        "ffn_sublayer_up_projection",
        up,
        golden_projection(shape, model, kOpFfnUp, normalized, layer),
        1
    ) && pass;

    tensor_t activated = accelerator.run_feature(
        kOpSiluMul,
        gate,
        &up,
        layer,
        position
    );
    pass = compare_tensors(
        "ffn_sublayer_silu_mul",
        activated,
        golden_silu_mul(gate, up),
        1
    ) && pass;

    tensor_t down = accelerator.run_feature(
        kOpFfnDown,
        activated,
        nullptr,
        layer,
        position
    );
    pass = compare_tensors(
        "ffn_sublayer_down_projection",
        down,
        golden_projection(shape, model, kOpFfnDown, activated, layer),
        4
    ) && pass;

    tensor_t residual = accelerator.run_feature(
        kOpResidualAdd,
        hidden,
        &down,
        layer,
        position
    );
    pass = compare_tensors(
        "ffn_sublayer_residual",
        residual,
        golden_residual(hidden, down),
        0
    ) && pass;

    std::cout
        << "FFN_SUBLAYER_PROFILE position=" << position
        << " seed=" << seed
        << " hidden_checksum=0x" << std::hex << tensor_checksum(hidden)
        << " normalized_checksum=0x" << tensor_checksum(normalized)
        << " gate_checksum=0x" << tensor_checksum(gate)
        << " up_checksum=0x" << tensor_checksum(up)
        << " activated_checksum=0x" << tensor_checksum(activated)
        << " down_checksum=0x" << tensor_checksum(down)
        << " residual_checksum=0x" << tensor_checksum(residual)
        << std::dec
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool run_nop_verification(accelerator_t& accelerator) {
    std::cout << "NOP launching\n";
    const decoded_status_t status = accelerator.run_nop();
    std::cout
        << "NOP returned status=" << status.code
        << " packets=" << status.completed_packets
        << " token_count=" << status.token_count
        << "\n";
    const bool pass =
        status.op == unsigned(kOpNop) &&
        status.code == 0 &&
        status.completed_packets == 0 &&
        status.last_task;
    std::cout << "NOP_VERIFY " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

bool run_nop_controller_only_verification(accelerator_t& accelerator) {
    std::cout << "NOP_CTRL_ONLY launching\n";
    const decoded_status_t status = accelerator.run_nop_controller_only();
    std::cout
        << "NOP_CTRL_ONLY returned status=" << status.code
        << " packets=" << status.completed_packets
        << " token_count=" << status.token_count
        << "\n";
    const bool pass =
        status.op == unsigned(kOpNop) &&
        status.code == 0 &&
        status.completed_packets == 0 &&
        status.last_task;
    std::cout
        << "NOP_CTRL_ONLY_VERIFY "
        << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool run_nop_controller_enqueue_only_verification(accelerator_t& accelerator) {
    std::cout << "NOP_CTRL_ENQUEUE_ONLY launching\n";
    const decoded_status_t status =
        accelerator.run_nop_controller_enqueue_only();
    std::cout
        << "NOP_CTRL_ENQUEUE_ONLY completed synthetic status="
        << status.code
        << " controller command reached COMPLETE\n";
    const bool pass =
        status.op == unsigned(kOpNop) &&
        status.code == 0 &&
        status.last_task;
    std::cout
        << "NOP_CTRL_ENQUEUE_ONLY_VERIFY "
        << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool run_prefill_vector_profile(
    const model_shape_t& shape,
    const model_data_t& model,
    accelerator_t& accelerator,
    unsigned int requested_token_count,
    uint32_t seed
) {
    const unsigned int token_count =
        requested_token_count == 0 ? 8 : requested_token_count;
    if (token_count == 0 || token_count > kMaxTokensPerLaunch) {
        throw std::runtime_error(
            "profile-prefill-vector --prefill-len must be in 1..8"
        );
    }
    const unsigned int layer = 0;
    bool pass = true;

    tensor_t hidden = make_random_tensor(
        token_count,
        shape.hidden_size,
        seed ^ 0x76100001u,
        -64,
        64
    );
    const tensor_t norm_weight = model.norm_row(layer, false);
    tensor_t normalized = accelerator.run_feature(
        kOpRmsNorm,
        hidden,
        &norm_weight,
        layer,
        0
    );
    pass = compare_tensors(
        "prefill_vector_rmsnorm",
        normalized,
        golden_rmsnorm(hidden, norm_weight),
        1
    ) && pass;

    tensor_t gate = make_random_tensor(
        token_count,
        shape.intermediate_size,
        seed ^ 0x76100002u,
        -64,
        64
    );
    tensor_t up = make_random_tensor(
        token_count,
        shape.intermediate_size,
        seed ^ 0x76100003u,
        -64,
        64
    );
    tensor_t activated = accelerator.run_feature(
        kOpSiluMul,
        gate,
        &up,
        layer,
        0
    );
    pass = compare_tensors(
        "prefill_vector_silu_mul",
        activated,
        golden_silu_mul(gate, up),
        1
    ) && pass;

    tensor_t rhs = make_random_tensor(
        token_count,
        shape.hidden_size,
        seed ^ 0x76100004u,
        -64,
        64
    );
    tensor_t residual = accelerator.run_feature(
        kOpResidualAdd,
        hidden,
        &rhs,
        layer,
        0
    );
    pass = compare_tensors(
        "prefill_vector_residual",
        residual,
        golden_residual(hidden, rhs),
        0
    ) && pass;

    std::cout
        << "PREFILL_VECTOR_PROFILE token_count=" << token_count
        << " rms_checksum=0x" << std::hex << tensor_checksum(normalized)
        << " silu_checksum=0x" << tensor_checksum(activated)
        << " residual_checksum=0x" << tensor_checksum(residual)
        << std::dec
        << " seed=" << seed
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool run_prefill_softmax_diagnostic(
    const model_shape_t& shape,
    const model_data_t& model,
    unsigned int requested_prefill_len,
    unsigned int requested_prefill_start,
    uint32_t seed
) {
    const unsigned int prefill_len =
        requested_prefill_len == 0 ? 8 : requested_prefill_len;
    if (
        prefill_len == 0 ||
        prefill_len > shape.max_seq_len ||
        requested_prefill_start >= prefill_len
    ) {
        throw std::runtime_error(
            "diagnose-prefill-softmax requires 0 <= prefill-start < "
            "prefill-len <= max_seq_len"
        );
    }
    const unsigned int token_count =
        prefill_len - requested_prefill_start;
    if (token_count == 0 || token_count > kMaxTokensPerLaunch) {
        throw std::runtime_error(
            "diagnose-prefill-softmax supports one block of 1..8 tokens"
        );
    }
    if (shape.num_layers != 1 || shape.num_kv_heads != 2) {
        throw std::runtime_error(
            "diagnose-prefill-softmax expects the one-layer, two-KV-head profile"
        );
    }

    const unsigned int layer = 0;
    const unsigned int head_dim = shape.head_dim();
    const unsigned int group_size = shape.gqa_group_size();
    const tensor_t hidden = make_random_tensor(
        prefill_len,
        shape.hidden_size,
        seed ^ 0x76000001u,
        -64,
        64
    );
    tensor_t cache_k0 = make_random_tensor(
        prefill_len, head_dim, seed ^ 0x76010001u, -32, 32
    );
    tensor_t cache_k1 = make_random_tensor(
        prefill_len, head_dim, seed ^ 0x76010002u, -32, 32
    );
    tensor_t cache_v0 = make_random_tensor(
        prefill_len, head_dim, seed ^ 0x76010003u, -32, 32
    );
    tensor_t cache_v1 = make_random_tensor(
        prefill_len, head_dim, seed ^ 0x76010004u, -32, 32
    );

    const tensor_t block_hidden = tensor_rows(
        hidden, requested_prefill_start, token_count
    );
    const tensor_t normalized = golden_rmsnorm(
        block_hidden, model.norm_row(layer, false)
    );
    tensor_t q = golden_projection(
        shape, model, kOpQProjection, normalized, layer
    );
    tensor_t k = golden_projection(
        shape, model, kOpKProjection, normalized, layer
    );
    tensor_t v = golden_projection(
        shape, model, kOpVProjection, normalized, layer
    );
    tensor_t query_block0(token_count * group_size, head_dim);
    tensor_t query_block1(token_count * group_size, head_dim);

    for (unsigned int local_token = 0;
         local_token < token_count;
         local_token++) {
        const unsigned int position = requested_prefill_start + local_token;
        tensor_t q_row = tensor_rows(q, local_token, 1);
        tensor_t k_row = tensor_rows(k, local_token, 1);
        const tensor_t v_row = tensor_rows(v, local_token, 1);
        profile_apply_rope(shape, q_row, k_row, position);
        tensor_store_rows(
            query_block0,
            local_token * group_size,
            profile_build_query_group(shape, q_row, 0)
        );
        tensor_store_rows(
            query_block1,
            local_token * group_size,
            profile_build_query_group(shape, q_row, 1)
        );
        const tensor_t k_payload = profile_build_kv_payload(shape, k_row);
        const tensor_t v_payload = profile_build_kv_payload(shape, v_row);
        for (unsigned int elem = 0; elem < head_dim; elem++) {
            cache_k0.at(position, elem) = k_payload.at(0, elem);
            cache_k1.at(position, elem) = k_payload.at(1, elem);
            cache_v0.at(position, elem) = v_payload.at(0, elem);
            cache_v1.at(position, elem) = v_payload.at(1, elem);
        }
    }

    bool current_within_tolerance = true;
    bool wide_within_tolerance = true;
    bool wide_pv_within_tolerance = true;
    bool wide_rescale_pv_within_tolerance = true;
    bool q2_14_within_tolerance = true;
    bool legacy_global_reference_within_tolerance = true;
    bool current_reference_within_tolerance = true;
    bool q2_14_reference_within_tolerance = true;
    for (unsigned int local_token = 0;
         local_token < token_count;
         local_token++) {
        const unsigned int position = requested_prefill_start + local_token;
        const tensor_t query0 = tensor_rows(
            query_block0, local_token * group_size, group_size
        );
        const tensor_t query1 = tensor_rows(
            query_block1, local_token * group_size, group_size
        );
        const tensor_t active_k0 = tensor_prefix_rows(cache_k0, position + 1);
        const tensor_t active_k1 = tensor_prefix_rows(cache_k1, position + 1);
        const tensor_t active_v0 = tensor_prefix_rows(cache_v0, position + 1);
        const tensor_t active_v1 = tensor_prefix_rows(cache_v1, position + 1);
        const std::string name =
            "prefill_softmax_pos" + std::to_string(position);
        const tensor_t global0 = golden_flash_attention(
            query0, active_k0, active_v0
        );
        const tensor_t global1 = golden_flash_attention(
            query1, active_k1, active_v1
        );
        const tensor_t reference0 = golden_flash_attention_reference(
            query0, active_k0, active_v0
        );
        const tensor_t reference1 = golden_flash_attention_reference(
            query1, active_k1, active_v1
        );
        const tensor_t tiled0 = golden_tiled_flash_attention(
            query0, active_k0, active_v0
        );
        const tensor_t tiled1 = golden_tiled_flash_attention(
            query1, active_k1, active_v1
        );
        const tensor_t q2_14_0 = golden_tiled_flash_attention_q2_14(
            query0, active_k0, active_v0
        );
        const tensor_t q2_14_1 = golden_tiled_flash_attention_q2_14(
            query1, active_k1, active_v1
        );
        current_within_tolerance = compare_tensors(
            name + "_port0_tiled_global",
            tiled0,
            global0,
            4
        ) && current_within_tolerance;
        current_within_tolerance = compare_tensors(
            name + "_port1_tiled_global",
            tiled1,
            global1,
            4
        ) && current_within_tolerance;
        wide_within_tolerance = compare_tensors(
            name + "_port0_wide_rescale_global",
            golden_tiled_flash_attention_wide_rescale(
                query0, active_k0, active_v0
            ),
            global0,
            4
        ) && wide_within_tolerance;
        wide_pv_within_tolerance = compare_tensors(
            name + "_port0_wide_pv_global",
            golden_tiled_flash_attention_wide_pv(
                query0, active_k0, active_v0
            ),
            global0,
            4
        ) && wide_pv_within_tolerance;
        wide_pv_within_tolerance = compare_tensors(
            name + "_port1_wide_pv_global",
            golden_tiled_flash_attention_wide_pv(
                query1, active_k1, active_v1
            ),
            global1,
            4
        ) && wide_pv_within_tolerance;
        wide_rescale_pv_within_tolerance = compare_tensors(
            name + "_port0_wide_rescale_pv_global",
            golden_tiled_flash_attention_wide_rescale_pv(
                query0, active_k0, active_v0
            ),
            global0,
            4
        ) && wide_rescale_pv_within_tolerance;
        q2_14_within_tolerance = compare_tensors(
            name + "_port0_q2_14_global",
            q2_14_0,
            global0,
            4
        ) && q2_14_within_tolerance;
        q2_14_within_tolerance = compare_tensors(
            name + "_port1_q2_14_global",
            q2_14_1,
            global1,
            4
        ) && q2_14_within_tolerance;
        legacy_global_reference_within_tolerance = compare_tensors(
            name + "_port0_legacy_global_reference",
            global0,
            reference0,
            4
        ) && legacy_global_reference_within_tolerance;
        legacy_global_reference_within_tolerance = compare_tensors(
            name + "_port1_legacy_global_reference",
            global1,
            reference1,
            4
        ) && legacy_global_reference_within_tolerance;
        current_reference_within_tolerance = compare_tensors(
            name + "_port0_tiled_reference",
            tiled0,
            reference0,
            4
        ) && current_reference_within_tolerance;
        current_reference_within_tolerance = compare_tensors(
            name + "_port1_tiled_reference",
            tiled1,
            reference1,
            4
        ) && current_reference_within_tolerance;
        q2_14_reference_within_tolerance = compare_tensors(
            name + "_port0_q2_14_reference",
            q2_14_0,
            reference0,
            5
        ) && q2_14_reference_within_tolerance;
        q2_14_reference_within_tolerance = compare_tensors(
            name + "_port1_q2_14_reference",
            q2_14_1,
            reference1,
            5
        ) && q2_14_reference_within_tolerance;
        wide_rescale_pv_within_tolerance = compare_tensors(
            name + "_port1_wide_rescale_pv_global",
            golden_tiled_flash_attention_wide_rescale_pv(
                query1, active_k1, active_v1
            ),
            global1,
            4
        ) && wide_rescale_pv_within_tolerance;
        wide_within_tolerance = compare_tensors(
            name + "_port1_wide_rescale_global",
            golden_tiled_flash_attention_wide_rescale(
                query1, active_k1, active_v1
            ),
            global1,
            4
        ) && wide_within_tolerance;
    }
    std::cout
        << "PREFILL_SOFTMAX_DIAGNOSTIC prefill_start="
        << requested_prefill_start
        << " prefill_len=" << prefill_len
        << " token_count=" << token_count
        << " tile_count=" << ceildiv(prefill_len, kAttentionTile)
        << " tiled_global_within_tolerance="
        << (current_within_tolerance ? 1 : 0)
        << " wide_rescale_global_within_tolerance="
        << (wide_within_tolerance ? 1 : 0)
        << " wide_pv_global_within_tolerance="
        << (wide_pv_within_tolerance ? 1 : 0)
        << " wide_rescale_pv_global_within_tolerance="
        << (wide_rescale_pv_within_tolerance ? 1 : 0)
        << " q2_14_global_within_tolerance="
        << (q2_14_within_tolerance ? 1 : 0)
        << " legacy_global_reference_within_tolerance="
        << (legacy_global_reference_within_tolerance ? 1 : 0)
        << " tiled_reference_within_tolerance="
        << (current_reference_within_tolerance ? 1 : 0)
        << " q2_14_reference_within_tolerance="
        << (q2_14_reference_within_tolerance ? 1 : 0)
        << " COMPLETE\n";
    return true;
}

bool run_prefill_block_profile(
    const model_shape_t& shape,
    const model_data_t& model,
    accelerator_t& accelerator,
    unsigned int requested_prefill_len,
    unsigned int requested_prefill_start,
    uint32_t seed
) {
    const unsigned int prefill_len =
        requested_prefill_len == 0 ? 8 : requested_prefill_len;
    if (prefill_len == 0 || prefill_len > shape.max_seq_len) {
        throw std::runtime_error(
            "profile-prefill-block --prefill-len must be in 1..max_seq_len"
        );
    }
    if (requested_prefill_start >= prefill_len) {
        throw std::runtime_error(
            "profile-prefill-block --prefill-start must be smaller than --prefill-len"
        );
    }
    if (shape.num_layers != 1 || shape.num_kv_heads != 2) {
        throw std::runtime_error(
            "profile-prefill-block currently expects the one-layer, two-KV-head profile"
        );
    }

    constexpr unsigned int block_size = kMaxTokensPerLaunch;
    const unsigned int layer = 0;
    const unsigned int head_dim = shape.head_dim();
    const unsigned int group_size = shape.gqa_group_size();
    tensor_t hidden = make_random_tensor(
        prefill_len,
        shape.hidden_size,
        seed ^ 0x76000001u,
        -64,
        64
    );
    const unsigned int processed_tokens =
        prefill_len - requested_prefill_start;
    tensor_t final_hidden(processed_tokens, shape.hidden_size);
    // Resume-mode profiling preloads a deterministic, non-zero KV prefix.
    // This lets a single target block exercise the same HBM reads and online
    // attention schedule as a long prompt without simulating every preceding
    // transformer block in hw_emu.  Rows produced by the target block replace
    // the corresponding fixture rows below.
    tensor_t cache_k0 = make_random_tensor(
        prefill_len, head_dim, seed ^ 0x76010001u, -32, 32
    );
    tensor_t cache_k1 = make_random_tensor(
        prefill_len, head_dim, seed ^ 0x76010002u, -32, 32
    );
    tensor_t cache_v0 = make_random_tensor(
        prefill_len, head_dim, seed ^ 0x76010003u, -32, 32
    );
    tensor_t cache_v1 = make_random_tensor(
        prefill_len, head_dim, seed ^ 0x76010004u, -32, 32
    );

    accelerator.initialize_kv_cache_heads(
        layer,
        tensor_prefix_rows(cache_k0, requested_prefill_start),
        tensor_prefix_rows(cache_k1, requested_prefill_start),
        tensor_prefix_rows(cache_v0, requested_prefill_start),
        tensor_prefix_rows(cache_v1, requested_prefill_start)
    );

    bool pass = true;
    unsigned int block_count_total = 0;
    unsigned int attention_calls = 0;
    unsigned int attention_mm_tasks = 0;
    unsigned int attention_packets = 0;

    for (unsigned int block_begin = requested_prefill_start;
         block_begin < prefill_len;
         block_begin += block_size) {
        const unsigned int token_count = std::min(
            block_size,
            prefill_len - block_begin
        );
        const std::string prefix =
            "prefill_block" + std::to_string(block_count_total);
        tensor_t block_hidden = tensor_rows(
            hidden,
            block_begin,
            token_count
        );
        tensor_t residual = block_hidden;
        const tensor_t attention_norm_weight = model.norm_row(layer, false);
        tensor_t normalized = accelerator.run_feature(
            kOpRmsNorm,
            block_hidden,
            &attention_norm_weight,
            layer,
            block_begin
        );
        pass = compare_tensors(
            prefix + "_attention_rmsnorm",
            normalized,
            golden_rmsnorm(block_hidden, attention_norm_weight),
            1
        ) && pass;

        tensor_t q = accelerator.run_feature(
            kOpQProjection,
            normalized,
            nullptr,
            layer,
            block_begin
        );
        tensor_t k = accelerator.run_feature(
            kOpKProjection,
            normalized,
            nullptr,
            layer,
            block_begin
        );
        tensor_t v = accelerator.run_feature(
            kOpVProjection,
            normalized,
            nullptr,
            layer,
            block_begin
        );
        pass = compare_tensors(
            prefix + "_q_projection",
            q,
            golden_projection(shape, model, kOpQProjection, normalized, layer),
            1
        ) && pass;
        pass = compare_tensors(
            prefix + "_k_projection",
            k,
            golden_projection(shape, model, kOpKProjection, normalized, layer),
            1
        ) && pass;
        pass = compare_tensors(
            prefix + "_v_projection",
            v,
            golden_projection(shape, model, kOpVProjection, normalized, layer),
            1
        ) && pass;

        tensor_t attention(token_count, shape.hidden_size);
        tensor_t query_block0(token_count * group_size, head_dim);
        tensor_t query_block1(token_count * group_size, head_dim);
        for (unsigned int local_token = 0;
             local_token < token_count;
             local_token++) {
            const unsigned int position = block_begin + local_token;
            tensor_t q_row = tensor_rows(q, local_token, 1);
            tensor_t k_row = tensor_rows(k, local_token, 1);
            tensor_t v_row = tensor_rows(v, local_token, 1);
            profile_apply_rope(shape, q_row, k_row, position);
            tensor_t query0 = profile_build_query_group(shape, q_row, 0);
            tensor_t query1 = profile_build_query_group(shape, q_row, 1);
            tensor_t k_payload = profile_build_kv_payload(shape, k_row);
            tensor_t v_payload = profile_build_kv_payload(shape, v_row);

            tensor_store_rows(
                query_block0, local_token * group_size, query0
            );
            tensor_store_rows(
                query_block1, local_token * group_size, query1
            );

            for (unsigned int elem = 0; elem < head_dim; elem++) {
                cache_k0.at(position, elem) = k_payload.at(0, elem);
                cache_k1.at(position, elem) = k_payload.at(1, elem);
                cache_v0.at(position, elem) = v_payload.at(0, elem);
                cache_v1.at(position, elem) = v_payload.at(1, elem);
            }
        }

        // Test-fixture handoff: make the newly projected K/V rows visible to
        // the controller-resident block-attention path.  Production prefill
        // will replace this host migration with an on-device projection-to-
        // cache handoff; controller execution and timing measured below do
        // not include this migration.
        accelerator.initialize_kv_cache_heads(
            layer,
            tensor_prefix_rows(cache_k0, block_begin + token_count),
            tensor_prefix_rows(cache_k1, block_begin + token_count),
            tensor_prefix_rows(cache_v0, block_begin + token_count),
            tensor_prefix_rows(cache_v1, block_begin + token_count)
        );
        operator_result_t decoded = accelerator.run_prefill_attention_block(
            query_block0,
            query_block1,
            block_begin,
            token_count,
            layer
        );
        bool block_pass = check_prefill_block_attention_status(
            "PREFILL_BLOCK_ATTENTION_STATUS",
            shape,
            decoded.status,
            block_begin,
            token_count,
            decoded.controller_ms
        );

        for (unsigned int local_token = 0;
             local_token < token_count;
             local_token++) {
            const unsigned int position = block_begin + local_token;
            const tensor_t decoded0 = tensor_rows(
                decoded.port0, local_token * group_size, group_size
            );
            const tensor_t decoded1 = tensor_rows(
                decoded.port1, local_token * group_size, group_size
            );
            const tensor_t query_token0 = tensor_rows(
                query_block0,
                local_token * group_size,
                group_size
            );
            const tensor_t query_token1 = tensor_rows(
                query_block1,
                local_token * group_size,
                group_size
            );
            const tensor_t active_k0 =
                tensor_prefix_rows(cache_k0, position + 1);
            const tensor_t active_k1 =
                tensor_prefix_rows(cache_k1, position + 1);
            const tensor_t active_v0 =
                tensor_prefix_rows(cache_v0, position + 1);
            const tensor_t active_v1 =
                tensor_prefix_rows(cache_v1, position + 1);
            const tensor_t global0 = golden_flash_attention(
                query_token0, active_k0, active_v0
            );
            const tensor_t global1 = golden_flash_attention(
                query_token1, active_k1, active_v1
            );
            const tensor_t reference0 = golden_flash_attention_reference(
                query_token0, active_k0, active_v0
            );
            const tensor_t reference1 = golden_flash_attention_reference(
                query_token1, active_k1, active_v1
            );
            const tensor_t q2_14_0 = golden_tiled_flash_attention_q2_14(
                query_token0, active_k0, active_v0
            );
            const tensor_t q2_14_1 = golden_tiled_flash_attention_q2_14(
                query_token1, active_k1, active_v1
            );

            const std::string position_name =
                prefix + "_attention_pos" + std::to_string(position);
            const bool q2_14_hw0 = compare_tensors(
                position_name + "_port0_q2_14_hw",
                decoded0,
                q2_14_0,
                4
            );
            const bool reference_hw0 = compare_tensors(
                position_name + "_port0_reference",
                decoded0,
                reference0,
                5
            );
            const bool q2_14_reference0 = compare_tensors(
                position_name + "_port0_q2_14_reference",
                q2_14_0,
                reference0,
                5
            );
            const bool q2_14_hw1 = compare_tensors(
                position_name + "_port1_q2_14_hw",
                decoded1,
                q2_14_1,
                4
            );
            const bool reference_hw1 = compare_tensors(
                position_name + "_port1_reference",
                decoded1,
                reference1,
                5
            );
            const bool q2_14_reference1 = compare_tensors(
                position_name + "_port1_q2_14_reference",
                q2_14_1,
                reference1,
                5
            );
            // Retain the historical global fixed-point golden as a
            // non-gating diagnostic: it saturates the unnormalised long-
            // context PV result to Q8.8 and is not a valid accuracy oracle.
            compare_tensors(
                position_name + "_port0_legacy_global_diagnostic",
                decoded0,
                global0,
                4
            );
            compare_tensors(
                position_name + "_port1_legacy_global_diagnostic",
                decoded1,
                global1,
                4
            );
            block_pass =
                q2_14_hw0 && q2_14_hw1 &&
                reference_hw0 && reference_hw1 &&
                q2_14_reference0 && q2_14_reference1 && block_pass;
            profile_scatter_attention_group_row(
                attention,
                local_token,
                decoded0,
                0,
                group_size,
                head_dim
            );
            profile_scatter_attention_group_row(
                attention,
                local_token,
                decoded1,
                1,
                group_size,
                head_dim
            );
        }
        pass = block_pass && pass;
        attention_calls++;
        attention_mm_tasks += decoded.status.mm_tasks;
        attention_packets += decoded.status.completed_packets;

        tensor_t projected = accelerator.run_feature(
            kOpOProjection,
            attention,
            nullptr,
            layer,
            block_begin
        );
        pass = compare_tensors(
            prefix + "_o_projection",
            projected,
            golden_projection(shape, model, kOpOProjection, attention, layer),
            1
        ) && pass;
        tensor_t after_attention = accelerator.run_feature(
            kOpResidualAdd,
            residual,
            &projected,
            layer,
            block_begin
        );
        pass = compare_tensors(
            prefix + "_attention_residual",
            after_attention,
            golden_residual(residual, projected),
            0
        ) && pass;

        residual = after_attention;
        const tensor_t ffn_norm_weight = model.norm_row(layer, true);
        normalized = accelerator.run_feature(
            kOpRmsNorm,
            after_attention,
            &ffn_norm_weight,
            layer,
            block_begin
        );
        pass = compare_tensors(
            prefix + "_ffn_rmsnorm",
            normalized,
            golden_rmsnorm(after_attention, ffn_norm_weight),
            1
        ) && pass;
        tensor_t gate = accelerator.run_feature(
            kOpFfnGate,
            normalized,
            nullptr,
            layer,
            block_begin
        );
        tensor_t up = accelerator.run_feature(
            kOpFfnUp,
            normalized,
            nullptr,
            layer,
            block_begin
        );
        pass = compare_tensors(
            prefix + "_ffn_gate",
            gate,
            golden_projection(shape, model, kOpFfnGate, normalized, layer),
            1
        ) && pass;
        pass = compare_tensors(
            prefix + "_ffn_up",
            up,
            golden_projection(shape, model, kOpFfnUp, normalized, layer),
            1
        ) && pass;
        tensor_t activated = accelerator.run_feature(
            kOpSiluMul,
            gate,
            &up,
            layer,
            block_begin
        );
        pass = compare_tensors(
            prefix + "_ffn_silu_mul",
            activated,
            golden_silu_mul(gate, up),
            1
        ) && pass;
        tensor_t down = accelerator.run_feature(
            kOpFfnDown,
            activated,
            nullptr,
            layer,
            block_begin
        );
        pass = compare_tensors(
            prefix + "_ffn_down",
            down,
            golden_projection(shape, model, kOpFfnDown, activated, layer),
            4
        ) && pass;
        tensor_t block_output = accelerator.run_feature(
            kOpResidualAdd,
            residual,
            &down,
            layer,
            block_begin
        );
        pass = compare_tensors(
            prefix + "_ffn_residual",
            block_output,
            golden_residual(residual, down),
            0
        ) && pass;
        tensor_store_rows(
            final_hidden,
            block_begin - requested_prefill_start,
            block_output
        );

        std::cout
            << "PREFILL_BLOCK_PROFILE block=" << block_count_total
            << " position_begin=" << block_begin
            << " token_count=" << token_count
            << " hidden_checksum=0x" << std::hex
            << tensor_checksum(block_output)
            << std::dec
            << " " << (pass ? "PASS" : "FAIL")
            << "\n";
        block_count_total++;
    }

    std::cout
        << "PREFILL_PROFILE prefill_len=" << prefill_len
        << " prefill_start=" << requested_prefill_start
        << " processed_tokens=" << processed_tokens
        << " block_size=" << block_size
        << " blocks=" << block_count_total
        << " attention_calls=" << attention_calls
        << " attention_mm_tasks=" << attention_mm_tasks
        << " attention_packets=" << attention_packets
        << " final_hidden_checksum=0x" << std::hex
        << tensor_checksum(final_hidden)
        << std::dec
        << " seed=" << seed
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

bool run_mm_wave_profile(
    const model_shape_t& shape,
    accelerator_t& accelerator,
    const std::string& op_name,
    unsigned int layer,
    unsigned int first_wave,
    unsigned int wave_count,
    unsigned int k_limit,
    unsigned int debug_stage,
    unsigned int core_mask,
    bool zero_weight_stream,
    bool single_launch,
    unsigned int token_count,
    uint32_t seed
) {
    operator_kind_t op = kOpQProjection;
    if (!parse_mm_profile_op(op_name, op)) {
        throw std::runtime_error("unknown --op " + op_name);
    }
    const unsigned int input_dim = mm_profile_input_dim(shape, op);
    const unsigned int output_dim = mm_profile_output_dim(shape, op);
    const unsigned int total_waves =
        unsigned(ceildiv(output_dim, kMmOutputsPerWave));
    if (first_wave >= total_waves) {
        throw std::runtime_error("--wave is outside the selected operator");
    }
    if (k_limit != 0) {
        if (k_limit > input_dim) {
            throw std::runtime_error("--k-limit exceeds the selected operator input dimension");
        }
        if ((k_limit % kMmInputBlock) != 0) {
            throw std::runtime_error("--k-limit must be a multiple of 64 to match 8x64 weight tiles");
        }
    }
    const unsigned int effective_k = k_limit == 0 ? input_dim : k_limit;
    if (core_mask == 0 || core_mask > 3u) {
        throw std::runtime_error("--profile-core-mask must be 1, 2, or 3");
    }
    const unsigned int active_core_count =
        ((core_mask & 1u) ? 1u : 0u) +
        ((core_mask & 2u) ? 1u : 0u);
    if (token_count == 0 || token_count > kMaxTokensPerLaunch) {
        throw std::runtime_error("--profile-tokens must be in 1..8");
    }

    tensor_t input = make_random_tensor(
        token_count,
        input_dim,
        seed ^ (unsigned(op) * 0x10101u)
    );

    const unsigned int iterations =
        wave_count == 0 ? total_waves - first_wave : wave_count;
    const unsigned int bounded_iterations =
        first_wave + iterations > total_waves ?
        total_waves - first_wave :
        iterations;
    if (single_launch) {
        const decoded_status_t status = accelerator.run_mm_wave_profile(
            op,
            input,
            layer,
            first_wave,
            bounded_iterations,
            k_limit,
            debug_stage,
            core_mask,
            zero_weight_stream
        );
        const bool debug_short_circuit =
            debug_stage == 1u ||
            debug_stage == 2u;
        const bool pass =
            status.op == unsigned(op) &&
            status.code == 0 &&
            status.output_waves == total_waves &&
            status.mm_tasks == (
                debug_short_circuit ? 0u : active_core_count * bounded_iterations
            ) &&
            status.completed_packets ==
                (
                    debug_short_circuit ?
                    0u :
                    active_core_count * kMmPacketsPerBlock * bounded_iterations
                ) &&
            status.last_task;
        std::cout
            << "MM_WAVE_PROFILE_SINGLE op=" << mm_profile_op_name(op)
            << " layer=" << layer
            << " first_wave=" << first_wave
            << " wave_count=" << bounded_iterations
            << "/" << total_waves
            << " k_limit=" << k_limit
            << " effective_k=" << effective_k
            << " debug_stage=" << debug_stage
            << " core_mask=" << core_mask
            << " zero_weight_stream=" << (zero_weight_stream ? 1 : 0)
            << " token_count=" << token_count
            << " output_dim=" << output_dim
            << " status=" << status.code
            << " mm_tasks=" << status.mm_tasks
            << " packets=" << status.completed_packets
            << " " << (pass ? "PASS" : "FAIL")
            << "\n";
        return pass;
    }
    bool pass = true;
    for (unsigned int index = 0; index < iterations; index++) {
        const unsigned int wave = first_wave + index;
        if (wave >= total_waves) {
            break;
        }
        const decoded_status_t status = accelerator.run_mm_wave_profile(
            op,
            input,
            layer,
            wave,
            1,
            k_limit,
            debug_stage,
            core_mask,
            zero_weight_stream
        );
        const bool debug_short_circuit =
            debug_stage == 1u ||
            debug_stage == 2u;
        const bool wave_pass =
            status.op == unsigned(op) &&
            status.code == 0 &&
            status.output_waves == total_waves &&
            status.mm_tasks == (
                debug_short_circuit ? 0u : active_core_count
            ) &&
            status.completed_packets ==
                (
                    debug_short_circuit ?
                    0u :
                    active_core_count * kMmPacketsPerBlock
                ) &&
            status.last_task;
        std::cout
            << "MM_WAVE_PROFILE op=" << mm_profile_op_name(op)
            << " layer=" << layer
            << " wave=" << wave
            << "/" << total_waves
            << " k_limit=" << k_limit
            << " effective_k=" << effective_k
            << " debug_stage=" << debug_stage
            << " core_mask=" << core_mask
            << " zero_weight_stream=" << (zero_weight_stream ? 1 : 0)
            << " token_count=" << token_count
            << " output_dim=" << output_dim
            << " status=" << status.code
            << " mm_tasks=" << status.mm_tasks
            << " packets=" << status.completed_packets
            << " " << (wave_pass ? "PASS" : "FAIL")
            << "\n";
        pass = wave_pass && pass;
    }
    std::cout
        << "MM_WAVE_PROFILE_SUMMARY op=" << mm_profile_op_name(op)
        << " first_wave=" << first_wave
        << " requested_wave_count=" << wave_count
        << " k_limit=" << k_limit
        << " effective_k=" << effective_k
        << " debug_stage=" << debug_stage
        << " core_mask=" << core_mask
        << " zero_weight_stream=" << (zero_weight_stream ? 1 : 0)
        << " token_count=" << token_count
        << " total_waves=" << total_waves
        << " " << (pass ? "PASS" : "FAIL")
        << "\n";
    return pass;
}

class qwen_executor_t {
public:
    qwen_executor_t(
        const model_shape_t& shape,
        const model_data_t& model,
        accelerator_t& accelerator,
        unsigned int layer_count,
        bool hardware_softmax,
        bool resident_layer,
        bool coarse_tasks
    )
        : shape_(shape),
          model_(model),
          accelerator_(accelerator),
          layer_count_(
              layer_count == 0 ?
              shape.num_layers :
              layer_count
          ),
          resident_layer_(resident_layer),
          coarse_tasks_(coarse_tasks) {
        if (layer_count_ > shape.num_layers) {
            throw std::runtime_error("--layers exceeds the selected profile");
        }
        std::cout
            << "using controller-side decode attention "
            << "(HBM KV cache + online softmax)\n";
        if (hardware_softmax) {
            std::cout
                << "--hardware-softmax is now implied by the decode path\n";
        }
        if (resident_layer_) {
            std::cout
                << "using one-launch resident decoder-layer scheduling "
                << "(on-chip intermediates + controller KV/softmax)\n";
        }
        if (coarse_tasks_) {
            std::cout
                << "using host-composed attention/FFN tasks "
                << "(HBM-resident boundary, no intermediate host copy)\n";
        }
        if (resident_layer_ && coarse_tasks_) {
            throw std::runtime_error(
                "--resident-layer and --coarse-tasks are mutually exclusive"
            );
        }
    }

    tensor_t run_token(unsigned int token_id, unsigned int position) {
        if (position >= shape_.max_seq_len) {
            throw std::runtime_error("sequence exceeds the selected profile");
        }

        tensor_t hidden = model_.embedding(token_id);
        if (coarse_tasks_) {
            composed_layer_result_t result =
                accelerator_.run_composed_decoder_stack(
                    hidden,
                    0,
                    layer_count_,
                    position,
                    true
                );
            hidden = result.output;
            std::cout
                << "COARSE_TASK_STACK layers=" << result.layer_count
                << " tasks=" << result.task_count
                << " position=" << position
                << " event_timing_domain=" << event_timing_domain()
                << " attention_controller_ms="
                << result.attention_controller_ms
                << " ffn_controller_ms="
                << result.ffn_controller_ms
                << " final_norm_controller_ms="
                << result.final_norm_controller_ms
                << " kernel_active_ms="
                << result.kernel_active_ms
                << " input_migration_ms="
                << result.input_migration_ms
                << " auxiliary_migration_ms="
                << result.auxiliary_migration_ms
                << " status_kernel_ms="
                << result.status_kernel_ms
                << " status_migration_ms="
                << result.status_migration_ms
                << " output_migration_ms="
                << result.output_migration_ms
                << " profiled_sequence_ms="
                << result.profiled_sequence_ms
                << " host_elapsed_ms="
                << result.host_elapsed_ms
                << " intermediate_host_copy=0\n";
        }
        for (unsigned int layer = 0; layer < layer_count_; layer++) {
            if (coarse_tasks_) {
                break;
            }
            if (resident_layer_) {
                hidden = accelerator_.run_decoder_layer(
                    hidden,
                    model_.norm_row(layer, false),
                    model_.norm_row(layer, true),
                    layer,
                    position
                );
                continue;
            }
            tensor_t residual = hidden;
            tensor_t norm_weight = model_.norm_row(layer, false);
            tensor_t normalized = accelerator_.run_feature(
                kOpRmsNorm,
                hidden,
                &norm_weight,
                layer,
                position
            );
            tensor_t q = accelerator_.run_feature(
                kOpQProjection,
                normalized,
                nullptr,
                layer,
                position
            );
            tensor_t k = accelerator_.run_feature(
                kOpKProjection,
                normalized,
                nullptr,
                layer,
                position
            );
            tensor_t v = accelerator_.run_feature(
                kOpVProjection,
                normalized,
                nullptr,
                layer,
                position
            );

            apply_rope(q, k, position);
            tensor_t attention = run_decode_attention(
                layer,
                position,
                q,
                k,
                v
            );
            tensor_t projected = accelerator_.run_feature(
                kOpOProjection,
                attention,
                nullptr,
                layer,
                position
            );
            hidden = accelerator_.run_feature(
                kOpResidualAdd,
                residual,
                &projected,
                layer,
                position
            );

            residual = hidden;
            norm_weight = model_.norm_row(layer, true);
            normalized = accelerator_.run_feature(
                kOpRmsNorm,
                hidden,
                &norm_weight,
                layer,
                position
            );
            tensor_t gate = accelerator_.run_feature(
                kOpFfnGate,
                normalized,
                nullptr,
                layer,
                position
            );
            tensor_t up = accelerator_.run_feature(
                kOpFfnUp,
                normalized,
                nullptr,
                layer,
                position
            );
            tensor_t activated = accelerator_.run_feature(
                kOpSiluMul,
                gate,
                &up,
                layer,
                position
            );
            tensor_t down = accelerator_.run_feature(
                kOpFfnDown,
                activated,
                nullptr,
                layer,
                position
            );
            hidden = accelerator_.run_feature(
                kOpResidualAdd,
                residual,
                &down,
                layer,
                position
            );
        }

        if (coarse_tasks_) {
            return hidden;
        }
        tensor_t final_norm = model_.final_norm_row();
        return accelerator_.run_feature(
            kOpRmsNorm,
            hidden,
            &final_norm,
            0,
            position
        );
    }

private:
    void apply_rope(
        tensor_t& q,
        tensor_t& k,
        unsigned int position
    ) const {
        const unsigned int head_dim = shape_.head_dim();
        const unsigned int half = head_dim / 2;
        for (unsigned int head = 0; head < shape_.num_heads; head++) {
            const unsigned int base = head * head_dim;
            for (unsigned int i = 0; i < half; i++) {
                const double exponent = double(2 * i) / head_dim;
                const double inverse_frequency =
                    1.0 / std::pow(kRopeTheta, exponent);
                const double angle = position * inverse_frequency;
                const double cosine = std::cos(angle);
                const double sine = std::sin(angle);
                const double x0 = dequantize_fix16(q.at(0, base + i));
                const double x1 =
                    dequantize_fix16(q.at(0, base + half + i));
                q.at(0, base + i) =
                    quantize_fix16(x0 * cosine - x1 * sine);
                q.at(0, base + half + i) =
                    quantize_fix16(x1 * cosine + x0 * sine);
            }
        }
        for (unsigned int head = 0; head < shape_.num_kv_heads; head++) {
            const unsigned int base = head * head_dim;
            for (unsigned int i = 0; i < half; i++) {
                const double exponent = double(2 * i) / head_dim;
                const double inverse_frequency =
                    1.0 / std::pow(kRopeTheta, exponent);
                const double angle = position * inverse_frequency;
                const double cosine = std::cos(angle);
                const double sine = std::sin(angle);
                const double x0 = dequantize_fix16(k.at(0, base + i));
                const double x1 =
                    dequantize_fix16(k.at(0, base + half + i));
                k.at(0, base + i) =
                    quantize_fix16(x0 * cosine - x1 * sine);
                k.at(0, base + half + i) =
                    quantize_fix16(x1 * cosine + x0 * sine);
            }
        }
    }

    tensor_t build_query_group(
        const tensor_t& q,
        unsigned int kv_head
    ) const {
        tensor_t group(
            shape_.gqa_group_size(),
            shape_.head_dim()
        );
        for (unsigned int row = 0;
             row < shape_.gqa_group_size();
             row++) {
            const unsigned int query_head =
                kv_head * shape_.gqa_group_size() + row;
            const unsigned int base = query_head * shape_.head_dim();
            for (unsigned int elem = 0;
                 elem < shape_.head_dim();
                 elem++) {
                group.at(row, elem) = q.at(0, base + elem);
            }
        }
        return group;
    }

    tensor_t build_kv_payload(const tensor_t& kv) const {
        if (kv.rows != 1 || kv.cols != shape_.kv_channels()) {
            throw std::runtime_error("KV payload source shape mismatch");
        }
        tensor_t payload(shape_.num_kv_heads, shape_.head_dim());
        for (unsigned int head = 0;
             head < shape_.num_kv_heads;
             head++) {
            const unsigned int source_base = head * shape_.head_dim();
            for (unsigned int elem = 0;
                 elem < shape_.head_dim();
                 elem++) {
                payload.at(head, elem) = kv.at(0, source_base + elem);
            }
        }
        return payload;
    }

    static void scatter_attention_group(
        tensor_t& attention,
        const tensor_t& group,
        unsigned int kv_head,
        unsigned int group_size,
        unsigned int head_dim
    ) {
        if (group.rows != group_size || group.cols != head_dim) {
            throw std::runtime_error("decode attention result shape mismatch");
        }
        for (unsigned int row = 0; row < group_size; row++) {
            const unsigned int query_head = kv_head * group_size + row;
            const unsigned int destination_base = query_head * head_dim;
            for (unsigned int elem = 0; elem < head_dim; elem++) {
                attention.at(0, destination_base + elem) =
                    group.at(row, elem);
            }
        }
    }

    tensor_t run_decode_attention(
        unsigned int layer,
        unsigned int position,
        const tensor_t& q,
        const tensor_t& k,
        const tensor_t& v
    ) const {
        if (
            q.rows != 1 ||
            q.cols != shape_.hidden_size ||
            k.rows != 1 ||
            k.cols != shape_.kv_channels() ||
            v.rows != 1 ||
            v.cols != shape_.kv_channels()
        ) {
            throw std::runtime_error("decode attention input shape mismatch");
        }

        const unsigned int group_size = shape_.gqa_group_size();
        const unsigned int head_dim = shape_.head_dim();
        tensor_t query0 = build_query_group(q, 0);
        tensor_t query1 = build_query_group(q, 1);
        tensor_t k_payload = build_kv_payload(k);
        tensor_t v_payload = build_kv_payload(v);

        operator_result_t decoded = accelerator_.run_attention(
            kOpDecodeSmoke,
            query0,
            query1,
            &k_payload,
            &v_payload,
            position,
            1,
            layer
        );

        tensor_t attention(1, shape_.hidden_size);
        scatter_attention_group(
            attention,
            decoded.port0,
            0,
            group_size,
            head_dim
        );
        scatter_attention_group(
            attention,
            decoded.port1,
            1,
            group_size,
            head_dim
        );
        return attention;
    }

    const model_shape_t& shape_;
    const model_data_t& model_;
    accelerator_t& accelerator_;
    unsigned int layer_count_;
    bool resident_layer_;
    bool coarse_tasks_;
};

bool get_option(
    int argc,
    const char* argv[],
    const std::string& option,
    std::string& value
) {
    for (int i = 1; i + 1 < argc; i++) {
        if (option == argv[i]) {
            value = argv[i + 1];
            return true;
        }
    }
    return false;
}

bool has_flag(
    int argc,
    const char* argv[],
    const std::string& flag
) {
    for (int i = 1; i < argc; i++) {
        if (flag == argv[i]) {
            return true;
        }
    }
    return false;
}

std::vector<unsigned int> parse_tokens(const std::string& text) {
    std::vector<unsigned int> tokens;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            throw std::runtime_error("empty token in --tokens");
        }
        tokens.push_back(unsigned(std::stoul(item)));
    }
    if (tokens.empty()) {
        throw std::runtime_error("--tokens must not be empty");
    }
    return tokens;
}

bool parse_mm_profile_op(
    const std::string& name,
    operator_kind_t& op
) {
    if (name == "q" || name == "q-proj" || name == "qproj") {
        op = kOpQProjection;
    } else if (name == "k" || name == "k-proj" || name == "kproj") {
        op = kOpKProjection;
    } else if (name == "v" || name == "v-proj" || name == "vproj") {
        op = kOpVProjection;
    } else if (name == "o" || name == "o-proj" || name == "oproj") {
        op = kOpOProjection;
    } else if (name == "ffn-gate" || name == "gate") {
        op = kOpFfnGate;
    } else if (name == "ffn-up" || name == "up") {
        op = kOpFfnUp;
    } else if (name == "ffn-down" || name == "down") {
        op = kOpFfnDown;
    } else {
        return false;
    }
    return true;
}

std::string mm_profile_op_name(operator_kind_t op) {
    switch (op) {
    case kOpQProjection: return "q";
    case kOpKProjection: return "k";
    case kOpVProjection: return "v";
    case kOpOProjection: return "o";
    case kOpFfnGate: return "ffn-gate";
    case kOpFfnUp: return "ffn-up";
    case kOpFfnDown: return "ffn-down";
    default: return "unknown";
    }
}

unsigned int mm_profile_input_dim(
    const model_shape_t& shape,
    operator_kind_t op
) {
    switch (op) {
    case kOpFfnDown:
        return shape.intermediate_size;
    case kOpQProjection:
    case kOpKProjection:
    case kOpVProjection:
    case kOpOProjection:
    case kOpFfnGate:
    case kOpFfnUp:
        return shape.hidden_size;
    default:
        throw std::runtime_error("operator is not an MM profile operator");
    }
}

unsigned int mm_profile_output_dim(
    const model_shape_t& shape,
    operator_kind_t op
) {
    switch (op) {
    case kOpKProjection:
    case kOpVProjection:
        return shape.kv_channels();
    case kOpFfnGate:
    case kOpFfnUp:
        return shape.intermediate_size;
    case kOpQProjection:
    case kOpOProjection:
    case kOpFfnDown:
        return shape.hidden_size;
    default:
        throw std::runtime_error("operator is not an MM profile operator");
    }
}

command_line_t parse_command_line(int argc, const char* argv[]) {
    command_line_t command;
    std::string value;
    get_option(argc, argv, "--mode", command.mode);
    get_option(argc, argv, "--profile", command.profile);
    get_option(argc, argv, "--data", command.data_dir);
    get_option(argc, argv, "--xclbin", command.xclbin);
    get_option(argc, argv, "--op", command.profile_op);
    if (get_option(argc, argv, "--tokens", value)) {
        command.tokens = parse_tokens(value);
    }
    if (get_option(argc, argv, "--layers", value)) {
        command.layer_count = unsigned(std::stoul(value));
    }
    if (get_option(argc, argv, "--max-new-tokens", value)) {
        command.max_new_tokens = unsigned(std::stoul(value));
    }
    if (get_option(argc, argv, "--wave", value)) {
        command.profile_wave = unsigned(std::stoul(value));
    }
    if (get_option(argc, argv, "--wave-count", value)) {
        command.profile_wave_count = unsigned(std::stoul(value));
    }
    if (get_option(argc, argv, "--k-limit", value)) {
        command.profile_k_limit = unsigned(std::stoul(value));
    }
    if (get_option(argc, argv, "--profile-debug-stage", value)) {
        command.profile_debug_stage = unsigned(std::stoul(value));
    }
    if (get_option(argc, argv, "--profile-core-mask", value)) {
        command.profile_core_mask = unsigned(std::stoul(value));
    }
    if (get_option(argc, argv, "--profile-tokens", value)) {
        command.profile_token_count = unsigned(std::stoul(value));
    }
    if (get_option(argc, argv, "--position", value)) {
        command.attention_position = unsigned(std::stoul(value));
    }
    if (get_option(argc, argv, "--prefill-len", value)) {
        command.attention_prefill_len = unsigned(std::stoul(value));
    }
    if (get_option(argc, argv, "--prefill-start", value)) {
        command.attention_prefill_start = unsigned(std::stoul(value));
    }
    get_option(argc, argv, "--phase", command.attention_phase);
    command.profile_zero_weight_stream = has_flag(
        argc,
        argv,
        "--profile-zero-weight-stream"
    );
    command.profile_single_launch = has_flag(
        argc,
        argv,
        "--profile-single-launch"
    );
    if (get_option(argc, argv, "--seed", value)) {
        command.random_seed = uint32_t(std::stoul(value));
    }
    command.zero_model = has_flag(argc, argv, "--zero-model");
    command.random_model = has_flag(argc, argv, "--random-model");
    command.verbose_ops = has_flag(argc, argv, "--verbose-ops");
    command.hardware_softmax = has_flag(
        argc,
        argv,
        "--hardware-softmax"
    );
    command.resident_layer = has_flag(
        argc,
        argv,
        "--resident-layer"
    );
    command.coarse_tasks = has_flag(argc, argv, "--coarse-tasks");
    command.tie_embeddings = has_flag(
        argc,
        argv,
        "--tie-embeddings"
    );
    command.skip_weight_preload = has_flag(
        argc,
        argv,
        "--skip-weight-preload"
    );
    command.load_only = has_flag(argc, argv, "--load-only");
    return command;
}

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "  --mode plan|inspect|run|generate|verify-random|verify-decode-smoke|verify-resident-layer|verify-composed-layer|verify-composed-stack|verify-nop|verify-nop-ctrl-only|verify-nop-ctrl-enqueue-only|profile-mm-wave|profile-attention|profile-attention-pd|profile-attention-block|profile-attention-sublayer|profile-ffn-sublayer|profile-prefill-block|profile-prefill-vector|diagnose-prefill-softmax\n"
        << "  --prefill-start N   First P-stage position (resume support)\n"
        << "  --profile small|medium|qwen-layer|qwen-layer-long|qwen2.5-3b\n"
        << "  --xclbin <file>          Required for run/generate\n"
        << "  --data <dir>             Packed Fix16 model directory\n"
        << "  --tokens <id,id,...>     Prompt token IDs\n"
        << "  --layers <n>             Limit executed decoder layers\n"
        << "  --max-new-tokens <n>     Greedy tokens for generate\n"
        << "  --op <name>              profile-mm-wave op: q,k,v,o,ffn-gate,ffn-up,ffn-down\n"
        << "  --wave <n>               First output wave for profile-mm-wave\n"
        << "  --wave-count <n>         Number of waves; 0 means all remaining waves one by one\n"
        << "  --k-limit <n>            Optional profile-mm-wave K dimension limit; 0 means full K, must be 64-aligned\n"
        << "  --profile-debug-stage <n>\n"
        << "                           Debug profile-mm-wave: 0=normal, 1=stop before load, 2=stop after load/clear\n"
        << "  --profile-core-mask <n>  Debug profile-mm-wave core mask: 1=core0, 2=core1, 3=both\n"
        << "  --profile-tokens <n>     Active MM rows for profile-mm-wave; 1..8\n"
        << "  --profile-single-launch  Run requested profile-mm-wave waves in one controller launch\n"
        << "  --profile-zero-weight-stream\n"
        << "                           Debug profile-mm-wave by generating zero weight packets on chip instead of reading HBM weights\n"
        << "  --position <n>           Target decode position for attention profile modes\n"
        << "  --prefill-len <n>        P/D attention profile prefill length; default 64 or max_seq-1\n"
        << "  --phase <p|d|pd>         P/D attention profile phase selection\n"
        << "  --seed <n>               Random model/verification seed\n"
        << "  --zero-model             Deterministic plumbing model\n"
        << "  --random-model           Deterministic non-zero in-memory model\n"
        << "  --tie-embeddings         Reuse embedding as the LM head\n"
        << "  --hardware-softmax       Deprecated; decode uses controller online softmax\n"
        << "  --resident-layer         Execute each decoder layer in one controller launch\n"
        << "  --coarse-tasks           Compose controller-resident attention and FFN tasks through HBM without an intermediate host copy\n"
        << "  --skip-weight-preload    Program xclbin without migrating weights\n"
        << "  --load-only              Stop after xclbin/context/buffer setup\n"
        << "  --verbose-ops            Print every controller launch\n";
}

void print_profile(const model_shape_t& shape) {
    const std::size_t shard_bytes =
        shape.weight_shard_words() * sizeof(word512_t);
    std::cout
        << "profile=" << shape.name << "\n"
        << "  vocab=" << shape.vocab_size
        << " hidden=" << shape.hidden_size
        << " intermediate=" << shape.intermediate_size
        << " layers=" << shape.num_layers << "\n"
        << "  heads=" << shape.num_heads
        << " kv_heads=" << shape.num_kv_heads
        << " head_dim=" << shape.head_dim()
        << " max_seq=" << shape.max_seq_len << "\n"
        << "  feature_stride_words=" << shape.feature_words_per_token()
        << " data_port_words=" << shape.data_port_words() << "\n"
        << "  weight_shard_words=" << shape.weight_shard_words()
        << " weight_shard_bytes=" << shard_bytes << "\n"
        << "  embedding_bytes="
        << shape.embedding_values() * sizeof(int16_t)
        << " norm_bytes="
        << shape.norm_values() * sizeof(int16_t)
        << "\n";
    if (shard_bytes > kHbmPseudoChannelBytes) {
        std::cout
            << "  BLOCKED: each shard exceeds one 256 MiB HBM "
            << "pseudo-channel in the current link map\n";
    }
}

void print_execution_plan(const model_shape_t& shape) {
    print_profile(shape);
    std::cout
        << "decoder token plan:\n"
        << "  embedding(host)\n"
        << "  repeat layer: RMS -> Q/K/V -> RoPE(host)\n"
        << "  attention: DECODE_SMOKE(controller writes HBM KV,"
        << " online-softmax + PV) -> O\n"
        << "  residual -> RMS -> Gate/Up -> SiLU-Mul -> Down -> residual\n"
        << "  final RMS -> dedicated or explicitly tied LM head(host)\n"
        << "hardware contract:\n"
        << "  one controller invocation launches cc8_ctrl + cc8_cu0"
        << " + cc8_cu1 + cc8_status\n"
        << "  feature tensors use fixed max-dimension token stride\n"
        << "  attention rows and K/V panels use tight 512-bit row packing\n"
        << "  the selected --profile must match the xclbin build profile\n";
}

void inspect_files(
    const model_shape_t& shape,
    const std::string& data_dir
) {
    print_profile(shape);
    const std::string embedding =
        data_dir + "/token_embedding.fix16.bin";
    const std::string norms =
        data_dir + "/norm_weights.fix16.bin";
    auto print_one = [](const std::string& path, std::size_t expected) {
        const bool exists = file_exists(path);
        const std::size_t actual = exists ? file_size(path) : 0;
        std::cout
            << (exists && actual == expected ? "OK      " : "MISSING ")
            << path
            << " expected="
            << expected
            << " actual="
            << actual
            << "\n";
    };
    print_one(
        embedding,
        shape.embedding_values() * sizeof(int16_t)
    );
    print_one(
        norms,
        shape.norm_values() * sizeof(int16_t)
    );
    for (unsigned int shard = 0; shard < kWeightShardCount; shard++) {
        print_one(
            shard_path(data_dir, shard),
            shape.weight_shard_words() * sizeof(word512_t)
        );
    }
    const std::string lm_head = data_dir + "/lm_head.fix16.bin";
    if (file_exists(lm_head)) {
        print_one(
            lm_head,
            shape.embedding_values() * sizeof(int16_t)
        );
    } else {
        std::cout
            << "OPTIONAL "
            << lm_head
            << " (required for generate unless --tie-embeddings)\n";
    }
}

uint64_t tensor_checksum(const tensor_t& tensor) {
    uint64_t checksum = 1469598103934665603ull;
    for (int16_t value : tensor.values) {
        checksum ^= uint16_t(value);
        checksum *= 1099511628211ull;
    }
    return checksum;
}

}  // namespace

int main(int argc, const char* argv[]) {
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    try {
        if (has_flag(argc, argv, "--help")) {
            print_usage(argv[0]);
            return 0;
        }

        const command_line_t command = parse_command_line(argc, argv);
        model_shape_t shape;
        if (!parse_profile(command.profile, shape)) {
            throw std::runtime_error("unknown --profile " + command.profile);
        }
        shape.validate();

        if (command.mode == "plan") {
            print_execution_plan(shape);
            return 0;
        }
        if (command.mode == "inspect") {
            inspect_files(shape, command.data_dir);
            return 0;
        }
        if (
            command.mode != "run" &&
            command.mode != "generate" &&
            command.mode != "verify-random" &&
            command.mode != "verify-decode-smoke" &&
            command.mode != "verify-resident-layer" &&
            command.mode != "verify-composed-layer" &&
            command.mode != "verify-composed-stack" &&
            command.mode != "verify-nop" &&
            command.mode != "verify-nop-ctrl-only" &&
            command.mode != "verify-nop-ctrl-enqueue-only" &&
            command.mode != "profile-mm-wave" &&
            command.mode != "profile-attention" &&
            command.mode != "profile-attention-pd" &&
            command.mode != "profile-attention-block" &&
            command.mode != "profile-attention-sublayer" &&
            command.mode != "profile-ffn-sublayer" &&
            command.mode != "profile-prefill-block" &&
            command.mode != "profile-prefill-vector" &&
            command.mode != "diagnose-prefill-softmax"
        ) {
            throw std::runtime_error("unknown --mode " + command.mode);
        }
        if (
            command.mode != "diagnose-prefill-softmax" &&
            command.xclbin.empty()
        ) {
            throw std::runtime_error("--xclbin is required for run/generate");
        }
        if (command.profile != kDeviceProfile) {
            throw std::runtime_error(
                "host/xclbin profile contract requires --profile " +
                std::string(kDeviceProfile) +
                "; requested " + command.profile
            );
        }
        if (
            command.tokens.size() + command.max_new_tokens >
            shape.max_seq_len
        ) {
            throw std::runtime_error("prompt and generation exceed max_seq_len");
        }
        for (unsigned int token : command.tokens) {
            if (token >= shape.vocab_size) {
                throw std::runtime_error("prompt token exceeds profile vocabulary");
            }
        }
        if (
            shape.weight_shard_words() * sizeof(word512_t) >
            kHbmPseudoChannelBytes
        ) {
            throw std::runtime_error(
                "selected profile exceeds the current one-bank-per-shard "
                "HBM link map"
            );
        }

        const bool thin_smoke_buffers =
            (command.mode == "verify-decode-smoke" ||
             command.mode == "profile-attention" ||
             command.mode == "profile-attention-pd" ||
             command.mode == "profile-prefill-vector" ||
             command.mode == "verify-nop" ||
             command.mode == "verify-nop-ctrl-only" ||
             command.mode == "verify-nop-ctrl-enqueue-only") &&
            command.skip_weight_preload;
        if (thin_smoke_buffers) {
            std::cout
                << "[qwen-host] using one-word placeholder weight buffers "
                << "for diagnostic smoke\n";
        }

        model_data_t model(shape, thin_smoke_buffers);
        if (command.zero_model && command.random_model) {
            throw std::runtime_error(
                "--zero-model and --random-model are mutually exclusive"
            );
        }
        if (
            !command.zero_model &&
            (
                command.mode == "verify-random" ||
                command.mode == "verify-decode-smoke" ||
                command.mode == "verify-resident-layer" ||
                command.mode == "verify-composed-layer" ||
                command.mode == "verify-composed-stack" ||
                command.mode == "verify-nop" ||
                command.mode == "verify-nop-ctrl-only" ||
                command.mode == "verify-nop-ctrl-enqueue-only" ||
                command.mode == "profile-mm-wave" ||
                command.mode == "profile-attention" ||
                command.mode == "profile-attention-pd" ||
                command.mode == "profile-attention-block" ||
                command.mode == "profile-attention-sublayer" ||
                command.mode == "profile-ffn-sublayer" ||
                command.mode == "profile-prefill-block" ||
                command.mode == "profile-prefill-vector" ||
                command.mode == "diagnose-prefill-softmax" ||
                command.random_model
            )
        ) {
            model.initialize_random_model(command.random_seed);
        } else {
            model.load(
                command.data_dir,
                command.zero_model,
                command.tie_embeddings,
                command.mode == "generate"
            );
        }
        if (command.mode == "diagnose-prefill-softmax") {
            return run_prefill_softmax_diagnostic(
                shape,
                model,
                command.attention_prefill_len,
                command.attention_prefill_start,
                command.random_seed
            ) ? 0 : 1;
        }
        accelerator_t accelerator(
            shape,
            model,
            command.xclbin,
            command.verbose_ops,
            command.skip_weight_preload
        );
        if (command.load_only) {
            std::cout << "QWEN_8X64_LOAD_ONLY PASS\n";
            return 0;
        }
        if (command.mode == "verify-random") {
            return run_random_verification(
                shape,
                model,
                accelerator,
                command.random_seed
            ) ? 0 : 1;
        }
        if (command.mode == "verify-decode-smoke") {
            return run_decode_smoke_verification(
                shape,
                accelerator,
                command.random_seed
            ) ? 0 : 1;
        }
        if (command.mode == "verify-resident-layer") {
            return run_resident_layer_verification(
                shape,
                model,
                accelerator,
                command.random_seed,
                command.attention_position
            ) ? 0 : 1;
        }
        if (command.mode == "verify-composed-layer") {
            return run_composed_layer_verification(
                shape,
                model,
                accelerator,
                command.random_seed,
                command.attention_position
            ) ? 0 : 1;
        }
        if (command.mode == "verify-composed-stack") {
            return run_composed_stack_verification(
                shape,
                model,
                accelerator,
                command.random_seed,
                command.attention_position
            ) ? 0 : 1;
        }
        if (command.mode == "verify-nop") {
            return run_nop_verification(accelerator) ? 0 : 1;
        }
        if (command.mode == "verify-nop-ctrl-only") {
            return run_nop_controller_only_verification(accelerator) ? 0 : 1;
        }
        if (command.mode == "verify-nop-ctrl-enqueue-only") {
            return run_nop_controller_enqueue_only_verification(accelerator) ?
                0 :
                1;
        }
        if (command.mode == "profile-mm-wave") {
            return run_mm_wave_profile(
                shape,
                accelerator,
                command.profile_op,
                0,
                command.profile_wave,
                command.profile_wave_count,
                command.profile_k_limit,
                command.profile_debug_stage,
                command.profile_core_mask,
                command.profile_zero_weight_stream,
                command.profile_single_launch,
                command.profile_token_count,
                command.random_seed
            ) ? 0 : 1;
        }
        if (command.mode == "profile-attention") {
            return run_attention_profile(
                shape,
                accelerator,
                command.attention_position,
                command.random_seed
            ) ? 0 : 1;
        }
        if (command.mode == "profile-attention-pd") {
            return run_attention_pd_profile(
                shape,
                accelerator,
                command.attention_prefill_len,
                command.attention_prefill_start,
                command.attention_phase,
                command.random_seed
            ) ? 0 : 1;
        }
        if (command.mode == "profile-attention-block") {
            if (command.skip_weight_preload) {
                throw std::runtime_error(
                    "profile-attention-block needs projection weights; "
                    "do not use --skip-weight-preload"
                );
            }
            return run_attention_block_profile(
                shape,
                model,
                accelerator,
                command.attention_position,
                command.random_seed
            ) ? 0 : 1;
        }
        if (command.mode == "profile-attention-sublayer") {
            if (command.skip_weight_preload) {
                throw std::runtime_error(
                    "profile-attention-sublayer needs projection weights; "
                    "do not use --skip-weight-preload"
                );
            }
            return run_attention_sublayer_profile(
                shape,
                model,
                accelerator,
                command.attention_position,
                command.random_seed
            ) ? 0 : 1;
        }
        if (command.mode == "profile-ffn-sublayer") {
            if (command.skip_weight_preload) {
                throw std::runtime_error(
                    "profile-ffn-sublayer needs projection weights; "
                    "do not use --skip-weight-preload"
                );
            }
            return run_ffn_sublayer_profile(
                shape,
                model,
                accelerator,
                command.attention_position,
                command.random_seed
            ) ? 0 : 1;
        }
        if (command.mode == "profile-prefill-block") {
            if (command.skip_weight_preload) {
                throw std::runtime_error(
                    "profile-prefill-block needs projection weights; "
                    "do not use --skip-weight-preload"
                );
            }
            return run_prefill_block_profile(
                shape,
                model,
                accelerator,
                command.attention_prefill_len,
                command.attention_prefill_start,
                command.random_seed
            ) ? 0 : 1;
        }
        if (command.mode == "profile-prefill-vector") {
            return run_prefill_vector_profile(
                shape,
                model,
                accelerator,
                command.attention_prefill_len,
                command.random_seed
            ) ? 0 : 1;
        }
        qwen_executor_t executor(
            shape,
            model,
            accelerator,
            command.layer_count,
            command.hardware_softmax,
            command.resident_layer,
            command.coarse_tasks
        );

        const auto begin = std::chrono::steady_clock::now();
        tensor_t hidden;
        std::vector<unsigned int> sequence = command.tokens;
        for (unsigned int position = 0;
             position < command.tokens.size();
             position++) {
            hidden = executor.run_token(sequence[position], position);
            std::cout
                << "position=" << position
                << " token=" << sequence[position]
                << " hidden_checksum=0x"
                << std::hex
                << tensor_checksum(hidden)
                << std::dec
                << "\n";
        }

        if (command.mode == "generate") {
            for (unsigned int step = 0;
                 step < command.max_new_tokens;
                 step++) {
                const unsigned int next = model.lm_head_argmax(hidden);
                sequence.push_back(next);
                const unsigned int position =
                    unsigned(sequence.size() - 1);
                std::cout
                    << "generated[" << step << "]=" << next << "\n";
                hidden = executor.run_token(next, position);
            }
        }

        const auto end = std::chrono::steady_clock::now();
        std::cout
            << "QWEN_8X64_HOST PASS elapsed_seconds="
            << std::chrono::duration<double>(end - begin).count()
            << " sequence=";
        for (std::size_t i = 0; i < sequence.size(); i++) {
            if (i != 0) {
                std::cout << ",";
            }
            std::cout << sequence[i];
        }
        std::cout << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "QWEN_8X64_HOST ERROR: " << error.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
}
