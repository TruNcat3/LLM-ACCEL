#include "xcl2.hpp"

#include <ap_fixed.h>
#include <ap_int.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kWeightShardCount = 16;
constexpr unsigned int kValuesPerWord = 32;
constexpr unsigned int kTokensPerPort = 4;
constexpr unsigned int kMaxTokensPerLaunch = 8;
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
    kOpSoftmax = 13
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

    std::size_t kv_cache_values() const {
        return
            std::size_t(num_layers) *
            max_seq_len *
            kv_channels();
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

struct command_line_t {
    std::string mode = "plan";
    std::string profile = "small";
    std::string data_dir = "data";
    std::string xclbin;
    std::vector<unsigned int> tokens{0};
    unsigned int layer_count = 0;
    unsigned int max_new_tokens = 0;
    uint32_t random_seed = 20260701u;
    bool zero_model = false;
    bool verbose_ops = false;
    bool hardware_softmax = false;
    bool tie_embeddings = false;
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
    const bool negative = clamped < golden_fm_t(0);
    golden_fm_t magnitude =
        negative ? golden_fm_t(-clamped) : clamped;
    golden_fm_t result =
        golden_fm_t(1) +
        magnitude * golden_fm_t(0.03125);
    for (unsigned int i = 0; i < 5; i++) {
        result *= result;
    }
    return negative ? golden_recip(result) : result;
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

int16_t saturating_add(int32_t lhs, int32_t rhs) {
    const int32_t sum = lhs + rhs;
    if (sum > std::numeric_limits<int16_t>::max()) {
        return std::numeric_limits<int16_t>::max();
    }
    if (sum < std::numeric_limits<int16_t>::min()) {
        return std::numeric_limits<int16_t>::min();
    }
    return int16_t(sum);
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
    explicit model_data_t(const model_shape_t& shape)
        : shape_(shape),
          embedding_(shape.embedding_values()),
          norm_weights_(shape.norm_values()) {
        for (auto& shard : weight_shards_) {
            shard.resize(shape.weight_shard_words());
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
        bool verbose_ops
    )
        : shape_(shape),
          model_(model),
          verbose_ops_(verbose_ops),
          data_words_{
              aligned_word_vector(shape.data_port_words()),
              aligned_word_vector(shape.data_port_words()),
              aligned_word_vector(shape.data_port_words()),
              aligned_word_vector(shape.data_port_words()),
              aligned_word_vector(shape.data_port_words()),
              aligned_word_vector(shape.data_port_words())
          },
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
        migrate_weights();
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

    operator_result_t run_attention(
        operator_kind_t op,
        const tensor_t& input0,
        const tensor_t& input1,
        const tensor_t* aux0,
        const tensor_t* aux1,
        unsigned int position,
        unsigned int tile_len
    ) {
        const unsigned int rows = shape_.gqa_group_size();
        const unsigned int source_cols =
            op == kOpAttentionQk ?
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
        if (op != kOpSoftmax) {
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
            0,
            rows,
            position,
            tile_len
        );
        check_status(op, status);

        unsigned int output_cols = 0;
        if (op == kOpAttentionQk) {
            output_cols = kAttentionTile;
        } else if (op == kOpAttentionPv) {
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

private:
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
    }

    void migrate_weights() {
        std::vector<cl::Memory> buffers;
        for (const auto& buffer : weight_buffers_) {
            buffers.push_back(buffer);
        }
        check_cl(
            transfer_queue_.enqueueMigrateMemObjects(buffers, 0),
            "migrate weight buffers"
        );
        check_cl(transfer_queue_.finish(), "finish weight migration");
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
        check_cl(
            transfer_queue_.enqueueMigrateMemObjects(to_device, 0),
            "migrate operator inputs"
        );
        check_cl(transfer_queue_.finish(), "finish operator input migration");

        cl::Event compute0_event;
        cl::Event compute1_event;
        cl::Event status_event;
        cl::Event controller_event;
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
        last_controller_ms_ = event_milliseconds(controller_event);

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
    double last_controller_ms_ = -1.0;

    std::array<aligned_word_vector, 6> data_words_;
    aligned_word_vector status_words_;
    std::array<cl_mem_ext_ptr_t, 6> data_ext_{};
    std::array<cl_mem_ext_ptr_t, kWeightShardCount> weight_ext_{};
    cl_mem_ext_ptr_t status_ext_{};
    std::array<cl::Buffer, 6> data_buffers_;
    std::array<cl::Buffer, kWeightShardCount> weight_buffers_;
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
            golden_accum_t banks[4] = {
                golden_accum_t(0),
                golden_accum_t(0),
                golden_accum_t(0),
                golden_accum_t(0)
            };
            for (unsigned int in = 0; in < spec.in_dim; in++) {
                const golden_fm_t activation =
                    fixed_from_raw<golden_fm_t>(input.at(token, in));
                const golden_weight_t weight =
                    fixed_from_raw<golden_weight_t>(
                        model.projection_weight_raw(
                            layer,
                            op,
                            out,
                            in
                        )
                    );
                const golden_accum_t product =
                    golden_accum_t(activation * weight);
                const unsigned int phase = in & 3u;
                if (in < 4) {
                    banks[phase] = product;
                } else {
                    banks[phase] += product;
                }
            }
            const golden_accum_t total =
                banks[0] + banks[1] + banks[2] + banks[3];
            output.at(token, out) =
                fixed_to_raw(golden_fm_t(total));
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

    std::cout
        << "RANDOM_VERIFY seed=" << seed
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
        bool hardware_softmax
    )
        : shape_(shape),
          model_(model),
          accelerator_(accelerator),
          hardware_softmax_(hardware_softmax),
          layer_count_(
              layer_count == 0 ?
              shape.num_layers :
              layer_count
          ),
          k_cache_(
              std::size_t(layer_count_) *
              shape.max_seq_len *
              shape.kv_channels(),
              0
          ),
          v_cache_(
              std::size_t(layer_count_) *
              shape.max_seq_len *
              shape.kv_channels(),
              0
          ) {
        if (layer_count_ > shape.num_layers) {
            throw std::runtime_error("--layers exceeds the selected profile");
        }
    }

    tensor_t run_token(unsigned int token_id, unsigned int position) {
        if (position >= shape_.max_seq_len) {
            throw std::runtime_error("sequence exceeds the selected profile");
        }

        tensor_t hidden = model_.embedding(token_id);
        for (unsigned int layer = 0; layer < layer_count_; layer++) {
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
            append_kv(layer, position, k, v);
            tensor_t attention = run_attention(layer, position, q);
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
    std::size_t cache_index(
        unsigned int layer,
        unsigned int position,
        unsigned int channel
    ) const {
        return
            (
                std::size_t(layer) * shape_.max_seq_len +
                position
            ) *
            shape_.kv_channels() +
            channel;
    }

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

    void append_kv(
        unsigned int layer,
        unsigned int position,
        const tensor_t& k,
        const tensor_t& v
    ) {
        for (unsigned int channel = 0;
             channel < shape_.kv_channels();
             channel++) {
            const std::size_t index = cache_index(
                layer,
                position,
                channel
            );
            k_cache_[index] = k.at(0, channel);
            v_cache_[index] = v.at(0, channel);
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

    tensor_t build_cache_panel(
        const std::vector<int16_t>& cache,
        unsigned int layer,
        unsigned int kv_head,
        unsigned int position_begin,
        unsigned int tile_len
    ) const {
        tensor_t panel(tile_len, shape_.head_dim());
        const unsigned int channel_base = kv_head * shape_.head_dim();
        for (unsigned int pos = 0; pos < tile_len; pos++) {
            for (unsigned int elem = 0;
                 elem < shape_.head_dim();
                 elem++) {
                panel.at(pos, elem) = cache[cache_index(
                    layer,
                    position_begin + pos,
                    channel_base + elem
                )];
            }
        }
        return panel;
    }

    static void cpu_softmax(std::vector<int16_t>& scores) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (int16_t score : scores) {
            maximum = std::max(maximum, dequantize_fix16(score));
        }
        std::vector<double> exponentials(scores.size());
        double sum = 0.0;
        for (std::size_t i = 0; i < scores.size(); i++) {
            exponentials[i] =
                std::exp(dequantize_fix16(scores[i]) - maximum);
            sum += exponentials[i];
        }
        for (std::size_t i = 0; i < scores.size(); i++) {
            scores[i] = quantize_fix16(exponentials[i] / sum);
        }
    }

    tensor_t run_attention(
        unsigned int layer,
        unsigned int position,
        const tensor_t& q
    ) {
        const unsigned int context_len = position + 1;
        const unsigned int group_size = shape_.gqa_group_size();
        const unsigned int head_dim = shape_.head_dim();
        tensor_t query0 = build_query_group(q, 0);
        tensor_t query1 = build_query_group(q, 1);
        std::vector<std::vector<int16_t> > probabilities(
            shape_.num_heads,
            std::vector<int16_t>(context_len, 0)
        );

        for (unsigned int tile_begin = 0;
             tile_begin < context_len;
             tile_begin += kAttentionTile) {
            const unsigned int tile_len =
                std::min(kAttentionTile, context_len - tile_begin);
            tensor_t k0 = build_cache_panel(
                k_cache_,
                layer,
                0,
                tile_begin,
                tile_len
            );
            tensor_t k1 = build_cache_panel(
                k_cache_,
                layer,
                1,
                tile_begin,
                tile_len
            );
            operator_result_t qk = accelerator_.run_attention(
                kOpAttentionQk,
                query0,
                query1,
                &k0,
                &k1,
                position,
                tile_len
            );
            for (unsigned int row = 0; row < group_size; row++) {
                for (unsigned int pos = 0; pos < tile_len; pos++) {
                    probabilities[row][tile_begin + pos] =
                        qk.port0.at(row, pos);
                    probabilities[group_size + row][tile_begin + pos] =
                        qk.port1.at(row, pos);
                }
            }
        }

        if (hardware_softmax_ && context_len <= kAttentionTile) {
            tensor_t score0(group_size, kAttentionTile);
            tensor_t score1(group_size, kAttentionTile);
            for (unsigned int row = 0; row < group_size; row++) {
                for (unsigned int pos = 0; pos < context_len; pos++) {
                    score0.at(row, pos) = probabilities[row][pos];
                    score1.at(row, pos) =
                        probabilities[group_size + row][pos];
                }
            }
            operator_result_t softmax = accelerator_.run_attention(
                kOpSoftmax,
                score0,
                score1,
                nullptr,
                nullptr,
                position,
                context_len
            );
            for (unsigned int row = 0; row < group_size; row++) {
                for (unsigned int pos = 0; pos < context_len; pos++) {
                    probabilities[row][pos] =
                        softmax.port0.at(row, pos);
                    probabilities[group_size + row][pos] =
                        softmax.port1.at(row, pos);
                }
            }
        } else {
            for (auto& row : probabilities) {
                cpu_softmax(row);
            }
        }

        std::vector<std::vector<int32_t> > accumulated(
            shape_.num_heads,
            std::vector<int32_t>(head_dim, 0)
        );
        for (unsigned int tile_begin = 0;
             tile_begin < context_len;
             tile_begin += kAttentionTile) {
            const unsigned int tile_len =
                std::min(kAttentionTile, context_len - tile_begin);
            tensor_t p0(group_size, kAttentionTile);
            tensor_t p1(group_size, kAttentionTile);
            for (unsigned int row = 0; row < group_size; row++) {
                for (unsigned int pos = 0; pos < tile_len; pos++) {
                    p0.at(row, pos) =
                        probabilities[row][tile_begin + pos];
                    p1.at(row, pos) =
                        probabilities[group_size + row][tile_begin + pos];
                }
            }
            tensor_t v0 = build_cache_panel(
                v_cache_,
                layer,
                0,
                tile_begin,
                tile_len
            );
            tensor_t v1 = build_cache_panel(
                v_cache_,
                layer,
                1,
                tile_begin,
                tile_len
            );
            operator_result_t pv = accelerator_.run_attention(
                kOpAttentionPv,
                p0,
                p1,
                &v0,
                &v1,
                position,
                tile_len
            );
            for (unsigned int row = 0; row < group_size; row++) {
                for (unsigned int elem = 0; elem < head_dim; elem++) {
                    accumulated[row][elem] += pv.port0.at(row, elem);
                    accumulated[group_size + row][elem] +=
                        pv.port1.at(row, elem);
                }
            }
        }

        tensor_t attention(1, shape_.hidden_size);
        for (unsigned int head = 0; head < shape_.num_heads; head++) {
            for (unsigned int elem = 0; elem < head_dim; elem++) {
                attention.at(0, head * head_dim + elem) =
                    saturating_add(accumulated[head][elem], 0);
            }
        }
        return attention;
    }

    const model_shape_t& shape_;
    const model_data_t& model_;
    accelerator_t& accelerator_;
    bool hardware_softmax_;
    unsigned int layer_count_;
    std::vector<int16_t> k_cache_;
    std::vector<int16_t> v_cache_;
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

command_line_t parse_command_line(int argc, const char* argv[]) {
    command_line_t command;
    std::string value;
    get_option(argc, argv, "--mode", command.mode);
    get_option(argc, argv, "--profile", command.profile);
    get_option(argc, argv, "--data", command.data_dir);
    get_option(argc, argv, "--xclbin", command.xclbin);
    if (get_option(argc, argv, "--tokens", value)) {
        command.tokens = parse_tokens(value);
    }
    if (get_option(argc, argv, "--layers", value)) {
        command.layer_count = unsigned(std::stoul(value));
    }
    if (get_option(argc, argv, "--max-new-tokens", value)) {
        command.max_new_tokens = unsigned(std::stoul(value));
    }
    if (get_option(argc, argv, "--seed", value)) {
        command.random_seed = uint32_t(std::stoul(value));
    }
    command.zero_model = has_flag(argc, argv, "--zero-model");
    command.verbose_ops = has_flag(argc, argv, "--verbose-ops");
    command.hardware_softmax = has_flag(
        argc,
        argv,
        "--hardware-softmax"
    );
    command.tie_embeddings = has_flag(
        argc,
        argv,
        "--tie-embeddings"
    );
    return command;
}

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "  --mode plan|inspect|run|generate|verify-random\n"
        << "  --profile small|medium|qwen-layer|qwen2.5-3b\n"
        << "  --xclbin <file>          Required for run/generate\n"
        << "  --data <dir>             Packed Fix16 model directory\n"
        << "  --tokens <id,id,...>     Prompt token IDs\n"
        << "  --layers <n>             Limit executed decoder layers\n"
        << "  --max-new-tokens <n>     Greedy tokens for generate\n"
        << "  --seed <n>               Random verification seed\n"
        << "  --zero-model             Deterministic plumbing model\n"
        << "  --tie-embeddings         Reuse embedding as the LM head\n"
        << "  --hardware-softmax       Diagnostic path for one attention tile\n"
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
        << "  repeat layer: RMS -> Q/K/V -> RoPE(host) -> KV(host)\n"
        << "  attention: QK(HLS) -> global softmax(host)"
        << " -> PV(HLS) -> O\n"
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
            command.mode != "verify-random"
        ) {
            throw std::runtime_error("unknown --mode " + command.mode);
        }
        if (command.xclbin.empty()) {
            throw std::runtime_error("--xclbin is required for run/generate");
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

        model_data_t model(shape);
        if (command.mode == "verify-random") {
            model.initialize_random_model(command.random_seed);
        } else {
            model.load(
                command.data_dir,
                command.zero_model,
                command.tie_embeddings,
                command.mode == "generate"
            );
        }
        accelerator_t accelerator(
            shape,
            model,
            command.xclbin,
            command.verbose_ops
        );
        if (command.mode == "verify-random") {
            return run_random_verification(
                shape,
                model,
                accelerator,
                command.random_seed
            ) ? 0 : 1;
        }
        qwen_executor_t executor(
            shape,
            model,
            accelerator,
            command.layer_count,
            command.hardware_softmax
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
