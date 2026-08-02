#ifndef LLM_FPGA_MODEL_CONFIG_HPP
#define LLM_FPGA_MODEL_CONFIG_HPP

#include <cstddef>

constexpr std::size_t ceildiv_size(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

#if defined(QWEN_TEST_COSIM_MEDIUM)
constexpr unsigned int VOCAB_SIZE = 128;
constexpr unsigned int HIDDEN_SIZE = 128;
constexpr unsigned int INTERMEDIATE_SIZE = 256;
constexpr unsigned int NUM_LAYERS = 2;
constexpr unsigned int NUM_ATTENTION_HEADS = 4;
constexpr unsigned int NUM_KEY_VALUE_HEADS = 2;
constexpr unsigned int MAX_SEQ_LEN = 64;
constexpr unsigned int TEST_PREFILL_LEN = 16;
constexpr unsigned int ATTENTION_TILE_SIZE = 16;
constexpr double ROPE_FREQ_RATIO = 0.001;
#elif defined(QWEN_TEST_MEDIUM_LAYER)
constexpr unsigned int VOCAB_SIZE = 128;
constexpr unsigned int HIDDEN_SIZE = 512;
constexpr unsigned int INTERMEDIATE_SIZE = 1024;
constexpr unsigned int NUM_LAYERS = 1;
constexpr unsigned int NUM_ATTENTION_HEADS = 8;
constexpr unsigned int NUM_KEY_VALUE_HEADS = 2;
constexpr unsigned int MAX_SEQ_LEN = 96;
constexpr unsigned int TEST_PREFILL_LEN = 32;
constexpr unsigned int ATTENTION_TILE_SIZE = 16;
constexpr double ROPE_FREQ_RATIO = 0.8058421877614819;
#elif defined(QWEN_TEST_QWEN_LAYER)
constexpr unsigned int VOCAB_SIZE = 128;
constexpr unsigned int HIDDEN_SIZE = 2048;
constexpr unsigned int INTERMEDIATE_SIZE = 11008;
constexpr unsigned int NUM_LAYERS = 1;
constexpr unsigned int NUM_ATTENTION_HEADS = 16;
constexpr unsigned int NUM_KEY_VALUE_HEADS = 2;
constexpr unsigned int MAX_SEQ_LEN = 96;
constexpr unsigned int TEST_PREFILL_LEN = 64;
constexpr unsigned int ATTENTION_TILE_SIZE = 32;
constexpr double ROPE_FREQ_RATIO = 0.8058421877614819;
#elif defined(QWEN_TEST_SMALL)
constexpr unsigned int VOCAB_SIZE = 64;
constexpr unsigned int HIDDEN_SIZE = 64;
constexpr unsigned int INTERMEDIATE_SIZE = 128;
constexpr unsigned int NUM_LAYERS = 2;
constexpr unsigned int NUM_ATTENTION_HEADS = 4;
constexpr unsigned int NUM_KEY_VALUE_HEADS = 2;
constexpr unsigned int MAX_SEQ_LEN = 32;
constexpr unsigned int TEST_PREFILL_LEN = 8;
constexpr unsigned int ATTENTION_TILE_SIZE = 8;
constexpr double ROPE_FREQ_RATIO = 0.001;
#else
constexpr unsigned int VOCAB_SIZE = 151936;
constexpr unsigned int HIDDEN_SIZE = 2048;
constexpr unsigned int INTERMEDIATE_SIZE = 11008;
constexpr unsigned int NUM_LAYERS = 36;
constexpr unsigned int NUM_ATTENTION_HEADS = 16;
constexpr unsigned int NUM_KEY_VALUE_HEADS = 2;
constexpr unsigned int MAX_SEQ_LEN = 2048;
constexpr unsigned int TEST_PREFILL_LEN = 16;
constexpr unsigned int ATTENTION_TILE_SIZE = 32;
constexpr double ROPE_FREQ_RATIO = 0.8058421877614819;
#endif

constexpr unsigned int HEAD_DIM = HIDDEN_SIZE / NUM_ATTENTION_HEADS;
constexpr unsigned int KV_CHANNELS = NUM_KEY_VALUE_HEADS * HEAD_DIM;
constexpr unsigned int GQA_GROUP_SIZE = NUM_ATTENTION_HEADS / NUM_KEY_VALUE_HEADS;
constexpr unsigned int ATTENTION_NUM_TILES = ceildiv_size(MAX_SEQ_LEN, ATTENTION_TILE_SIZE);
constexpr unsigned int TEST_PREFILL_LAST_ATTENTION_TILES = ceildiv_size(TEST_PREFILL_LEN, ATTENTION_TILE_SIZE);
constexpr unsigned int TEST_DECODE_ATTENTION_TILES = ceildiv_size(TEST_PREFILL_LEN + 1, ATTENTION_TILE_SIZE);
constexpr unsigned int NUM_WEIGHT_SHARDS = 16;
constexpr double RMS_NORM_EPS = 1e-6;
constexpr double ROPE_THETA = 1000000.0;

constexpr double sqrt_newton(double x, double guess, unsigned int iter) {
    return iter == 0 ? guess : sqrt_newton(x, 0.5 * (guess + x / guess), iter - 1);
}

constexpr double const_sqrt(double x) {
    return sqrt_newton(x, x > 1.0 ? x : 1.0, 24);
}

constexpr double ATTENTION_SCALE = 1.0 / const_sqrt(double(HEAD_DIM));

constexpr unsigned int PACKED_WEIGHT_TILE_OUT = 16;
constexpr unsigned int PACKED_WEIGHT_TILE_IN = 16;
constexpr unsigned int PACKED_WEIGHT_BLOCK_ELEMS = 32;
constexpr unsigned int PACKED_WEIGHT_TILE_ELEMS = PACKED_WEIGHT_TILE_OUT * PACKED_WEIGHT_TILE_IN;
constexpr unsigned int PACKED_WEIGHT_BLOCKS_PER_TILE = PACKED_WEIGHT_TILE_ELEMS / PACKED_WEIGHT_BLOCK_ELEMS;
constexpr unsigned int WEIGHT_SHARD_GROUPS = NUM_WEIGHT_SHARDS / PACKED_WEIGHT_BLOCKS_PER_TILE;

constexpr std::size_t packed_linear_tiles(std::size_t out_dim, std::size_t in_dim) {
    return ceildiv_size(out_dim, PACKED_WEIGHT_TILE_OUT) * ceildiv_size(in_dim, PACKED_WEIGHT_TILE_IN);
}

constexpr std::size_t packed_linear_elems(std::size_t out_dim, std::size_t in_dim) {
    return packed_linear_tiles(out_dim, in_dim) * PACKED_WEIGHT_TILE_ELEMS;
}

constexpr std::size_t Q_WEIGHT_OFFSET = 0;
constexpr std::size_t K_WEIGHT_OFFSET = Q_WEIGHT_OFFSET + packed_linear_elems(HIDDEN_SIZE, HIDDEN_SIZE);
constexpr std::size_t V_WEIGHT_OFFSET = K_WEIGHT_OFFSET + packed_linear_elems(KV_CHANNELS, HIDDEN_SIZE);
constexpr std::size_t O_WEIGHT_OFFSET = V_WEIGHT_OFFSET + packed_linear_elems(KV_CHANNELS, HIDDEN_SIZE);
constexpr std::size_t GATE_WEIGHT_OFFSET = O_WEIGHT_OFFSET + packed_linear_elems(HIDDEN_SIZE, HIDDEN_SIZE);
constexpr std::size_t UP_WEIGHT_OFFSET = GATE_WEIGHT_OFFSET + packed_linear_elems(INTERMEDIATE_SIZE, HIDDEN_SIZE);
constexpr std::size_t DOWN_WEIGHT_OFFSET = UP_WEIGHT_OFFSET + packed_linear_elems(INTERMEDIATE_SIZE, HIDDEN_SIZE);
constexpr std::size_t LAYER_WEIGHT_SIZE = DOWN_WEIGHT_OFFSET + packed_linear_elems(HIDDEN_SIZE, INTERMEDIATE_SIZE);
constexpr std::size_t LAYER_WEIGHT_TILES = LAYER_WEIGHT_SIZE / PACKED_WEIGHT_TILE_ELEMS;
constexpr std::size_t TOTAL_LAYER_WEIGHT_TILES = std::size_t(NUM_LAYERS) * LAYER_WEIGHT_TILES;
constexpr std::size_t TOTAL_LAYER_WEIGHT_ELEMS = TOTAL_LAYER_WEIGHT_TILES * PACKED_WEIGHT_TILE_ELEMS;
constexpr std::size_t WEIGHT_SHARD_BLOCKS = ceildiv_size(TOTAL_LAYER_WEIGHT_TILES, WEIGHT_SHARD_GROUPS);
constexpr std::size_t WEIGHT_SHARD_ELEMS = WEIGHT_SHARD_BLOCKS * PACKED_WEIGHT_BLOCK_ELEMS;

constexpr std::size_t TOKEN_EMBEDDING_ELEMS = std::size_t(VOCAB_SIZE) * HIDDEN_SIZE;
constexpr std::size_t NORM_WEIGHT_ELEMS = std::size_t(NUM_LAYERS) * 2 * HIDDEN_SIZE + HIDDEN_SIZE;
constexpr std::size_t KV_CACHE_ELEMS = std::size_t(NUM_LAYERS) * MAX_SEQ_LEN * KV_CHANNELS;
constexpr std::size_t SCRATCH_ELEMS = std::size_t(8) * HIDDEN_SIZE;

static_assert(HIDDEN_SIZE % NUM_ATTENTION_HEADS == 0, "hidden size must divide attention heads");
static_assert(NUM_ATTENTION_HEADS % NUM_KEY_VALUE_HEADS == 0, "GQA requires query heads to divide KV heads");
static_assert(HEAD_DIM % 2 == 0, "RoPE expects an even head dimension");
static_assert(ATTENTION_TILE_SIZE > 0, "attention tile must be nonzero");
static_assert(ATTENTION_TILE_SIZE <= MAX_SEQ_LEN, "attention tile must fit in the configured sequence length");
static_assert(NUM_WEIGHT_SHARDS % PACKED_WEIGHT_BLOCKS_PER_TILE == 0, "weight shard groups must cover tile blocks");
static_assert(TEST_PREFILL_LEN <= MAX_SEQ_LEN, "test prefill length must fit in the configured KV cache");
#if defined(QWEN_TEST_QWEN_LAYER)
static_assert(TEST_PREFILL_LAST_ATTENTION_TILES >= 2, "Qwen-layer prefill must cover at least two attention tile loads");
static_assert(TEST_DECODE_ATTENTION_TILES >= 3, "Qwen-layer decode must cover at least three attention tile loads");
#endif

#endif
