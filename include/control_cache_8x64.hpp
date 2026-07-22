#ifndef LLM_FPGA_CONTROL_CACHE_8X64_HPP
#define LLM_FPGA_CONTROL_CACHE_8X64_HPP

#include "compute_core_8x64_unified.hpp"
#include "mm_controller.hpp"
#include "weight_pipeline_config.hpp"

#ifndef CC8_RESIDENT_LAYER_ONLY
#define CC8_RESIDENT_LAYER_ONLY 0
#endif

constexpr unsigned int CC8_MM_CORE_COUNT = 2;
constexpr unsigned int CC8_OUTPUTS_PER_WAVE =
    CC8_MM_CORE_COUNT * MM_STREAM_8X64_OUTPUTS;
constexpr unsigned int CC8_OUTPUT_TILES_PER_WAVE =
    CC8_OUTPUTS_PER_WAVE / MM_PE_OUT;
constexpr unsigned int CC8_MAX_OUTPUT_WAVES =
    ceildiv(MAX_LINEAR_OUT_DIM, CC8_OUTPUTS_PER_WAVE);
constexpr unsigned int CC8_MAX_PACKED_WEIGHT_TILES =
    MAX_LINEAR_IN_BLOCKS * CC8_OUTPUT_TILES_PER_WAVE;
constexpr unsigned int CC8_TOKENS_PER_DATA_PORT =
    ceildiv(LINEAR_TOKEN_TILE_ACTIVE, 2u);
constexpr unsigned int CC8_FEATURE_WORDS_PER_TOKEN =
    ceildiv(MAX_LINEAR_IN_DIM, FM_BLOCK_SIZE);
constexpr unsigned int CC8_FEATURE_WORDS_PER_PORT =
    CC8_TOKENS_PER_DATA_PORT * CC8_FEATURE_WORDS_PER_TOKEN;
constexpr unsigned int CC8_DATA_PORT_WORDS =
    CC8_FEATURE_WORDS_PER_PORT > CU_MV_CACHE_FM_WORDS ?
    CC8_FEATURE_WORDS_PER_PORT :
    CU_MV_CACHE_FM_WORDS;
constexpr unsigned int CC8_GBUF_BLOCKS = MAX_LINEAR_IN_BLOCKS;
constexpr unsigned int CC8_HIDDEN_GBUF_BLOCKS =
    ceildiv(HIDDEN_SIZE, MM_PE_IN);
// The resident-only image accepts exactly one decode token per launch.  Keep
// the external HBM/AXI ABI sized for the general 8-token path, but do not
// replicate every internal feature bank eight times in that production
// image.  Diagnostic/non-resident builds retain all token rows.
constexpr unsigned int CC8_GBUF_TOKEN_ROWS =
    CC8_RESIDENT_LAYER_ONLY ? 1u : LINEAR_TOKEN_TILE_ACTIVE;
constexpr unsigned int CC8_ATTN_TILE = CU_SOFTMAX_ELEMS;
constexpr unsigned int CC8_HEAD_WORDS =
    ceildiv(HEAD_DIM, FM_BLOCK_SIZE);
constexpr unsigned int CC8_KV_CACHE_ROW_MAJOR_WORDS =
    NUM_LAYERS * MAX_SEQ_LEN * NUM_KEY_VALUE_HEADS * CC8_HEAD_WORDS;
constexpr unsigned int CC8_K_CACHE_POSITION_WORDS =
    ceildiv(MAX_SEQ_LEN, FM_BLOCK_SIZE);
constexpr unsigned int CC8_K_CACHE_TRANSPOSED_WORDS =
    NUM_LAYERS * NUM_KEY_VALUE_HEADS * HEAD_DIM *
    CC8_K_CACHE_POSITION_WORDS;
// Keep the original row-major region first for the diagnostic ABI and append
// a controller-private K^T region.  V uses only the row-major prefix.  Even
// the full 36-layer profile remains well below one 256 MiB HBM pseudo-channel.
constexpr unsigned int CC8_KV_CACHE_WORDS =
    CC8_KV_CACHE_ROW_MAJOR_WORDS + CC8_K_CACHE_TRANSPOSED_WORDS;
constexpr unsigned int CC8_ATTN_BUFFER_ELEMS =
    HEAD_DIM > CC8_ATTN_TILE ? HEAD_DIM : CC8_ATTN_TILE;
constexpr unsigned int CC8_ATTN_BUFFER_BLOCKS =
    ceildiv(CC8_ATTN_BUFFER_ELEMS, MM_PE_IN);
constexpr unsigned int CC8_ATTN_BUFFER_WORDS =
    ceildiv(CC8_ATTN_BUFFER_ELEMS, FM_BLOCK_SIZE);
constexpr unsigned int CC8_FLASH_ACCUM_BLOCKS =
    ceildiv(HEAD_DIM, CU_VEC_LANES);
constexpr unsigned int CC8_ATTN_PACKET_GROUPS =
    MM_STREAM_8X64_WEIGHT_GROUPS;
constexpr unsigned int CC8_ATTN_PV_WAVES =
    ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS);
constexpr unsigned int CC8_ATTN_PV_PACKET_DEPTH =
    CC8_ATTN_PV_WAVES * CC8_ATTN_TILE;
constexpr unsigned int CC8_ATTN_WEIGHT_PACKET_BITS =
    CU_VEC_LANES * wt_linear_t::width;
using cc8_attention_weight_packet_word_t =
    ap_uint<CC8_ATTN_WEIGHT_PACKET_BITS>;
constexpr unsigned int CC8_LAYER_NORM_WORDS =
    ceildiv(HIDDEN_SIZE, FM_BLOCK_SIZE);
constexpr unsigned int CC8_ROPE_HALF_ELEMS = HEAD_DIM / 2;
constexpr unsigned int CC8_ROPE_VALUE_BANKS = 2;
constexpr unsigned int CC8_ROPE_WORDS =
    ceildiv(CC8_ROPE_HALF_ELEMS, FM_BLOCK_SIZE);
constexpr unsigned int CC8_ROPE_COS_WORD_OFFSET = CC8_LAYER_NORM_WORDS;
constexpr unsigned int CC8_ROPE_SIN_WORD_OFFSET =
    CC8_ROPE_COS_WORD_OFFSET + CC8_ROPE_WORDS;
constexpr unsigned int CC8_RESIDENT_LAYER_MM_WAVE_SLOTS_MAX =
    4 * ceildiv(HIDDEN_SIZE, CC8_OUTPUTS_PER_WAVE) +
    2 * ceildiv(KV_CHANNELS, CC8_OUTPUTS_PER_WAVE) +
    2 * ceildiv(INTERMEDIATE_SIZE, CC8_OUTPUTS_PER_WAVE) +
    ATTENTION_NUM_TILES *
        (1 + ceildiv(HEAD_DIM, MM_STREAM_8X64_OUTPUTS));
constexpr unsigned int CC8_RESIDENT_LAYER_VECTOR_TASKS = 5;
constexpr unsigned int CC8_RESIDENT_LAYER_CORE0_TASKS_MAX =
    CC8_RESIDENT_LAYER_MM_WAVE_SLOTS_MAX +
    CC8_RESIDENT_LAYER_VECTOR_TASKS;

static_assert(CC8_MM_CORE_COUNT == 2, "v1 control/cache core drives two MM cores");
static_assert(CC8_OUTPUTS_PER_WAVE == 128, "two 8x64 cores produce one 8x128 wave");
static_assert(FM_BLOCK_SIZE == 2 * CU_VEC_LANES, "one 512-bit word holds two vec16 packets");
static_assert(NUM_KEY_VALUE_HEADS == CC8_MM_CORE_COUNT, "two unified cores map the two Qwen KV groups");
static_assert(GQA_GROUP_SIZE <= MM_STREAM_8X64_TOKENS, "one core must hold one GQA group");
static_assert(CC8_ATTN_PACKET_GROUPS == 4, "attention packet panel matches four CU weight streams");
static_assert(CC8_ATTN_TILE == 2 * FM_BLOCK_SIZE, "one attention tile maps to four 16-lane packets");
static_assert(CC8_ATTN_WEIGHT_PACKET_BITS == 256, "attention packet bank stores one CU weight packet");
static_assert(CC8_ROPE_VALUE_BANKS == 2, "RoPE cache has low/high half-head banks");
static_assert(2 * CC8_ROPE_HALF_ELEMS == HEAD_DIM, "RoPE requires an even head dimension");
static_assert(
    CC8_K_CACHE_TRANSPOSED_WORDS <= CC8_KV_CACHE_ROW_MAJOR_WORDS,
    "transposed K cache must fit within one additional row-major-sized region"
);
static_assert(CC8_ROPE_SIN_WORD_OFFSET + CC8_ROPE_WORDS <= CC8_DATA_PORT_WORDS,
              "resident layer aux0 must hold norm and RoPE coefficients");
#if CC8_RESIDENT_LAYER_ONLY
static_assert(
    CC8_RESIDENT_LAYER_CORE0_TASKS_MAX <= CU8_MAX_TASKS_PER_LAUNCH,
    "one-launch resident layer exceeds compute service task capacity"
);
#endif

struct cc8_attention_panel_t {
    wt_block_t word[CC8_ATTN_TILE][CC8_HEAD_WORDS];
};

struct cc8_attention_qk_packet_panel_t {
    cc8_attention_weight_packet_word_t
        packet[CC8_ATTN_PACKET_GROUPS][HEAD_DIM];
};

struct cc8_attention_pv_packet_panel_t {
    cc8_attention_weight_packet_word_t
        packet[CC8_ATTN_PACKET_GROUPS][CC8_ATTN_PV_PACKET_DEPTH];
};

struct cc8_current_kv_words_t {
    fm_word_t word[NUM_KEY_VALUE_HEADS][CC8_HEAD_WORDS];
};

// Resident decode keeps RoPE operands in two scalar half-head banks.  RoPE
// consumes one value from each half and writes both results every cycle, so
// the split maps naturally to independent 1R1W BRAM banks and avoids a
// 256/512-bit packed-word read-modify-write path.
struct cc8_resident_query_buffer_t {
    fm_t value
        [GQA_GROUP_SIZE]
        [CC8_ROPE_VALUE_BANKS]
        [CC8_ROPE_HALF_ELEMS];
};

struct cc8_resident_k_buffer_t {
    fm_t value
        [NUM_KEY_VALUE_HEADS]
        [CC8_ROPE_VALUE_BANKS]
        [CC8_ROPE_HALF_ELEMS];
};

struct cc8_rope_coefficient_buffer_t {
    fm_t value
        [CC8_ROPE_VALUE_BANKS]
        [CC8_ROPE_HALF_ELEMS];
};

struct cc8_resident_probability_buffer_t {
    fm_t value
        [GQA_GROUP_SIZE]
        [CC8_ATTN_TILE];
};

struct cc8_attention_buffer_t {
    mm_input_block_t block
        [MM_STREAM_8X64_TOKENS]
        [CC8_ATTN_BUFFER_BLOCKS];
};

struct cc8_flash_accumulator_t {
    fm_accum_t value
        [MM_STREAM_8X64_TOKENS]
        [CC8_FLASH_ACCUM_BLOCKS]
        [CU_VEC_LANES];
};

enum cc8_operator_t {
    CC8_OP_NOP = 0,
    CC8_OP_Q_PROJECTION = 1,
    CC8_OP_K_PROJECTION = 2,
    CC8_OP_V_PROJECTION = 3,
    CC8_OP_O_PROJECTION = 4,
    CC8_OP_FFN_GATE = 5,
    CC8_OP_FFN_UP = 6,
    CC8_OP_FFN_DOWN = 7,
    CC8_OP_RMSNORM = 8,
    CC8_OP_SILU_MUL = 9,
    CC8_OP_RESIDUAL_ADD = 10,
    CC8_OP_ATTN_QK = 11,
    CC8_OP_ATTN_PV = 12,
    CC8_OP_SOFTMAX = 13,
    CC8_OP_ATTN_FLASH = 14,
    CC8_OP_DECODE_SMOKE = 15,
    CC8_OP_DECODER_LAYER = 16
};

enum cc8_status_code_t {
    CC8_STATUS_OK = 0,
    CC8_STATUS_BAD_OPERATOR = 1,
    CC8_STATUS_BAD_TOKEN_COUNT = 2,
    CC8_STATUS_BAD_TILE_LENGTH = 3,
    CC8_STATUS_BAD_POSITION = 4,
    CC8_STATUS_BAD_LAYER = 5
};

struct cc8_operator_spec_t {
    bool uses_mm;
    bool uses_vector;
    mm_projection_kind_t projection;
    cu8_mode_t compute_mode;
    unsigned int in_dim;
    unsigned int out_dim;
    unsigned int vector_inputs;
};

struct cc8_status_packet_t {
    cc8_operator_t op;
    cc8_status_code_t status;
    unsigned int token_count;
    unsigned int output_waves;
    unsigned int dispatched_mm_tasks;
    unsigned int dispatched_vector_tasks;
    unsigned int completed_output_packets;
    bool last_task;
};

template <unsigned int BLOCK_COUNT>
struct cc8_feature_buffer_t {
    static constexpr unsigned int kBlockCount = BLOCK_COUNT;
    mm_input_block_t block[CC8_GBUF_TOKEN_ROWS][BLOCK_COUNT];
};

// The generic/legacy scheduler still needs the maximum linear dimension.
// The resident layer uses narrow hidden-state caches alongside exactly two
// intermediate-width ping-pong banks so residual/norm state is not
// accidentally implemented at FFN width.
using cc8_global_buffer_t = cc8_feature_buffer_t<CC8_GBUF_BLOCKS>;
using cc8_hidden_buffer_t = cc8_feature_buffer_t<CC8_HIDDEN_GBUF_BLOCKS>;

void cc8_status_sink(
    hls::stream<cc8_status_packet_t>& status_stream,
    fm_word_t status_output[1]
);

bool cc8_get_operator_spec(
    cc8_operator_t op,
    cc8_operator_spec_t& spec
);

void control_cache_8x64_dual_core(
    hls::stream<cu8_task_t>& core0_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core0_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core0_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core0_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core0_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core0_result_stream,
    hls::stream<cu8_task_t>& core1_task_stream,
    hls::stream<mm_stream_8x64_activation_packet_t>& core1_activation_stream,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream0,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream1,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream2,
    hls::stream<mm_stream_8x64_weight_packet_t>& core1_weight_stream3,
    hls::stream<cu_vec16_packet_t>& core1_vector_input0_stream,
    hls::stream<cu_vec16_packet_t>& core1_vector_input1_stream,
    hls::stream<cu_vec16_packet_t>& core1_result_stream,
    hls::stream<cc8_status_packet_t>& status_stream,
    fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT],
    fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT],
    const fm_word_t input_port0[CC8_DATA_PORT_WORDS],
    const fm_word_t input_port1[CC8_DATA_PORT_WORDS],
    const fm_word_t aux_port0[CC8_DATA_PORT_WORDS],
    const fm_word_t aux_port1[CC8_DATA_PORT_WORDS],
    unsigned int operator_kind,
    unsigned int layer_id,
    unsigned int token_count,
    unsigned int position,
    unsigned int tile_len,
    QWEN_WEIGHT_SHARD_PARAMS,
    fm_word_t kv_cache_k[CC8_KV_CACHE_WORDS],
    fm_word_t kv_cache_v[CC8_KV_CACHE_WORDS]
);

#endif
