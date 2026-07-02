#include "vitis_stream_8x64.hpp"

#include <cstdio>

static fm_word_t output_port0[CC8_FEATURE_WORDS_PER_PORT];
static fm_word_t output_port1[CC8_FEATURE_WORDS_PER_PORT];
static fm_word_t input_port0[CC8_DATA_PORT_WORDS];
static fm_word_t input_port1[CC8_DATA_PORT_WORDS];
static fm_word_t aux_port0[CC8_DATA_PORT_WORDS];
static fm_word_t aux_port1[CC8_DATA_PORT_WORDS];

static wt_block_t shard0[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard1[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard2[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard3[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard4[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard5[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard6[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard7[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard8[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard9[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard10[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard11[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard12[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard13[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard14[WEIGHT_SHARD_BLOCKS];
static wt_block_t shard15[WEIGHT_SHARD_BLOCKS];

int main() {
    hls::stream<cu8_nk_task_word_t> core0_task_stream;
    hls::stream<cu8_nk_activation_word_t> core0_activation_stream;
    hls::stream<cu8_nk_weight_word_t> core0_weight_stream0;
    hls::stream<cu8_nk_weight_word_t> core0_weight_stream1;
    hls::stream<cu8_nk_weight_word_t> core0_weight_stream2;
    hls::stream<cu8_nk_weight_word_t> core0_weight_stream3;
    hls::stream<cu8_nk_vector_word_t> core0_vector_input0_stream;
    hls::stream<cu8_nk_vector_word_t> core0_vector_input1_stream;
    hls::stream<cu8_nk_vector_word_t> core0_result_stream;
    hls::stream<cu8_nk_task_word_t> core1_task_stream;
    hls::stream<cu8_nk_activation_word_t> core1_activation_stream;
    hls::stream<cu8_nk_weight_word_t> core1_weight_stream0;
    hls::stream<cu8_nk_weight_word_t> core1_weight_stream1;
    hls::stream<cu8_nk_weight_word_t> core1_weight_stream2;
    hls::stream<cu8_nk_weight_word_t> core1_weight_stream3;
    hls::stream<cu8_nk_vector_word_t> core1_vector_input0_stream;
    hls::stream<cu8_nk_vector_word_t> core1_vector_input1_stream;
    hls::stream<cu8_nk_vector_word_t> core1_result_stream;
    hls::stream<cu8_nk_status_word_t> status_stream;

    control_cache_8x64_dual_core_nk(
        core0_task_stream,
        core0_activation_stream,
        core0_weight_stream0,
        core0_weight_stream1,
        core0_weight_stream2,
        core0_weight_stream3,
        core0_vector_input0_stream,
        core0_vector_input1_stream,
        core0_result_stream,
        core1_task_stream,
        core1_activation_stream,
        core1_weight_stream0,
        core1_weight_stream1,
        core1_weight_stream2,
        core1_weight_stream3,
        core1_vector_input0_stream,
        core1_vector_input1_stream,
        core1_result_stream,
        status_stream,
        output_port0,
        output_port1,
        input_port0,
        input_port1,
        aux_port0,
        aux_port1,
        CC8_OP_NOP,
        0,
        1,
        0,
        0,
        shard0,
        shard1,
        shard2,
        shard3,
        shard4,
        shard5,
        shard6,
        shard7,
        shard8,
        shard9,
        shard10,
        shard11,
        shard12,
        shard13,
        shard14,
        shard15
    );

    int errors = 0;
    if (core0_task_stream.empty() || core1_task_stream.empty()) {
        errors++;
    } else {
        cu8_task_t task0 =
            unpack_cu8_nk_task(core0_task_stream.read());
        cu8_task_t task1 =
            unpack_cu8_nk_task(core1_task_stream.read());
        if (task0.mode != CU8_MODE_STOP ||
            task1.mode != CU8_MODE_STOP ||
            !task0.last_task ||
            !task1.last_task) {
            errors++;
        }
    }
    if (status_stream.empty()) {
        errors++;
    } else {
        cu8_nk_status_word_t status = status_stream.read();
        if (status.range(31, 0).to_uint() != CC8_OP_NOP ||
            status.range(63, 32).to_uint() != CC8_STATUS_OK ||
            !status[224]) {
            errors++;
        }
    }

    std::printf(
        "CONTROL CACHE 8X64 NK CSIM %s\n",
        errors == 0 ? "PASS" : "FAIL"
    );
    return errors == 0 ? 0 : 1;
}
