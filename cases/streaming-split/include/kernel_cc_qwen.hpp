#ifndef LLM_FPGA_CC_KERNEL_QWEN_HPP
#define LLM_FPGA_CC_KERNEL_QWEN_HPP

/*
============================================================================
LLM-FPGA-CC Qwen-Normal version: standard Qwen dimensions with full pipeline optimization.
============================================================================

Features:
1. Standard Qwen dimensions (HIDDEN_SIZE=2048).
2. Single-transformer-layer test.
3. Full five-stage pipeline (IF-ID-EX-MEM-WB).
4. Multilevel ping-pong buffering.
5. Two- to three-round attention tile swap tests.
6. Prefill and Decode scenarios.

============================================================================
*/

#include "hardware.hpp"
#include "datatypes.hpp"
#include <hls_stream.h>

// ============================================================================
// Five-stage pipeline data abstractions.
// ============================================================================

// IF (Instruction Fetch): abstract token read.
struct if_stage_t {
    unsigned int token_id;
    unsigned int tile_idx;
    bool valid;

    void init() {
        #pragma HLS inline
        token_id = 0;
        tile_idx = 0;
        valid = false;
    }

    void set(unsigned int tid, unsigned int tidx) {
        #pragma HLS inline
        token_id = tid;
        tile_idx = tidx;
        valid = true;
    }
};

// ID (Instruction Decode): abstract data attributes.
struct id_stage_t {
    fm_t input_block[32];
    unsigned int output_channel;
    bool valid;

    void init() {
        #pragma HLS inline
        for (unsigned int i = 0; i < 32; i++) {
            #pragma HLS unroll
            input_block[i] = fm_t(0);
        }
        output_channel = 0;
        valid = false;
    }
};

// EX (Execute): MAC computation.
struct ex_stage_t {
    fm_accum_t accumulators[32];
    bool valid;

    void init() {
        #pragma HLS inline
        for (unsigned int i = 0; i < 32; i++) {
            #pragma HLS unroll
            accumulators[i] = fm_accum_t(0);
        }
        valid = false;
    }
};

// MEM (Memory): access ping-pong buffers.
struct mem_stage_t {
    fm_t results[32];
    bool buffer_select;  // Ping-pong buffer selection.
    bool valid;

    void init() {
        #pragma HLS inline
        for (unsigned int i = 0; i < 32; i++) {
            #pragma HLS unroll
            results[i] = fm_t(0);
        }
        buffer_select = false;
        valid = false;
    }
};

// WB (Write Back): write results.
struct wb_stage_t {
    fm_t output_data[32];
    unsigned int write_addr;
    bool valid;

    void init() {
        #pragma HLS inline
        for (unsigned int i = 0; i < 32; i++) {
            #pragma HLS unroll
            output_data[i] = fm_t(0);
        }
        write_addr = 0;
        valid = false;
    }
};

// ============================================================================
// Five-stage pipeline core for one token.
// ============================================================================

struct five_stage_core_t {
    if_stage_t if_stage;
    id_stage_t id_stage;
    ex_stage_t ex_stage;
    mem_stage_t mem_stage;
    wb_stage_t wb_stage;

    // Pipeline registers.
    void pipeline_stages() {
        #pragma HLS pipeline II=1
        // HLS infers inter-stage registers automatically.
    }

    // Initialize all stages.
    void init() {
        #pragma HLS inline
        if_stage.init();
        id_stage.init();
        ex_stage.init();
        mem_stage.init();
        wb_stage.init();
    }

    // IF stage: read token.
    void if_stage_compute(unsigned int token_id, unsigned int tile_idx, const fm_t input[32]) {
        if_stage.set(token_id, tile_idx);

        // Forward to the ID stage.
        for (unsigned int i = 0; i < 32; i++) {
            #pragma HLS unroll
            id_stage.input_block[i] = input[i];
        }
        id_stage.output_channel = tile_idx % 32;
        id_stage.valid = true;
    }

    // ID stage: decode data attributes.
    void id_stage_compute() {
        if (!id_stage.valid) return;

        // Forward decoded attributes to the EX stage.
        ex_stage.valid = true;
    }

    // EX stage: MAC computation.
    void ex_stage_compute(const wt_block_t& weights) {
        if (!ex_stage.valid) return;

        #pragma HLS pipeline II=1
        for (unsigned int i = 0; i < 32; i++) {
            #pragma HLS unroll
            fm_accum_t product = fm_accum_t(id_stage.input_block[i]) * fm_accum_t(weights[i]);
            ex_stage.accumulators[i] += product;
        }
    }

    // MEM stage: ping-pong buffer access.
    void mem_stage_compute() {
        if (!ex_stage.valid) return;

        #pragma HLS pipeline II=1
        for (unsigned int i = 0; i < 32; i++) {
            #pragma HLS unroll
            mem_stage.results[i] = fm_t(ex_stage.accumulators[i]);
        }

        // Switch the ping-pong buffer.
        mem_stage.buffer_select = !mem_stage.buffer_select;
        mem_stage.valid = true;
    }

    // WB stage: write results.
    void wb_stage_compute(unsigned int base_addr) {
        if (!mem_stage.valid) return;

        wb_stage.write_addr = base_addr;

        #pragma HLS pipeline II=1
        for (unsigned int i = 0; i < 32; i++) {
            #pragma HLS unroll
            wb_stage.output_data[i] = mem_stage.results[i];
        }
        wb_stage.valid = true;
    }

    // Execute the complete pipeline.
    void execute_pipeline(
        unsigned int token_id,
        unsigned int tile_idx,
        const fm_t input[32],
        const wt_block_t& weights,
        unsigned int output_addr
    ) {
        // Execute all five stages concurrently.
        if_stage_compute(token_id, tile_idx, input);
        id_stage_compute();
        ex_stage_compute(weights);
        mem_stage_compute();
        wb_stage_compute(output_addr);
    }
};

// ============================================================================
// Multilevel ping-pong buffering system.
// ============================================================================

struct multi_level_buffers_t {
    // Level 1: Input ping-pong buffer
    fm_t input_buffer[2][8][32];  // Two buffers with eight lanes each.

    // Level 2: Weight ping-pong buffer
    wt_block_t weight_buffer[2][16];  // Two buffers with 16 shards each.

    // Level 3: Output ping-pong buffer
    fm_t output_buffer[2][8][32];  // Two buffers with eight lanes each.

    unsigned int input_buffer_sel;
    unsigned int weight_buffer_sel;
    unsigned int output_buffer_sel;

    void init() {
        #pragma HLS inline
        input_buffer_sel = 0;
        weight_buffer_sel = 0;
        output_buffer_sel = 0;
    }

    void switch_input_buffer() {
        #pragma HLS inline
        input_buffer_sel = 1 - input_buffer_sel;
    }

    void switch_weight_buffer() {
        #pragma HLS inline
        weight_buffer_sel = 1 - weight_buffer_sel;
    }

    void switch_output_buffer() {
        #pragma HLS inline
        output_buffer_sel = 1 - output_buffer_sel;
    }
};

// ============================================================================
// Qwen-Normal qkv_tile_kernel declaration.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// 16-port variant for testing 16 data channels.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_16ports(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7,
    fm_t* output_token8, fm_t* output_token9, fm_t* output_token10, fm_t* output_token11,
    fm_t* output_token12, fm_t* output_token13, fm_t* output_token14, fm_t* output_token15,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    const fm_t* input_token8, const fm_t* input_token9, const fm_t* input_token10, const fm_t* input_token11,
    const fm_t* input_token12, const fm_t* input_token13, const fm_t* input_token14, const fm_t* input_token15,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// Five-stage pipeline variant with complete separation between compute and AXI interfaces.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_pipeline(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// Final streaming-interface variant.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_stream_final(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// Independent streaming-interface variant.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_independent_stream(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// Fully parallel tree-accumulator variant using 4096 DSPs.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_parallel_tree(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// Optimized architecture variant with controller-managed reduction.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_optimized(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// 16-core variant targeting 4096 DSPs.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_16cores_4096dsp(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7,
    fm_t* output_token8, fm_t* output_token9, fm_t* output_token10, fm_t* output_token11,
    fm_t* output_token12, fm_t* output_token13, fm_t* output_token14, fm_t* output_token15,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    const fm_t* input_token8, const fm_t* input_token9, const fm_t* input_token10, const fm_t* input_token11,
    const fm_t* input_token12, const fm_t* input_token13, const fm_t* input_token14, const fm_t* input_token15,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// High-bandwidth 16-core variant with large BRAM/URAM buffers.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_16cores_cache(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7, fm_t* output_token8,
    fm_t* output_token9, fm_t* output_token10, fm_t* output_token11, fm_t* output_token12,
    fm_t* output_token13, fm_t* output_token14, fm_t* output_token15,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    const fm_t* input_token8, const fm_t* input_token9, const fm_t* input_token10, const fm_t* input_token11,
    const fm_t* input_token12, const fm_t* input_token13, const fm_t* input_token14, const fm_t* input_token15,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// High-bandwidth 16-core V2 variant using direct arrays.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_16cores_cache_v2(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7, fm_t* output_token8,
    fm_t* output_token9, fm_t* output_token10, fm_t* output_token11, fm_t* output_token12,
    fm_t* output_token13, fm_t* output_token14, fm_t* output_token15,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    const fm_t* input_token8, const fm_t* input_token9, const fm_t* input_token10, const fm_t* input_token11,
    const fm_t* input_token12, const fm_t* input_token13, const fm_t* input_token14, const fm_t* input_token15,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// Simplified-cache 16-core variant based on the validated 4096-DSP architecture plus BRAM buffers.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_16cores_cache_simple(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7, fm_t* output_token8,
    fm_t* output_token9, fm_t* output_token10, fm_t* output_token11, fm_t* output_token12,
    fm_t* output_token13, fm_t* output_token14, fm_t* output_token15,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    const fm_t* input_token8, const fm_t* input_token9, const fm_t* input_token10, const fm_t* input_token11,
    const fm_t* input_token12, const fm_t* input_token13, const fm_t* input_token14, const fm_t* input_token15,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// Cache-optimized 16-core variant with large on-chip ping-pong buffers.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_16cores_cache_optimized(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7, fm_t* output_token8,
    fm_t* output_token9, fm_t* output_token10, fm_t* output_token11, fm_t* output_token12,
    fm_t* output_token13, fm_t* output_token14, fm_t* output_token15,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    const fm_t* input_token8, const fm_t* input_token9, const fm_t* input_token10, const fm_t* input_token11,
    const fm_t* input_token12, const fm_t* input_token13, const fm_t* input_token14, const fm_t* input_token15,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// Effective-cache 16-core variant with large global-array buffers.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_16cores_cache_effective(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7, fm_t* output_token8,
    fm_t* output_token9, fm_t* output_token10, fm_t* output_token11, fm_t* output_token12,
    fm_t* output_token13, fm_t* output_token14, fm_t* output_token15,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    const fm_t* input_token8, const fm_t* input_token9, const fm_t* input_token10, const fm_t* input_token11,
    const fm_t* input_token12, const fm_t* input_token13, const fm_t* input_token14, const fm_t* input_token15,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

extern "C" void qkv_tile_kernel_cc_qwen_16cores_cache_bram_optimized(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7, fm_t* output_token8,
    fm_t* output_token9, fm_t* output_token10, fm_t* output_token11, fm_t* output_token12,
    fm_t* output_token13, fm_t* output_token14, fm_t* output_token15,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    const fm_t* input_token8, const fm_t* input_token9, const fm_t* input_token10, const fm_t* input_token11,
    const fm_t* input_token12, const fm_t* input_token13, const fm_t* input_token14, const fm_t* input_token15,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

extern "C" void qkv_tile_kernel_cc_qwen_16cores_cache_fixed(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7, fm_t* output_token8,
    fm_t* output_token9, fm_t* output_token10, fm_t* output_token11, fm_t* output_token12,
    fm_t* output_token13, fm_t* output_token14, fm_t* output_token15,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    const fm_t* input_token8, const fm_t* input_token9, const fm_t* input_token10, const fm_t* input_token11,
    const fm_t* input_token12, const fm_t* input_token13, const fm_t* input_token14, const fm_t* input_token15,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

extern "C" void qkv_tile_kernel_cc_qwen_16cores_cached(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7, fm_t* output_token8,
    fm_t* output_token9, fm_t* output_token10, fm_t* output_token11, fm_t* output_token12,
    fm_t* output_token13, fm_t* output_token14, fm_t* output_token15,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    const fm_t* input_token8, const fm_t* input_token9, const fm_t* input_token10, const fm_t* input_token11,
    const fm_t* input_token12, const fm_t* input_token13, const fm_t* input_token14, const fm_t* input_token15,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// BRAM-optimized 16-core V2 variant that reduces over-partitioning while retaining 4096 DSPs.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_16cores_bram_opt_v2(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7, fm_t* output_token8,
    fm_t* output_token9, fm_t* output_token10, fm_t* output_token11, fm_t* output_token12,
    fm_t* output_token13, fm_t* output_token14, fm_t* output_token15,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    const fm_t* input_token8, const fm_t* input_token9, const fm_t* input_token10, const fm_t* input_token11,
    const fm_t* input_token12, const fm_t* input_token13, const fm_t* input_token14, const fm_t* input_token15,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// BRAM-port-optimized 16-core variant with additional partitioning to eliminate II violations.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_16cores_bram_port_opt(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7, fm_t* output_token8,
    fm_t* output_token9, fm_t* output_token10, fm_t* output_token11, fm_t* output_token12,
    fm_t* output_token13, fm_t* output_token14, fm_t* output_token15,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    const fm_t* input_token8, const fm_t* input_token9, const fm_t* input_token10, const fm_t* input_token11,
    const fm_t* input_token12, const fm_t* input_token13, const fm_t* input_token14, const fm_t* input_token15,
    unsigned int token_count,
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15
);

// ============================================================================
// Compute/cache-decoupled architecture declaration.
// ============================================================================

extern "C" void qkv_tile_kernel_cc_qwen_16cores_cache_controller(
    fm_t* output_token0, fm_t* output_token1, fm_t* output_token2, fm_t* output_token3,
    fm_t* output_token4, fm_t* output_token5, fm_t* output_token6, fm_t* output_token7, fm_t* output_token8,
    fm_t* output_token9, fm_t* output_token10, fm_t* output_token11, fm_t* output_token12,
    fm_t* output_token13, fm_t* output_token14, fm_t* output_token15,
    const fm_t* input_token0, const fm_t* input_token1, const fm_t* input_token2, const fm_t* input_token3,
    const fm_t* input_token4, const fm_t* input_token5, const fm_t* input_token6, const fm_t* input_token7,
    const fm_t* input_token8, const fm_t* input_token9, const fm_t* input_token10, const fm_t* input_token11,
    const fm_t* input_token12, const fm_t* input_token13, const fm_t* input_token14, const fm_t* input_token15,
    unsigned int token_count,
    hls::stream<fm_t>& compute_input_stream,
    hls::stream<fm_accum_t>& compute_output_stream
);

extern "C" void qkv_tile_kernel_cc_qwen_16cores_compute_core(
    hls::stream<fm_t>& input_stream,
    hls::stream<fm_accum_t>& output_stream,
    const wt_block_t* weights,
    unsigned int num_iterations
);

extern "C" void qkv_tile_kernel_cc_qwen_16cores_compute_core_16x(
    hls::stream<fm_t> input_streams[16],
    hls::stream<fm_accum_t> output_streams[16],
    const wt_block_t* shard0, const wt_block_t* shard1, const wt_block_t* shard2,
    const wt_block_t* shard3, const wt_block_t* shard4, const wt_block_t* shard5,
    const wt_block_t* shard6, const wt_block_t* shard7, const wt_block_t* shard8,
    const wt_block_t* shard9, const wt_block_t* shard10, const wt_block_t* shard11,
    const wt_block_t* shard12, const wt_block_t* shard13, const wt_block_t* shard14,
    const wt_block_t* shard15,
    unsigned int num_iterations
);

#endif // LLM_FPGA_CC_KERNEL_QWEN_HPP
