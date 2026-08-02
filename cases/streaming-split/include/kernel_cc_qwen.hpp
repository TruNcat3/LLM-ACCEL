#ifndef LLM_FPGA_CC_KERNEL_QWEN_HPP
#define LLM_FPGA_CC_KERNEL_QWEN_HPP

/*
============================================================================
LLM-FPGA-CC Qwen-Normal版本 - 正常Qwen尺寸 + 完整Pipeline优化
============================================================================

特性:
1. 正常Qwen尺寸 (HIDDEN_SIZE=2048)
2. 单层transformer layer测试
3. 完整五级流水线优化 (IF-ID-EX-MEM-WB)
4. 多级双缓冲 (ping-pong buffers)
5. 2-3轮attention tile换入换出测试
6. Prefill + Decode双场景

============================================================================
*/

#include "hardware.hpp"
#include "datatypes.hpp"
#include <hls_stream.h>

// ============================================================================
// 五级流水线数据抽象
// ============================================================================

// IF (Instruction Fetch) - 抽象token读取
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

// ID (Instruction Decode) - 抽象数据属性
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

// EX (Execute) - MAC计算
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

// MEM (Memory) - 访问双缓冲
struct mem_stage_t {
    fm_t results[32];
    bool buffer_select;  // ping-pong buffer选择
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

// WB (Write Back) - 结果写回
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
// 五级流水线核心 - 单token处理
// ============================================================================

struct five_stage_core_t {
    if_stage_t if_stage;
    id_stage_t id_stage;
    ex_stage_t ex_stage;
    mem_stage_t mem_stage;
    wb_stage_t wb_stage;

    // 流水线寄存器
    void pipeline_stages() {
        #pragma HLS pipeline II=1
        // Stage间寄存器由HLS自动推断
    }

    // 初始化所有stage
    void init() {
        #pragma HLS inline
        if_stage.init();
        id_stage.init();
        ex_stage.init();
        mem_stage.init();
        wb_stage.init();
    }

    // IF Stage: 读取token
    void if_stage_compute(unsigned int token_id, unsigned int tile_idx, const fm_t input[32]) {
        if_stage.set(token_id, tile_idx);

        // 传递到ID stage
        for (unsigned int i = 0; i < 32; i++) {
            #pragma HLS unroll
            id_stage.input_block[i] = input[i];
        }
        id_stage.output_channel = tile_idx % 32;
        id_stage.valid = true;
    }

    // ID Stage: 解码数据属性
    void id_stage_compute() {
        if (!id_stage.valid) return;

        // 数据属性抽象完成，传递到EX stage
        ex_stage.valid = true;
    }

    // EX Stage: MAC计算
    void ex_stage_compute(const wt_block_t& weights) {
        if (!ex_stage.valid) return;

        #pragma HLS pipeline II=1
        for (unsigned int i = 0; i < 32; i++) {
            #pragma HLS unroll
            fm_accum_t product = fm_accum_t(id_stage.input_block[i]) * fm_accum_t(weights[i]);
            ex_stage.accumulators[i] += product;
        }
    }

    // MEM Stage: 双缓冲访问
    void mem_stage_compute() {
        if (!ex_stage.valid) return;

        #pragma HLS pipeline II=1
        for (unsigned int i = 0; i < 32; i++) {
            #pragma HLS unroll
            mem_stage.results[i] = fm_t(ex_stage.accumulators[i]);
        }

        // 切换ping-pong buffer
        mem_stage.buffer_select = !mem_stage.buffer_select;
        mem_stage.valid = true;
    }

    // WB Stage: 写回结果
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

    // 完整流水线执行
    void execute_pipeline(
        unsigned int token_id,
        unsigned int tile_idx,
        const fm_t input[32],
        const wt_block_t& weights,
        unsigned int output_addr
    ) {
        // 五级流水线并行执行
        if_stage_compute(token_id, tile_idx, input);
        id_stage_compute();
        ex_stage_compute(weights);
        mem_stage_compute();
        wb_stage_compute(output_addr);
    }
};

// ============================================================================
// 多级双缓冲系统
// ============================================================================

struct multi_level_buffers_t {
    // Level 1: Input ping-pong buffer
    fm_t input_buffer[2][8][32];  // 2个buffer，每个8个lane

    // Level 2: Weight ping-pong buffer
    wt_block_t weight_buffer[2][16];  // 2个buffer，每个16个shards

    // Level 3: Output ping-pong buffer
    fm_t output_buffer[2][8][32];  // 2个buffer，每个8个lane

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
// Qwen-Normal版qkv_tile_kernel函数声明
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
// 16端口版本函数声明 - 专门用于16数据通道测试
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
// 五级流水线版本函数声明 - 计算核心与AXI接口完全分离
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
// Stream接口最终版本函数声明 - 基于用户关键洞察
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
// 独立Stream接口版本函数声明 - 基于用户关键洞察
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
// 完全并行树形累加器版本函数声明 - 真正的4096个DSP
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
// 优化架构版本函数声明 - 统一控制器管理规约
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
// 16核心版本函数声明 - 实现4096个DSP目标
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
// 16核心高带宽版本函数声明 - 大容量BRAM/URAM缓存
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
// 16核心高带宽版本V2函数声明 - 直接数组设计
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
// 16核心简化缓存版本函数声明 - 基于成功4096DSP架构 + BRAM缓存
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
// 16核心缓存优化版本函数声明 - 片上大规模双缓存系统
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
// 16核心有效缓存优化版本函数声明 - 全局数组大缓存
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
// 16核心BRAM优化V2版本函数声明 - 减少过度分区 + 保持4096 DSP
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
// 16核心BRAM端口优化版本函数声明 - 增加BRAM分区消除II违规
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
// 计算-缓存分离架构函数声明
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