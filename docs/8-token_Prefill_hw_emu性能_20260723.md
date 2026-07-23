# 8-token Prefill hw_emu 性能（2026-07-23）

## 测试配置

本次测试使用 Qwen 单层标准尺寸：

| 项目 | 值 |
| --- | ---: |
| hidden / intermediate | 2048 / 11008 |
| Q heads / KV heads / head dim | 16 / 2 / 128 |
| token block | 8 |
| compute | 2 × 8×64 MAC CU |
| 双 CU 峰值 | 1024 MAC/cycle |
| kernel clock | 200 MHz |
| weight FIFO depth / load II | 2 / 2 |
| wave result FIFO / cross-wave DATAFLOW | 33 / 1 |

随机 Fix16 输入和权重的 seed 为 `20260722`。测试不是缩小矩阵，而是完整执行
一个 8-token block 的 Q/K/V/O、Gate/Up/Down，并逐 token 执行 position 0..7
的 causal FlashAttention。

## 功能结果

以下路径全部通过：

- RMSNorm：16384 个值；
- Q：16384，K/V：各 2048；
- position 0..7 attention：每个 position 6 个 MM task、192 个 packet；
- O projection 和 attention residual：各 16384；
- Gate/Up/SiLU-mul：各 88064；
- Down 和最终 residual：各 16384。

总计完成 48 个 attention MM task、1536 个 packet，最终 hidden checksum 为
`0xb72a92cb5224f0c7`。仿真器正常退出，无 timeout 或 stream deadlock。

## 周期与利用率

`profile_kernels.csv` 中 controller、两个 compute CU 和 status kernel 的
Running Time 均为 3176.997 us。按 xclbin 中记录的 200 MHz kernel clock：

```text
active cycles = 3176.997 us × 200 MHz = 635399.4 cycles
```

有效 MAC 口径如下：

```text
Q/K/V/O + Gate/Up/Down = 616,562,688 MAC
valid causal QK + PV    =     147,456 MAC
total useful           = 616,710,144 MAC
```

因此：

| 指标 | 结果 |
| --- | ---: |
| active interval | 3.176997 ms |
| effective throughput | 194.12 GMAC/s |
| theoretical peak @ 200 MHz | 204.8 GMAC/s |
| useful-MAC array efficiency | 94.78% |
| amortized time/token/layer | 397.1 us |
| one-layer token throughput | 2518 token-layer/s |

向量操作会消耗 active cycles，但没有计入 useful MAC。Attention 只统计有效
causal 元素，没有把 64-entry tile padding 计为有效工作，因此上述利用率没有
通过虚增操作量获得。

## Batch 收益

同一 xclbin 的 2-token 回归为 2806.029 us，即 561,206 cycles、54.94 GMAC/s
和 26.82% 利用率。8-token 相比它：

- active interval 仅增加 13.2%；
- 有效吞吐提高 3.53 倍；
- 每 token 摊销时间降低 3.53 倍；
- 阵列利用率从 26.82% 提升到 94.78%。

结果证明当前 8-row 数据通路需要 M=8 才能完整填充。对 prefill 而言，块级
GBUF、II=2 weight producer 和跨 wave DATAFLOW 已经能够把大部分 load/drive/
collect 开销隐藏在计算之后。

## 结果边界

1. 这里的数值是 hw_emu kernel active interval，不是 XSim 在 CPU 上运行约
   4 小时的墙钟时间，也不是物理板卡实测。
2. `profile-prefill-block` 为方便逐阶段 golden check，当前仍由 Host 分算子
   下发；profile 不包含 Host 检查及部分 PCIe migration 时间。生产版本需要把
   同一序列固化为 controller 内一次整层任务。
3. Attention 只覆盖 position 0..7，均位于第一个 64-token tile。长 context
   会增加 QK/PV、KV HBM 和 online softmax 开销。
4. 若仅按当前单层 active cycles 外推 36 层，200 MHz 下约为 114.4 ms/8-token
   block，即约 69.9 prefill token/s；该估计不包含 embedding、final norm、
   LM head、sampling、层间 Host 开销和长 context 增量。

## HLS 资源与部署状态

该 hw_emu xclbin 使用 `CC8_RESIDENT_LAYER_ONLY=0`，以保留逐算子 profile 和
golden check 所需的所有诊断通路。修复后的 controller NK 顶层 HLS 报告为：

| 项目 | 结果 | U50 全器件占比 |
| --- | ---: | ---: |
| estimated period / Fmax | 3.839 ns / 260.48 MHz | - |
| BRAM18 | 1586 | 59% |
| DSP | 248 | 4% |
| FF | 2,405,836 | 137% |
| LUT | 1,021,326 | 117% |

因此当前结论应拆开理解：

- 多 token 计算、KV/attention 协议和流水重叠在 RTL hw_emu 中正确；
- 8-token shape 对双 8×64 CU 的利用率接近峰值；
- 诊断 controller 本身资源超限，尚不能通过 U50 真实布局布线。

资源收敛的目标不是改变已验证的数据通路，而是将 Host 分算子的 profile 流程
固化为 `CC8_RESIDENT_LAYER_ONLY=1` 风格的静态 prefill layer task，共享 GBUF，
并在编译期裁剪互斥操作入口。完成后必须重新执行 CSynth、deadlock-on CoSim、
hw_emu 和真实 `TARGET=hw` 实现。

## 验证链

多 token 修复后的验证顺序为：

```text
15-case CSim PASS
-> deadlock detection 开启的 C/RTL CoSim 13/13 PASS
-> controller CSynth/XO export PASS（260.48 MHz HLS estimate；诊断资源超限）
-> 2-token hw_emu PASS
-> 8-token 标准尺寸 hw_emu PASS
```

公开仓库不提交生成的 XO、xclbin、波形和多 GB 仿真目录，只保留实现、构建
入口和经过核对的结果摘要。
