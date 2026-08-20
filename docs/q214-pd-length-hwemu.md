# Q2.14 Multi-Length Prefill/Decode Hardware-Emulation Results

## Scope

This experiment validates the Q2.14 online-softmax/FlashAttention path and the
complete Qwen2.5-3B decoder layer at four context lengths: 64, 256, 512, and
1024.  The Vitis 2022.2 hardware-emulation build runs at a modeled 200 MHz and
contains one controller, two 8x64 compute CUs, and one status sink.

- Prefill (`P`) evaluates the final 8-row query block ending at the stated context
  length.  For example, P1024 processes query positions 1016 through 1023
  against up to 1024 KV entries.
- Decode (`D`) evaluates one new token after the stated cached context.  For
  example, D1024 processes position 1024 against 1025 KV entries.
- Weights, input activations, and KV data are deterministic non-zero Fix16
  random data generated with seed `20260722`.
- The Attention probability representation is Q2.14.  Precision is checked
  both against the bit-accurate tiled Q2.14 model and against the high-precision
  reference.

The isolated xclbin is:

```text
vitis_8x64/build.qwen_layer_long.prefill_layer.d2.ii2.wr33.cw1.scratch.q214exp18.hw_emu.xilinx_u50_gen3x16_xdma_5_202210_1/qwen_8x64_dual.xclbin
```

## Performance

The host invokes the hardware operators that compose one decoder layer. The
reported interval is the sum of their controller-kernel active times, including
RMSNorm, Q/K/V projections, tiled Attention, O projection, residual,
Gate/Up/SiLU/Down FFN, and the final residual. Latency is computed at 200 MHz.
Modeled useful-MAC efficiency divides the shape-counted useful MACs by the
measured cycles and the peak of two 8x64 compute CUs, or 1,024 MAC/cycle. It is
not post-route physical utilization.

This is a **kernel-only, host-orchestrated layer profile**, not yet a single
autonomous controller launch. Host scheduling gaps, transfers between
operator calls, CPU RoPE/test-fixture packing, KV fixture migration, and CPU
golden checks are outside the reported interval. The dense and attention
arithmetic listed above executes in the FPGA kernels.

| Phase | Context | Active query rows / block | Cycles | Latency (ms) | Useful GMAC/s | Modeled useful-MAC efficiency | Target-block query-row/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| P | 64 | 8 | 637,103 | 3.1855 | 194.174 | 94.812% | 2,511.37 |
| P | 256 | 8 | 685,489 | 3.4274 | 182.304 | 89.016% | 2,334.10 |
| P | 512 | 8 | 749,407 | 3.7470 | 168.994 | 82.516% | 2,135.02 |
| P | 1024 | 8 | 877,846 | 4.3892 | 148.090 | 72.310% | 1,822.64 |
| D | 64 | 1 | 568,040 | 2.8402 | 27.229 | 13.296% | 352.09 |
| D | 256 | 1 | 590,794 | 2.9540 | 26.447 | 12.913% | 338.53 |
| D | 512 | 1 | 622,231 | 3.1112 | 25.448 | 12.426% | 321.42 |
| D | 1024 | 1 | 683,754 | 3.4188 | 23.771 | 11.607% | 292.50 |

The measured cycle growth is close to linear in the number of 64-entry KV
tiles.  A P tile costs approximately 16.0k additional cycles, while a D tile
costs approximately 7.7k cycles.  P remains efficient because eight query rows
fill the 8x64 datapath.  D uses only one query row, so its shape-normalized work
is valid but its full-array modeled utilization remains around 12%.

`Active query rows / block = 8` means eight consecutive rows of one sequence,
not an eight-sequence batch or eight decoded outputs; it also does not assert
that the complete prompt contains only eight tokens. For P1024, the measured
block contains positions 1016--1023 and reads a causal KV history of up to 1024
entries. A complete 1024-token prefill contains 128 such context-dependent
blocks. Consequently, `target-block query-row/s` is a one-layer block throughput,
not full-prompt or whole-model generation throughput.

The FPGA executes RMSNorm, all projections, QK/PV with online softmax,
residual operators, and the FFN. The host still performs operator sequencing,
RoPE, boundary packing, deterministic KV-prefix preload, and projection-to-KV
fixture migration. CPU golden computation is validation only and does not feed
the accelerator result. The coarse-task resident runtime reported separately
moves RoPE/KV ownership into the controller and removes intermediate hidden-
state host copies.

## Precision and Protocol Coverage

| Phase | Context | Q2.14 checks | Q2.14 failures | Max Q2.14 raw error | Reference checks | Reference failures | Max reference raw error |
|---|---:|---:|---:|---:|---:|---:|---:|
| P | 64 | 16 | 0 | 1 | 16 | 0 | 3 |
| P | 256 | 16 | 0 | 1 | 16 | 0 | 5 |
| P | 512 | 16 | 0 | 1 | 16 | 0 | 4 |
| P | 1024 | 16 | 0 | 1 | 16 | 0 | 5 |
| D | 64 | 2 | 0 | 1 | 2 | 0 | 2 |
| D | 256 | 2 | 0 | 1 | 2 | 0 | 3 |
| D | 512 | 2 | 0 | 0 | 2 | 0 | 2 |
| D | 1024 | 2 | 0 | 0 | 2 | 0 | 2 |

All eight launches returned exit status zero.  Q/K/V/O, Attention, FFN, both
residual paths, and the final hidden checksum passed in every case.  Attention
task and packet accounting also matched exactly:

| Phase | Context | Tiles | MM tasks | Result packets |
|---|---:|---:|---:|---:|
| P | 64 | 1 | 48/48 | 1,536/1,536 |
| P | 256 | 4 | 192/192 | 6,144/6,144 |
| P | 512 | 8 | 384/384 | 12,288/12,288 |
| P | 1024 | 16 | 768/768 | 24,576/24,576 |
| D | 64 | 2 | 96/96 | 3,072/3,072 |
| D | 256 | 5 | 240/240 | 7,680/7,680 |
| D | 512 | 9 | 432/432 | 13,824/13,824 |
| D | 1024 | 17 | 816/816 | 26,112/26,112 |

## Reproduction and Raw Evidence

```bash
scripts/launch_vitis_8x64_pd_sweep_tmux.sh \
  --prefix q214_pd_20260811_1405 \
  --build-dir vitis_8x64/build.qwen_layer_long.prefill_layer.d2.ii2.wr33.cw1.scratch.q214exp18.hw_emu.xilinx_u50_gen3x16_xdma_5_202210_1 \
  --profile qwen-layer-long \
  --seed 20260722
```

Published evidence:

- [`results/q214-pd-20260811/performance.tsv`](../results/q214-pd-20260811/performance.tsv)
- [`results/q214-pd-20260811/precision.tsv`](../results/q214-pd-20260811/precision.tsv)
- [`results/q214-pd-20260811/raw/`](../results/q214-pd-20260811/raw/)
  contains the eight authoritative `profile_kernels.csv` files and host logs.

The published `profile_kernels.csv` files are the authoritative source for the
active cycle counts.

## Conclusion

The Q2.14 blockwise online-softmax implementation is functionally stable for
both P and D through a 1024-token context.  The complete layer has no observed
stream deadlock, task-count mismatch, packet loss, or precision-gate failure.
The primary remaining performance limitation is the one-row Decode shape,
not the long-context Attention protocol.
