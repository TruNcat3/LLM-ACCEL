# Case 2: Streaming Split Architecture (cc + V8-2_s)

## Overview

A **separated streaming architecture** where a model-aware control/cache
core (`control_cache_core`, "cc") orchestrates a fixed compute core
(`qkv_tile_kernel_cc_qwen_small_core_v8_2_s`, "V8-2_s") through packed
AXI streams. The compute core is a regular tile processor that knows nothing
about model semantics; all dimension scaling, op sequencing, and data
movement are handled by cc's `operator_program`.

This case demonstrates how a **fixed compute core** can serve arbitrary LLM
dimensions (Qwen hidden=2048) through an `operator_program` schedule of
accumulate ops, without redesigning the datapath for each projection size.

## Key design decisions

### 1. Fixed compute core (V8-2_s)

| Parameter | Value | Rationale |
| --- | --- | --- |
| `INPUT_DIM` | 16 | Reduction depth per op (balance: 4 too many ops, 32 too few lanes) |
| `OUTPUT_DIM` | 64 | Output columns per matmul instance |
| `NUM_CORES` | 2 | Independent core pairs sharing one activation |
| `NUM_LANES` | 1 | Lanes per core (DSP budget: 2×1×16×64 = 2048) |
| `NUM_TILES` | 16 | Tile depth (group = 2×16 = 32 = Q 2048/64, 100% output util) |
| `WT_BLOCKS_PER_LANE` | 32 | Weight blocks per lane per op |

**Key properties:**
- `matmul_16x64_tpl`: 16-input × 64-output MAC, 1024 DSP, II=1 pipeline
- 2 cores (2048 DSP total), shared activation block
- `acc` uses `ap_fixed<48,24>` (P2 wide accumulator): eliminates per-step
  saturation, single round/sat at output
- `LANE_MM_stream` macro: always `+=` (no branch), accumulate via caller-side
  clear — avoids HLS pipeline scheduling explosion from conditional branches
- `compute_stream_core`: fused read+compute (no `local_input_buffer`),
  eliminates read/write port conflict on shared BRAM

### 2. Control/cache core (cc) — dataflow orchestrator

cc is a **3-process dataflow DAG** connected by internal `hls::stream`:

```
cc_dispatch (op_program m_axi → iparam/oparam streams)
    ├── cc_input_path (hidden_in + 4×weight_hbm → input/weight/ctrl streams)
    └── cc_output_path (collect → store_out 512-bit packed → hidden_out)
```

**`operator_program` format** (flat uint32 array, 6 fields per op):

| Field | Purpose |
| --- | --- |
| `op_ctrl` | Activation type (NONE/RELU/GELU/SILU/SOFTMAX/LAYERNORM) + ACCUMULATE/FINALIZE flags |
| `weight_offset` | This op's weight slice in `weight_hbm` (wt_block granularity) |
| `input_source` | SRC_HBM (reload) or reuse `gbuf_in` |
| `input_hbm_offset` | Hidden input element offset |
| `output_dest` | DST_HBM or DST_GBUF_FEEDBACK |
| `output_hbm_offset` | Output element offset (fm_accum_t granularity) |

**Accumulate sequence** (large-dimension reduction):
```
op0:  ctrl=0x00  (clear + compute, weight slice 0)
op1:  ctrl=0x40  (accumulate, weight slice 1)
...
opN:  ctrl=0xC0  (accumulate + finalize, output)
```
Total reduction = `num_ops × INPUT_DIM`. Verified: 512 op → reduction 2048
(sw_emu `sample=2048 ✅`).

### 3. Weight multi-bank (4 PC parallel)

4 independent `weight_hbm` m_axi ports (HBM[2:5]), each reading
`WT_PER_V82/4 = 16` wt_blocks per op in parallel. cc sends 4 weight streams
to V8-2_s, which loads them in 16 cycles (vs 66 cycles single-stream,
**3.67× speedup**).

```
conn_v8_2x2.cfg:
  sp=cc_0.weight_hbm_0:HBM[2]
  sp=cc_0.weight_hbm_1:HBM[3]
  sp=cc_0.weight_hbm_2:HBM[4]
  sp=cc_0.weight_hbm_3:HBM[5]
  stream_connect=cc_0.weight_stream_0:v82_0.weight_stream_0
  stream_connect=cc_0.weight_stream_1:v82_0.weight_stream_1
  stream_connect=cc_0.weight_stream_2:v82_0.weight_stream_2
  stream_connect=cc_0.weight_stream_3:v82_0.weight_stream_3
```

### 4. 512-bit packed I/O

- **Input packet** (`in_pkt_axis` = `ap_axiu<512>`): 1 tile × 2 lanes ×
  16 inputs, lane-major, zero metadata. `read` 128→16 cycles (8×).
- **Output** (`hidden_out` = `ap_uint<512>*`): 16 fm_accum_t per 512-bit word,
  `store_out` 8192→512 cycles (16×).
- **Weight** (`weight_axis` = `ap_axiu<512>`): 1 wt_block (32 × 16-bit) per
  packet.

## Performance and validation evidence

### Single-op HLS schedule (300 MHz target)

| Stage | Cycles | Notes |
| --- | --- | --- |
| `load_weights_stream` (4 parallel) | 18 | 4 streams × 16 blocks, II=1 |
| `compute_stream` (II=2) | 36 | 16 tiles, 2 cores, mm_results RMW port limit |
| `activate_tiles_core` | 187 | layernorm/gelu/silu, II=1 |
| `write_stream` | ~128 | 2 lanes × 16 tiles × 512-bit |
| **accumulate op** | **54** | load_w + compute |
| **FINALIZE op** | **~369** | + activate + write |

### Analytical full-layer projection (Qwen-3B, hidden=2048)

| Metric | Value |
| --- | --- |
| Total ops/layer | ~3140 |
| Layer latency | ~178K cyc ≈ 0.59 ms |
| 36-layer decode | ~21 ms/token ≈ **50 token/s** |
| vs D=4 original | **4.3× speedup** |

### HLS resource estimate (xcu50)

| Resource | Approximate usage | Fraction of full xcu50 |
| --- | ---: | ---: |
| DSP | 2,838 | 47.7% |
| FF | 676K | 38.8% |
| LUT | 342K | 39.3% |
| BRAM18 | 256 | 9.5% |

Percentages use the full xcu50 capacities (5,952 DSP, 1,743K FF, 871K LUT,
and 2,688 BRAM18). The figures are pre-route HLS estimates; they are not a
placed-and-routed utilization report. Likewise, the 36-layer row above is an
analytical composition of the single-op schedule rather than a measured
end-to-end HW-Emu or board throughput result.

### Validation

| Level | Status |
| --- | --- |
| csynth | ✅ load 18cyc, compute II=2, activate II=1, All constraints |
| sw_emu | ✅ 7-op miss=0/2048, 16-chunk sample=256 |
| hw_emu | ✅ 0% stall, no deadlock, miss=0 |

## Optimization history

The path from D=4 single-stream to D=16 + 4-PC:

| Step | Change | Effect |
| --- | --- | --- |
| store_out 512-bit | 32-bit serial → 512-bit packed | 16× (8192→512cyc) |
| cc dataflow | Sequential op_loop → 3-process DAG | store/send overlap |
| compute streaming | Delete local_input_buffer, fuse read+compute | Remove port conflict |
| mm_results cyclic64 | cyclic 32→64 | II 4→2 |
| param local | Kernel-top load param→register | activate II 67→1 |
| clear parallel | c,l unroll + i unroll64 | 458× (8192→18cyc) |
| activate cyclic64 | local_output cyclic 32→64 | II 2→1 |
| hls::recip | ap_fixed / → float recip IP | ~15cyc saved |
| **INPUT_DIM=16** | 4→16, lanes 4→1 | op count 4×, output 100% |
| **weight 4 PC** | 1→4 m_axi + 4 streams | load 66→18cyc (3.67×) |

## Files

| File | Description |
| --- | --- |
| `kernel/control_cache_core.cpp` | cc dataflow orchestrator (dispatch + input_path + output_path) |
| `kernel/qkv_tile_kernel_cc_qwen_small_core_v8_2_s.cpp` | V8-2_s fixed compute core (streaming matmul + activate + write) |
| `include/control_cache_packets.hpp` | Dimension constants, packet types, pack/unpack, op_task_t |
| `include/kernel_cc_qwen.hpp` | fm_t, fm_accum_t, wt_block_t definitions |
| `host/host_v8_2x2.cpp` | 7-op integrated layer host (Q/K/V/O/Gate/Up/Down) |
| `host/host_accum.cpp` | Parameterized accumulate host (variable reduction depth) |
| `conn_v8_2x2.cfg` | Vitis connectivity (7 m_axi sp + 8 stream_connect) |
| `tcl/run_v8_2_s_csynth.tcl` | V8-2_s HLS synthesis script |
| `tcl/run_cc_csynth.tcl` | cc HLS synthesis script |
