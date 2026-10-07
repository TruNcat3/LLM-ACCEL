# Architecture

[Documentation index](README.md) | [Designs](README.md#designs) |
[Design space](design-space.md) | [Evaluation](experiments.md) |
[Source map](repository-map.md)

CoWave keeps model-aware scheduling at a controller boundary and regular
arithmetic in stream compute kernels. The public designs share that boundary
idea but do not share a source closure, numeric contract, or evidence claim.
This page describes the mechanisms common enough to compare; family details
live in the [design pages](README.md#designs).

## Family boundaries

| Canonical design | Source boundary | Compute and numeric boundary | Evidence boundary |
| --- | --- | --- | --- |
| [`cowave-fix16-2-8-64`](designs/fix16.md) | Root `kernel/`, `include/`, `host/`, `common/` | 2 stream compute CUs, logical 8 rows x 64 output columns; signed `ap_fixed`, Q2.14 attention probabilities | Bounded CSim/RTL CoSim/HLS and resident HW-Emu packages |
| [`cowave-streaming-split`](designs/streaming-split.md) | `cases/streaming-split/` | Independent `cc` plus fixed V8-2_s core; fixed-point packets with a float-assisted normalization helper | Case-local CSim/sw_emu/HLS and analytical composition |
| [`cowave-int4-4-8-128`](designs/quantized.md) | `cases/quantized-layer/` complete-layer source | W4A4 signed INT4; profile label counts 4 compute CUs, logical 8 rows, 128 output columns | Named full-layer source/evidence package; separate from component probes |
| [`cowave-int8-4-4-128`](designs/quantized.md) | `cases/quantized-layer/` complete-layer source | W8A8 signed INT8; profile label counts 4 compute CUs, logical 4 rows, 128 output columns | Named full-layer source/evidence package; separate from component probes |
| `cowave-quantized-blocks` | `cases/quantized-block/` isolated kernels | W4A4 8x64 and W8A8 4x128 component tiles | CSim/RTL CoSim/HLS/resource probes only |

The names count compute CUs and logical matrix dimensions only. A controller,
status sink, or other service kernel is not a CU in the name. DSP packing
changes physical resource accounting; it does not multiply the logical work
again. Existing `8x64` build paths and exported stream ABIs remain unchanged.

## Control and compute split

The common data path has three responsibilities:

1. A controller owns model semantics, external-memory policy, task order, and
   state that must survive a kernel call.
2. Stream compute kernels consume regular fixed-width packets and return
   bounded result streams. They do not decide layer order or HBM ownership.
3. Finite FIFOs make backpressure, drainage, and deadlock visible in CSim and
   RTL CoSim instead of hiding them in an unbounded test harness.

The root resident design uses one model-aware controller/cache kernel, two
regular compute CUs, and a status sink. The streaming split uses a different
`control_cache_core` and fixed V8-2_s service. Quantized kernels expose
controller-facing streams but do not own the root controller's HBM or KV
cache. These are independent designs, not numbered generations.

## Resident state and task flow

Only `cowave-fix16-2-8-64` currently publishes the production coarse-task
contract. Two HBM feature-buffer pairs form a ping-pong boundary:

```text
initial hidden --one H2D--> pair B

for each decoder layer:
  Task 18: pair B -> pair A   attention and residual
  Task 19: pair A -> pair B   FFN and residual

Task 20: pair B -> pair A     final RMSNorm
pair A --one D2H--> final hidden
```

Task 18 owns RMSNorm, Q/K/V, RoPE, KV append/read, tiled online attention, O
projection, and residual. Task 19 owns RMSNorm, Gate/Up, SiLU-Mul, Down, and
residual. Task 20 is model-level final normalization. The Host submits stable
operation, layer, position, query-row, and HBM-pair metadata; raw KV and
weight addresses remain controller state. Intermediate hidden tensors and KV
stay in controller-managed HBM.

The static descriptor program is an orchestration contract, not a new kernel
ABI. Detailed task counts, pair transitions, capacity guards, and result
boundaries are maintained in the [coarse-task runtime](coarse-task-runtime.md).

## Stream and packet boundaries

Each family uses explicit packet widths, but the ABIs are not interchangeable:

| Family | Representative packets | Ownership rule |
| --- | --- | --- |
| Fix16 resident | 160-bit task, 128-bit activation, 256-bit weight, 416-bit vector/result, 64-bit status | Compute CUs have no HBM master; controller drives task order |
| Streaming split | 512-bit input/weight packets, two 1,024-bit output halves, 32-bit control | `cc` owns operation order and four weight-port mapping |
| Quantized blocks | 128-bit task, 32-bit activation, 256-bit weight, 448/576-bit result | Scale fields and block metadata are transported; component probes do not apply scales |

The packet boundary limits control fan-out and makes stream drainage testable.
It does not remove reduction dependencies, placement pressure, or the need for
finite-buffer evidence. A source file under `cases/` is not implicitly part of
the root binary.

## Memory and scheduling

Resident hidden-state transfer and compute dispatch occur at block/wave
granularity. The tested Fix16 projection path overlaps a bounded sequence of
load, issue, compute, collect, and commit stages across waves:

```text
load -> issue/drive -> compute -> collect -> commit
```

FIFO depths, weight-load II, and cross-wave dataflow are configuration choices,
not universal architecture claims. A changed depth or overlap setting needs
finite-FIFO RTL CoSim and its own evidence identity. Prompt blocks and decode
descriptors remain ordered by the Host task program; a local cross-wave overlap
does not prove arbitrary cross-request overlap.

The streaming split has its own dispatch/input/output dataflow graph and a
fixed compute service. Quantized blocks are bounded stream kernels; their
replication and HBM connectivity are planning questions, not a linked runtime.

## Numeric representation

| Design | Payload and accumulator contract | Important distinction |
| --- | --- | --- |
| Fix16 resident | `ap_fixed<16,8>` activations, `ap_fixed<16,4>` weights, `ap_fixed<32,16>` accumulators, Q2.14 `ap_fixed<16,2>` probabilities | Fixed point, not IEEE FP16; controller owns online attention state |
| Streaming split | `ap_fixed<16,8>`, `<16,4>`, `<32,16>`, internal `<48,24>` | `hls::recip` follows a float conversion in normalization; not an end-to-end Fix16-equivalent path |
| W4A4 component | Signed `ap_int<4>` operands; source-selected 20/24-bit accumulator/output widths | Four INT4 products per DSP are physical packing; scale metadata is not applied |
| W8A8 component | Signed `ap_int<8>` operands; source-selected 28/32-bit accumulator/output widths | One INT8 product per DSP in the checked block; scale metadata is not applied |

The logical work denominator is declared from compute count and logical matrix
shape. DSP packing is reported separately. Vector operations, padding,
controller/status traffic, host work, and CPU-oracle work must not be silently
folded into a matrix-product peak.

## Attention and KV ownership

The resident controller computes attention tile by tile with running score
maximum, rescaled normalization sum, and rescaled PV accumulator. It does not
materialize the full score matrix in external memory. QK/PV arithmetic is
delegated to compute CUs; cache addressing, RoPE, and normalization state remain
controller responsibilities.

The Q2.14 report is an operator diagnostic with a Host-orchestrated boundary;
it is not a second resident implementation. The streaming split, quantized
full-layer profiles, and quantized component blocks do not publish this same
root controller-owned online-attention implementation.

## Evidence stages

```text
CSim -> RTL CoSim -> HLS CSynth -> Vitis HW-Emu -> physical implementation
```

- CSim checks arithmetic, routing, and packet semantics.
- Finite-FIFO RTL CoSim checks progress, feedback, and deadlock for named
  bounds.
- HLS CSynth reports local II, resource, and timing estimates; it is not
  place-and-route closure.
- Vitis HW-Emu checks a linked multi-kernel image and modeled CU intervals; it
  is not board latency, power, or routed frequency.
- A physical implementation needs a separate result package.

Use [experiments](experiments.md) for comparable claims and the family README
for scoped commands. Do not transfer resident results to streaming or
quantized sources merely because packet widths or matrix shapes look similar.
