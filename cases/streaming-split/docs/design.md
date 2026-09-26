# Streaming Split Design

[Case overview](../README.md) | [Implementation map](../../../docs/implementations.md) |
[Repository map](../../../docs/repository-map.md) | [Architecture](../../../docs/architecture.md) |
[Design space](../../../docs/design-space.md) | [Environment](../../../docs/environment.md)

## Overview

Streaming split is an alternative implementation family. A model-aware
control/cache kernel (`cc`) schedules a regular V8-2_s tile kernel through
AXI streams. It is a case-local design study, not a second name for the root
Fix16 resident implementation and not a replacement for its KV-cache runtime.

The public boundary is deliberately narrow: the sources exercise a bounded
operator program and accumulation path. They do not publish a complete model
graph, checkpoint accuracy, or end-to-end board throughput.

## Key design decisions

### 1. Fixed compute core (V8-2_s)

The compute source fixes the reduction and output shape instead of specializing
the datapath for each model projection:

| Parameter | Source value | Meaning |
| --- | ---: | --- |
| `INPUT_DIM` | 16 | Reduction elements per operation |
| `OUTPUT_DIM` | 64 | Output columns per operation |
| `NUM_CORES` | 2 | Independent compute cores |
| `NUM_LANES` | 1 | Lanes per core (`TOTAL_LANES=2`) |
| `NUM_TILES` | 16 | Input tile count |
| `WT_BLOCKS_PER_LANE` | 32 | 512-bit weight packets per lane |

Each operation performs a 16-by-64 fixed-point matrix product. The source
uses `ap_fixed<48,24>` for its internal accumulator and converts the result to
the 32-bit `fm_accum_t` output format after the reduction.

### 2. Control/cache core (`cc`)

`control_cache_core.cpp` is a three-process dataflow graph:

```
cc_dispatch -> cc_input_path -> V8-2_s
     |             |
     +--------> cc_output_path -> hidden_out
```

The `operator_program` is a flat six-word record per operation. Its fields
select activation/control flags, weight offset, input source and offset,
output destination, and output offset. An operation sequence can therefore
clear, accumulate, and finalize a large reduction without changing the tile
kernel shape. `SRC_PREV_GBUF` and `DST_GBUF_FEEDBACK` remain source-level
placeholders; this case does not claim a complete on-chip model schedule.

### 3. Weight multi-bank

Four independent weight memory ports (`HBM[2:5]`) feed four weight streams.
Each operation sends 16 weight blocks per port, so the source can present the
64-block V8-2_s weight slice in parallel. This is a connectivity and packet
organization choice local to this family; it is not evidence for the resident
family's HBM scheduler.

### 4. Packed stream interfaces

The packet widths are defined in `include/control_cache_packets.hpp`:

| Stream | Width | Contents |
| --- | ---: | --- |
| Input | 512 bits | Two lane-major copies of 16 `fm_t` values |
| Weight | 512 bits | 32 `wt_linear_t` values |
| Output low/high | 1024 bits each | 32 `fm_accum_t` values per stream |
| Control | 32 bits | One `op_ctrl` value per operation |

Output metadata is implicit in packet order. The two output streams are kept
at 1024 bits because the combined result and metadata packet would exceed the
stream-width limit used by this design.

## Numeric boundary

The formats below are taken from `include/kernel_cc_qwen.hpp` and the compute
kernel. They describe packet and accumulator representation, not an accuracy
claim:

| Type | Source format | Role |
| --- | --- | --- |
| `fm_t` | `ap_fixed<16,8>` | Activation/input packets |
| `wt_linear_t` | `ap_fixed<16,4>` | Weight packets |
| `fm_accum_t` | `ap_fixed<32,16>` | Output packets and host decoding |
| Internal matmul accumulator | `ap_fixed<48,24>` | Unsaturated reduction |

The activation helpers call `hls::recip` after converting a fixed-point value
to `float`. Thus the family has fixed-point storage and stream payloads, but
it is not an end-to-end equivalent of the root Fix16 numerical path.

## Performance and validation evidence

The case's evidence is intentionally scoped to the bounded design:

| Evidence level | Supports | Does not support |
| --- | --- | --- |
| CSim and software emulation | Packet ordering, operation-program routing, and finite accumulation checks | Model accuracy or resident-runtime behavior |
| HLS synthesis | Local schedule, resource, and timing estimates for `cc` and V8-2_s | Routed utilization, board frequency, or system throughput |
| Case emulation checks | The tested finite stream graph and deadlock-free termination | A complete Qwen model execution |
| Analytical composition | A projection from local operation schedules | Measured full-layer or board throughput |

Do not combine these rows with the resident family's Task 18/19/20,
KV-ownership, or Fix16 end-to-end evidence. The shared architectural idea is
streamed model-aware scheduling; the implementation and evidence boundaries
are different. See [`docs/experiments.md`](../../../docs/experiments.md) for
the repository-wide evidence hierarchy.

<a id="quick-start"></a>
## Reproduce

Use the canonical bounded build and software-emulation commands in the
[case overview](../README.md). The HLS-only entry points are:

```bash
cd cases/streaming-split
source ../../scripts/setup_environment.sh
../../scripts/check_environment.sh hls
vitis_hls -f tcl/run_v8_2_s_csynth.tcl
vitis_hls -f tcl/run_cc_csynth.tcl
```

The Vitis software-emulation launcher, host checks, and connectivity file are
also listed in the overview. Generated HLS projects, emulation binaries, and
traces are not public evidence artifacts.

## Files

| File | Role |
| --- | --- |
| `kernel/control_cache_core.cpp` | Control/cache dataflow orchestrator |
| `kernel/qkv_tile_kernel_cc_qwen_small_core_v8_2_s.cpp` | Fixed V8-2_s compute core |
| `include/control_cache_packets.hpp` | Dimensions, packet types, and operation ABI |
| `include/kernel_cc_qwen.hpp` | Fixed-point payload and accumulator types |
| `host/host_v8_2x2.cpp` | Seven-operation bounded host check |
| `host/host_accum.cpp` | Parameterized accumulation check |
| `conn_v8_2x2.cfg` | HBM and stream connectivity |
| `tcl/run_v8_2_s_csynth.tcl` | Compute-core HLS entry point |
| `tcl/run_cc_csynth.tcl` | Control/cache HLS entry point |
