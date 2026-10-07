# Quantized Matrix Blocks

[Repository](../../README.md) | [Implementation map](../../docs/implementations.md) |
[Repository map](../../docs/repository-map.md) | [Design space](../../docs/design-space.md) |
[Environment](../../docs/environment.md) | [Usage](../../docs/usage.md)

This case is `cowave-quantized-blocks`, a component study. It contains isolated
controller-facing integer matrix blocks for resource and protocol study. The
blocks are not connected to the Fix16 resident controller, its HBM scheduler,
or its end-to-end runtime.

## Scope and candidates

The public sources contain two bounded kernels:

| Candidate | Shape | Operands | Products per K | Physical DSPs | Public status |
| --- | ---: | --- | ---: | ---: | --- |
| W4A4 packed block | 8x64 | Signed INT4 activations and weights | 512 | 128 | Public kernel and bounded CSim/RTL CoSim |
| W8A8 decode block | 4x128 | Signed INT8 activations and weights | 512 | 512 | Public kernel and bounded CSim/RTL CoSim |
| W8A4 packed row | 8x64 reference shape | Mixed INT8/INT4 | 512 | - | Historical external reference only; no public kernel in this case |

The W4A4 source packs four signed INT4 products into each physical DSP
multiply. W8A8 exposes four output groups of 32 weights across four weight
streams and maps one signed INT8 product per DSP. The W8A4 row is retained only
to explain a historical packing comparison; it has no source, testbench,
reproduction command, or release evidence here and must not be read as a third
candidate.

## Numeric and packet contract

The task and stream layouts are defined in the case headers:

| Field or stream | W4A4 | W8A8 |
| --- | --- | --- |
| Task word | 128 bits | 128 bits |
| Activation word | 32 bits, eight signed INT4 values | 32 bits, four signed INT8 values |
| Weight word(s) | One 256-bit word | Four 256-bit words |
| Output word | 448 bits, 24-bit output values plus metadata | 576 bits, 32-bit output values plus metadata |
| Selected internal accumulator | `ap_int<20>` | `ap_int<28>` |

The task word carries `k_count`, output element base, block ID, end-of-stream
state, and two 16-bit scale fields. The scale fields are metadata in these
probes; the kernels deliberately do not apply scales or zero points. The
selected single-bank variants use one accumulator bank and close the integer
feedback loop at `II=1`. The output widths remain 24 bits (W4A4) and 32 bits
(W8A8), respectively.

The source declares an HLS/test `K` range up to 4096 and the testbenches
exercise finite accumulation and metadata cases. These are block-level
arithmetic and stream contracts, not a quantization policy or model-accuracy
result.

## Replication planning

`scripts/plan_quantized_cus.sh` can sum per-CU HLS estimates and emit an
`nk_line` under a device resource cap:

```bash
make test_quantized_cu_planner
make quantized_cu_plan QUANT_KERNEL=w4a4 QUANT_KERNEL_COUNT=auto
make quantized_cu_plan QUANT_KERNEL=w8a8 QUANT_KERNEL_COUNT=auto

# Compute-only planning, without the fixed resident resource proxy.
QUANT_FIXED_PROFILE=none \
  make quantized_cu_plan QUANT_KERNEL=w4a4 QUANT_KERNEL_COUNT=auto
```

The planner output is a capacity model. A four-CU row is a resource sum, not a
linked design: each CU still needs controller task, activation, weight, and
output streams, output partitioning, connectivity, placement, and route
closure. Do not combine planner rooflines or HLS estimates with resident
performance measurements.

## Reproduce

Use the repository environment contract before launching Vitis HLS:

```bash
source scripts/setup_environment.sh
scripts/check_environment.sh hls

# Run the bounded candidates and their accumulator variants.
scripts/run_quantized_block_regression.sh csim
scripts/run_quantized_block_regression.sh synth
scripts/run_quantized_block_regression.sh cosim
```

The regression script selects the case-local Tcl files and enables the
deadlock monitor for RTL CoSim. It writes generated projects, logs, and traces
outside the published evidence tree. The exact validated single-bank summary
is recorded in [`quantized-single-bank-20260907`](../../results/quantized-single-bank-20260907/).

## Evidence boundary

The public package supports the following claims:

| Evidence | Supports | Does not support |
| --- | --- | --- |
| CSim | Integer arithmetic, packet layout, and metadata semantics for finite K | Controller integration or model accuracy |
| Deadlock-enabled RTL CoSim | Finite-stream completion for the tested W4A4/W8A8 variants | Routed system behavior or board timing |
| HLS CSynth | Local II, accumulator/resource, and timing estimates | Post-route Fmax, device utilization, or throughput |
| CU planner | Resource-capacity and replication arithmetic | A complete multi-kernel link |

This matrix-block case has no full-layer HW-Emu or physical-board result,
scale application, checkpoint accuracy result, or full-model INT4/INT8 claim.
The separate [full-layer development study](../../docs/quantized-layer-progress.md)
publishes W4A4 controller/compute measurements and replayable analysis; that
system's hardware source/build closure is not contained in this case. The
current source boundary here remains the two kernels and headers in this
directory; see the [implementation map](../../docs/implementations.md).

## Source map

| File | Role |
| --- | --- |
| `kernel/mm_stream_8x64_int4x4_block.cpp` | W4A4 packed block kernel |
| `kernel/mm_stream_4x128_int8x8_block.cpp` | W8A8 four-stream block kernel |
| `include/mm_stream_8x64_int4x4_block.hpp` | W4A4 dimensions and packet widths |
| `include/mm_stream_4x128_int8x8_block.hpp` | W8A8 dimensions and packet widths |
| `include/mm_stream_quantized_nk.hpp` | Shared task word and scale metadata |
| `tests/mm_stream_8x64_int4x4_block_tb.cpp` | W4A4 bounded testbench |
| `tests/mm_stream_4x128_int8x8_block_tb.cpp` | W8A8 bounded testbench |
| `tcl/run_cosim_mm_stream_8x64_int4x4_block*.tcl` | W4A4 HLS entry points |
| `tcl/run_cosim_mm_stream_4x128_int8x8_block*.tcl` | W8A8 HLS entry points |
