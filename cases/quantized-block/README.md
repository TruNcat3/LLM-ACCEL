# Quantized Block Candidates

[Repository](../../README.md) | [Design space](../../docs/design-space.md) |
[Environment](../../docs/environment.md) | [Usage](../../docs/usage.md)

This case contains isolated controller-facing matrix-multiply blocks for the
next quantized implementation. They use the same fixed-width, block-level
stream discipline as the resident R1 path, but they are not yet connected to
the Qwen controller, HBM scheduler, or the published end-to-end runtime.

## Candidates

| Candidate | Tile | Activation/weight packet | Logical products per K | Physical DSPs | Packing | HLS II/depth | Estimated Fmax | Status |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| W4A4 packed block, single 20-bit accumulator bank | 8x64 | 32-bit / 256-bit | 512 | 128 | 4 products/DSP | 1 / 5 | 440.53 MHz | CSim + RTL CoSim pass |
| W8A8 decode block, single 28-bit accumulator bank | 4x128 | 32-bit / four 256-bit streams | 512 | 512 | 1 product/DSP | 1 / 3 | 563.70 MHz | CSim + RTL CoSim pass |
| W8A4 packed reference (2-wave) | 8x64 | block-level | 512 | 128 | 2 products/DSP | 1 / 7 | 440.33 MHz | Existing reference |

The W4A4 design packs two signed INT4 activations and two signed INT4 weights
into one DSP multiply and extracts the four cross-products. The W8A8 design
uses a 4x128 rectangle to expose a decode-friendly output shape; INT8 products
are mapped one per DSP. Both selected kernels use one accumulator bank because
the integer feedback closes at II=1 without the four rotating banks used by the
floating-point pipeline. The W8A4 row is included to make the packing
distinction explicit: it is two products per DSP, not four.

## Interface contract

Each block starts with one 128-bit task word containing K length, output base,
block ID, end-of-stream state, and two 16-bit scale fields. Scales are carried
as metadata and are deliberately not applied in these arithmetic probes. The
controller remains responsible for quantization policy, zero points, HBM
traffic, and accumulation format selection.

W4A4 consumes one 32-bit activation word and one 256-bit weight word per K
step, then emits sixteen 24-bit accumulators plus metadata in a 448-bit packet.
W8A8 consumes one 32-bit activation word and four 256-bit weight words per K
step, then emits sixteen 32-bit accumulators plus metadata in a 576-bit packet.
Both kernels accept bounded K values and emit deterministic block metadata.

## CU replication planning

The tile shape and the number of replicated compute units are independent
parameters. The resource planner reserves the published R1 controller and
status-sink estimates, applies an 85% whole-U50 cap by default, and chooses the
largest count that fits all BRAM, DSP, FF, and LUT limits:

```bash
make quantized_cu_plan QUANT_KERNEL=w4a4 QUANT_KERNEL_COUNT=auto
make quantized_cu_plan QUANT_KERNEL=w8a8 QUANT_KERNEL_COUNT=auto

# Explore compute-only capacity or add a known controller stream limit.
QUANT_FIXED_PROFILE=none \
  make quantized_cu_plan QUANT_KERNEL=w4a4 QUANT_KERNEL_COUNT=auto
QUANT_INPUT_BITS_PER_CYCLE_CAP=2112 \
  make quantized_cu_plan QUANT_KERNEL=w8a8 QUANT_KERNEL_COUNT=auto
```

| Budget profile | Candidate | Selected CUs | Modeled total LUT | LUT use | Aggregate HLS roofline |
| --- | --- | ---: | ---: | ---: | ---: |
| Resident R1 plus status, 85% cap | W4A4 single-bank | 4 | 637,849 | 73.17% | 902.205 GMAC/s |
| Resident R1 plus status, 85% cap | W8A8 single-bank | 4 | 604,849 | 69.39% | 1,154.458 GMAC/s |
| Compute only, 85% cap | W4A4 single-bank | 4 (topology cap) | 175,880 | 20.18% | 902.205 GMAC/s |

The four-CU rows are resource sums, not a completed link. The fixed profile is
measured from a two-CU R1 controller; a four-CU integration still needs new
stream ports, dispatch logic, connectivity, placement, and routing closure.
The planner emits an `nk_line`, but every CU still requires a unique
task/activation/weight/output stream set and an output-block partition in the
controller. A requested count above the modeled limit is rejected. Run
`make test_quantized_cu_planner` to check this contract.

The earlier implementation used four rotating accumulator banks. Binding those
updates to DSP48 reduced LUT at the cost of four copies of the accumulator
datapath and thousands of extra DSPs. A shared four-bank mux also reduced LUT,
but produced a 90-stage high-fanout pipeline. The selected single-bank design
keeps the external packet widths while using verified 20-bit (W4A4) and 28-bit
(W8A8) internal accumulators for the bounded `K<=4096` range:

| Variant | Per-CU DSP | Per-CU FF | Per-CU LUT | Est. Fmax | Verification |
| --- | ---: | ---: | ---: | ---: | --- |
| W4A4 four-bank LUT baseline | 128 | 59,417 | 192,809 | 440.53 MHz | superseded |
| W4A4 four-bank DSP diagnostic | 2,176 | 59,417 | 129,321 | 440.53 MHz | pass, not selected |
| W4A4 single-bank narrow | 128 | 19,607 | 43,970 | 440.53 MHz | CSim + deadlock-on CoSim |
| W8A8 four-bank LUT baseline | 512 | 76,459 | 225,343 | 521.69 MHz | superseded |
| W8A8 four-bank DSP diagnostic | 2,560 | 68,091 | 128,975 | 528.23 MHz | pass, not selected |
| W8A8 single-bank narrow | 512 | 24,167 | 35,720 | 563.70 MHz | CSim + deadlock-on CoSim |

The selected single-bank variants are the planner defaults. Both pass CSim and
deadlock-enabled RTL CoSim with the same finite-stream testbench; no stream
depth or protocol relaxation was used. These remain pre-link estimates until
the controller stream replication and post-route timing are complete.

```bash
QUANT_ACCUM_IMPL=dsp \
  make quantized_cu_plan QUANT_KERNEL=w4a4 QUANT_KERNEL_COUNT=auto

QUANT_ACCUM_IMPL=dsp QUANT_NARROW_ACCUM=1 QUANT_RESOURCE_CAP_PCT=90 \
  make quantized_cu_plan QUANT_KERNEL=w8a8 QUANT_KERNEL_COUNT=auto
```

## Reproduce

The source is self-contained apart from the repository's HLS fixed-point type
definitions. Use the reference Vitis 2022.2 environment from
[`docs/environment.md`](../../docs/environment.md):

```bash
cd /path/to/LLM-ACCEL

# Set this only when the toolchain is not auto-discovered.
export VITIS_ENV_SCRIPT=/path/to/Vitis/2022.2/settings64.sh
source scripts/setup_environment.sh

# CSim, synthesis, and RTL CoSim (deadlock detection remains enabled).
scripts/run_vitis_hls.sh \
  cases/quantized-block/tcl/run_cosim_mm_stream_8x64_int4x4_block.tcl
scripts/run_vitis_hls.sh \
  cases/quantized-block/tcl/run_cosim_mm_stream_4x128_int8x8_block.tcl
```

The public checkout also provides a regression launcher for the baseline,
diagnostic DSP, and selected single-bank variants:
`scripts/run_quantized_block_regression.sh [csim|synth|cosim|all]`. To run only
the two optimized candidates, use the same Vitis 2022.2 environment and invoke
their dedicated Tcl entry points:

```bash
HLS_EXTRA_CFLAGS="-DMM_STREAM_QUANTIZED_SINGLE_ACCUM_BANK -DMM_STREAM_QUANTIZED_NARROW_ACCUM" \
  scripts/run_vitis_hls.sh \
  cases/quantized-block/tcl/run_cosim_mm_stream_8x64_int4x4_block_single_narrow.tcl

HLS_EXTRA_CFLAGS="-DMM_STREAM_QUANTIZED_SINGLE_ACCUM_BANK -DMM_STREAM_QUANTIZED_NARROW_ACCUM" \
  scripts/run_vitis_hls.sh \
  cases/quantized-block/tcl/run_cosim_mm_stream_4x128_int8x8_block_single_narrow.tcl
```

The public case intentionally does not store generated HLS projects, reports,
XOs, or simulator traces.

## Evidence boundary

The reported resource and Fmax values are Vitis HLS CSynth estimates for
`xcu50-fsvh2104-2-e` at a 3.333 ns target clock. The testbenches cover three
bounded K cases and exact integer accumulation/metadata checks. RTL CoSim was
run with finite streams and the common deadlock monitor enabled. There is no
HW Emu, routed system link, physical-board measurement, scale application,
checkpoint accuracy result, or full-model INT4/INT8 claim yet.

The next integration gate is to connect one candidate to the controller's
block loader, parameterize its stream ports by the selected CU count, and run
the same functional and residency checks used by R1. Resource/timing must then
be re-evaluated on the complete multi-kernel link.
