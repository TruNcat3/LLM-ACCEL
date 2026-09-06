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
| W4A4 packed block | 8x64 | 32-bit / 256-bit | 512 | 128 | 4 products/DSP | 1 / 5 | 440.53 MHz | CSim + RTL CoSim pass |
| W8A8 decode block | 4x128 | 32-bit / four 256-bit streams | 512 | 512 | 1 product/DSP | 1 / 3 | 521.69 MHz | CSim + RTL CoSim pass |
| W8A4 packed reference (2-wave) | 8x64 | block-level | 512 | 128 | 2 products/DSP | 1 / 7 | 440.33 MHz | Existing reference |

The W4A4 design packs two signed INT4 activations and two signed INT4 weights
into one DSP multiply and extracts the four cross-products. The W8A8 design
uses a 4x128 rectangle to expose a decode-friendly output shape; INT8 products
are mapped one per DSP. The W8A4 row is included to make the packing distinction
explicit: it is two products per DSP, not four.

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

For an existing development checkout, the combined regression launcher is
`scripts/run_quantized_block_regression.sh [csim|synth|cosim|all]`. The public
case intentionally does not store generated HLS projects, reports, XOs, or
simulator traces.

## Evidence boundary

The reported resource and Fmax values are Vitis HLS CSynth estimates for
`xcu50-fsvh2104-2-e` at a 3.333 ns target clock. The testbenches cover three
bounded K cases and exact integer accumulation/metadata checks. RTL CoSim was
run with finite streams and the common deadlock monitor enabled. There is no
HW Emu, routed system link, physical-board measurement, scale application,
checkpoint accuracy result, or full-model INT4/INT8 claim yet.

The next integration gate is to connect one candidate to the controller's
block loader and run the same functional and residency checks used by R1,
followed by a resource/timing re-evaluation of the complete multi-kernel link.
