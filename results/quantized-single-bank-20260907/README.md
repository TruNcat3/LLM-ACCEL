# Quantized Single-Bank Accumulator Evidence

This package records the Q1 controller-facing INT4/INT8 block-kernel update
validated on 2026-09-07 with Vitis HLS 2022.2 for `xcu50-fsvh2104-2-e`.

The selected implementation removes the four rotating accumulator banks that
were inherited from the floating-point pipeline. The integer feedback path
closes at `II=1` with one LUT-based accumulator bank per logical output lane.
W4A4 retains four logical products per DSP through signed INT4 outer-product
packing; W8A8 retains the decode-oriented 4x128 tile and maps one INT8 product
per DSP. Internal widths are 20 bits for W4A4 and 28 bits for W8A8, sufficient
for the declared `K<=4096` range while preserving the external result packet.

Both selected kernels pass the three-case CSim and Verilog RTL CoSim tests with
the finite-stream deadlock monitor enabled. The largest observed HLS stream
depth is 128. CSynth estimates are summarized in `resource.tsv`.

The four-CU rows are resource sums, not a completed system implementation. They
reuse the published two-CU R1 controller/status estimate as a fixed proxy. A
four-CU Q1 system still requires controller stream replication, output-column
partitioning, HBM connectivity, HW Emu, and Vitis post-route timing closure.
No physical-board or full-model INT4/INT8 claim is made here.

Reproduce from a full development checkout with:

```bash
source /home/hepc/env/vitis_env_22.sh
make hls_csim_mm_stream_8x64_int4x4_block_single_narrow
make hls_csynth_mm_stream_8x64_int4x4_block_single_narrow
make hls_cosim_mm_stream_8x64_int4x4_block_single_narrow
make hls_csim_mm_stream_4x128_int8x8_block_single_narrow
make hls_csynth_mm_stream_4x128_int8x8_block_single_narrow
make hls_cosim_mm_stream_4x128_int8x8_block_single_narrow
```

The source and testbench are in [`cases/quantized-block/`](../../cases/quantized-block/).
