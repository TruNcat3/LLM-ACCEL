# CoWave case index

[Repository](../README.md) | [Documentation](../docs/README.md) |
[Design catalog](../docs/implementations.md) | [Usage](../docs/usage.md) |
[Evidence](../results/README.md)

The root source is [`cowave-fix16-2-8-64`](../docs/designs/fix16.md). This
directory contains independent source boundaries: a streaming control/compute
split, quantized matrix components, and the public quantized full-layer source.
No case inherits the root controller's ABI, KV ownership, or evidence unless
its own README and result manifest say so.

| Canonical design | Case source | Purpose | Evidence boundary |
| --- | --- | --- | --- |
| [`cowave-streaming-split`](../docs/designs/streaming-split.md) | [`streaming-split/`](streaming-split/README.md) | Fixed V8-2_s compute service with a separate control/cache orchestrator | CSim/sw_emu, HLS, bounded emulation, and analytical composition |
| [`cowave-int4-4-8-128`](../docs/designs/quantized.md) | [`quantized-layer/`](quantized-layer/README.md) | Public W4A4 complete-layer controller/compute source | Its own source closure and separately identified result packages |
| [`cowave-int8-4-4-128`](../docs/designs/quantized.md) | [`quantized-layer/`](quantized-layer/README.md) | Public W8A8 complete-layer controller/compute source | Its own source closure and separately identified result packages |
| `cowave-quantized-blocks` | [`quantized-block/`](quantized-block/README.md) | Isolated W4A4/W8A8 matrix blocks, packet contracts, and resource planning | Component arithmetic, finite-stream progress, HLS estimates, and CU-capacity planning |

The canonical quantized names count compute CUs and logical rows/output
columns. They do not count controller or status kernels. DSP packing affects
physical DSP usage only; it must not be multiplied into logical work twice.
The `quantized-block` W4A4 source remains an 8x64 component and its W8A8
source remains a 4x128 component. Those source shapes and the profile labels
are related but not interchangeable.

## How to choose a case

- Choose `streaming-split` to study the independent control/cache and fixed
  compute dataflow. Its [case README](streaming-split/README.md) is the direct
  build and `sw_emu` route.
- Choose `quantized-block` to reproduce bounded W4A4/W8A8 arithmetic, stream
  packets, accumulator variants, or resource planning. Its source does not
  apply scales or implement a model-level runtime.
- Choose `quantized-layer` to reproduce the public full-layer source. Its
  [README](quantized-layer/README.md) owns the build commands, configuration
  labels, and source/evidence boundary. The September 30 archive is a frozen
  snapshot and must not be silently attributed to this source.

The [Vitis workflow tutorial](https://github.com/Reconfigurable-Computing/Vitis_workflow)
remains a general learning resource. Case READMEs retain their original
license and authorship context; this index only describes navigation and
scope.
