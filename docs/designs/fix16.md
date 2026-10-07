# `cowave-fix16-2-8-64`

[Design index](../README.md#designs) | [Architecture](../architecture.md) |
[Design space](../design-space.md) | [Reproduce](../reproduction-resident.md) |
[Evaluation](../experiments.md) | [Source map](../repository-map.md)

`cowave-fix16-2-8-64` is the canonical name for the root resident design.
The name describes two compute CUs, eight logical query rows, and 64 output
columns. The controller/cache kernel and completion sink are not counted as
compute CUs. Existing Makefile targets, kernel entry points, connectivity
files, result directories, and the exported stream ABI retain their historical
`8x64` paths for compatibility.

## Numeric contract

This design is fixed point, not IEEE FP16. The main payloads are signed
`ap_fixed` values (`ap_fixed<16,8>` activations, `ap_fixed<16,4>` linear
weights, and `ap_fixed<32,16>` accumulators); the online-attention probability
path uses Q2.14 (`ap_fixed<16,2>`). The name `Fix16` refers to these source
types and must not be expanded to "FP16".

The modeled logical matrix peak is `2 * 8 * 64 = 1,024` products per cycle.
Vector work, controller/status traffic, padding, and host work are separate
from this matrix numerator. A resource report is not a replacement for this
declared workload denominator.

## Boundary and ownership

- The controller owns HBM residency, task sequencing, RoPE, KV addressing,
  online attention state, and layer subgraphs.
- The two regular compute CUs consume fixed-width streams and do not own HBM.
- The status sink drains completion packets; it is not a third compute CU.
- Host code submits a static Task 18/19/20 descriptor program. Intermediate
  hidden tensors and KV state stay in controller-managed HBM.

The packet widths and task semantics are stable contracts. Read the
[coarse-task runtime](../coarse-task-runtime.md) for descriptor fields,
ping-pong transitions, timing boundaries, and acceptance gates.

## Source and reproduction

The implementation is the root `kernel/`, `include/`, `host/`, `common/`,
`tests/`, and build configuration. The [repository map](../repository-map.md)
identifies ownership without changing the historical path names.

The shortest bounded route is the [resident reproduction guide](../reproduction-resident.md).
It starts with CSim/finite-FIFO CoSim and then gives profile-matched HW-Emu
commands. Run the [environment preflight](../environment.md) first. No
physical board is required for CSim, HLS, CoSim, or HW-Emu.

## Evidence boundary

Published Fix16 packages cover bounded one- and two-layer workloads and
operator/protocol diagnostics. Their modeled CU intervals are not physical
board timing, power, or a trained-checkpoint accuracy claim. The [evaluation
report](../experiments.md) is the comparison entry; immutable raw evidence is
indexed by [`results/README.md`](../../results/README.md).

Historical labels such as `P8`, `G2`, and `D1` remain in result package paths
and diagnostic records. New reports should use explicit prompt rows, context,
sequence batch, decode forwards, and layer count; see the
[reference index](../reference.md).
