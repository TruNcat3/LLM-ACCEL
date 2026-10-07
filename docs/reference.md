# Reference and history

[Documentation index](README.md) | [Catalog](implementations.md) |
[Evaluation](experiments.md) | [Repository map](repository-map.md)

Use this page after the short Overview, Architecture, Designs, Design Space,
Evaluation, and Getting Started path. It keeps compatibility identifiers and
long diagnostic material available without presenting them as current design
names.

## Stable contracts

- [Controller-resident runtime](coarse-task-runtime.md): Task 18/19/20,
  HBM ping-pong, descriptor fields, timing, and acceptance gates.
- [Repository and source map](repository-map.md): module ownership, build
  inputs, case boundaries, and script families.
- [Case index](../cases/README.md): independent streaming-split,
  quantized-blocks, and complete-layer quantized source boundaries.
- [Release workflow](release-workflow.md): source/evidence identity,
  immutable result packages, and publication checks.

## Diagnostics and reproduction detail

- [Fix16 resident reproduction](reproduction-resident.md): complete root
  build, HW-Emu, archive, and checkpoint command history.
- [Diagnostics](reproduction-diagnostics.md): Q2.14 sweep, package
  verification, checksum checks, and result-reading rules.
- [Q2.14 report](q214-pd-length-hwemu.md): dated operator diagnostic with
  local `P<n>`/`D<n>` notation.

## Historical records

- [Detailed experiment record](experiment-details.md): original result tables,
  dated resource qualifications, and historical interpretations.
- [Coarse-task runtime history](coarse-task-runtime-history.md): relocated
  command blocks and result anchors retained for cited links.
- [September 30 quantized study](quantized-layer-progress.md): frozen W4A4
  snapshot interpretation and its separate public source boundary.

Historical result directories under [`results/`](../results/README.md) are
immutable. A legacy path or label may remain necessary to reproduce an old
package; it is not evidence that the old label is a current implementation
family. New prose uses the canonical names in the implementation catalog and
explicit workload fields instead of numbered generations.
