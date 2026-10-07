# CoWave documentation

[Repository](../README.md) | [Implementation catalog](implementations.md) |
[Evidence index](../results/README.md) | [Reference and history](reference.md)

This is the short route through the repository. Read the pages in order when
learning the design; use the family README or reproduction guide linked in the
same row when running it. The [implementation catalog](implementations.md) is
the authoritative source for names, configuration labels, legacy aliases, and
evidence stages.

Start with [Summary and example](summary-example.md) for a compact overview
and a publication-only validation example.

## Overview

The root [README](../README.md) gives the research question and the selected
results. The catalog separates an implementation family from a build
configuration, workload, and immutable result package. Existing source and
result paths remain valid compatibility identifiers.

## Architecture

[Architecture](architecture.md) explains the shared controller/stream-compute
mechanism, HBM/KV ownership, packet boundaries, backpressure, and numeric
differences. It does not repeat case-local commands or historical result
tables.

## Designs

These pages are the canonical design entry points. Each has a direct
reproduction link and states its source and evidence boundary:

| Design | Source and direct reproduction |
| --- | --- |
| [`cowave-fix16-2-8-64`](designs/fix16.md) | Root `kernel/`, `include/`, `host/`; [resident recipe](reproduction-resident.md) |
| [`cowave-streaming-split`](designs/streaming-split.md) | [`cases/streaming-split/`](../cases/streaming-split/); [case README](../cases/streaming-split/README.md) |
| [`cowave-int4-4-8-128`](designs/quantized.md) | W4A4 complete-layer source; [quantized-layer README](../cases/quantized-layer/README.md) |
| [`cowave-int8-4-4-128`](designs/quantized.md) | W8A8 complete-layer source; [quantized-layer README](../cases/quantized-layer/README.md) |
| `cowave-quantized-blocks` | Isolated W4A4/W8A8 matrix probes; [component README](../cases/quantized-block/README.md) |

The profile numbers count compute CUs and logical matrix dimensions. They do
not count controller/status kernels, and DSP packing never multiplies the
logical product count a second time. `Fix16` means signed `ap_fixed` source
types, not IEEE FP16. The root build and kernel ABI retain their historical
`8x64` paths.

## Design space

[Design space](design-space.md) records candidates and tradeoffs: partitioning,
array shape, packet granularity, attention state, scheduling, numeric formats,
and quantized packing. The page states what is selected, what remains a
component candidate, and which alternatives are not measured.

## Evaluation

[Experiments](experiments.md) is the release-first evidence map. It uses one
common vocabulary for prompt rows, context, layers, timing boundaries, logical
work, and modeled efficiency. Immutable raw packages are indexed by
[`results/README.md`](../results/README.md). The [dated quantized study](quantized-layer-progress.md)
keeps its September 30 snapshot identity separate from the public
`cases/quantized-layer/` source.

## Getting started

1. Run the [environment preflight](environment.md).
2. Pick a design above and follow its direct reproduction link, or use the
   [usage route map](usage.md) for root, streaming-split, and quantized paths.
3. Record the source/configuration identity, workload fields, timing boundary,
   and result package with any measurement.

The introductory [Vitis workflow tutorial](https://github.com/Reconfigurable-Computing/Vitis_workflow)
remains the learning resource for Kernel, Host, emulation, and build flow.
It is not a CoWave build dependency.

## Reference and history

[Reference and history](reference.md) points to the maintained coarse-task
contract, source map, diagnostics, resident command history, Q2.14 report,
long experiment record, and release workflow. Historical labels and numbered
sections remain only where a cited result or command needs them; new design
prose uses the catalog names and explicit workload fields.
