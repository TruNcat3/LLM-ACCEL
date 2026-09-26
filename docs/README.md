# CoWave documentation

[Repository](../README.md) | [Implementation map](implementations.md) |
[Repository map](repository-map.md) | [Evidence](../results/README.md)

The documentation has four layers: the root explains the research; design
pages explain the architecture; reproduction pages specify executable flows;
result packages preserve evidence. Implementation families, build profiles and
test workloads are separate concepts.

## Read the research

| Read in this order | Question answered |
| --- | --- |
| [Implementation catalog](implementations.md) | Which families exist, which profiles belong to them, and what do old names mean? |
| [Architecture](architecture.md) | What is the common idea, and how does the resident implementation realize it? |
| [Design space](design-space.md) | What alternatives exist for each compute, storage and scheduling decision? |
| [Coarse-task runtime](coarse-task-runtime.md) | How do Host requests compose resident layer subgraphs and KV state? |
| [Experiments](experiments.md) | What is demonstrated, at what dimensions and timing boundary? |

## Find and reproduce an implementation

| Family / purpose | Source orientation | Execution entry |
| --- | --- | --- |
| Fix16 resident | [Root source map](repository-map.md#resident-source) | [Resident reproduction](reproduction-resident.md) |
| Streaming split | [Case overview](../cases/streaming-split/README.md) | Case-local reproduction instructions |
| Quantized matrix blocks | [Case overview](../cases/quantized-block/README.md) | Matrix regression and resource-planning instructions |
| Operator / precision diagnostics | [Context-sweep definition](q214-pd-length-hwemu.md) | [Diagnostic recipes](reproduction-diagnostics.md) |
| Small-shape protocol tests | [Runtime validation](coarse-task-runtime.md) | [Bounded reproduction](reproduction-resident.md) |

Begin with [environment setup](environment.md). Each recipe identifies its
source/profile and test scope. The publication-only checks inspect archived
evidence; HLS/RTL and HW Emu reproduction require the vendor toolchain.

## Read the evidence

[results/README.md](../results/README.md) indexes every immutable package.
The [experimental report](experiments.md) explains which measurements can be
compared and links to detailed tables. Historical and diagnostic data remain
available in the [detailed experimental record](experiment-details.md) and
[runtime history](coarse-task-runtime-history.md).

A workload needs explicit prompt tokens, active query rows, sequence batch,
layers and KV context. HW-Emu cycles differ from simulator wall time; modeled
useful-MAC efficiency differs from PE occupancy. See the
[catalog's notation](implementations.md#workload-notation) before comparing
old P/D labels.

## Maintain the repository

- [Repository map](repository-map.md): module ownership, source locations,
  build configuration and script categories.
- [Release workflow](release-workflow.md): synchronize a reviewed development
  snapshot, validate its source/evidence relationship, and publish a coherent
  public update.
- [Citation](../CITATION.cff) and [licenses](../LICENSE): authorship and reuse.

Update stable design contracts when the implementation changes; keep run-local
status in experiment evidence. Do not use a dated “next step” as the public
definition of a design. Setup and commands belong in reproduction pages, and
a measurement belongs to one explicitly identified experiment.
