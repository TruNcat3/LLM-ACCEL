# Alternative implementations and compute studies

[Repository](../README.md) | [Implementation catalog](../docs/implementations.md) |
[Design space](../docs/design-space.md) | [Reproduction](../docs/usage.md)

The resident Fix16 system lives in the root source directories, described by
the [source map](../docs/repository-map.md#resident-source). This directory
contains two other public implementation families.

| Family | Research purpose | What is included | Evidence boundary |
| --- | --- | --- | --- |
| [Streaming split](streaming-split/README.md) | Explore a fixed compute core with a control/cache orchestrator | Case-local kernels, packets, Host tests, connectivity and HLS/software-emulation recipes | Operator/accumulation validation and analytical full-layer projections |
| [Quantized matrix blocks](quantized-block/README.md) | Study DSP packing, rectangular PE shapes and accumulator resource cost | W4A4/W8A8 kernels, bounded tests, HLS variants and CU-planning support | Component correctness, schedules and resource estimates |

Each case specifies its own task contract and parameters. Model-layer
scheduling or numerical correctness demonstrated by the root resident system
does not automatically apply to another case. Likewise, a matrix component's
II=1 or four-products-per-DSP result is not a measured full-layer throughput.

New implementation families should provide the same four entry points:
design boundary, source ownership, reproducible commands and scoped evidence.
Feature toggles and new workloads within one family belong to that family's
configuration/evidence tables, not a new numbered case.
