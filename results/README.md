# Published Experimental Evidence

This directory contains compact, versioned evidence packages for the claims in
the root README and the experimental report. Raw HW-Emu CU profiles, Host
excerpts, HLS reports, derived TSV rows, and SHA-256 manifests are kept
together so that a table can be audited without retaining a generated Vitis
project.

## Evidence index

| Artifact | Workload | Evidence source | Timed scope | Primary supported claim |
| --- | --- | --- | --- | --- |
| [`q214-pd-20260811/`](q214-pd-20260811/) | Standard Qwen layer, P/D contexts 64--1024 | Vitis 2022.2 HW Emu CU profiles | Host-orchestrated operator-level controller intervals | Context scaling and useful-MAC efficiency of the diagnostic datapath |
| [`coarse-task-20260816/`](coarse-task-20260816/) | Small two-layer Task-18/19/20 and serial P2/G3 | RTL CoSim, HW Emu, HLS CSynth | Common four-CU modeled interval | Cross-task/cross-layer HBM residency and controller-owned KV |
| [`block-prefill-20260817/`](block-prefill-20260817/) | Small P8, P16, P11 tail, and P8/G2 | RTL CoSim, HW Emu, HLS CSynth | Common four-CU modeled interval | 1--8-row block semantics, causal KV state, and finite-FIFO closure |
| [`q214-resident-fix-20260818/`](q214-resident-fix-20260818/) | Standard Qwen-shaped P8 Task 18 -> 19 -> 20 | Vitis 2022.2 HW Emu CU profile and fixed-point oracle | Common four-CU modeled interval | 16,384-value numerical closure with no intermediate Host copy |
| [`qwen3b-e2e-20260820/`](qwen3b-e2e-20260820/) | Standard-shape Qwen2.5-3B P8/G2/L1 composition | Vitis 2022.2 HW Emu CU profile, fixed-point oracle, and HLS CSynth | Common four-CU modeled interval | Six-task Prefill-plus-real-D1 closure: 4,096 values exact, 1,190,693 cycles, and 56.904% modeled useful-MAC efficiency |

The Qwen2.5-3B package is a bounded one-layer generation-path gate using
deterministic random Fix16 weights and tied embeddings. It proves the six-task
P8/G2 composition and its Host/accelerator ownership boundary; it does not
claim checkpoint accuracy, a 36-layer run, or physical-board performance. An
in-progress run is never represented as a published result.

## Measurement policy

- `P8` means one sequence with eight active prefill query rows in one block.
  It is not batch eight and does not mean eight decoded outputs.
- HW-Emu CU Running Time is modeled RTL evidence, not XSim CPU wall time and
  not physical-board latency.
- HW-Emu CU intervals exclude Host embedding, LM-head, sampling, setup,
  weight preload, and post-inference CPU golden arithmetic. The common
  four-CU profiler field does not separately resolve inter-task issue gaps.
- A common four-CU interval does not resolve separable per-CU occupancy or
  inter-task issue gaps. Efficiency using this scope is labeled modeled
  useful-MAC efficiency.
- HLS CSynth tables are resource and local timing estimates. They are not
  post-route utilization or timing closure.
- CPU fixed-point oracles validate arithmetic after the inference boundary and
  are excluded from accelerator useful work.
- Random deterministic Fix16 weights validate shape, arithmetic, and protocol;
  they are not checkpoint-level model-accuracy evidence.

## Integrity

Run the repository helper from the project root:

```bash
make test_publication_release
```

The verifier accepts both historical repository-root-relative manifests and
the archive-relative manifests emitted by the current atomic E2E archiver.
Raw Host logs, CU profiles, and numeric rows are not silently rewritten when
terminology is refined. A schema label may be clarified only when the artifact
README records the change, its complete checksum manifest is regenerated, and
the raw-to-derived-table verifier still reproduces every numeric value.
