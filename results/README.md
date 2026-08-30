# Published Experimental Evidence

[Repository](../README.md) | [Documentation](../docs/README.md) |
[Experiments](../docs/experiments.md) | [Setup](../docs/environment.md) |
[License](../LICENSE)

This directory contains compact, versioned evidence packages for the claims in
the root README and the experimental report. Raw HW-Emu CU profiles, Host
excerpts, HLS reports, derived TSV rows, and SHA-256 manifests are kept
together so that a table can be audited without retaining a generated Vitis
project.

## Evidence index

| Design | Artifact | Workload | Evidence source | Timed scope | Primary supported claim |
| --- | --- | --- | --- | --- | --- |
| D1 | [`q214-pd-20260811/`](q214-pd-20260811/) | Standard Qwen layer, P/D contexts 64--1024 | Vitis 2022.2 HW Emu CU profiles | Host-orchestrated operator-level controller intervals | Context scaling and useful-MAC efficiency of the diagnostic datapath |
| P1 | [`coarse-task-20260816/`](coarse-task-20260816/) | Small two-layer Task-18/19/20 and serial P2/G3 | RTL CoSim, HW Emu, HLS CSynth | Common four-CU modeled interval | Cross-task/cross-layer HBM residency and controller-owned KV |
| P1 | [`block-prefill-20260817/`](block-prefill-20260817/) | Small P8, P16, P11 tail, and P8/G2 | RTL CoSim, HW Emu, HLS CSynth | Common four-CU modeled interval | 1--8-row block semantics, causal KV state, and finite-FIFO closure |
| R1 | [`q214-resident-fix-20260818/`](q214-resident-fix-20260818/) | Standard Qwen-shaped P8 Task 18 -> 19 -> 20 | Vitis 2022.2 HW Emu CU profile and fixed-point oracle | Common four-CU modeled interval | 16,384-value numerical closure with no intermediate Host copy |
| R1 | [`qwen3b-e2e-20260820/`](qwen3b-e2e-20260820/) | Standard-shape Qwen2.5-3B P8/G2/L1 composition | Vitis 2022.2 HW Emu CU profile, fixed-point oracle, and HLS CSynth | Common four-CU modeled interval | Six-task Prefill-plus-real-D1 closure: 4,096 values exact, 1,190,693 cycles, and 56.904% modeled useful-MAC efficiency |
| R1 | [`qwen3b-e2e-l2-20260821/`](qwen3b-e2e-l2-20260821/) | Standard-shape Qwen2.5-3B P8/G2/L2 composition | Vitis 2022.2 HW Emu CU profile, fixed-point oracle, and HLS CSynth | Common four-CU modeled interval | Ten-task cross-layer Prefill-plus-real-D1 closure: 4,096 values exact, 2,319,441.4 cycles, and 58.424% modeled useful-MAC efficiency |
| R1-D | [`qwen3b-checkpoint-20260830/`](qwen3b-checkpoint-20260830/) | Standard-shape Qwen2.5-3B P8 checkpoint localization | Vitis 2022.2 HW Emu with per-task Host readback | Numerical checkpoints only; no performance claim | Layers 0--2 bit-exact; first one-unit divergence at layer 3 Attention (22/16,384 values) |

The Qwen2.5-3B packages are bounded one- and two-layer generation-path gates
using deterministic random Fix16 weights and tied embeddings. Together they
prove P8/G2 task composition, cross-layer HBM residency, and the
Host/accelerator ownership boundary; they do not claim checkpoint accuracy, a
36-layer run, or physical-board performance. An in-progress run is never
represented as a published result.

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

Run the repository helper from the project root after the `hls` preflight.
The aggregate gate compiles Host-only contracts but launches neither synthesis
nor simulation:

```bash
source scripts/setup_environment.sh
scripts/check_environment.sh hls
make test_publication_release
```

For evidence checks on a machine without AMD/Xilinx tools, use
`scripts/check_environment.sh publication`, `make test_publication_tree`, and
`make verify_result_checksums` instead.

The verifier accepts both historical repository-root-relative manifests and
the archive-relative manifests emitted by the current atomic E2E archiver.
Raw Host logs, CU profiles, and numeric rows are not silently rewritten when
terminology is refined. A schema label may be clarified only when the artifact
README records the change, its complete checksum manifest is regenerated, and
the raw-to-derived-table verifier still reproduces every numeric value.

Unless an artifact states otherwise, documentation, figures, and experimental
evidence are licensed under
[CC BY-NC 4.0](../LICENSES/CC-BY-NC-4.0.md). Attribution should name Teng Wang
and LLM-ACCEL and identify any modifications.
