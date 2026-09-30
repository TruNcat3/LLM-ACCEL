# Published Experimental Evidence

[Repository](../README.md) | [Documentation](../docs/README.md) |
[Implementation map](../docs/implementations.md) |
[Experiments](../docs/experiments.md) | [Setup](../docs/environment.md) |
[License](../LICENSE)

This directory contains nine immutable evidence packages. Each package keeps
the evidence appropriate to its stage, such as HW-Emu profiles and Host
excerpts where applicable, HLS/RTL reports, derived rows, source identity, and
the SHA-256 manifest needed to audit its claim without retaining a generated
Vitis project. Package paths and machine-facing labels are preserved;
the public family/scope names below come from the [implementation map](../docs/implementations.md).

## Evidence index

The table separates the quantized development comparison, released Fix16
resident mainline, diagnostics, protocol tests and component evidence.
“Common four-CU interval” means the same run-local profiler Running
Time for controller, both compute CUs, and status sink; it does not resolve
per-CU occupancy or Host gaps.
The W4 full-layer development archive instead measures each phase from the
earliest start through the latest end across its controller and four compute
CUs, using RTL pin transitions. Its hardware source release is pending.

| Package | Hardware family | Evidence scope | Workload | Evidence source and measured boundary | Supported claim |
| --- | --- | --- | --- | --- | --- |
| [`quantized-layer-w4-20260930/`](quantized-layer-w4-20260930/) | Development quantized full-layer system; hardware source release pending | Matched four-profile comparison | P66+D1, one layer, one sequence, four W4 compute CUs | RTL phase intervals; P+D excludes the Host gap; saved traces/dumps and replayable analysis | 171,520 values exact per profile; Integrated reduces total cycles by 14.28% from its matched Baseline; no trained-model or board-performance claim |
| [`q214-resident-fix-20260818/`](q214-resident-fix-20260818/) | Fix16 resident | Released single-layer gate | Standard Qwen-shaped P8, Tasks 18/19/20 | Vitis 2022.2 HW-Emu common four-CU interval; Host setup/embedding/LM head and CPU oracle excluded | 16,384-value exact numerical closure; 651,621 modeled cycles at 200 MHz projection; no intermediate Host copy |
| [`qwen3b-e2e-20260820/`](qwen3b-e2e-20260820/) | Fix16 resident | Released bounded generation | P8/G2/L1, one sequence, one real D1 forward | Vitis 2022.2 HW-Emu common four-CU interval; Host embedding/sampling/validation excluded | Six-task P8 plus real D1 closure; 4,096 values exact; 1,190,693 modeled cycles |
| [`qwen3b-e2e-l2-20260821/`](qwen3b-e2e-l2-20260821/) | Fix16 resident | Released bounded generation | P8/G2/L2, one sequence, two decoder layers | Vitis 2022.2 HW-Emu common four-CU interval; Host embedding/sampling/validation excluded | Ten-task cross-layer closure; 4,096 values exact; 2,319,441.4 modeled cycles |
| [`q214-pd-20260811/`](q214-pd-20260811/) | Fix16 resident | Operator diagnostics | Q2.14 P/D contexts 64, 256, 512, 1024 | Host-orchestrated aggregate `cc8_ctrl` Running Time for sequential operator calls; same interval is reported for listed CUs, while Host gaps, fixture migration, and CPU golden checks are excluded | Context scaling, precision checks, and modeled useful-MAC efficiency of the diagnostic datapath |
| [`coarse-task-20260816/`](coarse-task-20260816/) | Fix16 resident | Small-shape protocol tests | Small two-layer Task 18/19/20 and serial prompt/decode composition | RTL CoSim, HW-Emu, HLS; common four-CU modeled interval; 300-MHz XSim cycles projected to 200 MHz; Host embedding/sampling excluded | Cross-task/cross-layer HBM residency and controller-owned KV; not Qwen throughput |
| [`block-prefill-20260817/`](block-prefill-20260817/) | Fix16 resident | Small-shape protocol tests | Small P8, P16, P11 tail, and P8/G2 block contracts | RTL CoSim, HW-Emu, HLS; common four-CU modeled interval; 300-MHz XSim cycles projected to 200 MHz | One-to-eight-row block semantics, causal KV state, and finite-stream closure |
| [`qwen3b-checkpoint-20260830/`](qwen3b-checkpoint-20260830/) | Fix16 resident | Checkpoint diagnostics | P8 with per-task Host readback | Vitis 2022.2 HW-Emu with intentional checkpoint D2H after each task; no performance interval | Layers 0--2 bit-exact; first one-unit divergence at layer 3 Attention; no checkpoint-accuracy or throughput claim |
| [`quantized-single-bank-20260907/`](quantized-single-bank-20260907/) | Quantized matrix blocks | Published component candidate | Controller-facing W4A4/W8A8 single-bank kernels | Vitis HLS 2022.2 CSynth plus bounded deadlock-enabled RTL CoSim; resource rows are local/four-CU sums, not system timing | `II=1`, 3/3 CoSim transactions, and local resource estimates; controller integration, full-layer performance, and deployable system release remain open |

The **Streaming split** family has no published full-system measurement in this
directory. Its analytical projections and source boundary are documented in
[`cases/streaming-split/docs/design.md`](../cases/streaming-split/docs/design.md)
and must not be mixed with measured Fix16 resident rows. The quantized
matrix-block package remains component evidence. The W4 full-layer development
package provides separate controller/compute HW-Emu measurements, with its
hardware source release and physical implementation still pending.

## Measurement policy

- In resident packages, `P8` is eight consecutive query rows from one sequence
  in one prefill block; it is not batch eight or eight decoded outputs. `G2`
  includes the prompt sample plus one real one-row decode forward. In the Q2.14
  package, `P<n>` and `D<n>` are local context labels; see
  [`docs/q214-pd-length-hwemu.md`](../docs/q214-pd-length-hwemu.md).
- HW-Emu CU Running Time is modeled RTL evidence, not XSim CPU wall time and
  not physical-board latency. A 300-MHz XSim interval projected to 200 MHz is
  labeled target-equivalent/modelled, not routed timing.
- Common four-CU intervals exclude Host embedding, LM-head, sampling, setup,
  weight preload, and post-inference CPU golden arithmetic unless a package
  explicitly says otherwise. Operator diagnostics have an additional
  Host-orchestration boundary; checkpoint packages intentionally add per-task
  Host readback and make no timing claim.
- In the Q2.14 package, each row is derived from the authoritative per-case
  `cc8_ctrl` Running Time for the operator-call sum. Numerically matching CU
  rows do not establish separate CU occupancy or inter-task issue gaps.
- Modeled useful-MAC efficiency divides shape-counted useful MAC by the
  measured modeled interval and the declared peak: 1,024 MAC/cycle for the
  two-compute-CU Fix16 resident system, or 4,096 MAC/cycle for the four-compute-CU
  W4 full-layer development system. It does not measure PE occupancy, power,
  PCIe latency, or physical utilization.
- HLS CSynth tables are local resource and timing estimates, not post-route
  utilization or timing closure. Four-CU rows in the quantized matrix-block package are
  arithmetic resource sums and not an integrated system implementation.
- CPU fixed-point oracles validate arithmetic after inference and are excluded
  from accelerator useful work. Deterministic random Fix16 weights validate
  shape, arithmetic, and protocol; they are not trained-checkpoint accuracy.
- No package claims a 36-layer measured run or a physical-board result. The L2
  package is a bounded two-layer composition; the checkpoint package stops at
  the first observed layer-3 Attention divergence.

## Integrity

Run the repository helper from the project root after the `hls` preflight. The
aggregate gate compiles Host-only contracts but launches neither synthesis nor
simulation:

```bash
source scripts/setup_environment.sh
scripts/check_environment.sh hls
make test_publication_release
```

For evidence checks on a machine without AMD/Xilinx tools, use
`scripts/check_environment.sh publication`, `make test_publication_tree`, and
`make verify_result_checksums` instead.

The verifier accepts both historical repository-root-relative manifests and
archive-relative manifests emitted by the current atomic E2E archiver. Raw
Host logs, CU profiles, and numeric rows are not silently rewritten when
terminology is refined. A schema label may be clarified only when the artifact
README records the change, its complete checksum manifest is regenerated, and
the raw-to-derived-table verifier still reproduces every numeric value.

Unless an artifact states otherwise, documentation, figures, and experimental
evidence are licensed under
[CC BY-NC 4.0](../LICENSES/CC-BY-NC-4.0.md). Attribution should name Teng Wang
and LLM-ACCEL and identify any modifications.
