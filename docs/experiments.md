# Experimental Results

[Documentation index](README.md) | [Implementation map](implementations.md) |
[Evidence index](../results/README.md) | [Detailed report](experiment-details.md) |
[Setup](environment.md) | [Reproduction](usage.md) | [Repository](../README.md)

This page is the release-first evidence map. It names the source family,
workload, timing boundary, and claim supported by each published package. It
does not turn an operator diagnostic, small-shape protocol test, checkpoint
readback, HLS estimate, or isolated kernel into a whole-system release.

## Evidence taxonomy

The public repository has three hardware implementation families:

| Family | Source boundary | Public evidence status |
| --- | --- | --- |
| **Fix16 resident** | Controller-resident hidden/KV state, Tasks 18/19/20, and two regular 8x64 compute CUs | Current released mainline for bounded P8 and P8/G2/L1/L2 HW-Emu evidence |
| **Streaming split** | Earlier control/cache plus fixed-compute split in [`cases/streaming-split/`](../cases/streaming-split/) | Alternative family; full-layer values are analytical projections, not released full-system measurements |
| **Quantized matrix blocks** | Isolated INT4/INT8 matrix blocks in [`cases/quantized-block/`](../cases/quantized-block/) | Component candidates with HLS and bounded RTL evidence; no full-layer release |

The following names are evidence scopes under Fix16 resident, not additional
hardware generations:

- **Fix16 operator diagnostics:** Host-sequenced operator calls and Q2.14
  context-length measurements.
- **Small-shape protocol tests:** Reduced profiles for finite FIFOs, block
  tails, HBM residency, controller-owned KV, and task composition.
- **Checkpoint diagnostics:** Per-task Host readback used to localize numeric
  drift; it is not a performance result or checkpoint-accuracy release.

The [implementation map](implementations.md) is authoritative for legacy
aliases such as `R1`, `D1`, `P1`, `S1`, and `Q1`. In new prose, `D1` is only a
workload name for one real one-row decode forward; it is not an implementation
family.

## Released resident results

These are the current Fix16 resident packages, ordered from the single-layer
gate to bounded generation composition. All three use a common four-CU
hardware-emulation interval; the interval is modeled RTL evidence, not physical
board timing.

| Package | Workload | Numerical/protocol result | Measured boundary and limitation |
| --- | --- | --- | --- |
| [`q214-resident-fix-20260818/`](../results/q214-resident-fix-20260818/) | Standard Qwen-shaped P8, one layer, Tasks 18/19/20 | 16,384 values exact; 651,621 cycles at 200 MHz projection; 189.285 GMAC/s; 92.424% modeled efficiency | Common four-CU modeled interval; Host embedding/LM head, setup, and CPU oracle excluded; no physical-board or 36-layer claim |
| [`qwen3b-e2e-20260820/`](../results/qwen3b-e2e-20260820/) | P8/G2/L1, one sequence, one real D1 forward | 4,096 values exact; six tasks; 1,190,693 modeled cycles; 56.904% modeled efficiency | Common four-CU interval; Host embedding, sampling, and validation excluded; output-token rate includes prefill and is not steady D1 |
| [`qwen3b-e2e-l2-20260821/`](../results/qwen3b-e2e-l2-20260821/) | P8/G2/L2, one sequence, two decoder layers | 4,096 values exact; ten tasks; 2,319,441.4 modeled cycles; 58.424% modeled efficiency | Same boundary as L1; bounded two-layer composition, not a 36-layer or physical-board result |

The resident packages use deterministic random Fix16 weights. The L1/L2
generation packages additionally use tied embeddings. Their CPU fixed-point
oracle runs after inference and is excluded from the accelerator useful-work
numerator. They validate shape, arithmetic, task composition, and HBM/KV
ownership; they do not establish trained-model checkpoint accuracy.

## Other released evidence

| Package | Family / evidence scope | Workload and source | Supported claim | Boundary or non-claim |
| --- | --- | --- | --- | --- |
| [`q214-pd-20260811/`](../results/q214-pd-20260811/) | Fix16 operator diagnostics | Host-orchestrated Q2.14 operator path; P/D contexts 64, 256, 512, 1024 | Context scaling, precision checks, and modeled useful-MAC efficiency | Aggregate `cc8_ctrl` Running Time for sequential operator calls; Host gaps, fixture migration, and CPU oracle excluded; not resident common-four-CU timing |
| [`coarse-task-20260816/`](../results/coarse-task-20260816/) | Fix16 small-shape protocol tests | Small two-layer Task 18/19/20 and serial prompt/decode composition | Cross-task/cross-layer HBM residency and controller-owned KV | Not Qwen2.5-3B throughput, 36-layer performance, or trained-model accuracy |
| [`block-prefill-20260817/`](../results/block-prefill-20260817/) | Fix16 small-shape protocol tests | Small P8, P16, P11 tail, and P8/G2 block contracts | One-to-eight-row semantics, causal KV state, and finite-stream closure | Not standard large-shape performance, 36-layer timing, or physical-board timing |
| [`qwen3b-checkpoint-20260830/`](../results/qwen3b-checkpoint-20260830/) | Fix16 checkpoint diagnostics | P8 with per-task Host readback | Layers 0--2 bit exact; first observed one-unit divergence at layer 3 Attention | Not throughput, full 36-layer numerical closure, or checkpoint accuracy |
| [`quantized-single-bank-20260907/`](../results/quantized-single-bank-20260907/) | Quantized matrix blocks | Controller-facing W4A4 and W8A8 single-bank kernels | `II=1`, 3/3 bounded CoSim cases, local HLS estimates, and resource sums | Published component evidence; controller integration, HW-Emu, full-layer timing, and deployable system release remain open |

The streaming-split family is documented in
[`cases/streaming-split/docs/design.md`](../cases/streaming-split/docs/design.md).
Its projected full-layer values are analytical and must not be mixed with the
measured resident packages above.

## Measurement conventions

The following terms are deliberately explicit because historical reports use
short local labels:

- `P8` in resident packages means eight consecutive query rows from one
  sequence in one block. It is not sequence batch eight and not eight decoded
  outputs. `G2` means the prompt sample plus one real one-row decode forward.
- A common four-CU HW-Emu interval is the same run-local profiler Running Time
  for controller, both compute CUs, and status sink. It does not resolve
  separable CU occupancy or inter-task issue gaps. OpenCL event and Host wall
  times under HW Emu are simulator proxies, not device latency.
- Where a package records a 300-MHz XSim clock, its cycles are first derived
  from that run-local clock and then projected to the 200-MHz implementation
  target. A 200-MHz table is therefore a modeled target-equivalent latency,
  not a routed clock measurement.
- Modeled useful-MAC efficiency is shape-counted useful MAC divided by the
  measured modeled interval and the declared two-CU peak of 1,024 MAC/cycle.
  Padding, vector work, Host operations, and CPU-oracle work are not silently
  added to the numerator; this metric is not physical utilization or power.
- Operator diagnostics and production resident runs have different Host
  boundaries. Q2.14 includes Host sequencing, packing, KV fixture migration,
  and golden checks outside its controller intervals; the resident Task
  18/19/20 path keeps intermediate hidden state and KV in controller-owned HBM.
- The CPU fixed-point oracle is an out-of-band correctness check. It can prove
  the recorded output comparison for the selected workload, but it does not
  prove a performance result or change the timing boundary.
- Released end-to-end resident evidence reaches two decoder layers (`L2`). A
  `P8/G2/L36` task expansion or a checkpoint run stopped at layer 3 is not a
  measured 36-layer result.
- The historical full-profile composition completed its 146-task scheduling
  contract, but its numeric gate was not accepted. Preserve that as protocol
  evidence only; do not promote it to a 36-layer numerical or performance
  result.
- No package in this repository is a physical-board timing or power result.
  HLS CSynth is local timing/resource evidence; quantized four-CU rows are
  resource sums and not a placed-and-routed system.

For the Q2.14 report, `P<n>` and `D<n>` are a local historical context
convention: `P1024` is the final eight query rows at positions 1016--1023,
and `D1024` is one decode row at position 1024 against 1025 KV entries. They
are not universal prompt length, sequence batch, or implementation names. See
[`q214-pd-length-hwemu.md`](q214-pd-length-hwemu.md) for the complete local
definition.

## Detailed and historical records

The original long report, including dated comparisons, resource tables,
commands, package paths, and historical next/in-progress language, is retained
in [`experiment-details.md`](experiment-details.md). The maintained
coarse-task contract is [`coarse-task-runtime.md`](coarse-task-runtime.md);
its relocated result tables and commands remain in
[`coarse-task-runtime-history.md`](coarse-task-runtime-history.md).

## Legacy anchors

The headings below remain as redirects so links into the former long report
continue to resolve without making historical sections look current.

<details>
<summary>Show legacy heading redirects</summary>

## 1. Reporting policy

[Historical detail](experiment-details.md#1-reporting-policy).

## 2. Reference configuration

[Historical detail](experiment-details.md#2-reference-configuration).

## 3. Verification summary

[Historical detail](experiment-details.md#3-verification-summary).

## 4. Single-token resident layer

[Historical detail](experiment-details.md#4-single-token-resident-layer).

## 5. Projection steady state

[Historical detail](experiment-details.md#5-projection-steady-state).

## 6. Attention scaling experiments

[Historical detail](experiment-details.md#6-attention-scaling-experiments).

## 7. Eight-row prefill-block baseline

[Historical detail](experiment-details.md#7-eight-row-prefill-block-baseline).

### Functional workload

[Historical detail](experiment-details.md#functional-workload).

### Cycle calculation

[Historical detail](experiment-details.md#cycle-calculation).

### Resource qualification

[Historical detail](experiment-details.md#resource-qualification).

## 8. Q2.14 multi-length P/D sweep

[Historical detail](experiment-details.md#8-q214-multi-length-pd-sweep).

### Measurement boundary

[Historical detail](experiment-details.md#measurement-boundary).

## 9. Coarse-task resident runtime

[Historical detail](experiment-details.md#9-coarse-task-resident-runtime).

## 10. Interpretation

[Historical detail](experiment-details.md#10-interpretation).

## 11. Current experimental boundaries

[Historical detail](experiment-details.md#11-current-experimental-boundaries).

## 12. Next experiments

[Historical detail](experiment-details.md#12-next-experiments).

</details>
