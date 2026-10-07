# Evaluation and experiments

[Documentation index](README.md) | [Architecture](architecture.md) |
[Design space](design-space.md) | [Evidence index](../results/README.md) |
[Getting started](usage.md) | [Reference/history](reference.md)

This page is the release-first evaluation map. It keeps source family,
configuration, workload, timing boundary, and evidence stage together. It does
not turn a component probe, analytical projection, HLS estimate, or diagnostic
into a whole-system result.

## Evaluation vocabulary

Use these fields in new reports:

| Field | Meaning |
| --- | --- |
| `prompt_tokens` | Total prompt length supplied to the workload |
| `query_rows` | Active rows in the current prefill/decode block |
| `sequence_batch` | Number of independent sequences; not the same as rows |
| `context` | KV entries visible to the current query |
| `layers` | Decoder layers actually executed |
| `decode_forwards` | Real one-row decode forwards after prompt processing |
| `timing_scope` | Explicit Host, controller, CU, or common interval boundary |

Historical `P8`, `G2`, and `D1` labels remain in immutable package paths. In
resident packages, `P8` is eight rows from one sequence, `G2` is the prompt
sample plus one real decode forward, and `D1` is one decode row. Q2.14's
`P1024`/`D1024` are local context labels for one final block and one decode row;
they are not universal prompt lengths or implementation names.

## Public design families

| Canonical design | Source and tested scope | Current evidence boundary |
| --- | --- | --- |
| [`cowave-fix16-2-8-64`](designs/fix16.md) | Root controller-resident hidden/KV state, Tasks 18/19/20, two 8x64 compute CUs | Bounded resident HW-Emu, CSim, finite-FIFO CoSim, HLS, and operator/protocol diagnostics |
| [`cowave-streaming-split`](designs/streaming-split.md) | Case-local control/cache plus fixed V8-2_s service | CSim/sw_emu, HLS, bounded emulation, and analytical composition; no released full-layer system claim |
| [`cowave-int4-4-8-128`](designs/quantized.md) | Public W4A4 complete-layer source in `cases/quantized-layer/` | P66+D1 full-layer evidence with profile/source identity; component probes are separate |
| [`cowave-int8-4-4-128`](designs/quantized.md) | Public W8A8 complete-layer source in `cases/quantized-layer/` | P66+D1 full-layer evidence with profile/source identity; component probes are separate |
| `cowave-quantized-blocks` | Isolated W4A4/W8A8 matrix blocks in `cases/quantized-block/` | Component stream/resource evidence only |

The profile dimensions count compute CUs and logical products only. Controller
and status kernels are excluded. DSP packing changes physical resource cost,
not logical work, and is never counted twice.

## Released resident evidence

These immutable packages use the common four-CU modeled HW-Emu interval for the
resident controller, two compute CUs, and status sink. The interval is not
physical-board timing.

| Package | Workload | Recorded result | Boundary |
| --- | --- | --- | --- |
| [`q214-resident-fix-20260818`](../results/q214-resident-fix-20260818/) | Qwen-shaped P8, one layer, Tasks 18/19/20 | 16,384 values exact; 651,621 cycles at modeled 200 MHz; 189.285 GMAC/s; 92.424% modeled efficiency | Host embedding/LM head, setup, and CPU oracle excluded; no board or 36-layer claim |
| [`qwen3b-e2e-20260820`](../results/qwen3b-e2e-20260820/) | P8/G2/L1, one sequence, one real D1 | 4,096 values exact; six tasks; 1,190,693 modeled cycles; 56.904% modeled efficiency | Host embedding, sampling, and validation excluded; output-token rate includes prefill |
| [`qwen3b-e2e-l2-20260821`](../results/qwen3b-e2e-l2-20260821/) | P8/G2/L2, one sequence, two layers | 4,096 values exact; ten tasks; 2,319,441.4 modeled cycles; 58.424% modeled efficiency | Bounded two-layer composition, not 36-layer or board timing |

The resident packages use deterministic random Fix16 weights; L1/L2 also use
tied embeddings. They validate shape, arithmetic, task composition, and HBM/KV
ownership. They do not establish trained-checkpoint accuracy.

## Other published scopes

| Package | Scope | Supported claim | Not a claim |
| --- | --- | --- | --- |
| [`q214-pd-20260811`](../results/q214-pd-20260811/) | Fix16 Q2.14 operator diagnostics at contexts 64/256/512/1024 | Context scaling, precision checks, modeled useful-MAC efficiency | Resident common-four-CU production timing |
| [`coarse-task-20260816`](../results/coarse-task-20260816/) | Small Task 18/19/20 composition | HBM residency and controller-owned KV | Qwen2.5-3B throughput or trained accuracy |
| [`block-prefill-20260817`](../results/block-prefill-20260817/) | Small P8/P16/P11 block contracts | Row/tail semantics and finite-stream closure | Standard large-shape or board timing |
| [`qwen3b-checkpoint-20260830`](../results/qwen3b-checkpoint-20260830/) | Per-task P8 checkpoint diagnostics | Layers 0--2 bit exact; first one-unit divergence at layer 3 Attention | Throughput or full-model accuracy |
| [`quantized-single-bank-20260907`](../results/quantized-single-bank-20260907/) | W4A4/W8A8 component blocks | II=1, bounded CoSim, local HLS estimates, resource sums | Controller integration, HW-Emu, or full-layer timing |

The streaming split's case README and design record own its local evidence;
its analytical compositions must not be mixed with the resident rows above.

## Current quantized mainline package

The [`quantized-layer-20261007`](../results/quantized-layer-20261007/) package
records the public complete-layer W4/W8 P66+D1 runs. Its recorded modeled
cycles are:

| Profile | Configuration | Workload | Prefill cycles | Decode cycles | P+D cycles | P+D modeled efficiency | Source/evidence boundary |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `cowave-int4-4-8-128` | `integrated-rms2-silu4-prefill-overlap` | P66 + one D1 | 2,288,540 | 187,844 | 2,476,384 | 50.999764% | Four-CU complete-layer source; see package manifest |
| `cowave-int4-4-8-128` | `integrated-rms2-silu4-prefill-overlap-wave` | P66 + one D1 | 2,284,511 | 187,557 | 2,472,068 | 51.088805% | Four-CU complete-layer source; see package manifest |
| `cowave-int8-4-4-128` | `integrated-rms2-silu4-prefill-overlap` | P66 + one D1 | 3,592,954 | 187,220 | 3,780,174 | 66.819675% | Four-CU complete-layer source; see package manifest |
| `cowave-int8-4-4-128` | `integrated-rms2-silu4-prefill-overlap-wave` | P66 + one D1 | 3,582,758 | 186,332 | 3,769,090 | 67.016176% | Four-CU complete-layer source; see package manifest |

These rows are package claims for the named source/configuration and are not
board timing or a 36-layer projection. Use the package's profile identity and
raw evidence rather than inferring additional rows here.
The package reports the four-CU HLS estimate exceeding the device BRAM/LUT
capacity before platform resources; it therefore makes no routed,
deployability, PE-occupancy, power, or board claim.

## Quantized full-layer evidence and source identity

The dated [`quantized-layer-w4-20260930`](../results/quantized-layer-w4-20260930/)
package is a frozen W4A4 development snapshot. It records a matched P66+D1
comparison across Baseline, Attention, block pipeline, and Integrated decode:
all four runs compare 171,520 hidden/KV values exactly against the production C
model; Integrated reduces the recorded Prefill+Decode cycles from 3,326,289 to
2,851,423 (14.28%); its modeled Prefill/Decode efficiencies are 46.998% and
9.239%. These values belong to the frozen source identity in that package.

The public quantized-layer source is a separate closure under
[`cases/quantized-layer/`](../cases/quantized-layer/). The newer
[`quantized-layer-20261007`](../results/quantized-layer-20261007/) package
records its W4/W8 P66+D1 evidence and configuration identities. A new source
does not inherit the September snapshot's results: use the package manifest,
checksums, and source identity before attributing any row. The dated progress
page retains the detailed comparison and its historical pending-work context.

## Metric conventions

For a declared logical matrix shape, the compute numerator is based on logical
products per cycle. The canonical profile names give compute count, rows, and
output columns. DSP packing is a resource optimization and must not multiply
that numerator again. Vector operations, padding, Host work, setup, and CPU
oracle work are excluded unless a package explicitly says otherwise.

Modeled useful-MAC efficiency is:

```text
shape-counted useful logical MACs
---------------------------------
declared logical peak * named modeled interval
```

Resident package rows use the common profiler Running Time for controller,
compute CUs, and status sink. Q2.14 operator rows sum sequential controller
intervals and exclude Host gaps, fixture migration, and golden checks. HLS
resource/Fmax values are local estimates. HW-Emu CPU wall time is simulator
runtime, not device latency; a 300-MHz XSim interval projected to 200 MHz is a
modeled target-equivalent value, not routed timing.

No package in this repository is a physical-board power or timing result.

## Reanalysis and detailed history

The dated W4 package can be reanalyzed without Vitis:

```bash
bash results/quantized-layer-w4-20260930/verify.sh
```

Use [experiment-details](experiment-details.md) for the original long tables,
[coarse-task-runtime-history](coarse-task-runtime-history.md) for relocated
runtime commands, and [Q2.14 diagnostics](q214-pd-length-hwemu.md) for the
local context sweep. Those pages preserve cited detail; this page remains the
single comparison entry for current reading.
