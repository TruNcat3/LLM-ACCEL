# Design Space and Alternatives

[Documentation index](README.md) | [Implementation map](implementations.md) |
[Repository map](repository-map.md) | [Architecture](architecture.md) |
[Experiments](experiments.md) | [Setup](environment.md) |
[Repository](../README.md)

This document records the design choices that separate CoWave's three public
implementation families. It distinguishes the shared architectural idea, the
choices made by each actual source tree, and alternatives that remain open.
The [implementation map](implementations.md) defines public family names and
the difference between an implementation family and a narrower evidence scope.

## Shared architectural idea

All three families use a model-aware or controller-facing boundary around
regular stream compute:

- model and memory policy should not be replicated inside every arithmetic
  island;
- packet widths and ownership transitions should be explicit at kernel
  boundaries;
- finite streams must be checked for progress and backpressure, not only
  functional output;
- arithmetic shape, memory traffic, and evidence stage are separate design
  dimensions.

The root **Fix16 resident** implementation is the selected end-to-end boundary.
**Streaming split** is an earlier alternative architecture with a different
controller and compute core. **Quantized matrix blocks** are isolated INT4/INT8
kernel candidates, not a published full-layer implementation.

## 1. Kernel partitioning

| Family or boundary | Actual choice | What it does not claim |
| --- | --- | --- |
| Fix16 resident | One model-aware controller/cache kernel, two stream-only 8x64 compute CUs, and a status sink | The compute CUs do not own HBM or model-layer semantics |
| Streaming split | `control_cache_core` (`cc`) schedules a fixed V8-2_s compute core through an `operator_program` | It is not a second profile of the resident Task 18/19/20 runtime |
| Quantized matrix blocks | Standalone controller-facing W4A4 and W8A8 stream kernels | Four-CU resource sums are not a linked system or full model |

Alternatives considered in the resident family were a monolithic decoder kernel
and many operator-specific kernels. The former increases the control cone and
synthesis/debug burden; the latter fragments launches and data movement. The
controller-plus-stream-compute boundary is selected for the root family.

## 2. Compute-array shape

| Family | Shape | Rationale and limitation |
| --- | --- | --- |
| Fix16 resident | 2 x 8x64 CUs, 1,024 modeled MAC/cycle peak | Eight rows fit a prefill block; one-row decode cannot fill the array without independent row work |
| Streaming split | 2 cores x 1 lane x 16x64, 2,048 modeled DSP lanes | Larger reduction depth reduces operation count; its resource/frequency tradeoff is case-local |
| Quantized matrix blocks | W4A4 8x64; W8A8 4x128 | W4A4 packs four signed INT4 products per DSP; W8A8 exposes a decode-oriented rectangle |

An array shape is not a workload claim. Prompt rows, sequence batch, decode
forwards, layer count, and context are recorded separately in the workload and
evidence documents. A larger array alone does not solve M=1 decode utilization.

## 3. Cross-kernel protocol granularity

| Alternative | Observation | Public choice |
| --- | --- | --- |
| Fine-grained or bit-level control | Many narrow pipelines and large fan-out networks | Rejected for the root boundary |
| C++ structs across XO boundaries | Tool-dependent packing and fragile ABI | Rejected at exported stream boundaries |
| Fixed-width block packets | Regular interfaces and explicit compatibility checks | Used by all three families, with family-specific widths |

The root Fix16 ABI uses 160-bit task, 128-bit activation, 256-bit weight,
416-bit vector/result, and 64-bit status packets. Streaming split uses 512-bit
input/weight packets, two 1,024-bit output halves, and 32-bit control. Quantized
blocks use a 128-bit task word, 32-bit activation, 256-bit weight, and 448/576-bit
output words. These contracts are similar in purpose but not interchangeable.

## 4. Operator implementation choices

The resident family keeps model semantics in the controller and uses the
compute CUs for regular matrix/vector work:

| Operator or state | Resident choice | Alternative or boundary |
| --- | --- | --- |
| Dense projection | Unified 8x64 matrix engine with output-column partitioning | Operator-specific arrays duplicate control and memory policy |
| RMSNorm/residual/gated activation | CU vector tasks with resident operands | Host preprocessing or repeated external-memory round trips |
| RoPE and KV cache | Controller-local transform and controller-managed HBM cache | Host-managed KV would put PCIe traffic in the decode loop |
| Attention normalization | Tiled online maximum/sum/PV state | Materialized score matrix or a second normalization pass |
| Quantized arithmetic | Separate W4A4/W8A8 block candidates | Not inferred from Fix16 measurements and not yet controller integrated |

The streaming split implements a different operator contract: `cc` selects an
operation and data addresses, while V8-2_s applies activation/normalization
choices to the streamed result. It does not reproduce the resident controller's
online-attention/KV ownership.

## 5. Weight delivery pipeline

The Fix16 resident loader reads 512-bit weight blocks and emits regular 256-bit
row packets. The selected producer II is 2; eight output slots are interleaved
so each slot receives a block every 16 cycles, matching the consumer schedule.
Increasing the producer to II=1 would add queued state without improving the
current consumer rate. The bounded FIFO and source-profile details are kept in
the [architecture](architecture.md) and [runtime report](coarse-task-runtime.md).

Streaming split uses four weight HBM ports and four weight streams. Each
operation receives 16 weight blocks per stream, making the case's
single-stream versus four-PC comparison a bandwidth experiment inside that
family. Quantized blocks have bounded 256-bit weight words; their controller
loader, scale application, and model-level accumulation remain open.

## 6. Wave scheduling

The resident projection path uses bounded cross-wave dataflow. The tested
intra-projection invariant is:

```text
load -> issue/drive -> compute -> collect -> commit
```

Different waves occupy those stages concurrently after fill. `II=2` weight
loading, finite result FIFOs, and deadlock-enabled RTL CoSim are part of the
contract. This is not a promise of generalized cross-block or cross-request
overlap: prompt blocks and decode descriptors remain ordered in the Host task
program, and cross-block residency has its own evidence scope.

The streaming split has a three-process dataflow graph (`dispatch`,
`input_path`, and `output_path`) that overlaps operation issue, packed input and
weight movement, compute, and output storage. Its schedule is specific to the
case-local `operator_program`; it does not establish the resident pipeline
invariant for a different implementation.

Multi-wave repeat commands and deeper FIFOs remain alternatives, not release
defaults. Any change must be evaluated for finite-buffer progress, II, timing,
and resources together.

## 7. Attention alternatives

| Candidate | Storage/traffic | Decision |
| --- | --- | --- |
| Materialized score matrix | Extra score storage and traffic | Not selected for the resident controller |
| Tiled scores with a second pass | Bounded temporary storage but rereads tile state | Reference alternative |
| Online normalization and PV accumulation | Bounded running state with rescaling dependencies | Selected for Fix16 resident |

Online attention keeps the running maximum, normalization sum, and PV
accumulator per query group. Its current bottleneck is the carried PV
dependency; widening the packet ABI does not remove that dependency. Any
partial-accumulator or tree-reduction alternative must preserve the fixed-point
error envelope and cross-tile normalization semantics.

## 8. Prefill scheduling

The resident family has two intentionally different evidence scopes:

- **Fix16 operator diagnostics** invoke individual operators so each tensor can
  be checked and profiled. This is useful visibility evidence, not a separate
  implementation generation.
- **Small-shape protocol tests** exercise finite FIFOs, block tails, residency,
  and controller-owned KV at reduced shapes. They are protocol evidence, not a
  separate implementation generation.

The production resident path uses static coarse tasks. A prompt block submits
Task 18 and Task 19 for each layer; Task 20 materializes the final normalized
hidden state. The full task program, block composition, and timing boundaries
are maintained in [coarse-task-runtime.md](coarse-task-runtime.md) and
[experiments.md](experiments.md), rather than duplicated here.

## 9. Decode utilization candidates

The resident 8x64 array exposes an M=1 shape limit. Candidates that address
that limit are:

- multi-request batching, so independent rows share resident weights;
- speculative or multi-token blocks when the algorithm exposes independent
  candidate rows;
- row remapping or Split-M when the workload has independent row work.

Controller state alone cannot fill unused token rows. A future policy must be
measured against the same useful-work, timing-boundary, and evidence-stage
definitions used by the released reports.

## 10. Runtime task granularity

| Boundary | Benefit | Cost | Status |
| --- | --- | --- | --- |
| Host submits every operator | Maximum visibility and simple golden checks | Host round trips fragment a layer | Fix16 diagnostics only |
| One host command runs a whole model | Lowest launch overhead | Large dynamic controller and weak request/sampling control | Not the current target |
| Host composes coarse resident subgraphs | Static controller schedule with flexible layer/request composition | Requires explicit residency handles and task contracts | Selected Fix16 resident boundary |

The selected descriptor program carries operation, layer, position, active rows,
and HBM-pair IDs. It never exposes controller-owned KV or local intermediates.
The streaming split's `operator_program` is a distinct operator-level contract;
the quantized blocks currently expose only bounded matrix operations.

<a id="quantized-arithmetic-candidates"></a>

## 11. Numeric representation and quantized candidates

The source-defined numeric boundaries are:

| Family | Source-defined format | Evidence boundary |
| --- | --- | --- |
| Fix16 resident | Signed `ap_fixed` activations/weights/accumulators, with Q2.14 attention probabilities | Fixed-point CSim/oracles, RTL CoSim, HLS, and bounded HW-Emu; not IEEE FP16 |
| Streaming split | `fm_t ap_fixed<16,8>`, `wt_linear_t ap_fixed<16,4>`, `fm_accum_t ap_fixed<32,16>`, internal `ap_fixed<48,24>`; `hls::recip` receives float conversion | Fixed-point packet storage with a mixed helper path; do not merge its numbers with Fix16 resident claims |
| Quantized matrix blocks | W4A4 signed 4-bit operands and W8A8 signed 8-bit operands, with source-defined integer accumulators and output widths | Isolated CSim/RTL CoSim/HLS probes; scale fields are metadata and full-layer integration is open |

The public quantized source case contains isolated kernels and CU planning
tools. Separately, the [full-layer development study](quantized-layer-progress.md)
now provides a matched W4A4 comparison of Baseline, Attention, block pipeline
and Integrated decode. It tests controller-owned scheduling and KV state with
four compute CUs. Its trace analysis and numerical evidence are published;
the corresponding full-layer hardware source release remains pending.

This comparison distinguishes improving one service from improving the whole
layer. Attention packing/prefetch, projection overlap and decode row reuse
address different dependencies. Subsequent experiments vary AXI outstanding
requests, SiLU/RMS lanes and Attention wave overlap; their completed component
results do not yet establish the gain of the combined full-layer design.

## 12. Evaluation principles

Designs are compared across independent dimensions:

1. arithmetic and functional agreement at the declared numeric boundary;
2. finite-buffer progress and deadlock detection;
3. useful work, cycles, and shape-normalized efficiency within a declared timing
   boundary;
4. HLS II, resource estimates, and timing structure;
5. linked-system, routed, and physical-board evidence as separate later gates.

An optimization is not accepted solely because one dimension improves. The
[experiments report](experiments.md) owns measured resident results, while the
[streaming split case](../cases/streaming-split/docs/design.md) and
[quantized matrix-block case](../cases/quantized-block/README.md) own their
family-specific evidence and limitations.
