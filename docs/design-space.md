# Design space

[Documentation index](README.md) | [Architecture](architecture.md) |
[Design entries](README.md#designs) | [Evaluation](experiments.md) |
[Source map](repository-map.md)

This page records candidates, tradeoffs, and evidence boundaries. It is not a
catalog of numbered generations. The [implementation catalog](implementations.md)
owns canonical names and configuration labels; the [architecture](architecture.md)
owns the mechanisms shared by the source families.

## Partitioning the kernels

| Candidate | Benefit | Cost or boundary | CoWave choice |
| --- | --- | --- | --- |
| Monolithic decoder kernel | Fewer exported interfaces | Large control cone, long synthesis/debug path, weak local evidence | Not selected for the root resident path |
| Operator-specific kernels | Small local kernels | Host launches and external-memory round trips fragment a layer | Retained for the operator diagnostic scope only |
| Model-aware controller plus regular stream compute | Keeps scheduling/state in one owner and arithmetic regular | Needs explicit task/packet contracts and finite-FIFO validation | Selected by `cowave-fix16-2-8-64` |
| Fixed controller plus service core | Reuses a compute core across an operator program | Different dataflow and numeric boundary from the resident runtime | `cowave-streaming-split` |
| Complete-layer quantized controller/compute | Makes W4A4/W8A8 schedules measurable at the model-layer boundary | Needs source closure, scale policy, and profile-specific validation | `cowave-int4-4-8-128` and `cowave-int8-4-4-128` |
| Isolated quantized matrix block | Makes packing/accumulator choices measurable | No model-level state, scale application, or integrated schedule | `cowave-quantized-blocks` component route |

The controller boundary is useful only when its ownership is explicit. A
case-local kernel or a resource planner is not automatically an integrated
implementation.

## Array shape and logical work

The public names encode compute count and logical matrix shape:

| Design | Name fields | Shape decision |
| --- | --- | --- |
| Fix16 resident | `2-8-64` | Two CUs, 8 logical rows, 64 output columns; modeled matrix peak is `2 * 8 * 64` |
| W4A4 profile | `4-8-128` | Four CUs, 8 logical rows, 128 logical output columns |
| W8A8 profile | `4-4-128` | Four CUs, 4 logical rows, 128 logical output columns |

The root implementation's physical source and build paths retain `8x64`.
Quantized component headers retain their actual 8x64 W4A4 and 4x128 W8A8
tiles. The canonical profile labels are a comparison vocabulary, not a claim
that a four-CU linked image can be rebuilt from a component directory.

For every candidate, logical work is computed from the declared rows and
output columns. W4A4 DSP packing recovers multiple INT4 products in one
physical multiply, but packing is not an additional logical-work multiplier.
Controller/status kernels and service traffic are outside the compute-CU
count.

## Packet and protocol granularity

| Protocol choice | Why it helps | Why it is limited |
| --- | --- | --- |
| Bit-level or fine-grained control | Potentially precise scheduling | High fan-out and difficult cross-kernel debugging |
| C++ structs at exported boundaries | Convenient source syntax | Tool-dependent packing and fragile ABI |
| Fixed-width block packets | Explicit widths, regular FIFOs, easy compatibility checks | More packing/unpacking code and family-specific ABIs |

CoWave uses fixed-width packets in all families. The root task, activation,
weight, vector/result, status, streaming-split, and quantized widths are listed
in [architecture](architecture.md). Similar widths do not make the ABIs
interchangeable.

## Memory and residency

The resident alternative is a Host-managed pipeline versus controller-owned
state:

| Candidate | Strength | Weakness | Boundary |
| --- | --- | --- | --- |
| Host submits each operator | Simple golden checks and visibility | Intermediate transfers fragment latency and decode | Fix16 operator diagnostics |
| Host composes resident subgraphs | Short descriptor program; hidden/KV stay on device | Requires pair transitions, capacity guards, and task acceptance tests | Selected Fix16 runtime |
| Controller owns every model policy | Minimal Host scheduling | Larger controller and harder dynamic policy | Not required by current root contract |

The resident runtime selects a middle boundary: Host owns request/model policy
and sampling; the controller owns layer subgraphs, KV, and intermediate hidden
state. See [coarse-task runtime](coarse-task-runtime.md) for the stable contract.

## Weight delivery and scheduling

The root loader reads 512-bit HBM blocks and emits regular 256-bit row packets.
The selected producer II and bounded FIFOs rate-match the current consumer;
II=1, deeper queues, and wider packet paths are alternatives that may add
state or timing pressure without increasing useful work.

The resident projection schedule overlaps:

```text
load -> issue/drive -> compute -> collect -> commit
```

Cross-wave overlap is bounded to the named projection fixture. Prompt-block
order and decode descriptors remain explicit. A deeper FIFO or changed overlap
flag requires deadlock-enabled RTL CoSim and fresh evidence.

The streaming split uses a separate three-process dispatch/input/output graph
and four weight ports. Its single-stream versus multi-port comparison is
case-local. Quantized block replication is a resource/connectivity planning
problem; it is not a resident schedule.

## Attention alternatives

| Candidate | Storage and dependency | Decision |
| --- | --- | --- |
| Materialized score matrix | Extra score storage and external traffic | Not selected for the resident controller |
| Tiled scores with a second pass | Bounded temporary storage; rereads tile state | Reference alternative |
| Online normalization and PV accumulation | Running max/sum/PV state; carried reduction dependency | Selected for Fix16 resident |

Online attention avoids a full score matrix but leaves a carried PV dependency.
Changing packet width alone does not remove that dependency. Q2.14 precision
and Host orchestration are diagnostic dimensions, not new designs.

## Prefill and decode shape

Eight rows fill the resident array for a standard prefill block. A one-row
decode cannot fill all rows without independent row work or a different mapping
policy. Candidate responses include independent decode requests, query-row
batching, larger arrays, and a scheduler that mixes work; each changes memory
traffic, ordering, or numerical comparison scope.

The public workload vocabulary keeps prompt tokens, active query rows, sequence
batch, context, sampled outputs, decode forwards, and layers separate. `P8`,
`G2`, and `D1` remain historical result shorthand only.

## Numeric choices

| Numeric candidate | Advantages | Risks and evidence boundary |
| --- | --- | --- |
| Signed Fix16 / wider fixed accumulators | Predictable widths, HLS-friendly streams, exact fixed-point oracle | Not IEEE FP16; Q2.14 attention has its own rounding boundary |
| W4A4 packed integers | Lower operand storage and multiple INT4 products per DSP | Scale/zero-point policy, accumulator width, and sign handling need independent tests |
| W8A8 integers | Simple signed products and decode-oriented 4x128 block | More physical DSPs; full-layer scale/application and routing remain separate |
| Float-assisted helper path | Convenient reciprocal/normalization helper | Breaks equivalence with an all-fixed-point path; streaming split reports it explicitly |

The quantized component task carries scale metadata but does not apply it.
Component CSim/CoSim therefore cannot establish quantized model accuracy.

## How candidates are accepted

Every candidate is evaluated at the smallest scope that can falsify its
claim:

- CSim for arithmetic, packet layout, sign/rounding, and finite K semantics;
- finite-FIFO RTL CoSim for progress and deadlock;
- HLS CSynth for local II/resource/timing estimates;
- linked HW-Emu for named system protocols and modeled CU intervals;
- physical implementation for routing, frequency, power, and board timing.

A planner roofline is not a measurement. A component II is not full-layer
throughput. A hardware source snapshot and a dated result package must have
matching identities before a result is attributed to that source. See
[experiments](experiments.md), the [reference index](reference.md), and the
[release workflow](release-workflow.md).
