# Resident coarse-task runtime

[Documentation index](README.md) | [Architecture](architecture.md) |
[Design entry](designs/fix16.md) | [Experiments](experiments.md) |
[Evidence index](../results/README.md) | [Setup](environment.md) |
[Reproduction](usage.md) | [Historical runtime detail](coarse-task-runtime-history.md)

This page defines the maintained runtime contract for
[`cowave-fix16-2-8-64`](designs/fix16.md). It describes
who owns each operation, how hidden state and KV stay resident, how a Host task
program is composed, and how timing evidence is bounded. It is not a result
table or a live queue; released numeric claims belong to the immutable packages
listed in [`results/README.md`](../results/README.md).

## Motivation

Operator-by-operator Host orchestration exposes every intermediate tensor to
Host scheduling and transfer overhead. A single whole-model command removes
that boundary but turns the controller into a large dynamic runtime. The
resident runtime uses an explicit middle boundary: the Host composes a short
descriptor program, while the controller owns the static operator schedule,
HBM access, on-chip buffering, online softmax, and KV-cache update inside each
task.

The current decode path defines three Host-visible operations:

| ID | Task | Controller-resident subgraph |
| ---: | --- | --- |
| 18 | Attention sublayer | RMSNorm, Q/K/V projections, RoPE, KV append/read, blockwise QK/online-softmax/PV, O projection, residual |
| 19 | FFN sublayer | RMSNorm, Gate/Up projections, SiLU-Mul, Down projection, residual |
| 20 | Final norm | Model-level final RMSNorm after the last decoder layer |

Operation 16, the earlier single-launch decoder-layer path, remains a
compatibility/equivalence reference only. It is not a separate public design.

## HBM-resident composition

Two HBM feature-buffer pairs form a device-side ping-pong boundary:

```text
initial hidden --one H2D--> HBM pair B

for each decoder layer:
  Task 18: pair B -> pair A     attention residual
  Task 19: pair A -> pair B     completed layer

Task 20:   pair B -> pair A     final normalized hidden

HBM pair A --one D2H--> final hidden
```

Every layer finishes in pair B, so the next Task 18 consumes the previous
layer's result directly. Only a 64-byte completion record is returned after a
task. Intermediate hidden state is neither migrated to the CPU nor repacked by
Host code. K and V remain in controller-managed HBM allocations, and RoPE is
applied inside Task 18 before cache append.

The Host still chooses task sequence, layer index, prompt policy, and sampling.
This preserves software ownership of request/model policy while removing
intermediate operator round trips.

### Static Host task program

The sequence is materialized as a small Host-side descriptor program before the
controller invocation. A descriptor contains stable orchestration metadata:

| Field | Meaning |
| --- | --- |
| `op` | Task 18, 19, or 20 |
| `layer` | Decoder-layer index; zero for model-level final norm |
| `position` | First sequence position for this query block |
| `query_tokens` | One decode row or one to eight prefill rows |
| `input_pair` / `output_pair` | HBM hidden-state ping-pong pair, encoded as 0 or 1 |

The descriptor does not contain raw weight or KV-cache addresses. Those remain
part of persistent controller state and the fixed XRT kernel binding. The Host
uses one generic issue loop to bind the selected HBM pair, invoke the
controller, validate its status record, and advance to the next descriptor.
Only a program marked for final materialization performs a hidden-state D2H.

Runtime provenance is part of the result contract. A binary built with this
executor emits `host_task_program=static_descriptor_v1`; its archive is valid
only when every `COARSE_TASK_PROGRESS` record carries and passes the descriptor
pair checks (`1 -> 0` for Attention/final norm and `0 -> 1` for FFN). A
pre-versioning run is classified as `legacy_equivalent_sequence` when pair
trace verification is unavailable; it must not be presented as evidence for
the new executor.

[`include/host_coarse_task_program.hpp`](../include/host_coarse_task_program.hpp)
is standard C++14 and can be tested without XRT or HLS. Its source contract
freezes task counts, validates adjacent HBM boundaries, rejects sequence
overflow, and verifies that production Host code does not migrate KV inside the
task loop. `build_coarse_decoder_program()` constructs metadata and
`accelerator_t::run_coarse_task_program()` validates/issues it; the existing
`run_composed_decoder_stack()` API is a compatibility wrapper.

## Generation composition and task count

Use explicit workload fields in new reports. Let `prompt_tokens` be the total
prompt length, `block_rows <= 8` the active query rows per prefill block,
`sampled_outputs` the number of sampled outputs, and `layers` the selected
decoder-layer count. The Host submits:

```text
prompt_blocks = ceil(prompt_tokens / block_rows)
decode_forwards = max(sampled_outputs - 1, 0)
coarse_tasks = prompt_blocks * (2 * layers) + 1
             + decode_forwards * (2 * layers + 1)
```

Each prompt block runs Task 18 and Task 19 for every layer. Only the final
prompt block runs Task 20 and materializes the hidden row for the vocabulary
head; earlier blocks release their hidden output after updating controller-
owned KV state. The first sampled token comes from the final prompt hidden, so
at least two sampled outputs are needed to exercise a real decode forward.

Historical package shorthand is retained locally: in the resident `P8/G2`
packages, `P8` means `prompt_tokens=8`, `block_rows=8`, and one sequence;
`G2` means the prompt sample plus one real decode forward; and `D1` means one
real one-row decode forward. In the small block protocol packages, `P8` may
be one block within a longer P16 or P11 prompt. These labels never mean batch
eight or eight parallel outputs. A `P8/G2/L1` request therefore has six tasks;
`P8/G2/L2` has ten. A `P8/G2/L36` expansion has 146 tasks as a contract
calculation, not a released 36-layer measurement. The Q2.14 report's `P1024`
and `D1024` are separate local context labels for its final block and one
decode row; they do not override these resident workload fields.

## Timing and provenance contract

The Host-visible inference boundary and the CU-trace boundary are separate:

| Stage | Owner | Host inference wall time | Common CU-trace interval |
| --- | --- | ---: | ---: |
| Prompt/decode embedding lookup | Host | yes | no |
| Task 18/19/20 execution | Controller + compute CUs | yes | yes |
| Intermediate hidden and KV movement | Controller/HBM | yes | yes |
| Completion synchronization and final hidden D2H | Host/XRT | yes | no |
| LM-head argmax/sampling | Host | yes | no |
| Model allocation and one-time weight preload | Host/XRT setup | reported separately | no |
| CPU fixed-point correctness oracle | Host, after inference | reported separately | no |

For the released resident HW-Emu packages, the authoritative modeled interval
is the common profiler Running Time for controller, both compute CUs, and the
status sink. Identical top-level rows do not resolve per-CU occupancy,
inter-task issue gaps, or raw waveform stages. Host/OpenCL event durations under
HW Emu follow simulator wall time and are retained as diagnostics, not device
latency.

When an image runs a 300-MHz XSim kernel clock, cycles are derived from that
run-local clock and projected to the 200-MHz implementation target. The
projected value is modeled target-equivalent latency, not routed timing. HLS
CSynth period/resource values are local estimates; resource sums do not prove
placement, routing, or physical-board timing.

The useful-MAC numerator is shape-counted arithmetic for the declared workload.
Vector work, padding, Host work, setup, and CPU-oracle work are excluded unless
a package explicitly states otherwise. Modeled useful-MAC efficiency divides
that numerator by the common modeled interval and the declared two-CU peak of
1,024 MAC/cycle. It is not PE occupancy, power, or physical utilization.

## Full-profile HBM capacity guard

The `qwen2.5-3b` Host plan checks aggregate allocation against grouped
connectivity rather than checking each logical shard against one pseudo-channel.
Each logical weight shard is 346,816,512 bytes and is striped over three
pseudo-channels (805,306,368 bytes). A shared group contains shard `i`, shard
`i + 8`, and, for the first two groups, one K or V cache. The worst-case
payload is 769,130,496 bytes, leaving 36,175,872 bytes of headroom; the Host
rejects initialization if either the per-shard or shared-group guard fails.

The full-profile compute XO is matched to the maximum online-attention
descriptor count of the selected controller launch. A shorter-context XO can
preserve the stream ABI while stopping before `last_task` at long context, so
the build wrapper must reject that reuse. Passing this capacity/descriptor
guard is a runtime-configuration check, not a 36-layer performance result.

## Verification interfaces

The model Host exposes these contract paths:

- `verify-composed-layer`: deterministic random-weight Task 18 followed by
  Task 19 against a CPU fixed-point golden layer;
- `verify-composed-prefill-block`: one Task-18/19/20 sequence over one to eight
  consecutive query rows, including causal RoPE, KV append/read, online
  attention, and final RMSNorm;
- `verify-composed-prefill-stack`: the same block through all layers in the
  selected profile followed by final RMSNorm and a CPU comparison;
- `verify-composed-stack`: the selected-layer task program followed by Task 20,
  with the `2 * layers + 1` task contract checked;
- `--coarse-tasks` and `--mode generate --coarse-tasks`: normal and generation
  paths using the same HBM-resident program.

The command reference belongs to [`usage.md`](usage.md). The historical
runtime archive retains the original build and run blocks for provenance; they
are not repeated here.

## Acceptance gates

| Gate | Required evidence | Scope boundary |
| --- | --- | --- |
| Host contract | Descriptor construction, pair transitions, task count, status records | Does not establish RTL timing |
| CSim | Operator/task semantics and CPU-golden output | Does not establish finite-buffer progress |
| Finite-FIFO RTL CoSim | Deadlock detection and task drainage for the named fixture | Bounded fixture evidence is not whole-system performance |
| HW Emu protocol | XRT ABI, connectivity, controller-owned hidden/KV, and named numerical checks | Modeled RTL only; not board timing |
| Standard resident performance | Common four-CU interval, source/artifact identity, oracle result, and explicit Host boundary | Released rows are bounded one/two-layer workloads |
| HLS/resource | Controller/compute/status local estimates and selected profile identity | Not post-route or physical implementation |

Released evidence pointers and exact numeric claims are maintained in the
[evidence index](../results/README.md), not copied into this contract.

## Resource contract

Resource qualification is profile-specific. The controller, compute CUs, and
status sink must be built from the same source/configuration identity as the
reported resident workload. Four-CU rows in the quantized package are arithmetic
resource sums and must not be read as an integrated system. The standard
resident package HLS rows are local CSynth estimates and still require a
physical implementation gate for routing, final frequency, power, and board
measurement.

## Measurement contract

A resident result may report three related but non-interchangeable quantities:

1. Host inference wall time, including task issue, synchronization, final D2H,
   embedding, and sampling where the package says so.
2. The common modeled CU interval used for throughput and useful-MAC efficiency.
3. Post-inference CPU-oracle and archive/checksum time, which is validation
   only.

Operator diagnostics use a different boundary: Host sequencing, packing,
historical KV preload, projection-to-cache fixture migration, and golden checks
remain outside their controller intervals. Do not compare those operator rows
to production resident request latency without stating the boundary.

## Historical result anchors

The original result sections remain in
[`coarse-task-runtime-history.md`](coarse-task-runtime-history.md). These
redirect headings preserve old links while making the dated tables visibly
historical.

<details>
<summary>Show historical result redirects</summary>

## Validation gates

[Historical result](coarse-task-runtime-history.md#validation-gates).

## Standard Qwen-layer P8 HW-Emu result

[Historical result](coarse-task-runtime-history.md#standard-qwen-layer-p8-hw-emu-result).

## Standard-shape P8/G2 end-to-end HW-Emu result

[Historical result](coarse-task-runtime-history.md#standard-shape-p8g2-end-to-end-hw-emu-result).

## Small-profile HW-Emu result

[Historical result](coarse-task-runtime-history.md#small-profile-hw-emu-result).

## Resource gate

[Historical result](coarse-task-runtime-history.md#resource-gate).

## Measurement boundary

[Historical result](coarse-task-runtime-history.md#measurement-boundary).

</details>
