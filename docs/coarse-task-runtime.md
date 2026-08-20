# Controller-Resident Coarse-Task Runtime

## Motivation

Operator-by-operator host orchestration exposes every intermediate tensor to
host scheduling and transfer overhead. A single whole-model command removes
that overhead but turns the controller into a large dynamic runtime. LLM-ACCEL
uses an intermediate boundary: the host composes a short sequence of
coarse-grained tasks, while the controller owns the static operator schedule,
HBM access, on-chip buffering, online softmax, and KV-cache update inside each
task.

The current decode path defines three host-visible operations:

| ID | Task | Controller-resident subgraph |
| ---: | --- | --- |
| 18 | Attention sublayer | RMSNorm, Q/K/V projections, RoPE, KV append/read, blockwise QK/online-softmax/PV, O projection, residual |
| 19 | FFN sublayer | RMSNorm, Gate/Up projections, SiLU-Mul, Down projection, residual |
| 20 | Final norm | Model-level final RMSNorm after the last decoder layer |

Operation 16, the earlier single-launch decoder-layer path, remains as a
compatibility and equivalence reference.

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
layer's result directly. Only a 64-byte completion record is returned after
each task. The intermediate hidden tensor is neither migrated to the CPU nor
repacked by host code. K and V remain in controller-managed HBM allocations,
and RoPE is applied inside Task 18 before the cache append.

The host still chooses the task sequence and layer index. This keeps request,
sampling, and model-level policy in software while removing host round trips
inside each subgraph.

## Generation composition and task count

Let `P` be the prompt length, `B <= 8` the active-query-row block size, `G`
the number of sampled tokens, and `L` the selected decoder-layer count. The
Host submits

```text
prompt blocks = ceil(P / B)
decode forwards = max(G - 1, 0)
coarse tasks = prompt_blocks * (2 * L) + 1
             + decode_forwards * (2 * L + 1)
```

Each prompt block runs Task 18 and Task 19 for every layer. Only the final
prompt block runs Task 20 and materializes a hidden row for the vocabulary
head; earlier blocks release their hidden output after updating
controller-owned KV state. The first sampled token comes from that final
prompt hidden, so a request must use at least `G2` to exercise a real decode
forward. Every subsequent forward runs `2 * L + 1` tasks and materializes one
final hidden row.

For the bounded full-shape gate, `P8/G2/L1` therefore means one eight-row
prefill forward plus one one-row decode forward: six coarse tasks and two
output materializations. It is one sequence, not batch eight and not eight
tokens generated in parallel. The corresponding `P8/G2/L36` expansion uses
146 tasks while preserving the same three-operation interface.

The implemented post-initialization inference boundary is:

| Stage | Owner | Included in Host inference wall time | Included in CU-trace cycles |
| --- | --- | ---: | ---: |
| Prompt/decode embedding lookup | Host | yes | no |
| Task 18/19/20 execution | Controller + compute CUs | yes | yes |
| Intermediate hidden and KV movement | Controller/HBM | yes | yes |
| Completion synchronization and final hidden D2H | Host/XRT | yes | no |
| LM-head argmax/sampling | Host | yes | no |
| Model allocation and one-time weight preload | Host/XRT setup | no, reported separately | no |
| CPU fixed-point correctness oracle | Host, after inference | no, reported separately | no |

Vitis HW Emu Host wall time follows simulator execution and is therefore not a
physical latency estimate. Publication performance rows use the common
profiler-reported running time for controller, both compute CUs, and status
sink. The
Host still reports inference-only, whole-process, LM-head, setup, and
post-inference validation times so the boundary remains auditable and can be
reused unchanged for a future physical-board run.

### Full-profile HBM capacity guard

The `qwen2.5-3b` host plan checks aggregate allocation against the actual
grouped connectivity rather than checking each logical shard against one
256-MiB pseudo-channel. Each logical weight shard is 346,816,512 bytes and is
striped over three pseudo-channels (805,306,368 bytes). A shared group contains
shard `i`, shard `i + 8`, and, for the first two groups, one K or V cache. The
worst-case payload is 769,130,496 bytes, leaving 36,175,872 bytes of headroom.
The host rejects initialization if either the per-shard or shared-group guard
fails.

The full-profile build also synthesizes a profile-matched compute XO. The
compute service loop is statically bounded by the maximum number of online
attention descriptors in one controller launch: the 96-position single-layer
profile reserves 328 descriptors, whereas the 2048-position profile reserves
1,792. Reusing the shorter-context XO would preserve the stream ABI but could
stop consuming tasks before `last_task` at long context, so the Qwen2.5-3B
build wrapper explicitly forbids that reuse.

## Verification interfaces

The model host exposes four relevant paths:

- `--mode verify-composed-layer`: deterministic random-weight Task 18 followed
  by Task 19, compared with a CPU fixed-point golden layer;
- `--mode verify-composed-prefill-block`: one Task-18/19/20 sequence over
  1--8 consecutive query rows, including per-row causal RoPE, KV append/read,
  online attention, and final RMSNorm;
- `--mode verify-composed-prefill-stack`: the same 1--8-row block through all
  layers in the selected profile, followed by final RMSNorm and a full CPU
  fixed-point comparison;
- `--mode verify-composed-stack`: all layers in the selected profile followed
  by Task 20, with the `2 * layers + 1` task contract checked explicitly;
- `--coarse-tasks`: use the same HBM-resident task sequence in the normal
  `run` or `generate` host path;
- `--mode generate --coarse-tasks`: compose prompt and generated-token
  forwards while reporting the software/hardware boundary explicitly.

The build helper accepts both the standard `qwen-layer` profile and the small
two-layer contract profile:

```bash
source /home/hepc/env/vitis_env_22.sh

# Closed-loop CSim or RTL CoSim with source-fingerprint protection.
make hls_csim_closed_loop_8x64_composed_layer
make hls_cosim_closed_loop_8x64_composed_layer

# Build and run the two-layer/five-task contract test.
VITIS_8X64_MODEL_PROFILE=small \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh all
VITIS_8X64_MODEL_PROFILE=small \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-stack

# Exercise prompt/decode composition across positions and controller-owned KV.
VITIS_8X64_MODEL_PROFILE=small \
CC8_RESIDENT_TOKEN_ROWS=8 \
VITIS_8X64_E2E_TOKENS=0,1,2,3,4,5,6,7 \
VITIS_8X64_E2E_MAX_NEW_TOKENS=2 \
VITIS_8X64_E2E_LAYERS=2 \
VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8 \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-generate

# Build exact controller/compute XOs for the 8-row contract and verify a block.
VITIS_8X64_MODEL_PROFILE=small \
CC8_RESIDENT_TOKEN_ROWS=8 \
VITIS_8X64_BUILD_EXACT_COMPUTE_XO=1 \
VITIS_8X64_RESIDENT_VARIANT_TAG=block_prefill_q214_resident_fix \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh all
VITIS_8X64_MODEL_PROFILE=small \
CC8_RESIDENT_TOKEN_ROWS=8 \
VITIS_8X64_BUILD_EXACT_COMPUTE_XO=1 \
VITIS_8X64_RESIDENT_VARIANT_TAG=block_prefill_q214_resident_fix \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-block

# Reuse that image for an exact two-layer-by-eight-row stack gate.
VITIS_8X64_MODEL_PROFILE=small \
CC8_RESIDENT_TOKEN_ROWS=8 \
VITIS_8X64_BUILD_EXACT_COMPUTE_XO=1 \
VITIS_8X64_RESIDENT_VARIANT_TAG=block_prefill_q214_resident_fix \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-block-stack

# Run a standard-dimension Attention+FFN layer.
VITIS_8X64_MODEL_PROFILE=qwen-layer \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-composed
```

The CoSim Tcl flow fingerprints the design, testbench, and shared headers
before synthesis and rechecks them before RTL simulation. A source change
during a long synthesis therefore invalidates the run instead of silently
mixing old RTL with a new testbench.

## Validation gates

| Gate | Evidence |
| --- | --- |
| Vitis 2022.2 host compilation | PASS |
| Controller route CSim | PASS, 21 cases including Tasks 18/19/20 |
| Closed-loop three-transaction CSim | PASS, Task 18 -> 19 -> 20 |
| Closed-loop RTL CoSim | PASS, 7,983 total cycles, deadlock detection enabled |
| 8-row block closed-loop CSim | PASS, Task 18 -> 19 -> 20, eight active query rows |
| 8-row block closed-loop RTL CoSim | PASS, 25,591 total cycles, deadlock detection enabled |
| Persistent Norm/RoPE/KV focused test | PASS, three independent norm rows and banked RoPE/KV checked |
| Small two-layer/five-task HW Emu | PASS, 64/64 values exact, `intermediate_host_copy=0` |
| Small prompt/decode composition HW Emu | PASS, 4 forwards/20 tasks, controller-owned KV, no intermediate host copy |
| Small 8-row Task-18/19/20 HW Emu | PASS, 512 values, max raw error 10 within tolerance 32 |
| Small two-layer 8-row stack HW Emu | PASS, 512 values, max raw error 10 within tolerance 64, five tasks, no intermediate Host copy |
| Small block-prompt/decode composition HW Emu | PASS, P8 + G2, 2 forwards/10 tasks, controller-owned KV |
| Small multi-block prompt composition HW Emu | PASS, P16 as blocks 0--7 and 8--15, 10 tasks, controller-owned KV |
| Small exact multi-block sequence HW Emu | PASS, P16 as two P8 blocks, nine tasks, 512-value final-tail CPU golden, max raw error 10/64 |
| Small tail-block prompt composition HW Emu | PASS, P11 as blocks 0--7 and 8--10, 10 tasks, controller-owned KV |
| Standard Qwen2.5-3B layer-shape 8-row HW Emu | PASS, Task 18/19/20, 16,384 values exact, 651,621 cycles, no intermediate Host copy |
| Standard-shape Qwen2.5-3B P8/G2/L1 HW Emu | PASS, six tasks, two forwards, 4,096 values exact, one real D1 forward |

The original one-row RTL CoSim records three passing transactions with
minimum/average/maximum latency of 1,204/2,664/4,497 cycles. The current Q2.14
8-row CoSim records 2,354/8,533/19,057 cycles and 25,591 total cycles. Both
exercise finite stream depths with RTL deadlock detection enabled. The previous
Q8.8 probability-buffer image completed the same sequence in 25,411 cycles.

## Standard Qwen-layer P8 HW-Emu result

The release image executes one standard Qwen-shaped layer over an 8-row
prefill block as three coarse tasks:

```text
Task 18: RMSNorm -> Q/K/V -> RoPE -> KV/online attention -> O -> residual
Task 19: RMSNorm -> Gate/Up -> SiLU-Mul -> Down -> residual
Task 20: final RMSNorm
```

| Evidence source | Host computation in timed interval | Scope | Active query rows / block | Layers | Tasks | Common HW-Emu CU running time | Cycles at 200 MHz | Useful GMAC/s | Modeled useful-MAC efficiency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vitis 2022.2 HW Emu CU trace | No | Task 18 -> 19 -> 20 | 8 | 1 | 3 | 3,258.106 us | 651,621 | 189.285 | 92.424% |

Random Fix16 seed `20260718` passes all 16,384 final-hidden comparisons with
maximum raw error zero at tolerance 32. The Host observes exactly three
completion records; the result contract reports `intermediate_host_copy=0`
and `kv_cache_owner=controller`. The CU trace gives the same interval for the
controller, both compute CUs, and the status sink.

The efficiency numerator is 616,710,144 useful MAC, while the denominator is
1,024 MAC/cycle times 651,621 cycles. Host/OpenCL event durations are not used
because they track simulator wall time in HW Emu. This single-layer result does
not include embedding, LM head, sampling, or 36-layer generation. Its complete
log, profile, manifests, and checksums are in the
[Q2.14 release artifact](../results/q214-resident-fix-20260818/).

## Standard-shape P8/G2 end-to-end HW-Emu result

The next release gate composes the production Host boundary around the same
coarse tasks. With `sequence_batch=1`, `P8` is one eight-row prefill block from
one sequence. Its prompt forward produces the first sample; one real D1
forward produces the second. Each forward issues Task 18, Task 19, and Task
20, so the L1 request contains six tasks.

| Evidence source | Host computation in timed interval | Scope | Prompt rows | Sampled outputs | Layers | Tasks | Common HW-Emu CU running time | Cycles at 200 MHz | Useful GMAC/s | Modeled useful-MAC efficiency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vitis 2022.2 HW Emu CU trace | No | P8 prefill + one real D1 | 8 | 2 | 1 | 6 | 5,953.465 us | 1,190,693 | 116.540 | 56.904% |

The deterministic random-Fix16/tied-embedding model passes both CPU-golden
steps after production inference: 4,096 checked values, maximum raw error zero
at tolerance 32, and an identical sampled-token sequence. No intermediate
hidden tensor is copied through Host memory; the controller owns RoPE, online
attention, HBM-resident KV, and subgraph scheduling. Host embedding,
coarse-task issue/status, and LM-head/argmax remain software responsibilities.
The common four-CU interval excludes that Host computation and cannot resolve
per-CU occupancy or individual inter-task gaps.

The measured request rate is 335.939 output tokens/s including prefill and is
therefore not a steady D1 rate. This result uses standard dimensions but one
decoder layer and deterministic random weights: it is neither checkpoint
accuracy nor a 36-layer or physical-board result. The complete evidence and
source provenance are in
[`results/qwen3b-e2e-20260820/`](../results/qwen3b-e2e-20260820/).

## Small-profile HW-Emu result

The small profile is a cross-layer residency test rather than a performance
proxy for Qwen2.5-3B. It contains two decoder layers with 64 hidden values and
runs five commands:

```text
layer 0: Task 18 -> Task 19
layer 1: Task 18 -> Task 19
model:   Task 20
```

Random Fix16 seed `20260718` passed all 64 final-hidden comparisons with zero
raw error. The host observed five expected tasks and no intermediate hidden
copy. A separate one-layer Task-18/19 run passed the same checks.

| Scope | Layers | Tasks | Common HW-Emu CU running time | XSim cycles | Projected latency at 200 MHz |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention+FFN layer | 1 | 2 | 47.665 us | 14,300 | 71.498 us |
| Stack plus final norm | 2 | 5 | 95.809 us | 28,743 | 143.714 us |
| Serial prompt/decode composition | 2 | 20 across 4 forwards | 365.248 us | 109,574 | 547.872 us |
| 8-row block plus final norm | 1 | 3 | 104.843 us | 31,453 | 157.265 us |
| Two-layer 8-row stack plus final norm | 2 | 5 | 185.655 us | 55,697 | 278.483 us |
| Block prompt P8 plus G2 | 2 | 10 across 2 forwards | 258.503 us | 77,551 | 387.755 us |
| Multi-block prompt P16 plus G1 | 2 | 10 across 2 forwards | 364.087 us | 109,226 | 546.131 us |
| Exact multi-block P16 final-tail golden | 2 | 9 across 2 blocks | 357.259 us | 107,178 | 535.889 us |
| Tail-block prompt P11 plus G1 (8+3) | 2 | 10 across 2 forwards | 296.956 us | 89,087 | 445.434 us |

The U50 HW-Emu image generates a 300-MHz XSim kernel clock (3.333 ns), even
though the physical implementation target is 200 MHz. Cycles are therefore
derived from the run-local `profile_kernels.csv` at 300 MHz and then projected
to 200 MHz. OpenCL event times and host wall time under HW Emu are simulator
wall-time proxies, not physical-device latency.

The serial composition row uses two prompt tokens and samples three tokens. Four
decoder forwards are required: two prompt forwards and two generated-token
forwards; the final sampled token is returned without an unnecessary extra
forward. Each forward runs two layers plus final norm, so the host observes 20
coarse tasks. The block composition row instead processes all eight prompt
tokens in one forward and performs one decode forward to produce two sampled
tokens; each forward runs two layers plus final norm, for ten tasks. Both are
protocol/residency tests rather than Qwen2.5-3B throughput claims.

The P16/G1 generation case exercises two consecutive prefill blocks at
positions 0--7 and 8--15 as a KV-residency/progress contract. A separate exact
P16 sequence run now closes the numerical gate: the second Task 18 reads the
first block's controller-managed cache, the first block omits Task 20 and D2H,
and the final 512 values pass the cross-block CPU fixed-point golden with
maximum raw error 10 within tolerance 64.

The P11/G1 tail-block gate uses positions 0--7 and 8--10. Its second Task 18
advertises `query_tokens=3`, so it covers valid-row propagation, causal KV
addressing, and finite-stream completion for a non-full final block. Like the
P16/G1 row, it is a protocol/residency result rather than a cross-block CPU
golden comparison.

## Resource gate

The standard-dimension controller synthesis changes as follows. The
coarse-task column is the previous release, and the persistent column is the
current implementation with model-initialized Norm/RoPE storage:

| Metric | Previous coarse-task controller | Persistent auxiliary-state controller | Change |
| --- | ---: | ---: | ---: |
| BRAM18 | 488 | 488 | 0 |
| DSP | 102 | 102 | 0 |
| FF | 446,088 | 446,153 | +65 (+0.015%) |
| LUT | 406,157 | 406,003 | -154 (-0.038%) |
| HLS estimated period | 3.746 ns | 3.746 ns | unchanged |

The published 2026-08-18 exact 8-row `qwen-layer` specialization increases
resident storage and adds a second vector-access path. Its historical HLS
estimates are:

| Design | BRAM18 | DSP | FF | LUT | Estimated period |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8-row Q2.14 controller | 1,212 | 123 | 468,481 | 453,487 | 3.746 ns |
| One exact compute CU | 48 | 677 | 197,555 | 117,643 | 2.433 ns |
| Status sink | 0 | 0 | 4,867 | 8,532 | 2.433 ns |
| Controller + two compute CUs + status | 1,308 | 1,477 | 868,458 | 697,305 | local estimates |

The profile-matched `qwen2.5-3b` build used by the new P8/G2 release gate has
the following independently regenerated estimates:

| Design | Instances | BRAM18 | DSP | FF | LUT | Estimated period |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Controller | 1 | 1,212 | 126 | 429,922 | 453,437 | 3.746 ns |
| Compute CU | 2 | 96 | 1,354 | 395,134 | 235,298 | 2.433 ns |
| Status sink | 1 | 0 | 0 | 4,867 | 8,532 | 2.433 ns |
| Whole system | 4 | 1,308 | 1,480 | 829,923 | 697,267 | 3.746 ns maximum |

The current totals are 48.661% BRAM18, 24.866% DSP, 47.605% FF, and 79.991%
LUT of the full U50. `scripts/report_qwen3b_hls_resources.sh` parses the three
underlying CSynth reports and derives the two-CU/system rows; the E2E archiver
stores that exact TSV beside the run evidence. The controller alone still
estimates approximately 104% of one SLR's LUT capacity, so the whole-device
resource gate is not a physical placement proof. The 200-MHz link target has
local HLS timing margin relative to the 3.746-ns estimate; final closure still
requires place-and-route.

## Measurement boundary

For HW Emu, the common profiler-reported running time of all four deployed CUs
is the authoritative modeled RTL measurement used here. The release reporter
requires exactly one top-level row for every CU and rejects times that differ
by more than 0.001 us. This field is not described as a raw waveform interval
or as Host wall time, and the identical per-function rows do not support a
claim about separable CU active occupancy or inter-task issue gaps. The reported
efficiency is useful MAC divided by the two-array peak over this common modeled
interval. Host-reported OpenCL event durations are retained only as diagnostics
because they follow simulator wall time. On a physical device,
event profiling represents the device timeline and host elapsed time becomes
the launch- and PCIe-inclusive measurement.

Weight, Norm/RoPE tables, and persistent KV allocation/preload are
model-initialization costs. All `(2 * layers + 1)` norm rows and the complete
position-indexed RoPE table are initialized once. A task sequence therefore
reports `auxiliary_migration_ms=0`; Task 18 reads RoPE and appends K/V directly
through controller HBM ports. The Host-observed sequence includes the initial
hidden migration, task/status synchronizations, and final hidden migration. It
is reported beside, rather than conflated with, the common modeled four-CU
interval.

In `generate --coarse-tasks`, token embedding and LM-head/sampling remain host
operations. All selected decoder layers, final normalization, hidden-state
residency, and KV-cache updates execute through the coarse-task accelerator
path. Blockwise prefill is now integrated for 1--8 consecutive query rows;
the remaining software arithmetic boundary is the vocabulary head/sampling.
