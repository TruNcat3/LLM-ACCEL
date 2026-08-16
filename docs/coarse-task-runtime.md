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

### Full-profile HBM capacity guard

The `qwen2.5-3b` host plan checks aggregate allocation against the actual
grouped connectivity rather than checking each logical shard against one
256-MiB pseudo-channel. Each logical weight shard is 346,816,512 bytes and is
striped over three pseudo-channels (805,306,368 bytes). A shared group contains
shard `i`, shard `i + 8`, and, for the first two groups, one K or V cache. The
worst-case payload is 769,130,496 bytes, leaving 36,175,872 bytes of headroom.
The host rejects initialization if either the per-shard or shared-group guard
fails.

## Verification interfaces

The model host exposes four relevant paths:

- `--mode verify-composed-layer`: deterministic random-weight Task 18 followed
  by Task 19, compared with a CPU fixed-point golden layer;
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
VITIS_8X64_E2E_TOKENS=0,1 \
VITIS_8X64_E2E_MAX_NEW_TOKENS=3 \
VITIS_8X64_E2E_LAYERS=2 \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-generate

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
| Persistent Norm/RoPE/KV focused test | PASS, three independent norm rows and banked RoPE/KV checked |
| Small two-layer/five-task HW Emu | PASS, 64/64 values exact, `intermediate_host_copy=0` |
| Small prompt/decode composition HW Emu | PASS, 4 forwards/20 tasks, controller-owned KV, no intermediate host copy |
| Standard Qwen2.5-3B layer-shape HW Emu | In progress for this release |

The authoritative RTL CoSim report records three passing transactions with
minimum/average/maximum latency of 1,204/2,664/4,497 cycles. The finite stream
depth configuration is exercised with deadlock detection enabled.

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

| Scope | Layers | Tasks | XSim active interval | XSim cycles | Projected latency at 200 MHz |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention+FFN layer | 1 | 2 | 47.665 us | 14,300 | 71.498 us |
| Stack plus final norm | 2 | 5 | 95.809 us | 28,743 | 143.714 us |
| Serial prompt/decode composition | 2 | 20 across 4 forwards | 365.248 us | 109,574 | 547.872 us |

The U50 HW-Emu image generates a 300-MHz XSim kernel clock (3.333 ns), even
though the physical implementation target is 200 MHz. Cycles are therefore
derived from the run-local `profile_kernels.csv` at 300 MHz and then projected
to 200 MHz. OpenCL event times and host wall time under HW Emu are simulator
wall-time proxies, not physical-device latency.

The composition row uses two prompt tokens and samples three tokens. Four
decoder forwards are required: two prompt forwards and two generated-token
forwards; the final sampled token is returned without an unnecessary extra
forward. Each forward runs two layers plus final norm, so the host observes 20
coarse tasks. This is a protocol and residency test, not a Qwen2.5-3B
throughput claim. Prompt processing is still serial-token prefill in this path.

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

Including two existing compute CUs gives an HLS system estimate of
approximately 584 BRAM18, 1,456 DSP, 841k FF, and 642k LUT: about 22%, 24%,
48%, and 74% of the full U50 respectively. This passes the functional HW-Emu
gate, but controller placement and routing remain risks for a physical build.

## Measurement boundary

For HW Emu, the run-local CU interval is the authoritative modeled RTL
measurement. Host-reported OpenCL event durations are retained only as
diagnostics because they follow simulator wall time. On a physical device,
event profiling represents the device timeline and host elapsed time becomes
the launch- and PCIe-inclusive measurement.

Weight, Norm/RoPE tables, and persistent KV allocation/preload are
model-initialization costs. All `(2 * layers + 1)` norm rows and the complete
position-indexed RoPE table are initialized once. A task sequence therefore
reports `auxiliary_migration_ms=0`; Task 18 reads RoPE and appends K/V directly
through controller HBM ports. The host-observed sequence includes the initial
hidden migration, task/status synchronizations, and final hidden migration.
Kernel-active cycles are reported separately so that datapath and
software/runtime overhead are not conflated.

In `generate --coarse-tasks`, token embedding and LM-head/sampling remain host
operations. All selected decoder layers, final normalization, hidden-state
residency, and KV-cache updates execute through the coarse-task accelerator
path. This explicit boundary is retained until blockwise prefill and an
accelerator-side vocabulary head are integrated.
