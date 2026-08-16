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

## Verification interfaces

The model host exposes three relevant paths:

- `--mode verify-composed-layer`: deterministic random-weight Task 18 followed
  by Task 19, compared with a CPU fixed-point golden layer;
- `--mode verify-composed-stack`: all layers in the selected profile followed
  by Task 20, with the `2 * layers + 1` task contract checked explicitly;
- `--coarse-tasks`: use the same HBM-resident task sequence in the normal
  `run` or `generate` host path.

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
| Small two-layer/five-task HW Emu | PASS, 64/64 values exact, `intermediate_host_copy=0` |
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
| Attention+FFN layer | 1 | 2 | 48.598 us | 14,579 | 72.897 us |
| Stack plus final norm | 2 | 5 | 96.724 us | 29,017 | 145.086 us |

The U50 HW-Emu image generates a 300-MHz XSim kernel clock (3.333 ns), even
though the physical implementation target is 200 MHz. Cycles are therefore
derived from the run-local `profile_kernels.csv` at 300 MHz and then projected
to 200 MHz. OpenCL event times and host wall time under HW Emu are simulator
wall-time proxies, not physical-device latency.

## Resource gate

The standard-dimension controller synthesis changes as follows relative to the
earlier resident single-layer controller:

| Metric | Earlier resident controller | Coarse-task controller | Change |
| --- | ---: | ---: | ---: |
| BRAM18 | 488 | 488 | 0 |
| DSP | 90 | 102 | +12 |
| FF | 421,880 | 446,088 | +24,208 (+5.7%) |
| LUT | 390,283 | 406,157 | +15,874 (+4.1%) |
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

Weight and persistent KV allocation/preload are model-initialization costs.
The composed sequence measurement includes the initial hidden migration,
per-layer auxiliary-data migration, all task/status synchronizations, and the
final hidden migration in its host-observed interval. Kernel-active cycles are
reported separately so that datapath and software/runtime overhead are not
conflated.
