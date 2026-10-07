# Diagnostics and evidence reproduction

[Usage navigator](usage.md) | [`cowave-fix16-2-8-64` recipes](reproduction-resident.md) |
[Environment](environment.md) | [Release workflow](release-workflow.md) |
[Q2.14 report](q214-pd-length-hwemu.md) | [Reference/history](reference.md)

This page collects bounded `cowave-fix16-2-8-64` operator diagnostics,
historical Q2.14 prefill/decode checks, and read-only evidence verification.
These are scopes of the resident family, not additional designs. The
streaming-split and quantized families keep their complete commands in their
case READMEs; the complete-layer quantized profiles use
[`cases/quantized-layer/README.md`](../cases/quantized-layer/README.md).

## Run the 8-row diagnostic prefill block

The operator-level prefill profile checks arithmetic and packet payloads. It is
not the resource-pruned resident deployment image and its cycles must not be
combined with coarse-task E2E rows.

```bash
make vitis_8x64_run_qwen \
  TARGET=hw_emu \
  FREQUENCY=200 \
  VITIS_8X64_MODEL_PROFILE=qwen-layer \
  RUN_TIMEOUT=604800 \
  QWEN_ARGS="--mode profile-prefill-block --profile qwen-layer --prefill-len 8 --random-model --seed 20260722"
```

The host should end with a passing prefill profile, the expected 48 attention
matrix-multiply tasks, and 1536 completed packets. The test is diagnostic: it
does not establish controller-owned KV residency or a complete model result.

## Run the Q2.14 P/D context-length sweep

Build the prefill-specialized HW-Emu image and matching host:

```bash
VITIS_8X64_MODEL_PROFILE=qwen-layer-long \
CC8_PREFILL_VARIANT=q214exp18 \
  scripts/build_vitis_8x64_prefill_eval_hwemu.sh all
```

The build script accepts `compute-xo`, `control-xo`, `status-xo`, `link`,
`host`, `all`, and `run`. Launch one isolated HW-Emu process for each phase
and context length:

```bash
scripts/launch_vitis_8x64_pd_sweep_tmux.sh \
  --prefix q214_pd \
  --build-dir <generated-hw_emu-build> \
  --profile qwen-layer-long \
  --seed 20260722

scripts/watch_vitis_8x64_pd_sweep_tmux.sh q214_pd 3600
```

The sweep covers prefill and decode at contexts 64, 256, 512, and 1024.
Prefill runs the final eight-row block at each length; decode runs one new
token. The published package is
[`results/q214-pd-20260811/`](../results/q214-pd-20260811/), with the raw
logs and profiles under its `raw/` directory.

Rebuild its performance and precision tables from raw evidence without a new
simulation:

```bash
make verify_q214_pd_release
```

The verifier requires all eight passing raw cases, one common four-CU profile
interval per case, and byte-identical derived TSV files. That common interval
is a modeled device span; it excludes host scheduling, migration, fixture
packing, and golden checks.

## Standard-dimension closure checks

These checks distinguish the historical Q2.14 payload path from direct
arithmetic and audit the full-profile Host task plan:

```bash
make test_q214_payload_golden
make test_qwen3b_e2e_plan
```

After the exact profile-matched `qwen-layer` image exists, the standard P8
Task-18/19/20 HW-Emu gate can be launched with its dedicated tmux wrapper:

```bash
scripts/run_vitis_8x64_qwen_exact_p8_tmux.sh
```

This wrapper is a long-running gate and requires the exact compute XO,
`CC8_RESIDENT_TOKEN_ROWS=8`, `VITIS_8X64_BUILD_EXACT_COMPUTE_XO=1`, and a
profile-matched build directory. Use the bounded resident commands first.

## Historical resident package verification

The resident package verifier checks source and artifact manifests, raw Host
progress, validation rows, performance rows, and generated-artifact identity:

```bash
make verify_q214_resident_release
```

By default it verifies the archived snapshot as a historical package. To
require that the current checkout still matches every path in its historical
source manifest:

```bash
Q214_VERIFY_CURRENT_SOURCE=1 make verify_q214_resident_release
```

That strict form is expected to fail after source evolution; failure does not
invalidate the archived package. The package is
[`results/q214-resident-fix-20260818/`](../results/q214-resident-fix-20260818/).

## Publication tree and checksums

These read-only checks inspect documentation, local links, result manifests,
and checksums without starting Vitis synthesis, RTL CoSim, or HW-Emu:

```bash
scripts/check_environment.sh publication
make test_publication_tree
make verify_result_checksums
```

The full non-simulator release gate also compiles host-only contracts and
requires the HLS/XRT headers:

```bash
source scripts/setup_environment.sh
scripts/check_environment.sh hls
make test_publication_release
```

Root manifest regeneration, staging, installation, and publication ownership
are documented in the [release workflow](release-workflow.md). A result
directory is immutable: corrected evidence or a changed interpretation gets a
new package or an index/documentation change rather than an in-place rewrite.

## Archive and reporting boundaries

For a completed Qwen2.5-3B run, the resident guide retains the exact status,
archive, render, install, watcher, and scaling commands:

- [Inspect, archive, and render a resident run](reproduction-resident.md#inspect-and-archive-a-completed-run)
- [Compare layer-count gates](reproduction-resident.md#compare-layer-count-gates)
- [Expected resident checkpoints](reproduction-resident.md#expected-checkpoints)

The archive must identify the Host executable, xclbin, emulation configuration,
source snapshot, workload (`prompt_tokens`, `sampled_output_tokens`, layers,
block rows), and timing scope. Host inference, accelerator CU interval, and
post-inference CPU-golden time are separate fields.

## Reading results

### Functional output

The host must end with a named `PASS` line and the expected task, packet, and
post-inference validation counts. A CPU golden is an out-of-band numerical
oracle; it is not part of the modeled accelerator interval.

### Hardware-emulation cycles

Use `profile_kernels.csv` for simulated kernel-active time. Convert an
explicitly identified interval using:

```text
cycles = running_time_us * frequency_MHz
```

Do not report XSim CPU wall time as accelerator latency. The Q2.14 profile
contains multiple host-submitted calls; its published value is the accumulated
controller-event interval. The resident coarse-task profile reports one common
interval for controller, two compute CUs, and status sink; it does not expose
separate per-CU occupancy or task-issue gaps.

`P8` names exactly eight active query rows from one sequence; smaller resident
blocks are supported but are not P8. `G2` produces two sampled outputs: the
first follows the prompt forward and the second includes one actual D1 decode
forward; `G1` is prefill-only. A CPU golden is an out-of-band numerical
oracle. `intermediate_host_copy=0` and `kv_cache_owner=controller` are
residency assertions. A physical-board run requires a separate result package
and must not be substituted for HW-Emu evidence.

### HLS reports

For each exported kernel, retain the estimated clock period, achieved loop II,
BRAM/DSP/FF/LUT/URAM totals, and warnings about array replication, mux growth,
or FIFO depth. HLS latency bounds describe local structure; they are not a
replacement for a measured integrated HW-Emu interval.

## Generated artifacts

HLS/Vivado projects, XO/xclbin files, DCPs, waveforms, emulator output,
executables, checkpoints, and generated weights are excluded from Git. A
result package may retain compact logs, reports, profiles, source manifests,
and SHA-256 identities for those external artifacts.

## Quantized candidate boundary

The isolated `cowave-quantized-blocks` route is intentionally separate from
these Fix16 diagnostics:

```bash
make test_quantized_cu_planner
scripts/run_quantized_block_regression.sh csim
```

Use the [quantized matrix-block README](../cases/quantized-block/README.md)
for W4A4/W8A8 HLS, single-bank variants, DSP diagnostics, synthesis, RTL
CoSim, planner overrides, packet shapes, and resource estimates. Use the
[public quantized-layer README](../cases/quantized-layer/README.md) for the
complete-layer profiles and their current P66+D1 evidence. Component tests
have no scale application, controller integration, full-model accuracy claim,
or board measurement.
