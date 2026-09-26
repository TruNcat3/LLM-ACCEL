# Fix16 Resident Reproduction

[Usage navigator](usage.md) | [Environment](environment.md) |
[Implementation map](implementations.md) | [Experiments](experiments.md) |
[Release workflow](release-workflow.md)

This page contains the complete command history for the Fix16 resident family:
the controller, two 8x64 compute CUs, controller-owned KV state, and the
coarse Task 18/19/20 runtime. Operator diagnostics, Q2.14 length sweeps, and
publication-package checks are evidence scopes of this family; they are
collected in [reproduction diagnostics](reproduction-diagnostics.md).

Generated HLS/Vitis products belong in the selected work root and are excluded
from Git. Run one expensive Vitis or XSim job at a time on a shared host.

## Profiles

Set `VITIS_8X64_MODEL_PROFILE` for Makefile targets and the resident wrapper:

| Profile | Main use | Shape |
| --- | --- | --- |
| `small` | protocol, residency, block-tail, and two-layer checks | reduced hidden/intermediate dimensions |
| `medium` | larger local protocol experiments | intermediate development shape |
| `qwen-layer` | standard one-layer and P8 gates | hidden 2048, intermediate 11008, max sequence 96 |
| `qwen-layer-long` | Q2.14 P/D operator sweep | hidden 2048, max sequence 2048 |
| `qwen2.5-3b` | full-shape coarse-task E2E | hidden 2048, 36 layers, max sequence 2048 |

`qwen-layer` and `qwen2.5-3b` share the hidden/intermediate/head geometry, but
they differ in layer count and sequence capacity (one layer versus 36, and
96-position versus 2048-position bounds). A qwen-layer compute XO must not be
reused for the full shape. The full-shape wrapper rebuilds a profile-matched
compute XO.

## Validation ladder

### Focused and closed-loop CSim

These commands compile and run bounded C++ testbenches:

```bash
make hls_csim_compute
make hls_csim_control
make hls_csim_closed_loop_8x64_resident_layer
make hls_csim_closed_loop_8x64_composed_layer
make hls_csim_closed_loop_8x64_resident_prefill_block
```

The CSim tests cover controller dispatch, two compute paths, feedback-stream
ordering, status behavior, and resident block/KV transitions. They do not
model RTL FIFO backpressure; finite-FIFO behavior and deadlock/progress
evidence are checked by the RTL CoSim commands below.

### RTL CoSim with finite streams

```bash
scripts/run_hls_resident_layer_cosim.sh
make hls_cosim_closed_loop_8x64_resident_layer
make hls_cosim_closed_loop_8x64_composed_layer
make hls_cosim_closed_loop_8x64_resident_prefill_block
```

Deadlock detection stays enabled. A pass requires completed RTL transactions
and a passing C post-check, not only successful synthesis. The CoSim flow also
rejects source changes between RTL preparation and simulation using SHA-256
fingerprints.

### Export and link a multi-kernel image

```bash
make vitis_8x64_xo VITIS_8X64_MODEL_PROFILE=qwen-layer

make vitis_8x64_link \
  TARGET=hw_emu FREQUENCY=200 \
  VITIS_8X64_MODEL_PROFILE=qwen-layer

make vitis_8x64_qwen_host vitis_8x64_emconfig \
  TARGET=hw_emu VITIS_8X64_MODEL_PROFILE=qwen-layer
```

The Makefile exports controller, compute, and status kernels with matching
model and packet configurations. `TARGET=sw_emu` and `TARGET=hw` are also
available through the Makefile for smoke runs; the resident convenience flow
is intentionally `hw_emu`-only until its functional loop passes.

## Run the resource-pruned resident layer

The wrapper isolates products by profile, token-row count, FIFO settings, and
variant tag. The default action builds the selected image; `run` executes the
single-token resident check after those inputs exist:

```bash
VITIS_8X64_MODEL_PROFILE=qwen-layer \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh all
VITIS_8X64_MODEL_PROFILE=qwen-layer \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run
```

The underlying host arguments are:

```text
--mode verify-resident-layer
--profile qwen-layer
--random-model
--seed 20260718
--position 0
```

## Run the coarse-task resident runtime

Task 18 is Attention, Task 19 is FFN, and Task 20 is final RMSNorm. Hidden
state and KV remain in accelerator memory across the selected task sequence.
The quick host-only contracts are safe to run before a tool build:

```bash
make test_coarse_task_program
make test_host_task_program_trace_contract
make test_coarse_task_residency_contract
```

Run a one-layer composed Attention+FFN pair:

```bash
VITIS_8X64_MODEL_PROFILE=qwen-layer \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-composed
```

Reproduce the profile-matched standard P8 image and its exact block gate:

```bash
VITIS_8X64_MODEL_PROFILE=qwen-layer \
CC8_RESIDENT_TOKEN_ROWS=8 \
VITIS_8X64_BUILD_EXACT_COMPUTE_XO=1 \
VITIS_8X64_RESIDENT_VARIANT_TAG=block_prefill_q214_resident_fix \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh all

VITIS_8X64_MODEL_PROFILE=qwen-layer \
CC8_RESIDENT_TOKEN_ROWS=8 \
VITIS_8X64_BUILD_EXACT_COMPUTE_XO=1 \
VITIS_8X64_RESIDENT_VARIANT_TAG=block_prefill_q214_resident_fix \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-block
```

The reduced two-layer and block-sequence scopes use the same family and are
not separate implementation generations:

```bash
# Two layers and Task 20 in the small cross-layer contract profile.
VITIS_8X64_MODEL_PROFILE=small \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh all
VITIS_8X64_MODEL_PROFILE=small \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-stack

# One P8 prefill block and one real decode forward through two layers.
VITIS_8X64_MODEL_PROFILE=small \
CC8_RESIDENT_TOKEN_ROWS=8 \
VITIS_8X64_E2E_TOKENS=0,1,2,3,4,5,6,7 \
VITIS_8X64_E2E_MAX_NEW_TOKENS=2 \
VITIS_8X64_E2E_LAYERS=2 \
VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8 \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-generate

# Exact P8 block through one and two small-profile layers.
VITIS_8X64_MODEL_PROFILE=small \
CC8_RESIDENT_TOKEN_ROWS=8 \
VITIS_8X64_BUILD_EXACT_COMPUTE_XO=1 \
VITIS_8X64_RESIDENT_VARIANT_TAG=block_prefill_q214_resident_fix \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-block
VITIS_8X64_MODEL_PROFILE=small \
CC8_RESIDENT_TOKEN_ROWS=8 \
VITIS_8X64_BUILD_EXACT_COMPUTE_XO=1 \
VITIS_8X64_RESIDENT_VARIANT_TAG=block_prefill_q214_resident_fix \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-block-stack

# Two P8 blocks; only the final block is materialized and CPU-golden checked.
VITIS_8X64_VERIFY_SEQUENCE_TOKENS=16 \
VITIS_8X64_VERIFY_BLOCK_SIZE=8 \
  scripts/run_vitis_8x64_prefill_sequence_verify_nohup.sh
```

The corresponding host modes are `verify-composed-layer`,
`verify-composed-prefill-block`, `verify-composed-prefill-stack`,
`verify-composed-prefill-sequence`, and `verify-composed-stack`. A passing
stack reports `2 * layers + 1` tasks, `intermediate_host_copy=0`, and a CPU
fixed-point final-hidden check. The generate path keeps embedding and LM-head
sampling on the host while decoder layers, final RMSNorm, hidden state, and KV
updates use the accelerator task sequence.

## Build and run the full Qwen2.5-3B shape

Use a large work root for HLS/Vitis products. The build wrapper owns its long
job with `tmux` and rebuilds the compute XO for `MAX_SEQ_LEN=2048`:

```bash
export VITIS_8X64_QWEN3B_WORK_ROOT=/fast-scratch/$USER/llm-accel-qwen3b
export VITIS_8X64_QWEN3B_TMP_ROOT=$VITIS_8X64_QWEN3B_WORK_ROOT/tmp

scripts/run_vitis_8x64_qwen3b_e2e_build_tmux.sh all
```

Choose a bounded run first. Set all workload fields explicitly so the run
header and result identity are unambiguous:

```bash
VITIS_8X64_E2E_PROMPT_TOKENS=8 \
VITIS_8X64_E2E_MAX_NEW_TOKENS=2 \
VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8 \
VITIS_8X64_E2E_LAYERS=1 \
  scripts/run_vitis_8x64_qwen3b_e2e_hwemu_tmux.sh

# The same Host and xclbin, with two exercised decoder layers.
VITIS_8X64_E2E_PROMPT_TOKENS=8 \
VITIS_8X64_E2E_MAX_NEW_TOKENS=2 \
VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8 \
VITIS_8X64_E2E_LAYERS=2 \
  scripts/run_vitis_8x64_qwen3b_e2e_hwemu_tmux.sh
```

The launcher defaults to `VITIS_8X64_E2E_LAYERS=36`; this is the full
long-running extension, not a quick check. Invoke it only when that workload
is intended, preferably with the remaining fields explicit:

```bash
VITIS_8X64_E2E_PROMPT_TOKENS=8 \
VITIS_8X64_E2E_MAX_NEW_TOKENS=2 \
VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8 \
VITIS_8X64_E2E_LAYERS=36 \
  scripts/run_vitis_8x64_qwen3b_e2e_hwemu_tmux.sh
```

A real packed checkpoint can replace deterministic random weights. The data
directory must already exist:

```bash
VITIS_8X64_E2E_MODEL_SOURCE=checkpoint \
VITIS_8X64_E2E_DATA_DIR=/path/to/packed/qwen2.5-3b \
VITIS_8X64_E2E_PROMPT_TOKENS=8 \
VITIS_8X64_E2E_MAX_NEW_TOKENS=2 \
VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8 \
VITIS_8X64_E2E_LAYERS=1 \
  scripts/run_vitis_8x64_qwen3b_e2e_hwemu_tmux.sh
```

## Checkpoint localization

Checkpoint mode intentionally copies every task output to the host. It is a
diagnostic scope, not a production residency or performance path. Reuse the
full-shape host created by the build wrapper; the old standalone
`/tmp/host_qwen_8x64_checkpoint.exe` name is not a repository artifact:

```bash
export VITIS_8X64_CHECKPOINT_HOST_EXE=\
"${VITIS_8X64_QWEN3B_WORK_ROOT}/build.hw_emu.${DEVICE}/host_qwen_8x64.exe"
VITIS_8X64_CHECKPOINT_TOKENS=8 \
VITIS_8X64_CHECKPOINT_LAYERS=3 \
  scripts/run_vitis_8x64_qwen3b_checkpoint_hwemu_tmux.sh

# Continue after collecting a known one-LSB difference, while retaining the
# strict mismatch count and maximum raw error in the log.
VITIS_8X64_CHECKPOINT_TOKENS=8 \
VITIS_8X64_CHECKPOINT_LAYERS=4 \
VITIS_8X64_CHECKPOINT_TOLERANCE=1 \
VITIS_8X64_CHECKPOINT_CONTINUE_ON_FAILURE=1 \
  scripts/run_vitis_8x64_qwen3b_checkpoint_hwemu_tmux.sh
```

The default checkpoint tolerance is zero. A nonzero difference is classified
as `rounding_within_tolerance` only when its maximum absolute raw Fix16 error
is within the explicit threshold; the strict mismatch count remains visible.

## Inspect and archive a completed run

Inspect a live run without starting another simulator:

```bash
scripts/status_vitis_8x64_qwen3b_e2e.sh \
  logs/qwen3b_e2e_hwemu_p8_g2_l1_<timestamp>.log
```

Archive a passing Host log and its exact generated artifacts. The optional
build log and source-root arguments preserve build provenance:

```bash
scripts/archive_vitis_8x64_e2e_run.sh \
  logs/qwen3b_e2e_hwemu_p8_g2_l1_<timestamp>.log \
  "${VITIS_8X64_QWEN3B_WORK_ROOT}/build.hw_emu.${DEVICE}" \
  results/qwen3b-e2e-<date> 200 200 1 \
  logs/qwen3b_e2e_all_<build-timestamp>.log
```

The archiver refuses incomplete or non-PASS runs and never overwrites an
archive. It records Host, xclbin, and emconfig SHA-256 identities; source and
build manifests make the generated image auditable. The archived trace scope
is the common four-CU running-time interval, not per-CU occupancy or physical
board timing.

For a running job, the watcher performs the same archive and release audit
after a zero exit:

```bash
scripts/watch_vitis_8x64_e2e_archive_tmux.sh \
  logs/qwen3b_e2e_hwemu_p8_g2_l1_<timestamp>.log \
  "${VITIS_8X64_QWEN3B_WORK_ROOT}/build.hw_emu.${DEVICE}" \
  /tmp/qwen3b-e2e-archive 3600 \
  logs/qwen3b_e2e_all_<build-timestamp>.log
```

Verify and render a completed archive:

```bash
scripts/verify_qwen3b_e2e_release.sh \
  logs/qwen3b_e2e_hwemu_p8_g2_l1_<timestamp>.log \
  "${VITIS_8X64_QWEN3B_WORK_ROOT}/build.hw_emu.${DEVICE}" \
  /path/to/qwen3b-e2e-archive \
  /path/to/release-source-root
scripts/render_vitis_8x64_e2e_result_markdown.sh \
  /path/to/qwen3b-e2e-archive > /tmp/qwen3b-e2e-README.md
```

Install a verified temporary archive atomically into `results/`:

```bash
scripts/install_vitis_8x64_e2e_result.sh \
  /tmp/llm_accel_qwen3b_e2e_p8_g2_l1_archive_<timestamp> \
  qwen3b-e2e-<date>
```

The result index still requires a deliberate entry in `results/README.md`.
The package must be reviewed as a claim with a stated workload and timing
boundary rather than treated as an automatically generated performance claim.

## Compare layer-count gates

After archiving passing rows, compare only packages with the same workload,
clock, artifact identities, and validation status:

```bash
scripts/report_vitis_8x64_e2e_trace.sh \
  /path/to/profile_kernels.csv qwen2.5-3b 8 2 1 8 200 200 1 \
  /path/to/qwen3b_e2e_host.log

scripts/report_vitis_8x64_e2e_scaling.sh \
  results/qwen3b-e2e-20260820/performance.tsv \
  results/qwen3b-e2e-l2-20260821/performance.tsv \
  results/qwen3b-e2e-l36-<date>/performance.tsv
```

The reporter's per-layer field normalizes the common four-CU interval; it is
not an isolated layer latency or occupancy measurement. `P8` names exactly
eight active query rows from one sequence; smaller resident blocks are
supported but are not P8. Decode remains one row (`D1`).

The profile-matched Qwen2.5-3B image uses a 200 MHz kernel target. Earlier
small-profile archives may retain their 300 MHz XSim frequency; pass the
recorded clock for that package rather than copying the full-shape value.

## Pipeline parameters

The main research settings are:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `CC8_WEIGHT_TILE_FIFO_DEPTH` | 2 | 512-bit weight-block FIFO depth |
| `CC8_WEIGHT_TILE_LOAD_II` | 2 | Weight producer initiation interval |
| `CC8_MM_WAVE_RESULT_FIFO_DEPTH` | 33 | Cross-wave result buffering |
| `CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW` | 1 | Overlap load/drive/collect |
| `CC8_ENABLE_MM_WAVE_REPEAT` | 0 | Multi-wave command reuse; disabled in release path |
| `FREQUENCY` | flow dependent | Requested kernel frequency in MHz |
| `THREADS` | 16 | HLS/Vivado worker count |

Example finite-FIFO experiment:

```bash
CC8_WEIGHT_TILE_FIFO_DEPTH=4 \
CC8_WEIGHT_TILE_LOAD_II=2 \
CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW=1 \
  scripts/run_hls_resident_layer_cosim.sh
```

Compare CoSim progress, HLS II/timing, resource reports, and HW-Emu cycles.
A deeper FIFO is not an optimization unless a measured bottleneck improves.

## Expected checkpoints

The public Fix16 packages use deterministic seeds. See the result package for
the authoritative raw values and checksums:

| Scope | Seed | Expected evidence |
| --- | ---: | --- |
| Resident single-token layer | 20260718 | 2048 outputs bit-for-bit |
| Coarse Task 18/19 layer | 20260718 | CPU golden pass; `intermediate_host_copy=0` |
| Small two-layer stack | 20260718 | five tasks; final hidden exact |
| Eight-row prefill block | 20260722 | checksum `0xb72a92cb5224f0c7` |
| Standard Qwen-layer P8 Task 18/19/20 | 20260718 | 16,384 values exact; three tasks |
| Qwen2.5-3B P8/G2/L1 | 20260718 | six tasks; two CPU-golden steps |
| Qwen2.5-3B P8/G2/L2 | 20260718 | ten tasks; two CPU-golden steps |

The historical Q2.14 P/D expected rows and full release checks are in
[reproduction diagnostics](reproduction-diagnostics.md).
