# Usage and Reproduction

## 1. Toolchain

The recorded experiments use:

- Ubuntu 20.04;
- Vitis, Vivado, and Vitis HLS 2022.2;
- XRT 2022.2;
- platform `xilinx_u50_gen3x16_xdma_5_202210_1` for Vitis integration.

Set the environment script without modifying repository files:

```bash
export VITIS_ENV_SCRIPT=/path/to/vitis_env_22.sh
source "$VITIS_ENV_SCRIPT"
```

The scripts default to `/home/hepc/env/vitis_env_22.sh` when the variable is
not supplied.

## 2. Repository profiles

| Profile | Purpose |
| --- | --- |
| `small` | Fast software/HLS development |
| `medium` | Larger protocol and pipeline checks |
| `qwen-layer` | Standard one-layer shape: 2048/11008 |
| `qwen-layer-long` | One-layer long-context profile with `max_seq_len=2048` |
| `qwen2.5-3b` | Full-model layout configuration |

Select a profile with `VITIS_8X64_MODEL_PROFILE`. Generated artifacts are
placed under profile-specific `vitis_8x64/` and `reports/` directories and are
ignored by Git.

## 3. Validation ladder

### Focused and closed-loop CSim

```bash
make hls_csim_compute
make hls_csim_control
make hls_csim_closed_loop_8x64_resident_layer
```

The closed loop instantiates the controller, two compute paths, feedback
streams, and status behavior in one testbench.

### RTL CoSim with bounded streams

```bash
scripts/run_hls_resident_layer_cosim.sh
```

Deadlock detection is enabled by default. Do not add
`-disable_deadlock_detection` when evaluating FIFO depth or scheduling
changes. A successful run must report completed RTL transactions and a passing
C post-check, not merely successful compilation.

### Export multi-kernel XO files

```bash
make vitis_8x64_xo \
  VITIS_8X64_MODEL_PROFILE=qwen-layer
```

This exports controller, compute, and status kernels with matching model and
packet configurations.

### Build hardware emulation

```bash
make vitis_8x64_link \
  TARGET=hw_emu \
  FREQUENCY=200 \
  VITIS_8X64_MODEL_PROFILE=qwen-layer

make vitis_8x64_qwen_host vitis_8x64_emconfig \
  TARGET=hw_emu \
  VITIS_8X64_MODEL_PROFILE=qwen-layer
```

### Run the resource-pruned resident layer

The convenience script isolates artifacts by pipeline configuration:

```bash
scripts/build_vitis_8x64_resident_layer_hwemu.sh all
scripts/build_vitis_8x64_resident_layer_hwemu.sh run
```

Equivalent model-level host arguments are:

```text
--mode verify-resident-layer
--profile qwen-layer
--random-model
--seed 20260718
--position 0
```

### Run the coarse-task resident runtime

The same image exposes an Attention+FFN pair and a complete selected-profile
stack followed by final RMSNorm:

```bash
# Task 18 -> Task 19, standard Qwen layer dimensions by default.
VITIS_8X64_MODEL_PROFILE=qwen-layer \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-composed

# Reproduce the publication image and the standard-dimension P8 numerical
# gate. All four variables are part of the artifact identity.
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

# Two layers and Task 20 in the small cross-layer contract profile.
VITIS_8X64_MODEL_PROFILE=small \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh all
VITIS_8X64_MODEL_PROFILE=small \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-stack

# One 8-row prefill block, then one decode forward for two sampled tokens.
VITIS_8X64_MODEL_PROFILE=small \
CC8_RESIDENT_TOKEN_ROWS=8 \
VITIS_8X64_E2E_TOKENS=0,1,2,3,4,5,6,7 \
VITIS_8X64_E2E_MAX_NEW_TOKENS=2 \
VITIS_8X64_E2E_LAYERS=2 \
VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8 \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-generate

# Verify one exact 8-row Task-18/19/20 block against the CPU fixed-point model.
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

# Verify the same 8-row block through both Small-profile layers (five tasks).
VITIS_8X64_MODEL_PROFILE=small \
CC8_RESIDENT_TOKEN_ROWS=8 \
VITIS_8X64_BUILD_EXACT_COMPUTE_XO=1 \
VITIS_8X64_RESIDENT_VARIANT_TAG=block_prefill_q214_resident_fix \
  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-block-stack

# Verify a 16-token sequence as two P8 blocks. The first block is released
# after KV update; only the second block is materialized and CPU-golden checked.
VITIS_8X64_VERIFY_SEQUENCE_TOKENS=16 \
VITIS_8X64_VERIFY_BLOCK_SIZE=8 \
  scripts/run_vitis_8x64_prefill_sequence_verify_nohup.sh
```

The corresponding host modes are `verify-composed-layer`,
`verify-composed-prefill-block`, `verify-composed-prefill-stack`,
`verify-composed-prefill-sequence`, and
`verify-composed-stack`. A passing stack reports `2 * layers + 1` tasks,
`intermediate_host_copy=0`, and a passing CPU fixed-point final-hidden check.
The normal `run` and `generate` paths select the same runtime with
`--coarse-tasks`. In the generate path, embedding and LM-head/sampling remain
on the host; decoder layers, final RMSNorm, intermediate hidden state, and KV
updates use the accelerator task sequence. With `--coarse-tasks`, the host
chunks a prompt into blocks of at most `--prefill-block-size` rows (1--8);
decode remains a one-row task.

### Build and run the full Qwen2.5-3B shape

The full-shape wrapper keeps large temporary HLS/Vitis products outside the
repository and owns long jobs with `tmux`:

```bash
# Synthesize qwen2.5-3b compute/controller/status XOs, link the four-CU
# HW-Emu image, and compile the matching Host. The compute XO is rebuilt for
# MAX_SEQ_LEN=2048; reusing the 96-position qwen-layer XO is not safe.
scripts/run_vitis_8x64_qwen3b_e2e_build_tmux.sh all

# Choose one run; do not launch both against the same build concurrently.
# Bounded full-shape release gate: P8 prompt plus one real D1 forward through
# one standard decoder layer, six coarse tasks, and a post-inference CPU
# fixed-point oracle. This is the practical RTL HW-Emu numerical/trace gate.
VITIS_8X64_E2E_LAYERS=1 \
  scripts/run_vitis_8x64_qwen3b_e2e_hwemu_tmux.sh

# Full 36-layer extension of the same contract: random deterministic Fix16
# weights and 146 coarse tasks. This is the launcher default and is expected
# to take much longer under RTL HW Emu.
scripts/run_vitis_8x64_qwen3b_e2e_hwemu_tmux.sh

# A real packed checkpoint can replace the deterministic random model.
VITIS_8X64_E2E_MODEL_SOURCE=checkpoint \
VITIS_8X64_E2E_DATA_DIR=/path/to/packed/qwen2.5-3b \
  scripts/run_vitis_8x64_qwen3b_e2e_hwemu_tmux.sh
```

`P8` means eight active query rows from one sequence; it is neither batch size
nor eight tokens generated in parallel. Decode remains `D1`. The first new
token is sampled from the prompt forward. Therefore G2 is the
smallest generation request that contains one actual decode forward; G1 is a
prefill/TTFT-only gate. The Host emits one `COARSE_TASK_PROGRESS` line after
each completed Task 18/19/20. Under RTL HW Emu, CPU wall time is simulator
runtime and must not be reported as accelerator latency. New launcher logs
also freeze SHA-256 identities for the Host executable, xclbin, and emulation
configuration so a performance row can be tied to the exact generated inputs.
The reporter validates the three identities as an all-or-none set and carries
them into its tab-separated output.

Inspect a live run without launching a second simulator:

```bash
scripts/status_vitis_8x64_qwen3b_e2e.sh \
  logs/qwen3b_e2e_hwemu_p8_g2_l1_<timestamp>.log
```

After a long run finishes, freeze its build log, raw Host log, run-local CU
profile, HLS resources, release-candidate source snapshot, generated artifact
identities, performance row, manifest, and checksums atomically:

```bash
scripts/archive_vitis_8x64_e2e_run.sh \
  logs/qwen3b_e2e_hwemu_p8_g2_l1_<timestamp>.log \
  /tmp/llm_accel_qwen3b_q214_resident_fix/build.hw_emu.xilinx_u50_gen3x16_xdma_5_202210_1 \
  results/qwen3b-e2e-<date> 200 200 1 \
  logs/qwen3b_e2e_all_<build-timestamp>.log
```

The archiver refuses incomplete/non-PASS runs and never overwrites an existing
archive directory. The Host/XCLBIN/emconfig SHA-256 values are authoritative
for the binary run. `source_manifest.tsv` is an archive-time snapshot of the
release-candidate worktree. `build_source_equivalence.tsv` reconstructs the
source root recorded by the HLS build log and requires every kernel, local
header, Tcl configuration, connectivity file, and build harness to match the
release worktree byte for byte. The publication-oriented Makefile remains a
separate `release_entrypoint`; Host binary identity is checked directly.
`manifest.tsv` also fixes the trace scope as
`common_four_CU_running_time`, marks per-CU occupancy and inter-task issue gaps
as unresolved by the profiler CSV, and records that the result is not a
physical-board measurement. It separately records the Host inference scope
(`embedding_plus_lm_head_argmax`), the accelerator scope
(`decoder_layers_final_norm_rope_online_attention_kv`), and the post-inference
CPU-golden scope.

After the archive passes its release audit, render its evidence page from the
machine-readable workload fields rather than transcribing table values by
hand:

```bash
scripts/verify_qwen3b_e2e_release.sh \
  logs/qwen3b_e2e_hwemu_p8_g2_l1_<timestamp>.log \
  /tmp/llm_accel_qwen3b_q214_resident_fix/build.hw_emu.xilinx_u50_gen3x16_xdma_5_202210_1 \
  /path/to/qwen3b-e2e-archive \
  /path/to/release-source-root
scripts/render_vitis_8x64_e2e_result_markdown.sh \
  /path/to/qwen3b-e2e-archive > /tmp/qwen3b-e2e-README.md
```

The release verifier derives `P`, `G`, `L`, block count, decode-forward count,
task sequence, checked-value count, and numerical tolerance from the immutable
Host-log header. The same contract therefore covers L1 qualification and later
standard-shape multi-layer runs without weakening topology, residency, or
post-inference numerical checks.

The rendered workload table distinguishes sequence batch, prompt length,
sampled outputs, real D1 forwards, prefill blocks, and maximum active query
rows per block. It also prints the Host/accelerator/CPU-golden boundary and
labels HW-Emu timing, HLS estimates, and physical-board evidence separately.
Install a verified temporary archive into the publication tree atomically:

```bash
scripts/install_vitis_8x64_e2e_result.sh \
  /tmp/llm_accel_qwen3b_e2e_p8_g2_l1_archive_<timestamp> \
  qwen3b-e2e-<date>
```

The installer refuses an invalid source package or existing destination,
renders the result README, rebuilds an archive-relative SHA-256 manifest over
every installed file, verifies it, and only then exposes the final directory.
The result must still be added explicitly to `results/README.md`; this keeps
the index claim and its measurement scope under review rather than generating
research prose mechanically.

For a running job, the hourly watcher performs the same archive and release
audit automatically after a zero exit:

```bash
scripts/watch_vitis_8x64_e2e_archive_tmux.sh \
  logs/qwen3b_e2e_hwemu_p8_g2_l1_<timestamp>.log \
  /tmp/llm_accel_qwen3b_q214_resident_fix/build.hw_emu.xilinx_u50_gen3x16_xdma_5_202210_1 \
  /tmp/qwen3b-e2e-archive 3600 \
  logs/qwen3b_e2e_all_<build-timestamp>.log
```

After a successful run, derive the modeled cycle/performance row from the
run-local CU profile and simultaneously verify the Host contract:

```bash
scripts/report_vitis_8x64_e2e_trace.sh \
  /path/to/profile_kernels.csv qwen2.5-3b 8 2 1 8 200 200 1 \
  /path/to/qwen3b_e2e_host.log
```

The profile-matched Qwen2.5-3B wrapper links `KERNEL_CLK` at 200 MHz, so both
frequency arguments are 200. Earlier Small-profile archives used a distinct
300-MHz HW-Emu image and retain 300 as their recorded XSim frequency. The new
launcher reads `KERNEL_CLK` from the XCLBIN into the Host log; the archiver
uses that value by default and rejects an inconsistent manual frequency. The
standalone reporter defaults an omitted XSim frequency to the stated target.

The report audits every `COARSE_TASK_PROGRESS` event: Task-18/19/20 opcode and
phase, layer bounds, active query rows, group start/end, controller duration,
output materialization, and non-final prompt-block release. It also rejects a
missing final Host PASS, setup/weight-preload evidence, incorrect task count,
Host-managed KV, intermediate Host copies, or an unexpected timing domain.
Standard-dimension single-block numerical closure can be rerun independently
with:

```bash
# Reproduce the raw-payload versus direct-arithmetic discriminator without
# launching RTL simulation.
make test_q214_payload_golden

# Compile the full-profile Host and audit its HBM/task execution plan.
make test_qwen3b_e2e_plan

# Run the complete standard P8 Task-18/19/20 HW-Emu gate.
scripts/run_vitis_8x64_qwen_exact_p8_tmux.sh
```

After archiving the passing Host log and run-local CU profile, verify every
release-critical source hash, raw checksum, validation row, performance row,
and artifact identity before committing:

```bash
make verify_q214_resident_release
```

The closed-loop finite-FIFO checks are:

```bash
make hls_csim_closed_loop_8x64_composed_layer
make hls_cosim_closed_loop_8x64_composed_layer
make hls_csim_closed_loop_8x64_resident_prefill_block
make hls_cosim_closed_loop_8x64_resident_prefill_block
```

The CoSim flow keeps deadlock detection enabled and rejects source changes
between RTL synthesis and simulation by comparing SHA-256 fingerprints.

### Run the 8-row diagnostic prefill block

```bash
make vitis_8x64_run_qwen \
  TARGET=hw_emu \
  FREQUENCY=200 \
  VITIS_8X64_MODEL_PROFILE=qwen-layer \
  RUN_TIMEOUT=604800 \
  QWEN_ARGS="--mode profile-prefill-block --profile qwen-layer --prefill-len 8 --random-model --seed 20260722"
```

The prefill profile is intentionally diagnostic: it performs operator-level
golden checks and should not be interpreted as a resource-pruned deployment
image.

### Run the Q2.14 P/D context-length sweep

Build the prefill-specialized hardware-emulation image and host:

```bash
VITIS_8X64_MODEL_PROFILE=qwen-layer-long \
CC8_PREFILL_VARIANT=q214exp18 \
scripts/build_vitis_8x64_prefill_eval_hwemu.sh all
```

Then launch one isolated hardware-emulation process per phase/length:

```bash
scripts/launch_vitis_8x64_pd_sweep_tmux.sh \
  --prefix q214_pd \
  --build-dir <generated-hw_emu-build> \
  --profile qwen-layer-long \
  --seed 20260722

scripts/watch_vitis_8x64_pd_sweep_tmux.sh q214_pd 3600
```

The sweep covers P/D at contexts 64, 256, 512, and 1024. Prefill runs the
final 8-row block at each length; Decode runs one new token. The publication
artifact and expected tables are in
[`results/q214-pd-20260811/`](../results/q214-pd-20260811/).

## 4. Pipeline parameters

| Variable | Default research setting | Meaning |
| --- | ---: | --- |
| `CC8_WEIGHT_TILE_FIFO_DEPTH` | 2 | 512-bit weight-block FIFO depth |
| `CC8_WEIGHT_TILE_LOAD_II` | 2 | Weight producer initiation interval |
| `CC8_MM_WAVE_RESULT_FIFO_DEPTH` | 33 | Cross-wave result buffering |
| `CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW` | 1 | Overlap load/drive/collect across waves |
| `CC8_ENABLE_MM_WAVE_REPEAT` | 0 | Multi-wave command reuse; disabled in release path |
| `FREQUENCY` | flow dependent | Requested Vitis kernel frequency in MHz |
| `THREADS` | 16 | HLS/Vivado worker count |

Example depth experiment:

```bash
CC8_WEIGHT_TILE_FIFO_DEPTH=4 \
CC8_WEIGHT_TILE_LOAD_II=2 \
CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW=1 \
scripts/run_hls_resident_layer_cosim.sh
```

Always compare CoSim progress, HLS II/timing, resource reports, and hw_emu
cycles. A deeper FIFO is not an optimization unless at least one measured
bottleneck improves.

## 5. Reading results

### Functional output

The model host prints named comparisons such as Q projection, attention output,
FFN paths, status counters, and final hidden checksums. A valid experiment must
end with `PASS` and process the expected task and packet counts.

### Hardware-emulation cycles

Use `profile_kernels.csv` from the simulation directory. The `Running Time`
field is a Vitis HW-Emu profiler measurement. The Q2.14 operator-level profile
contains multiple Host-submitted hardware calls, so that published number is
the accumulated controller-event time. In the coarse-task streaming image,
the profiler reports an identical interval for the controller, both compute
CUs, the status sink, and their internal functions. That common interval is
therefore used as a modeled device-span metric; it must not be interpreted as
separable per-CU active occupancy. Convert either explicitly identified scope
with the clock embedded in the xclbin:

```text
cycles = running_time_us * frequency_MHz
```

Do not use XSim CPU wall time as accelerator latency. Hardware emulation can
take hours while simulating only milliseconds of device time.

For the Q2.14 diagnostic table, active cycles exclude Host scheduling gaps,
buffer migration, CPU-side RoPE/test-fixture packing, KV fixture preload, and
golden checks. For the coarse-task runtime, use both the common four-CU running
time reported by the profiler and the Host-observed sequence fields. The common
field does not separately expose task-issue gaps. The Host fields include the
initial input, status synchronizations, and final output, but remain simulator
wall-time proxies under HW Emu. Norm/RoPE state is initialized once with model
storage; a valid task sequence reports `auxiliary_migration_ms=0`.

### HLS reports

For each exported kernel, inspect:

- estimated clock period and the named critical path;
- achieved loop initiation intervals;
- BRAM, DSP, FF, LUT, and URAM totals;
- unexpected array replication, mux growth, or FIFO warnings.

Top-level HLS latency for a free-running or command-bounded network may use
conservative tripcount bounds. Prefer measured hw_emu active cycles for the
integrated workload, while retaining HLS reports for local structure and
resource analysis.

## 6. Expected checkpoints

The published standard experiments use deterministic seeds:

| Experiment | Seed | Expected checkpoint |
| --- | ---: | --- |
| Resident single-token layer | 20260718 | 2048 outputs match bit-for-bit |
| Coarse Task 18/19 layer | 20260718 | golden PASS and `intermediate_host_copy=0` |
| Coarse two-layer Task 18/19 + Task 20 stack | 20260718 | 5/5 tasks; 64 outputs exact |
| 8-row prefill block | 20260722 | final checksum `0xb72a92cb5224f0c7` |
| Q2.14 P/D length sweep | 20260722 | 8/8 exits zero; Q2.14 max raw error <= 1 |
| Standard Qwen-layer P8 Task 18/19/20 | 20260718 | 16,384 values exact; 651,621 cycles; intermediate_host_copy=0 |
| Standard-shape Qwen2.5-3B P8/G2/L1 | 20260718 | 6/6 tasks; 2/2 CPU-golden steps; 4,096 values exact; 1,190,693 cycles |

The 8-row block run must also report 48 attention MM tasks and 1536 completed
packets.

## 7. Generated artifacts

The following large generated products are intentionally excluded from the
repository:

- HLS/Vivado project directories;
- XO, xclbin, DCP, waveform, and emulator output;
- executable Host binaries;
- model checkpoints and generated weight binaries.

Curated text reports, Host logs, and run-local CU profiles needed to reproduce
published tables are versioned under results/ together with SHA-256 manifests.
This keeps the repository source-oriented without separating claims from their
compact raw evidence. Large binaries should be archived separately and
associated with the cited Git commit and artifact-manifest identities.

The completed standard-shape P8/G2/L1 package is available under
[`results/qwen3b-e2e-20260820/`](../results/qwen3b-e2e-20260820/). Its `P8`
field means one sequence with eight prompt query rows; the package separately
records sequence batch, sampled outputs, and the single real D1 forward.

The publication tree and every versioned result package can be checked without
launching any Xilinx tool:

```bash
make test_publication_release

# The constituent checks can also be run independently.
make test_publication_tree
make verify_result_checksums

# Release maintainers stage the complete candidate before rebuilding the root
# manifest. The generator refuses unstaged or untracked release files.
git add -A
make regenerate_root_checksums
git add CHECKSUMS.sha256
VERIFY_ROOT_CHECKSUMS=1 make test_publication_tree
```

The aggregate target runs every non-simulator release gate: task-plan and
launcher contracts, the Task-18-to-Task-19 HBM-residency invariant, progress
and performance semantics, source provenance, the historical Q2.14 evidence
package, and the publication tree. The publication-tree target validates the
required English research documents, citation
metadata, Markdown fences, local links, conflict markers, and all per-result
SHA-256 manifests. The root-manifest check is opt-in because it is expected to
be stale while a release candidate is being assembled.

Historical result verifiers always validate their archived manifests, raw
profiles, and derived rows. Because the research source continues to evolve,
matching an old source snapshot against the current checkout is optional:

```bash
make verify_q214_pd_release
make verify_q214_resident_release
Q214_VERIFY_CURRENT_SOURCE=1 make verify_q214_resident_release
```

The strict second form is expected to pass only when the checkout corresponds
to that historical source snapshot; the default remains valid on later
releases without weakening the archived evidence checks.

## 8. Case 2: Streaming split architecture

The `cases/streaming-split/` directory contains a second architecture case
with its own kernels, host programs, and connectivity configuration. It is
self-contained and does not depend on the Case 1 source files.

### Quick validation

```bash
cd cases/streaming-split

# Compile both kernels (sw_emu)
v++ -c -t sw_emu --platform $PLATFORM -I include \
  --hls.clock 300000000:control_cache_core \
  -k control_cache_core kernel/control_cache_core.cpp -o build/cc.xo

v++ -c -t sw_emu --platform $PLATFORM -I include \
  --hls.clock 300000000:qkv_tile_kernel_cc_qwen_small_core_v8_2_s \
  -k qkv_tile_kernel_cc_qwen_small_core_v8_2_s \
  kernel/qkv_tile_kernel_cc_qwen_small_core_v8_2_s.cpp -o build/v82.xo

# Link with 4-PC weight connectivity
v++ -l -t sw_emu --platform $PLATFORM --config conn_v8_2x2.cfg \
  --kernel_frequency 300 build/cc.xo build/v82.xo -o build/v8_2x2.xclbin

# Build and run hosts
g++ -std=c++14 -O2 -I/opt/xilinx/xrt/include -I./include \
  host/host_v8_2x2.cpp host/xcl2.cpp -o build/host_v8_2x2 \
  -L/usr/lib/x86_64-linux-gnu -lOpenCL -lpthread

g++ -std=c++14 -O2 -I/opt/xilinx/xrt/include -I./include \
  host/host_accum.cpp host/xcl2.cpp -o build/host_accum \
  -L/usr/lib/x86_64-linux-gnu -lOpenCL -lpthread

# 7-op integrated layer (Q/K/V/O/Gate/Up/Down)
XCL_EMULATION_MODE=sw_emu ./build/host_v8_2x2 build/v8_2x2.xclbin

# Variable-depth accumulate (e.g., 16-chunk = reduction 256)
XCL_EMULATION_MODE=sw_emu ./build/host_accum build/v8_2x2.xclbin 16

# Full hidden=2048 reduction (512 op)
XCL_EMULATION_MODE=sw_emu ./build/host_accum build/v8_2x2.xclbin 512
```

### Expected output

All seven projections should report `miss=0/2048 ✅` and the accumulate
test should report `sample=<N*16>` for `N` chunks. See
[`docs/design.md`](../cases/streaming-split/docs/design.md) for architecture
details, performance tables, and the optimization history.
