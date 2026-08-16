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

### Run the 8-token diagnostic prefill

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
final 8-token block at each length; Decode runs one new token. The publication
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
field is simulated kernel-active time. The Q2.14 layer profile contains
multiple host-submitted hardware operator calls, so the published number is
the accumulated controller active time. Convert it with the clock embedded in
the xclbin:

```text
cycles = running_time_us * frequency_MHz
```

Do not use XSim CPU wall time as accelerator latency. Hardware emulation can
take hours while simulating only milliseconds of device time.

The active-cycle number excludes host scheduling gaps, buffer migration,
CPU-side RoPE/test-fixture packing, KV fixture preload, and golden checks. Use
host-observed elapsed time separately when evaluating the future coarse-task
end-to-end runtime.

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
| 8-token prefill | 20260722 | final checksum `0xb72a92cb5224f0c7` |
| Q2.14 P/D length sweep | 20260722 | 8/8 exits zero; Q2.14 max raw error <= 1 |

The 8-token run must also report 48 attention MM tasks and 1536 completed
packets.

## 7. Generated artifacts

The following are intentionally excluded from the repository:

- HLS/Vivado project directories and reports;
- XO, xclbin, DCP, waveform, and emulator output;
- executable host binaries and logs;
- model checkpoints and generated weight binaries.

This keeps the repository source-oriented. A publication artifact should
archive exact generated reports and binaries separately and associate them
with the cited Git commit.

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
