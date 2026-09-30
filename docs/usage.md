# Usage and Reproduction

[Documentation index](README.md) | [Implementation map](implementations.md) |
[Architecture](architecture.md) | [Experiments](experiments.md) |
[Environment](environment.md) | [Repository map](repository-map.md) |
[Release workflow](release-workflow.md) | [Repository](../README.md)

This is the route map for reproducing every public CoWave implementation
family in this checkout. The repository name remains `LLM-ACCEL` in command
paths and artifact names for compatibility with the published tooling.

## Start here

1. Read [Environment Setup](environment.md), run the vendor-free publication
   preflight directly, and source the setup helper only for HLS/HW-Emu work.
2. Select one implementation family below. Fix16 diagnostics and small-shape
   tests are evidence scopes of the Fix16 resident family, not additional
   families.
3. Keep the command's result package and timing boundary with the result. Use
   [Experiments](experiments.md) and the [evidence index](../results/README.md)
   to interpret historical claims.

| Family | Canonical source and route | Representative check | Evidence boundary |
| --- | --- | --- | --- |
| **Fix16 resident** | [`kernel/`](../kernel/), [`host/`](../host/), [resident reproduction](reproduction-resident.md) | bounded CSim, RTL CoSim, or P8 HW-Emu | Controller-resident Tasks 18/19/20; HW-Emu CU trace is modeled device time |
| **Streaming split** | [`cases/streaming-split/`](../cases/streaming-split/), [case README](../cases/streaming-split/README.md) | `sw_emu` seven-operation host | Earlier control/cache plus fixed compute core; full-layer numbers are analytical projections |
| **Quantized matrix blocks** | [`cases/quantized-block/`](../cases/quantized-block/), [case README](../cases/quantized-block/README.md) | planner contract or bounded CSim/RTL CoSim | Isolated INT4/INT8 blocks; no full-layer or board claim |

The [public implementation map](implementations.md) defines the names and
legacy aliases used by older result packages. The [architecture](architecture.md)
and [design space](design-space.md) pages explain how the three families differ.

## Environment preflight

From the repository root:

```bash
scripts/check_environment.sh publication
```

`publication` checks source/evidence tooling without vendor binaries. Use
`hls` before CSim, RTL CoSim, HLS synthesis, or host compilation, and use
`hw-emu` before Vitis linking or an emulation run:

```bash
source scripts/setup_environment.sh
scripts/check_environment.sh hls
scripts/check_environment.sh hw-emu
```

The supported reference stack is Vitis/Vivado/Vitis HLS 2022.2, XRT 2022.2,
and the published U50 evaluation platform
`xilinx_u50_gen3x16_xdma_5_202210_1`. A physical U50 is optional for all
software, HLS, CoSim, and HW-Emu routes. Set `VITIS_ENV_SCRIPT`, `XILINX_XRT`,
`DEVICE`, and `XPLATFORM` for another installation; do not copy a
machine-private setup path into a reproduction record. See
[Environment Setup](environment.md) for resource guards and platform checks.

## Route A: Fix16 resident

The short validation ladder is deliberately tool-local before it reaches a
linked system:

```bash
make hls_csim_compute
make hls_csim_control
make hls_csim_closed_loop_8x64_resident_layer
scripts/run_hls_resident_layer_cosim.sh
```

For a linked image, export the profile-matched XOs and build HW-Emu:

```bash
make vitis_8x64_xo VITIS_8X64_MODEL_PROFILE=qwen-layer
make vitis_8x64_link TARGET=hw_emu FREQUENCY=200 \
  VITIS_8X64_MODEL_PROFILE=qwen-layer
make vitis_8x64_qwen_host vitis_8x64_emconfig TARGET=hw_emu \
  VITIS_8X64_MODEL_PROFILE=qwen-layer
```

The resident convenience flow also supports `run`, `run-composed`,
`run-block`, `run-block-stack`, `run-block-sequence`, `run-stack`, and
`run-generate`. Use the detailed [resident reproduction guide](reproduction-resident.md)
for their complete argument recipes, profiles, pipeline parameters, expected
checkpoints, and historical variants.

For the full Qwen2.5-3B launcher, always bound the layer count in a first
reproduction. The launcher default is `L=36` and is a long RTL HW-Emu run, not
a quick check:

```bash
# Requires a profile-matched full-shape build; see the resident guide.
VITIS_8X64_E2E_PROMPT_TOKENS=8 \
VITIS_8X64_E2E_MAX_NEW_TOKENS=2 \
VITIS_8X64_E2E_PREFILL_BLOCK_SIZE=8 \
VITIS_8X64_E2E_LAYERS=1 \
  scripts/run_vitis_8x64_qwen3b_e2e_hwemu_tmux.sh
```

`L=2` is the next bounded scaling gate. The unqualified launcher invokes the
36-layer extension only when that long-run evidence is explicitly intended.

## Route B: streaming split

The case is self-contained. Its README is the canonical recipe and includes
the complete host builds, connectivity file, and expected output:

```bash
cd cases/streaming-split
source ../../scripts/setup_environment.sh
../../scripts/check_environment.sh hls
mkdir -p build
```

Continue with the [streaming-split reproduction](../cases/streaming-split/README.md).
The route compiles `control_cache_core` and
`qkv_tile_kernel_cc_qwen_small_core_v8_2_s`, links `conn_v8_2x2.cfg` at
300 MHz, and runs `host_v8_2x2` plus the variable-depth `host_accum` test in
`sw_emu`. Its reported full-layer estimate is not a linked HW-Emu or board
measurement. Detailed design and optimization history are in
[`cases/streaming-split/docs/design.md`](../cases/streaming-split/docs/design.md).

## Route C: quantized matrix blocks

This family is an isolated W4A4/W8A8 matrix-block study. Start with the
resource planner, which does not launch a simulator:

```bash
make test_quantized_cu_planner
make quantized_cu_plan QUANT_KERNEL=w4a4 QUANT_KERNEL_COUNT=auto
make quantized_cu_plan QUANT_KERNEL=w8a8 QUANT_KERNEL_COUNT=auto
```

For the bounded arithmetic and finite-stream evidence, use the case README's
[Reproduce](../cases/quantized-block/README.md#reproduce) section. The
regression launcher accepts `csim`, `synth`, `cosim`, or `all`:

```bash
scripts/run_quantized_block_regression.sh csim
```

The selected single-bank candidates have CSim and deadlock-enabled RTL CoSim
evidence. They are not connected to the Fix16 controller, HBM scheduler, or a
full-model runtime; do not combine their resource estimates with resident
performance claims.

To reanalyze the completed quantized full-layer development comparison using
only the archived inputs and Python standard library:

```bash
bash results/quantized-layer-w4-20260930/verify.sh
```

This rebuilds metrics and numerical comparisons from saved traces and dumps.
It does not launch or rebuild the full-layer accelerator; see the
[progress report](quantized-layer-progress.md) for its source-release boundary.

## Evidence and historical releases

Each result package records its source snapshot, generated-artifact identity,
raw logs, timing boundary, and checksums. These checks are read-only and do
not launch synthesis or simulation:

```bash
scripts/check_environment.sh publication
make test_publication_tree
make verify_result_checksums
```

The historical Fix16 Q2.14 packages can be re-audited independently:

```bash
make verify_q214_pd_release
make verify_q214_resident_release
# Optional strict comparison with the archived source snapshot.
Q214_VERIFY_CURRENT_SOURCE=1 make verify_q214_resident_release
```

Use [operator and publication diagnostics](reproduction-diagnostics.md) for
the Q2.14 P/D sweep, profile interpretation, release archive, and checksum
workflow. Use [resident reproduction](reproduction-resident.md) for the
complete coarse-task and Qwen3B command history. Historical source-level
context remains available in
[`docs/coarse-task-runtime-history.md`](coarse-task-runtime-history.md).

## Reading a result

`P8` names exactly eight active query rows from one sequence; the resident
protocol supports smaller blocks, but P8 is not batch eight. `G2` produces two
sampled outputs: the first follows the prompt forward and the second follows
one real D1 decode forward; `G1` is a prefill/TTFT-only gate.
`intermediate_host_copy=0` and
`kv_cache_owner=controller` are residency checks, not performance values.

HW-Emu CPU wall time is simulator runtime. Published Fix16 cycle rows come
from the explicitly named CU/profile scope, with
`cycles = running_time_us * frequency_MHz`; they are neither physical-board
latency nor per-CU occupancy unless the package says so. HLS resource and Fmax
values are estimates, and the streaming/quantized families have their own
boundaries.

For the complete release gate, source the environment and run:

```bash
source scripts/setup_environment.sh
scripts/check_environment.sh hls
make test_publication_release
```

This compiles host-only contracts and validates archived evidence; it does not
start Vitis synthesis or XSim. Release maintenance and future publication
workflow are intentionally kept separate from the reproduction routes.
