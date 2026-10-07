# Usage and reproduction

[Documentation index](README.md) | [Environment](environment.md) |
[Design entries](README.md#designs) | [Evaluation](experiments.md) |
[Reference/history](reference.md)

This page is the short command route. Use the linked family README or scoped
guide for the complete recipe, profile parameters, and expected checks. Keep
the source/configuration identity and timing boundary beside every result.

## Before any route

From the repository root, inspect the publication tree first:

```bash
scripts/check_environment.sh publication
```

For HLS, CSim, RTL CoSim, Vitis link, or HW-Emu, source the vendor setup and
run the corresponding preflight:

```bash
source scripts/setup_environment.sh
scripts/check_environment.sh hls       # HLS, CSim, RTL CoSim
scripts/check_environment.sh hw-emu    # linked HW-Emu image/run
```

The reference stack is Vitis/Vivado/Vitis HLS 2022.2, XRT 2022.2, and the U50
platform `xilinx_u50_gen3x16_xdma_5_202210_1`. See [environment](environment.md)
for custom paths and resource guards. A physical U50 is not required for
publication checks, HLS, CoSim, or HW-Emu.

## Route map

| Design | Direct entry | Smallest useful check | Boundary |
| --- | --- | --- | --- |
| [`cowave-fix16-2-8-64`](designs/fix16.md) | [Resident guide](reproduction-resident.md) | CSim or finite-FIFO RTL CoSim | Controller-resident Tasks 18/19/20; HW-Emu interval is modeled device time |
| [`cowave-streaming-split`](designs/streaming-split.md) | [Case README](../cases/streaming-split/README.md) | Case `sw_emu` host checks | Independent `cc` + V8-2_s; full-layer values are analytical projections |
| [`cowave-int4-4-8-128`](designs/quantized.md) | [Public quantized-layer README](../cases/quantized-layer/README.md) | Case-defined host/CSim/HW-Emu check | Complete-layer W4A4 source and its own result identity |
| [`cowave-int8-4-4-128`](designs/quantized.md) | [Public quantized-layer README](../cases/quantized-layer/README.md) | Case-defined host/CSim/HW-Emu check | Complete-layer W8A8 source and its own result identity |
| `cowave-quantized-blocks` | [Component README](../cases/quantized-block/README.md) | CSim or deadlock-enabled CoSim | Isolated W4A4/W8A8 streams; no integrated model claim |

The [implementation catalog](implementations.md) defines the names and legacy
aliases. The canonical profile numbers count compute CUs and logical rows/
output columns; controller/status kernels and DSP packing are excluded from
that count. Root Makefile targets and kernel ABI retain the historical `8x64`
paths.

## Design-point generator

The catalog is the shared input for documentation, the command helper and a
browser selector. Generate a resolved point before running a build:

```bash
python3 scripts/cowave.py point cowave-int4-4-8-128 \
  integrated-rms2-silu4-prefill-overlap --prefill 66 --weights random \
  --output /tmp/cowave-build
```

The result identifies the architecture, configuration, workload, source and
evidence links, and resolved commands. A recorded measurement applies only to
its archived workload; choosing another prompt length does not create a new
performance result. Fix16 and component entries link their own recipes where
the unified quantized builder does not apply.

For interactive selection, serve the checkout using Python's standard library:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/docs/design-point-generator.html` in a browser.
The [generated selector](design-point-generator.html) works without external
JavaScript libraries or a remote configuration service. It generates commands;
building remains an explicit terminal action. GitHub's file view shows the
HTML source, so use this local server to interact with it.

After editing `designs/catalog.json` or refreshing the public source manifest,
run `python3 scripts/cowave.py render`. `make test_design_catalog` checks that
the generated documentation and selector agree with the catalog.

## CoWave command helper

The repository-level helper exposes a short, dry-run-friendly route for the
catalogued designs and configurations:

```bash
python3 scripts/cowave.py list
python3 scripts/cowave.py show DESIGN CONFIG
python3 scripts/cowave.py build DESIGN CONFIG --phase host --dry-run \
  --prefill 66 --weights random --output /tmp/cowave-build
```

`build` accepts `--phase controller-xo|compute-xo|link|host|emconfig|run|all`,
defaults to the documented 200 MHz HW-Emu configuration, and accepts an
explicit output directory. A dry run resolves the design/configuration and
prints the command plan without launching a long accelerator job; use
`python3 scripts/cowave.py build --help` for the parser's current phase list.

## Root resident route

The historical root build path remains the compatibility route:

```bash
make hls_csim_compute
make hls_csim_control
make hls_csim_closed_loop_8x64_resident_layer
scripts/run_hls_resident_layer_cosim.sh
```

For a profile-matched linked image, use the existing Makefile targets:

```bash
make vitis_8x64_xo VITIS_8X64_MODEL_PROFILE=qwen-layer
make vitis_8x64_link TARGET=hw_emu FREQUENCY=200 \
  VITIS_8X64_MODEL_PROFILE=qwen-layer
make vitis_8x64_qwen_host vitis_8x64_emconfig TARGET=hw_emu \
  VITIS_8X64_MODEL_PROFILE=qwen-layer
```

The resident guide covers bounded layer/block checks, coarse-task generation,
checkpoint localization, archiving, and the long Qwen2.5-3B launcher. Start
with a bounded profile; the default 36-layer launcher is not a quick check.

## Streaming split route

The case README is canonical and includes the complete host builds,
connectivity, expected output, and `sw_emu` invocation:

```bash
cd cases/streaming-split
source ../../scripts/setup_environment.sh
../../scripts/check_environment.sh hls
```

The [case design record](../cases/streaming-split/docs/design.md) contains
operator details and optimization history. This route does not reuse resident
Task 18/19/20 or KV-ownership claims.

## Quantized full layer

The main quantized route builds the public complete-layer source for either
canonical profile. Follow the case README for the profile/configuration and
the source identity recorded with each result:

```bash
python3 scripts/cowave.py list
python3 scripts/cowave.py build cowave-int4-4-8-128 integrated-rms2-silu4-prefill-overlap \
  --phase host --dry-run --prefill 66 --weights random --output /tmp/cowave-int4
```

The W8A8 command uses `cowave-int8-4-4-128` with the matching configuration.
The 2026-10-07 W4/W8 package records P66+D1 evidence for its exact source and
configuration; the September 30 W4 archive is historical and must not be
silently attributed to the new source.

## Quantized matrix blocks

The component route runs planner, CSim, synthesis, and deadlock-enabled CoSim:

```bash
make test_quantized_cu_planner
make quantized_cu_plan QUANT_KERNEL=w4a4 QUANT_KERNEL_COUNT=auto
make quantized_cu_plan QUANT_KERNEL=w8a8 QUANT_KERNEL_COUNT=auto
scripts/run_quantized_block_regression.sh csim
```

Use the [quantized-block README](../cases/quantized-block/README.md) for
`synth`, `cosim`, accumulator variants, and packet source maps. This component
family is independent of both complete-layer profiles.

## Reading a result

`P8` is eight active rows from one sequence, not batch eight. `G2` includes one
real one-row decode forward after the prompt sample. HW-Emu CPU wall time is
simulator runtime. Published modeled cycles use the named profiler interval
and frequency; they are not board latency. HLS resource and Fmax values are
estimates. A CPU fixed-point oracle is an out-of-band correctness check, not
part of accelerator useful work.

For read-only package checks:

```bash
make test_publication_tree
make verify_result_checksums
```

Use [diagnostics](reproduction-diagnostics.md) for Q2.14 and archive-specific
verification, and [reference/history](reference.md) for the complete command
records that are intentionally kept out of this short route.
