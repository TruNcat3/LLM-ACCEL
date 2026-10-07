# Quantized full layer case

This directory is the public, reproducible source closure for the verified
quantized full-layer cases. The two base designs are W4
`cowave-int4-4-8-128` and W8 `cowave-int8-4-4-128`; the base design identity
is intentionally separate from the profile/configuration label. The closure
contains both W4 and W8 controller and compute sources, their shared headers,
Host/XRT sources, Tcl entry points, small C++ checks, and the scripts used to
build or inspect the design.

The published hardware bytes come from two frozen follow-up snapshots:

| precision | source snapshot | selected verified candidate |
| --- | --- | --- |
| W4 `cowave-int4-4-8-128` | `quantized_layer_combinations_20261005_082444_v4FXjs` | `integrated_rms2_overlap_ref` |
| W8 `cowave-int8-4-4-128` | `quantized_layer_combinations_20261003_225644_r6FNxq` | `integrated_rms2_overlap_ref` |

The required source files are byte-identical between those snapshots. The
per-precision snapshot choice is retained here because validation evidence was
archived independently. The released profile is `integrated` with SiLU lanes
4, RMS lanes 2, weight-read outstanding 32, attention-wave 0, and prefill FFN
overlap 1. The independently validated `integrated_rms2_overlap_wave` is also
recorded in the provenance and results package; it is a separate configuration
label over the same source closure, not a silent source substitution.

## Layout and dependencies

`kernel/` contains the full-layer controller, compute, matrix, stream, decode,
and integration translation units. `include/` is the complete transitive
header closure. `host/` and `common/include/` are the XRT Host program and its
shared helpers. `tests/` contains native arithmetic/protocol checks and the
full-layer CoSim test benches. `tcl/` contains the HLS project and CoSim
entry points. `config/` contains the two connection profiles separated by
precision: `cowave-int4-4-8-128.cfg` for W4 and
`cowave-int8-4-4-128.cfg` for W8.

Native checks and Host compilation need `g++`, C++14, and the `ap_*` headers
exposed by an activated Vitis HLS installation. Set `VITIS_HLS_INCLUDE` to an
include directory, or set `XILINX_HLS` to a Vitis HLS root; if neither is set,
the scripts derive the include directory from `vitis_hls`. Host compilation
additionally needs `XILINX_XRT` with `include/` and `lib/`, and a link stage
needs `XPLATFORM` pointing to the target `.xpfm`. An activated toolchain is
preferred; `VITIS_ENV_SCRIPT` may point to a site-specific setup script.

All generated projects, reports, logs, and binaries default below `.build/`.
Set `COWAVE_QUANTIZED_LAYER_OUTPUT_DIR` to move that tree. `TARGET`,
`DEVICE`, `FREQUENCY`, `THREADS`, `XPLATFORM`, and
`COWAVE_QUANTIZED_LAYER_CONFIG_DIR` are the relevant build overrides.

## Rebuild and checks

From this directory:

```text
make check
make dry-run PRECISION=w4
make dry-run PRECISION=w8 PHASE=host
make test-native
make host PRECISION=w4
make host PRECISION=w8
```

`make check` verifies the source hash manifest, shell/Python syntax, public
path portability, and the W4/W8 profile resolver. `make dry-run` only prints
the phase command and required tool variables; it does not start HLS, Vitis,
XSim, or HWEmu. `make test-native` compiles and runs the small C++ regression
set without XRT or hardware.

The phase-level build entry points are also available directly:

```text
scripts/build_vitis_quantized_w4_layer.sh controller-xo|compute-xo|host|link|emconfig|run|all
scripts/build_vitis_quantized_w8_layer.sh controller-xo|compute-xo|host|link|emconfig|run|all
```

For the complete integration test bench, use `make cosim PRECISION=w4
COSIM_STAGE=prepare`, then `make cosim PRECISION=w4 COSIM_STAGE=cosim` (or
`COSIM_STAGE=all`). The W8 path is identical. First run
`make build PRECISION=w4 PHASE=all` (or W8), which builds the XO/link image,
Host executable, and HWEmu `emconfig.json`; `make hwemu PRECISION=w4|w8` is a
run-only launcher for those existing artifacts and does not build them. The
performance wrapper requires completed C references, for example
`QUANTIZED_LAYER_REFERENCE_ROOT=/path/to/reference make performance ARGS=--check`.
Neither CoSim, HWEmu, nor performance is run by the source checks.

The default `make hwemu` workload is intentionally a short zero-weight smoke
run, not the published P66 random workload: W4 defaults to prefill 10 with
block size 8, while W8 defaults to prefill 6 with block size 4. To reproduce
the P66 performance path, use an activated Vitis environment and keep one
portable output root for every phase:

```text
source /path/to/vitis_env.sh
export VITIS_ENV_SCRIPT=/path/to/vitis_env.sh
export COWAVE_QUANTIZED_LAYER_OUTPUT_DIR="$PWD/.build/p66"
export VITIS_QUANTIZED_LAYER_DEBUG=1
make build PRECISION=w4 PHASE=all
make build PRECISION=w8 PHASE=all
export QUANTIZED_LAYER_EVAL_PREFILL=66
make reference
```

Wait for the printed `run_root` to contain
`QUANTIZED_LAYER_C_REFERENCES_COMPLETE` and
`pipeline_exit_status=0`, then set `reference_root` to that completed
`quantized_layer_numeric_*` directory and preflight before launching the
performance workers:

```text
export QUANTIZED_LAYER_REFERENCE_ROOT=/path/to/.build/p66/references/quantized_layer_numeric_YYYYmmdd_HHMMSS
QUANTIZED_LAYER_EVAL_PREFILL=66 make performance ARGS=--check
QUANTIZED_LAYER_EVAL_PREFILL=66 make performance
```

The debug build is required for the performance trace preflight. The final
launch uses random weights and P66 for both precisions; its evaluation output
is placed below `COWAVE_QUANTIZED_LAYER_OUTPUT_DIR`.

## Provenance

`provenance/source_origins.tsv` records the snapshot and candidate identity;
`provenance/verification.tsv` records the completed W4/W8 evidence and the
scope of this release; `provenance/source.sha256` verifies every published
source/config/script input. These records distinguish source identity from
the validation logs, which were produced independently for each precision.
The archived full-layer evidence is linked at
`../../results/quantized-layer-20261007` relative to this directory.
