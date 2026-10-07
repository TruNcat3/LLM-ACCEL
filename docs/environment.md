# Environment

[Documentation index](README.md) | [Usage](usage.md) |
[Design entries](README.md#designs) | [Reference/history](reference.md)

Use this page once before selecting a family route. Publication checks inspect
source and immutable evidence without vendor tools; HLS/CoSim/HW-Emu routes
need the vendor stack. The same shell setup serves the root resident,
streaming-split, and quantized cases, while each case keeps its own source and
evidence boundary.

## Learn the tool flow

The companion [Vitis workflow tutorial](https://github.com/Reconfigurable-Computing/Vitis_workflow)
covers Kernel, Host, emulation, multi-kernel connectivity, and Makefile flow.
Its examples reference older tool releases and are instructional; it is not a
CoWave dependency.

## Reference stack

| Component | Reference | Needed for |
| --- | --- | --- |
| Host OS | Ubuntu 20.04 x86-64 | Archived builds and host checks |
| Vitis / Vivado / Vitis HLS | 2022.2 | HLS, XO export, link, HW-Emu |
| XRT | 2022.2 / 2.14 | Host compilation, HW-Emu, board runtime |
| Platform | `xilinx_u50_gen3x16_xdma_5_202210_1` | Reference link and emulation identity |
| Target card | Alveo U50 | Physical-board validation only; optional for software/HLS/HW-Emu |

A different tool release or platform may work, but it is a different evidence
configuration and must not be described as reproducing an archived result.

## Preflight levels

Run the smallest mode that matches the intended route:

| Mode | Checks | Default guard |
| --- | --- | --- |
| `publication` | Shell, compiler, Git, Python, Tcl, Perl, ripgrep, checksums | 2 GiB memory and `/tmp` |
| `hls` | Publication tools plus Vitis/Vivado/HLS and headers | 50 GiB memory, 20 GiB `/tmp` |
| `hw-emu` | HLS stack, XRT, U50 platform tools, `tmux` | 80 GiB memory, 100 GiB `/tmp` |
| `board` | XRT management tools and render node | 4 GiB memory and `/tmp` |

```bash
scripts/check_environment.sh publication

source scripts/setup_environment.sh
scripts/check_environment.sh hls
scripts/check_environment.sh hw-emu
```

`hls` covers root and case-local CSim, CoSim, and synthesis. `hw-emu` is
needed for linked system runs. Quantized planning and CSim do not need
`hw-emu`. A launcher can impose a stricter guard than this preflight.

## Install prerequisites

Install the licensed AMD/Xilinx 2022.2 tools, XRT, and (for link/HW-Emu) the
U50 platform package. The repository does not redistribute vendor binaries,
board firmware, model checkpoints, or weights.

Equivalent host packages include:

```text
build-essential git make g++ python3 tcl perl ripgrep tmux
ocl-icd-opencl-dev
```

OpenCL development headers must provide `CL/cl2.hpp`; XRT must provide headers
and runtime libraries. Physical board use additionally needs compatible XRT
drivers and firmware.

## Configure the shell

The helper resolves common 2022.2 and XRT paths:

```bash
source scripts/setup_environment.sh
```

For a custom install, set paths before sourcing:

```bash
export VITIS_ENV_SCRIPT=/path/to/Vitis/2022.2/settings64.sh
export XILINX_XRT=/path/to/xrt
export DEVICE=xilinx_u50_gen3x16_xdma_5_202210_1
export XPLATFORM=/path/to/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm
source scripts/setup_environment.sh
```

The helper must be sourced. `XPLATFORM` is required for Vitis link and
`sw_emu`; HLS-only quantized checks still use the same tool setup.
The root Makefile preserves a nonempty exported `XPLATFORM`; automatic
detection is used only when no platform path is supplied. A setup failure
returns a nonzero status, including when `source` is used in an `if` condition.

## Generated data

The root Makefile writes generated products under `vitis_8x64/` and
`reports/`; full-shape wrappers use `/tmp` by default. Redirect large builds
before launching:

```bash
export VITIS_8X64_QWEN3B_WORK_ROOT=/fast-scratch/$USER/cowave-qwen3b
export VITIS_8X64_QWEN3B_TMP_ROOT=/fast-scratch/$USER/cowave-qwen3b/tmp
```

Generated XO/xclbin, waveform, executable, checkpoint, model-weight, and
temporary project files are intentionally excluded from Git. A result package
records source and generated-artifact identities rather than storing these
large files in the repository.

## Board boundary and failures

The board preflight proves only XRT command and device access:

```bash
xbmgmt examine
xbutil examine
```

It does not prove shell health, HBM, clocks, routing, or a particular xclbin.
Physical timing/power/throughput require a separate result package.

Common failures are usually a mismatched Vitis release in `PATH`, an unresolved
U50 `.xpfm`, missing OpenCL/XRT headers, or insufficient memory/scratch space.
Set `VITIS_ENV_SCRIPT` explicitly, verify `XPLATFORM`, and inspect the actual
filesystem behind `VITIS_8X64_QWEN3B_TMP_ROOT` before retrying. Resource
threshold overrides are diagnostic only:

```bash
LLM_ACCEL_SKIP_RESOURCE_CHECK=1 scripts/check_environment.sh hls
```

After this page passes, continue with [Usage and reproduction](usage.md).
