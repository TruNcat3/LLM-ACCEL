# Environment Setup

[Documentation index](README.md) | [Usage](usage.md) |
[Repository map](repository-map.md) | [Release workflow](release-workflow.md) |
[Repository](../README.md)

This page is the required starting point for reproduction. It separates the
software needed to inspect published evidence from the substantially heavier
environment needed to synthesize or emulate the accelerator. The same shell
contract serves the Fix16 resident root, the streaming-split case, and the
quantized matrix-block case; each family has its own source and evidence
boundary in [Usage](usage.md).

## 1. Reference stack

The released measurements were produced with the following compatible stack:

| Component | Reference configuration | Required for |
| --- | --- | --- |
| Host OS | Ubuntu 20.04, x86-64 | All recorded builds |
| Vitis / Vivado / Vitis HLS | 2022.2 | HLS, XO export, linking, HW Emu |
| XRT | 2022.2 / 2.14 | Host compilation, HW Emu, board runtime |
| Vitis platform | `xilinx_u50_gen3x16_xdma_5_202210_1` | Reference evaluation platform for linking and emulation |
| Target card | Alveo U50 | Optional physical-board validation only |

The FPGA card is not required for CSim, RTL CoSim, synthesis, or HW Emu. A
different tool release or platform may work, but it creates a new evidence
configuration and must not be presented as a reproduction of the archived
2022.2 results. The U50 is CoWave's reference evaluation platform for those
artifacts, not the design identity; exact platform and device names remain
necessary for reproducing the published configuration.

## 2. Reproduction levels

Choose the smallest preflight mode matching the intended work:

| Mode | What it checks | Default resource guard |
| --- | --- | --- |
| `publication` | Shell, compiler, Git, Python, Perl, ripgrep, checksums | 2 GiB memory and 2 GiB `/tmp` |
| `hls` | Publication tools plus Vitis/Vivado/HLS 2022.2 and development headers; covers root and case-local HLS/CoSim | 50 GiB memory and 20 GiB `/tmp` |
| `hw-emu` | HLS stack plus XRT, U50 reference evaluation platform, `emconfigutil`, `xclbinutil`, and `tmux`; needed by Fix16 system runs | 80 GiB memory and 100 GiB `/tmp` |
| `board` | XRT management tools and a render device node | 4 GiB memory and 2 GiB `/tmp` |

Every mode also checks the core shell and publication utilities. The table
lists the additional contract that distinguishes each level.

The `hw-emu` guard is deliberately sized for the standard Qwen2.5-3B build.
Small profiles and operator diagnostics may use less. Individual long-running
launchers retain their own authoritative guards and may refuse a run even after
a relaxed preflight. Quantized planner and CSim checks do not need `hw-emu`.

## 3. Install prerequisites

Install the AMD/Xilinx 2022.2 tools, XRT, and (for Vitis link/HW-Emu) the U50
reference evaluation platform using
their licensed installers and platform packages. The repository does not
redistribute vendor binaries, board firmware, or model checkpoints.

The non-vendor host packages correspond to the following Ubuntu packages or
equivalent tools:

```text
build-essential  git  make  g++  python3  perl  ripgrep  tmux
ocl-icd-opencl-dev
```

OpenCL development headers must provide `CL/cl2.hpp`. XRT must provide its
headers and runtime libraries. Board use additionally requires the XRT kernel
drivers and a platform/firmware installation compatible with the card. The
quantized planner itself is shell/AWK-only, but its HLS and CoSim recipes use
the same Vitis HLS 2022.2 environment.

## 4. Configure the shell

For a standard installation, the repository setup helper locates common
2022.2 and XRT paths automatically:

```bash
source scripts/setup_environment.sh
```

This step is needed before HLS, CoSim, Vitis link, or HW-Emu. A publication-
only audit does not need vendor tools; run
`scripts/check_environment.sh publication` directly as shown in Section 5.

For a custom installation, set explicit paths first:

```bash
export VITIS_ENV_SCRIPT=/path/to/Vitis/2022.2/settings64.sh
export XILINX_XRT=/path/to/xrt
export DEVICE=xilinx_u50_gen3x16_xdma_5_202210_1
export XPLATFORM=/path/to/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm
source scripts/setup_environment.sh
```

The helper must be sourced; executing it cannot modify the parent shell. It
exports the resolved environment script, tool roots, device name, and platform
path. Public launchers use the same resolver and do not depend on a
machine-private setup path. `XPLATFORM` is required for Vitis link and
`sw_emu`; HLS-only quantized checks still benefit from the same tool setup.

## 5. Run preflight

Run one check before the corresponding workflow:

```bash
# Inspect documentation and checksum-protected evidence.
scripts/check_environment.sh publication

# Compile Host code, run CSim/CoSim, or synthesize HLS.
scripts/check_environment.sh hls

# Build or execute the linked multi-kernel hardware emulator.
scripts/check_environment.sh hw-emu

# Inspect a physically installed card and XRT driver access.
scripts/check_environment.sh board
```

A preflight returns nonzero if a required command, version, header, platform,
device node, or resource guard is missing. Warnings identify differences from
the reference system without hiding a hard failure.

For diagnostic use only, resource thresholds may be overridden:

```bash
LLM_ACCEL_MIN_AVAILABLE_GIB=32 \
LLM_ACCEL_MIN_TMP_GIB=40 \
  scripts/check_environment.sh hw-emu

# Skip only the capacity check; all tool and platform checks still run.
LLM_ACCEL_SKIP_RESOURCE_CHECK=1 scripts/check_environment.sh hls
```

Lowering a preflight threshold does not override a launcher's own safety guard.

## 6. Generated data and storage

The default Makefile writes profile-specific generated products under
`vitis_8x64/` and `reports/`. Full-shape wrappers place large temporary trees
under `/tmp` by default. Override these before launching when `/tmp` is not the
desired high-capacity filesystem:

```bash
export VITIS_8X64_QWEN3B_WORK_ROOT=/fast-scratch/$USER/llm-accel-qwen3b
export VITIS_8X64_QWEN3B_TMP_ROOT=/fast-scratch/$USER/llm-accel-qwen3b/tmp
```

The Qwen build launcher forwards both paths, the selected device, thread
count, and resource thresholds explicitly into its tmux worker. This avoids a
long-lived tmux server silently reusing stale environment values. `TMPDIR` is
also bound to `VITIS_8X64_QWEN3B_TMP_ROOT`, so the 100-GiB capacity guard is
checked on the filesystem that actually owns build scratch data.

Generated XO, xclbin, waveform, executable, checkpoint, and build-tree files
are intentionally excluded from Git. A released evidence package records the
source and generated-artifact identities needed to audit a result.

## 7. Board-specific boundary

The `board` preflight proves only that XRT commands and a render node exist. It
does not prove that the card shell, management controller, clocks, HBM, or a
particular xclbin are healthy. Before physical execution, separately inspect:

```bash
xbmgmt examine
xbutil examine
```

Do not use board output as a replacement for the archived HW-Emu evidence.
Physical timing, power, and throughput require their own result package and
measurement boundary.

## 8. Common failures

- **A 2021.x or 2023.x executable appears first in `PATH`.** Set
  `VITIS_ENV_SCRIPT` explicitly, source the setup helper again, and rerun the
  `hls` preflight.
- **The U50 reference evaluation platform is unresolved.** Set `XPLATFORM` to
  the installed `.xpfm`;
  a device name alone is not sufficient for a reproducible link.
- **Host compilation cannot find OpenCL/XRT.** Install the OpenCL development
  package and verify `XILINX_XRT` before rebuilding.
- **HLS or HW Emu refuses to start.** Check both available memory and the
  filesystem holding the selected work root. Do not bypass a resource guard
  while another Vivado/XSim job is consuming the machine.
- **Board preflight lacks a render node.** Fix XRT driver/device permissions
  before debugging kernels or model data.

After the selected preflight passes, continue with
[Usage and Reproduction](usage.md). It routes to the Fix16 resident,
streaming-split, and quantized matrix-block families, and links to the
scoped [resident](reproduction-resident.md) and
[diagnostic](reproduction-diagnostics.md) command guides.
