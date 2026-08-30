# Qwen2.5-3B Prefill Checkpoint Localization

This package records a diagnostic HW-Emu sweep that reads the controller output
after every coarse task. The extra Host readback is intentional and is excluded
from production performance measurements. The package is evidence for numerical
localization, not a successful full-stack accuracy claim.

## Workload and identity

| Field | Value |
| --- | --- |
| Model shape | Qwen2.5-3B |
| Prompt | P8, one sequence, eight query rows |
| Model source | deterministic random Fix16, seed `20260718` |
| Target | Vitis 2022.2 HW Emu |
| Host task program | `static_descriptor_v1`, Task 18/19/20 |
| KV ownership | controller/HBM |
| Production hidden-state copy | none |
| Diagnostic checkpoint copy | one D2H copy after each task |
| XCLBIN SHA-256 | `adf21bc4cc20d5bec514e69bfbdb28e12e9572a33f1071c090f5003e4d904076` |
| Emulation configuration SHA-256 | `696c863021ad86e4740c840916408d9fdae97a9bbaadaefe1e43a1f0f79116ac` |
| Checkpoint Host executable SHA-256 | `c12086aea3513066713cb7b61518aa3abf5b2e32d333d92f067e3de5ca0159a9` |

The Host hash above is the identity recorded by the archived diagnostic runs.
`source_manifest.tsv` is the publication-time source snapshot after the
descriptor-executor refactor; it is retained for review but is not a
retroactive build-equivalence claim for that earlier uncommitted executable.

## Results

The CPU fixed-point oracle uses strict tolerance zero for every checkpoint.

| Run | Expected checkpoints | Completed | First failing checkpoint | Checked values | Mismatches | Max raw error | Result |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| P8, layers 0--2 | 7 | 7 | none | 16,384/checkpoint | 0 | 0 | PASS |
| P8, layers 0--3 | 9 | 7 | layer 3 Attention | 16,384/checkpoint | 22 | 1 | FIRST DIVERGENCE |

Layers 0--2 are bit-exact at both Attention and FFN checkpoints. The first
observed discrepancy is at layer 3 Attention and is only one raw fixed-point
unit for 22 values. This localizes the earliest detectable drift, but does not
explain the separate original L36 end-to-end failure whose maximum raw error was
65535. Full 36-layer numerical closure therefore remains open.

The checkpoint runs exited cleanly at the simulator level. The four-layer run
stopped deliberately at its first failed checkpoint, so it is not a performance
benchmark and has no CU trace suitable for throughput claims.

## Reproduction

Build the checkpoint-capable Host from the public source with Vitis 2022.2, then
point the launcher at the resulting executable and the existing HW-Emu image:

```bash
export VITIS_ENV_SCRIPT=/path/to/Vitis/2022.2/settings64.sh
VITIS_ENV_SCRIPT="${VITIS_ENV_SCRIPT}" \
  make vitis_8x64_qwen_host TARGET=hw_emu \
    VITIS_8X64_MODEL_PROFILE=qwen2.5-3b \
    VITIS_8X64_BUILD_DIR=/tmp/llm_accel_public_checkpoint_build
VITIS_8X64_CHECKPOINT_HOST_EXE=/tmp/llm_accel_public_checkpoint_build/host_qwen_8x64.exe \
VITIS_8X64_CHECKPOINT_LAYERS=3 \
  scripts/run_vitis_8x64_qwen3b_checkpoint_hwemu_tmux.sh
```

The launcher records tool inputs and hashes before execution. See
[`raw/`](raw/) for the complete Host output and `precision.tsv` for the compact
machine-readable summary.
