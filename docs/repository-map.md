# Repository and source map

[Documentation](README.md) | [Design catalog](implementations.md) |
[Usage](usage.md) | [Reference/history](reference.md)

The source tree has one root resident implementation and independent case
families. Historical build paths are retained for compatibility; the
canonical names and configuration labels are maintained by the
[implementation catalog](implementations.md).

```text
LLM-ACCEL/
|- docs/                                  design, evaluation, reproduction
|- kernel/, include/, host/, common/      cowave-fix16-2-8-64 root source
|- cases/
|  |- streaming-split/                    cowave-streaming-split
|  |- quantized-block/                    W4A4/W8A8 component blocks
|  `- quantized-layer/                    public full-layer quantized source
|- Makefile, conn_*.cfg, tcl/, scripts/   build and evaluation entry points
|- tests/                                  root contracts and bounded fixtures
|- results/                                immutable evidence packages
`- CHECKSUMS.sha256                       public-tree manifest
```

## Root resident source

| Responsibility | Source entry | Contract |
| --- | --- | --- |
| Formats and model dimensions | [`model_config.hpp`](../include/model_config.hpp), [`datatypes.hpp`](../include/datatypes.hpp), [`hardware.hpp`](../include/hardware.hpp) | Capacity, numeric types, and workload are separate axes |
| Host and buffers | [`host_qwen_8x64.cpp`](../host/host_qwen_8x64.cpp), [`host_coarse_task_program.hpp`](../include/host_coarse_task_program.hpp) | Task descriptors, initial input/final output, embedding, vocabulary head |
| Controller/cache | [`control_cache_8x64.cpp`](../kernel/control_cache_8x64.cpp), [`control_cache_8x64.hpp`](../include/control_cache_8x64.hpp) | HBM residency, wave dispatch, online attention, KV, layer subgraphs |
| Compute service | [`compute_core_8x64_unified.cpp`](../kernel/compute_core_8x64_unified.cpp), [`compute_stream.cpp`](../kernel/compute_stream.cpp) | Matrix/vector tasks and stream ordering |
| Matrix engine | [`mm_stream_8x64_fused_mac.cpp`](../kernel/mm_stream_8x64_fused_mac.cpp), [`mm_controller.cpp`](../kernel/mm_controller.cpp) | Tiled products, reduction, projection scheduling |
| Vitis wrappers and ABI | [`control_cache_8x64_nk.cpp`](../kernel/control_cache_8x64_nk.cpp), [`compute_core_8x64_nk.cpp`](../kernel/compute_core_8x64_nk.cpp), [`vitis_stream_8x64.hpp`](../include/vitis_stream_8x64.hpp) | Packed task/data words; historical kernel ABI remains stable |
| Completion and tests | [`cc8_status_sink.cpp`](../kernel/cc8_status_sink.cpp), [`closed_loop_8x64_cosim.cpp`](../kernel/closed_loop_8x64_cosim.cpp) | Status drainage and finite-FIFO fixtures; not extra production CUs |
| Pipeline configuration | [`stream_depth_config.hpp`](../include/stream_depth_config.hpp), [`weight_pipeline_config.hpp`](../include/weight_pipeline_config.hpp) | FIFO capacity, load II, and overlap options |

Read [architecture](architecture.md) for ownership and
[coarse-task runtime](coarse-task-runtime.md) for task ordering. The source
map identifies modules; it does not replace their ABI headers or build
manifests.

## Independent case sources

The [case index](../cases/README.md) is the entry for alternative families.

- `cases/streaming-split/` keeps its `cc`, V8-2_s kernel, packet headers,
  Host tests, connectivity, and Tcl flows together.
- `cases/quantized-block/` keeps the W4A4/W8A8 component kernels, task/packet
  headers, testbenches, Tcl variants, regression wrapper, and planner together.
- `cases/quantized-layer/` is a separate public full-layer source closure with
  its own README and configuration/build identity. It must not be reconstructed
  by combining the component case with a dated result archive.

Case-local files are not automatically part of the root binary. Use the
family README, source manifest, and command helper to establish closure.

## Configuration ownership

| Setting | Authority |
| --- | --- |
| Catalog names/configurations | [`designs/catalog.json`](../designs/catalog.json) and generated [`implementations.md`](implementations.md) |
| Root model profile | [`Makefile`](../Makefile), [`common_hls_model_profile.tcl`](../tcl/common_hls_model_profile.tcl), [`model_config.hpp`](../include/model_config.hpp) |
| Root FIFO/HLS overrides | [`common_hls_depth_config.tcl`](../tcl/common_hls_depth_config.tcl) and pipeline headers above |
| Root replication/HBM mapping | [`conn_u50_8x64_dual.cfg`](../conn_u50_8x64_dual.cfg), [`conn_u50_8x64_dual_full_resident.cfg`](../conn_u50_8x64_dual_full_resident.cfg) |
| Quantized block accumulator/replication | [`quantized-block/README.md`](../cases/quantized-block/README.md), [`plan_quantized_cus.sh`](../scripts/plan_quantized_cus.sh) |
| Quantized full-layer configurations | [`quantized-layer/README.md`](../cases/quantized-layer/README.md) and catalog config labels |
| Streaming-split flags | [`cases/streaming-split/docs/design.md`](../cases/streaming-split/docs/design.md) |
| Vendor tools/platform | [Environment](environment.md) |

A profile needs matching Host, controller/compute XO set, and connectivity. A
directory name or successful software compile does not prove ABI compatibility.

## Script families

| Responsibility | Representative entry |
| --- | --- |
| Environment | [`setup_environment.sh`](../scripts/setup_environment.sh), [`check_environment.sh`](../scripts/check_environment.sh) |
| Catalog helper | [`cowave.py`](../scripts/cowave.py) |
| Root HLS/CoSim | Makefile `hls_*`, [`run_hls_resident_layer_cosim.sh`](../scripts/run_hls_resident_layer_cosim.sh) |
| Root resident HW-Emu | [`build_vitis_8x64_resident_layer_hwemu.sh`](../scripts/build_vitis_8x64_resident_layer_hwemu.sh), Qwen3B build/run wrappers |
| Diagnostics | Q2.14 context sweep and checkpoint runners |
| Case evaluation | [`run_quantized_block_regression.sh`](../scripts/run_quantized_block_regression.sh), streaming case-local scripts |
| Evidence/release | `report_*`, `archive_*`, `verify_*`, `install_*`; see [release workflow](release-workflow.md) |

Script names containing `8x64`, `nohup`, or a dated profile identify command
compatibility or a historical run; they are not architecture versions.

## Documents and artifacts

Stable mechanism belongs in [architecture](architecture.md), stable choices in
[design space](design-space.md), commands in [usage](usage.md) and scoped
guides, and measured interpretation in [experiments](experiments.md). Raw logs,
tables, source identities, and checksums belong in immutable
[result packages](../results/README.md). Generated HLS/Vivado projects,
executables, XO/xclbin files, waveforms, checkpoints, and model weights remain
external artifacts.
