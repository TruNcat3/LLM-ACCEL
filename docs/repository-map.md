# Repository and source map

[Documentation](README.md) | [Implementation catalog](implementations.md) |
[Usage](usage.md) | [Release workflow](release-workflow.md)

The root source implements Fix16 resident. Alternative implementations live
under `cases/`; all families share the public documentation and evidence
index. Existing source paths also identify historical build manifests, so the
catalog supplies the hierarchy without changing those build identities.

```text
LLM-ACCEL/
├── README.md, CITATION.cff, LICENSE       research entry and attribution
├── docs/                                design, reproduction, interpretation
├── kernel/, include/, host/, common/     Fix16 resident implementation
├── cases/
│   ├── streaming-split/                  alternative control/compute split
│   └── quantized-block/                  quantized matrix components
├── Makefile, conn_*.cfg, tcl/, scripts/   build and experiment entry points
├── tests/                               root contracts and bounded fixtures
├── results/                             immutable evidence packages
└── CHECKSUMS.sha256                      public-tree content manifest
```

## Resident source

| Responsibility | Source entry | Contract |
| --- | --- | --- |
| Model dimensions and formats | [model_config.hpp](../include/model_config.hpp), [datatypes.hpp](../include/datatypes.hpp), [hardware.hpp](../include/hardware.hpp) | Separate compiled capacity, numeric formats and actual runtime workload |
| Host program and buffers | [host_qwen_8x64.cpp](../host/host_qwen_8x64.cpp), [host_coarse_task_program.hpp](../include/host_coarse_task_program.hpp) | Coarse task descriptors, initial input/final output, embedding and vocabulary head |
| Smoke / operator harness | [host_8x64.cpp](../host/host_8x64.cpp) | Bounded integration diagnostics, not automatically a full-model execution |
| Model-aware controller | [control_cache_8x64.cpp](../kernel/control_cache_8x64.cpp), [controller header](../include/control_cache_8x64.hpp) | HBM residency, wave dispatch, online attention, KV and layer subgraphs |
| Compute service | [compute_core_8x64_unified.cpp](../kernel/compute_core_8x64_unified.cpp), [compute_stream.cpp](../kernel/compute_stream.cpp) | Unified matrix/vector task execution and stream ordering |
| Matrix engine | [mm_stream_8x64_fused_mac.cpp](../kernel/mm_stream_8x64_fused_mac.cpp), [mm_controller.cpp](../kernel/mm_controller.cpp) | Tiled products, reduction and projection scheduling |
| Vitis boundaries | [controller wrapper](../kernel/control_cache_8x64_nk.cpp), [compute wrapper](../kernel/compute_core_8x64_nk.cpp), [stream ABI](../include/vitis_stream_8x64.hpp) | Packed inter-kernel task/data words |
| Completion sink | [cc8_status_sink.cpp](../kernel/cc8_status_sink.cpp) | Consume status output so finite streams drain |
| Closed-loop RTL fixture | [closed_loop_8x64_cosim.cpp](../kernel/closed_loop_8x64_cosim.cpp), [testbench](../tests/closed_loop_8x64_cosim_tb.cpp) | Test instrumentation and finite FIFO closure, not another production CU |
| Pipeline configuration | [stream_depth_config.hpp](../include/stream_depth_config.hpp), [weight_pipeline_config.hpp](../include/weight_pipeline_config.hpp) | FIFO capacity, load II and wave-overlap options |
| OpenCL helpers | [common/include](../common/include/) | Host support shared by root flows |

For the actual tensor ownership and task ordering, read
[architecture](architecture.md) and [coarse-task runtime](coarse-task-runtime.md).
The source map identifies modules; it does not replace their ABI definitions.

## Alternatives and components

[The case index](../cases/README.md) is the entry for both alternative families.

- Streaming split keeps its own control/cache kernel, compute kernel, packet
  header, Host tests, connectivity and Tcl flows together.
- Quantized matrix blocks keep their own kernels, task/packet headers, bounded
  testbenches and HLS Tcl variants together. The root regression wrapper and
  CU planner dispatch these case-local sources.

A case-local include or source file is not automatically part of the resident
binary. Use the actual build script and source manifest to determine that
closure.

The [quantized full-layer development archive](../results/quantized-layer-w4-20260930/)
contains measurement inputs and analysis tools. Its hardware source identities
refer to the frozen development snapshot; they do not make the root or
matrix-block case a source closure for rebuilding that system. See the
[progress report](quantized-layer-progress.md) for the release boundary.

## Configuration ownership

| Setting | Authority |
| --- | --- |
| Root model profile | [Makefile](../Makefile), [common_hls_model_profile.tcl](../tcl/common_hls_model_profile.tcl), [model_config.hpp](../include/model_config.hpp) |
| Root FIFO/HLS overrides | [common_hls_depth_config.tcl](../tcl/common_hls_depth_config.tcl) and the two pipeline headers above |
| Root CU replication and HBM mapping | [standard connectivity](../conn_u50_8x64_dual.cfg), [full-resident connectivity](../conn_u50_8x64_dual_full_resident.cfg) |
| Runtime arguments / number of layers actually executed | Selected Host and launcher; see [usage](usage.md) |
| Vendor tool paths and reference platform | [Environment setup](environment.md) |
| Quantized accumulator / replication choices | [Quantized case](../cases/quantized-block/README.md) and [CU planner](../scripts/plan_quantized_cus.sh) |
| Streaming-split reduction/control flags | [Case design](../cases/streaming-split/docs/design.md) |

A new profile needs a matching Host, compute/controller XO set and connectivity.
A directory name or successful software compilation does not prove ABI matching.

## Script families

| Responsibility | Representative entry points |
| --- | --- |
| Environment | [setup_environment.sh](../scripts/setup_environment.sh), [check_environment.sh](../scripts/check_environment.sh) |
| Root HLS / bounded CoSim | [run_vitis_hls.sh](../scripts/run_vitis_hls.sh), [resident CoSim](../scripts/run_hls_resident_layer_cosim.sh), Makefile `hls_*` targets |
| Resident and model-stack HW Emu | [resident builder](../scripts/build_vitis_8x64_resident_layer_hwemu.sh), [model builder](../scripts/run_vitis_8x64_qwen3b_e2e_build_tmux.sh), [model runner](../scripts/run_vitis_8x64_qwen3b_e2e_hwemu_tmux.sh) |
| Diagnostic runs | [P/D context sweep](../scripts/run_vitis_8x64_pd_length_sweep_hwemu.sh), [checkpoint runner](../scripts/run_vitis_8x64_qwen3b_checkpoint_hwemu_tmux.sh) |
| Case evaluation | [quantized regression](../scripts/run_quantized_block_regression.sh); streaming case-local flows |
| Long-run status | [E2E status](../scripts/status_vitis_8x64_qwen3b_e2e.sh), [archive watcher](../scripts/watch_vitis_8x64_e2e_archive_tmux.sh) |
| Evidence extraction and publication | `report_*`, `archive_*`, `verify_*`, `install_*`; see [release workflow](release-workflow.md) |

The older `run_hls_*_nohup.sh` and `run_vitis_8x64_*_nohup.sh` names are
launch wrappers, not architecture versions. Use the recipe for the desired
evidence boundary instead of selecting a script only by its name.

## Documents and artifacts

Stable design belongs in architecture, design-space and runtime pages.
Commands belong in usage and its scoped recipes. Measured interpretation
belongs in experiments; raw logs, tables, source identities and checksums
belong in immutable [result packages](../results/README.md).

Generated HLS/Vivado projects, executables, XO/xclbin files, WDB files and model
weights are external artifacts. Their identities can be recorded in a result
manifest without storing the large files in Git.
