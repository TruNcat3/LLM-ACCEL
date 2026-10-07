# cowave-streaming-split

[Repository](../../README.md) | [Repository map](../../docs/repository-map.md) |
[Implementation catalog](../../docs/implementations.md) |
[Design space](../../docs/design-space.md) | [Environment](../../docs/environment.md) |
[Detailed design](docs/design.md)

The streaming split is an alternative implementation family. It combines a
model-aware control/cache core (`control_cache_core`, or `cc`) with a fixed
V8-2_s compute core. It is independent of the root Fix16 resident controller
and does not inherit that family's Task 18/19/20, KV-ownership, or end-to-end
correctness claims.

## Design boundary

| Component | Public role | Source entry |
| --- | --- | --- |
| `cc` | Three-process dataflow orchestrator; reads hidden input, weights, and an `operator_program`, then routes packed streams | [`kernel/control_cache_core.cpp`](kernel/control_cache_core.cpp) |
| V8-2_s | Fixed 16-input by 64-output matmul core with two cores and one lane per core | [`kernel/qkv_tile_kernel_cc_qwen_small_core_v8_2_s.cpp`](kernel/qkv_tile_kernel_cc_qwen_small_core_v8_2_s.cpp) |
| Host programs | Seven-operator integration and parameterized accumulation checks | [`host/host_v8_2x2.cpp`](host/host_v8_2x2.cpp), [`host/host_accum.cpp`](host/host_accum.cpp) |
| Connectivity | Four weight HBM ports and stream links | [`conn_v8_2x2.cfg`](conn_v8_2x2.cfg) |

The `operator_program` is a flat six-field `uint32` record per operation. It
selects activation/finalize behavior, weight and hidden offsets, and whether
the caller clears or accumulates output. The compute core therefore remains a
regular stream service; model dimensions are expressed by repeated operations,
not by changing the core shape.

## Numeric boundary

The case source defines these payload and accumulator formats:

| Type | Source format | Use |
| --- | --- | --- |
| `fm_t` | `ap_fixed<16,8>` | Activations and input stream values |
| `wt_linear_t` | `ap_fixed<16,4>` | Linear weights |
| `fm_accum_t` | `ap_fixed<32,16>` | Stream outputs and Host decoding |
| Internal matmul accumulator | `ap_fixed<48,24>` | Unsaturated accumulation before one output conversion |

Activation softmax and layer-normalization helpers call `hls::recip` after
converting a fixed-point value to `float`. The family therefore has fixed-point
storage and packet formats, but it is not an end-to-end equivalent of the root
Fix16 numerical path. The case does not publish model-level accuracy evidence.

<a id="quick-start"></a>
## Reproduce the bounded case

Run from the repository checkout. These commands are case-local and use the
reference environment contract:

```bash
cd cases/streaming-split
source ../../scripts/setup_environment.sh
../../scripts/check_environment.sh hls
mkdir -p build

v++ -c -t sw_emu --platform "${XPLATFORM}" -I include \
  --hls.clock 300000000:control_cache_core \
  -k control_cache_core kernel/control_cache_core.cpp \
  -o build/control_cache_core.xo

v++ -c -t sw_emu --platform "${XPLATFORM}" -I include \
  --hls.clock 300000000:qkv_tile_kernel_cc_qwen_small_core_v8_2_s \
  -k qkv_tile_kernel_cc_qwen_small_core_v8_2_s \
  kernel/qkv_tile_kernel_cc_qwen_small_core_v8_2_s.cpp \
  -o build/v8_2_s.xo

v++ -l -t sw_emu --platform "${XPLATFORM}" --config conn_v8_2x2.cfg \
  --kernel_frequency 300 build/control_cache_core.xo build/v8_2_s.xo \
  -o build/v8_2x2.xclbin

g++ -std=c++14 -O2 -I"${XILINX_XRT}/include" -I./include \
  host/host_v8_2x2.cpp host/xcl2.cpp -o build/host_v8_2x2 \
  -L"${XILINX_XRT}/lib" -lOpenCL -lpthread
g++ -std=c++14 -O2 -I"${XILINX_XRT}/include" -I./include \
  host/host_accum.cpp host/xcl2.cpp -o build/host_accum \
  -L"${XILINX_XRT}/lib" -lOpenCL -lpthread

XCL_EMULATION_MODE=sw_emu ./build/host_v8_2x2 build/v8_2x2.xclbin
XCL_EMULATION_MODE=sw_emu ./build/host_accum build/v8_2x2.xclbin 16
```

The first Host check covers the seven Q/K/V/O/Gate/Up/Down operation sequence;
the second covers a 16-chunk accumulation path. HLS Tcl entry points and the
source-level details are in the [detailed design](docs/design.md).

## Evidence boundary

| Evidence | What it supports | What it does not support |
| --- | --- | --- |
| CSim/sw_emu | Packet routing, bounded accumulation, and the case-local Host checks | Root resident runtime behavior or model accuracy |
| HLS CSynth | Local II, resource, and timing estimates for V8-2_s and `cc` | Routed utilization, board frequency, or system throughput |
| Case emulation checks | The documented bounded stream path and no-stall/deadlock behavior for the tested setup | A complete Qwen model run or checkpoint result |
| Analytical composition | A projection made from the single-operation schedule | Measured full-layer HW-Emu or board throughput |

Generated XO/xclbin files, HLS projects, and traces are not committed. See
[`docs/design.md`](docs/design.md) for source-level details and
[`docs/architecture.md`](../../docs/architecture.md) for the cross-family
comparison.
