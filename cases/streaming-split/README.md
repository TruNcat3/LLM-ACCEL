# Case 2: Streaming Split Architecture

A separated streaming LLM accelerator where a **model-aware control/cache
core** (cc) orchestrates a **fixed compute core** (V8-2_s) through packed
AXI streams. All dimension scaling is handled by cc's `operator_program`
schedule — the compute core is never redesigned for different projection sizes.

## Quick start

```bash
# Set environment
export VITIS_ENV_SCRIPT=/path/to/vitis_2022_2_env.sh
source "$VITIS_ENV_SCRIPT"

# Compile both kernels (sw_emu)
v++ -c -t sw_emu --platform $PLATFORM -I include \
  --hls.clock 300000000:control_cache_core \
  -k control_cache_core kernel/control_cache_core.cpp \
  -o build/control_cache_core.xo

v++ -c -t sw_emu --platform $PLATFORM -I include \
  --hls.clock 300000000:qkv_tile_kernel_cc_qwen_small_core_v8_2_s \
  -k qkv_tile_kernel_cc_qwen_small_core_v8_2_s \
  kernel/qkv_tile_kernel_cc_qwen_small_core_v8_2_s.cpp \
  -o build/v8_2_s.xo

# Link
v++ -l -t sw_emu --platform $PLATFORM --config conn_v8_2x2.cfg \
  --kernel_frequency 300 \
  build/control_cache_core.xo build/v8_2_s.xo \
  -o build/v8_2x2.xclbin

# Build hosts
g++ -std=c++14 -O2 -I/opt/xilinx/xrt/include -I./include \
  host/host_v8_2x2.cpp host/xcl2.cpp -o build/host_v8_2x2 \
  -L/usr/lib/x86_64-linux-gnu -lOpenCL -lpthread
g++ -std=c++14 -O2 -I/opt/xilinx/xrt/include -I./include \
  host/host_accum.cpp host/xcl2.cpp -o build/host_accum \
  -L/usr/lib/x86_64-linux-gnu -lOpenCL -lpthread

# Run 7-op integrated layer test
XCL_EMULATION_MODE=sw_emu ./build/host_v8_2x2 build/v8_2x2.xclbin

# Run 16-chunk accumulate test (reduction 256)
XCL_EMULATION_MODE=sw_emu ./build/host_accum build/v8_2x2.xclbin 16
```

## Architecture

```
HBM ──→ cc (dataflow orchestrator)           V8-2_s (fixed compute core)
         ├─ dispatch (op_program)              ├─ load_weights (4 stream, 18cyc)
         ├─ input_path                         ├─ compute_stream (II=2, 36cyc)
         │   ├─ load_in (hidden_in HBM[0])     ├─ activate_tiles (187cyc)
         │   ├─ send_weight (4 m_axi →         └─ write_stream (128cyc)
         │   │   4 weight_stream)
         │   ├─ send_input (512-bit wide)
         │   └─ ctrl_stream
         └─ output_path
             ├─ collect (lo/hi stream)
             └─ store_out (512-bit packed)
```

**Key parameters:** INPUT_DIM=16, OUTPUT_DIM=64, 2 cores × 1 lane = 2048
matrix DSP lanes, and 4 weight m_axi ports (HBM[2:5]). Composing the HLS
single-op schedule analytically projects about 50 token/s for 36 Qwen-3B
decoder layers; this is not an end-to-end HW-Emu or board measurement.

See [design.md](docs/design.md) for full details.
