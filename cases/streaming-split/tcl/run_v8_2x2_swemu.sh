#!/bin/bash
# Phase D: Vitis 2022.2 sw_emu compile + link (control_cache_core + 2×V8-2_s)
# 板卡/XRT 是 2022, 故切换到 Vitis 2022.2 + 平台 xilinx_u50_gen3x16_xdma_5_202210_1
set -e
cd /home/wt/git/LLM-FPGA-CC
PLATFORM=/opt/xilinx/platforms/xilinx_u50_gen3x16_xdma_5_202210_1/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm
VPP=/tools/Xilinx/Vitis/2022.2/bin/v++
mkdir -p build_swemu logs

echo "=== v++ compile control_cache_core (sw_emu) ==="
$VPP -c -t sw_emu --platform $PLATFORM -I include \
  --hls.clock 300000000:control_cache_core \
  -k control_cache_core kernel/control_cache_core.cpp \
  -o build_swemu/control_cache_core.xo 2>&1 | tee logs/vpp_compile_cc.log

echo "=== v++ compile V8-2_s (sw_emu) ==="
$VPP -c -t sw_emu --platform $PLATFORM -I include \
  --hls.clock 300000000:qkv_tile_kernel_cc_qwen_small_core_v8_2_s \
  -k qkv_tile_kernel_cc_qwen_small_core_v8_2_s kernel/qkv_tile_kernel_cc_qwen_small_core_v8_2_s.cpp \
  -o build_swemu/v8_2_s.xo 2>&1 | tee logs/vpp_compile_v82.log

echo "=== v++ link (conn_v8_2x2.cfg → xclbin) ==="
$VPP -l -t sw_emu --platform $PLATFORM --config conn_v8_2x2.cfg \
  --kernel_frequency 300 \
  build_swemu/control_cache_core.xo build_swemu/v8_2_s.xo \
  -o build_swemu/v8_2x2.xclbin 2>&1 | tee logs/vpp_link.log

echo "=== sw_emu compile+link 完成 ==="
ls -la build_swemu/v8_2x2.xclbin
