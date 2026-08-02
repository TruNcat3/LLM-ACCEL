# V8-2_s 综合 (stream-weight 版, 确认 II=1 + 为 cosim 生成 RTL)
open_project qkv_tile_kernel_cc_qwen_small_core_v8_2_s -reset
set_top qkv_tile_kernel_cc_qwen_small_core_v8_2_s
add_files kernel/qkv_tile_kernel_cc_qwen_small_core_v8_2_s.cpp
open_solution "csynth_solution" -reset
set_part {xcu250-figd2104-2L-e}
create_clock -period 3.33 -name default
config_compile -name_max_length 256
config_rtl -reset control

puts "=========================================="
puts "V8-2_s 综合 (stream-weight, 路线2)"
puts "目标: compute_tiles II=1 + weight_stream 加载可综合"
puts "=========================================="

csynth_design

if {[file exists qkv_tile_kernel_cc_qwen_small_core_v8_2_s/csynth_solution/syn/report/csynth.rpt]} {
    puts "✅ V8-2_s 综合完成!"
} else {
    puts "❌ V8-2_s 综合失败!"
}
exit
