# control_cache_core 综合 — 复核 512-bit store_out 周期 (8192→~512)
open_project control_cache_core -reset
set_top control_cache_core
add_files kernel/control_cache_core.cpp
open_solution "csynth_solution" -reset
set_part {xcu250-figd2104-2L-e}
create_clock -period 3.33 -name default
config_compile -name_max_length 256
config_rtl -reset control

puts "=========================================="
puts "cc 综合 (512-bit store_out 打包写)"
puts "目标: store_out 从 8192cyc 降到 ~512cyc, 无 II 退化"
puts "=========================================="

csynth_design

if {[file exists control_cache_core/csynth_solution/syn/report/csynth.rpt]} {
    puts "✅ cc 综合完成!"
} else {
    puts "❌ cc 综合失败!"
}
exit
