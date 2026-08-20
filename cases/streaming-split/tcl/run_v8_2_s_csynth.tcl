# V8-2_s synthesis (stream-weight version): verify II=1 and generate RTL for co-simulation.
open_project qkv_tile_kernel_cc_qwen_small_core_v8_2_s -reset
set_top qkv_tile_kernel_cc_qwen_small_core_v8_2_s
add_files kernel/qkv_tile_kernel_cc_qwen_small_core_v8_2_s.cpp
open_solution "csynth_solution" -reset
set_part {xcu250-figd2104-2L-e}
create_clock -period 3.33 -name default
config_compile -name_max_length 256
config_rtl -reset control

puts "=========================================="
puts "V8-2_s synthesis (stream-weight, design path 2)"
puts "Goal: compute_tiles II=1 with synthesizable weight_stream loading"
puts "=========================================="

csynth_design

if {[file exists qkv_tile_kernel_cc_qwen_small_core_v8_2_s/csynth_solution/syn/report/csynth.rpt]} {
    puts "V8-2_s synthesis completed successfully."
} else {
    puts "V8-2_s synthesis failed."
}
exit
