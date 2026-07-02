cd [file dirname [file dirname [info script]]]

set project_name qwen_hls_control_cache_8x64_nk_prj
set kernel_name control_cache_8x64_dual_core_nk
set output_dir [file normalize vitis_8x64/xo]
set output_xo [file join $output_dir ${kernel_name}.xo]
set cflags "-I[file normalize include] -std=c++14 -DQWEN_TEST_SMALL"

file mkdir $output_dir
open_project -reset $project_name
set_top $kernel_name
add_files kernel/mm_controller.cpp -cflags $cflags
add_files kernel/control_cache_8x64.cpp -cflags $cflags
add_files kernel/control_cache_8x64_nk.cpp -cflags $cflags
open_solution -reset solution1 -flow_target vitis
set_part {xcu50-fsvh2104-2-e}
create_clock -period 3.333 -name default
config_interface -m_axi_alignment_byte_size=64
config_interface -m_axi_max_widen_bitwidth=512
config_compile -name_max_length=256
config_export -format xo -ipname $kernel_name
csynth_design
export_design -rtl verilog -format xo -output $output_xo
close_project

puts "Built $output_xo"
exit
