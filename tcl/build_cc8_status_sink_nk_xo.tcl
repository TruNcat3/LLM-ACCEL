cd [file dirname [file dirname [info script]]]

set project_name qwen_hls_cc8_status_sink_nk_prj
set kernel_name cc8_status_sink_nk
set output_dir [file normalize vitis_8x64/xo]
set output_xo [file join $output_dir ${kernel_name}.xo]
set cflags "-I[file normalize include] -std=c++14 -DQWEN_TEST_SMALL"

file mkdir $output_dir
open_project -reset $project_name
set_top $kernel_name
add_files kernel/cc8_status_sink.cpp -cflags $cflags
open_solution -reset solution1 -flow_target vitis
set_part {xcu50-fsvh2104-2-e}
create_clock -period 3.333 -name default
config_interface -m_axi_alignment_byte_size=64
config_interface -m_axi_max_widen_bitwidth=512
config_export -format xo -ipname $kernel_name
csynth_design
export_design -rtl verilog -format xo -output $output_xo
close_project

puts "Built $output_xo"
exit
