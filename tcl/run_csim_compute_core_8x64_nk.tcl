cd [file dirname [file dirname [info script]]]
source tcl/common_hls_model_profile.tcl
source tcl/common_hls_depth_config.tcl
set project_name [llm_fpga_profiled_project_name qwen_hls_compute_core_8x64_nk_csim_prj]
append project_name [llm_fpga_depth_project_suffix]
open_project -reset $project_name
set_top compute_core_8x64_unified_nk
set cflags "-I./include -std=c++14[llm_fpga_model_cflags][llm_fpga_depth_cflags]"
add_files kernel/mm_stream_8x64_fused_mac.cpp -cflags $cflags
add_files kernel/compute_stream.cpp -cflags $cflags
add_files kernel/compute_core_8x64_unified.cpp -cflags $cflags
add_files kernel/compute_core_8x64_nk.cpp -cflags $cflags
add_files -tb tests/compute_core_8x64_nk_tb.cpp -cflags $cflags
open_solution -reset solution1 -flow_target vitis
set_part {xcu50-fsvh2104-2-e}
create_clock -period 3.333 -name default
csim_design
close_project
exit
