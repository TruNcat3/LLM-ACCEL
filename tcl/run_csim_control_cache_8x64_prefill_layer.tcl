cd [file dirname [file dirname [info script]]]
source tcl/common_hls_model_profile.tcl
source tcl/common_hls_depth_config.tcl
set project_name [llm_fpga_profiled_project_name qwen_hls_control_cache_8x64_prefill_layer_csim_prj]
append project_name [llm_fpga_depth_project_suffix]
open_project -reset $project_name
set_top control_cache_8x64_dual_core
set cflags "-I./include -std=c++14 -DCC8_RESIDENT_LAYER_ONLY=0 -DCC8_PREFILL_LAYER_ONLY -DCC8_PREFILL_LAYER_TEST_ONLY[llm_fpga_model_cflags][llm_fpga_depth_cflags]"
add_files kernel/mm_controller.cpp -cflags $cflags
add_files kernel/control_cache_8x64.cpp -cflags $cflags
add_files -tb tests/control_cache_8x64_tb.cpp -cflags $cflags
open_solution -reset solution1 -flow_target vitis
set_part {xcu50-fsvh2104-2-e}
create_clock -period 3.333 -name default
csim_design
close_project
exit
