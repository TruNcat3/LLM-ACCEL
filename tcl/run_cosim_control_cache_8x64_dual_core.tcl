cd [file dirname [file dirname [info script]]]
source tcl/common_hls_model_profile.tcl
source tcl/common_hls_depth_config.tcl
set project_name [llm_fpga_profiled_project_name qwen_hls_cosim_control_cache_8x64_dual_core_prj]
append project_name [llm_fpga_depth_project_suffix]
set top_name control_cache_8x64_dual_core
set cflags "-I./include -std=c++14[llm_fpga_model_cflags][llm_fpga_depth_cflags]"
set design_files {
    kernel/mm_controller.cpp
    kernel/control_cache_8x64.cpp
}
set tb_file tests/control_cache_8x64_tb.cpp
source tcl/common_separated_cosim_flow.tcl
