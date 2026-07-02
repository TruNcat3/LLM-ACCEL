cd [file dirname [file dirname [info script]]]
set project_name qwen_hls_cosim_control_cache_8x64_dual_core_prj
set top_name control_cache_8x64_dual_core
set cflags "-I./include -std=c++14 -DQWEN_TEST_SMALL"
set design_files {
    kernel/mm_controller.cpp
    kernel/control_cache_8x64.cpp
}
set tb_file tests/control_cache_8x64_tb.cpp
source tcl/common_separated_cosim_flow.tcl
