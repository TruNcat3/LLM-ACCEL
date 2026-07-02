cd [file dirname [file dirname [info script]]]
set project_name qwen_hls_cosim_compute_core_8x64_unified_prj
set top_name compute_core_8x64_unified
set cflags "-I./include -std=c++14 -DQWEN_TEST_SMALL"
set design_files {
    kernel/mm_stream_8x64_fused_mac.cpp
    kernel/compute_stream.cpp
    kernel/compute_core_8x64_unified.cpp
}
set tb_file tests/compute_core_8x64_unified_tb.cpp
source tcl/common_separated_cosim_flow.tcl
