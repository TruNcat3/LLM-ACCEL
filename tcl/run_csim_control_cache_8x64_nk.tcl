cd [file dirname [file dirname [info script]]]
open_project -reset qwen_hls_control_cache_8x64_nk_csim_prj
set_top control_cache_8x64_dual_core_nk
set cflags "-I./include -std=c++14 -DQWEN_TEST_SMALL"
add_files kernel/mm_controller.cpp -cflags $cflags
add_files kernel/control_cache_8x64.cpp -cflags $cflags
add_files kernel/control_cache_8x64_nk.cpp -cflags $cflags
add_files -tb tests/control_cache_8x64_nk_tb.cpp -cflags $cflags
open_solution -reset solution1 -flow_target vitis
set_part {xcu50-fsvh2104-2-e}
create_clock -period 3.333 -name default
csim_design
close_project
exit
