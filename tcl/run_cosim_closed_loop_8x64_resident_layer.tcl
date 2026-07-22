cd [file dirname [file dirname [info script]]]
source tcl/common_hls_depth_config.tcl

open_project -reset qwen_hls_closed_loop_resident_layer_cosim_prj
set_top cc8_closed_loop_inner_cosim
set cflags "-I./include -std=c++14 -DQWEN_TEST_SMALL -DCC8_CLOSED_LOOP_RESIDENT_LAYER_COSIM -DCC8_RESIDENT_LAYER_ONLY=1[llm_fpga_depth_cflags]"
add_files kernel/mm_controller.cpp -cflags $cflags
add_files kernel/control_cache_8x64.cpp -cflags $cflags
add_files kernel/mm_stream_8x64_fused_mac.cpp -cflags $cflags
add_files kernel/compute_stream.cpp -cflags $cflags
add_files kernel/compute_core_8x64_unified.cpp -cflags $cflags
add_files kernel/cc8_status_sink.cpp -cflags $cflags
add_files kernel/closed_loop_8x64_cosim.cpp -cflags $cflags
add_files -tb tests/closed_loop_8x64_cosim_tb.cpp -cflags $cflags
open_solution -reset solution1 -flow_target vitis
set_part {xcu50-fsvh2104-2-e}
create_clock -period 3.333 -name default
config_interface -m_axi_alignment_byte_size=64
config_interface -m_axi_max_widen_bitwidth=512
config_compile -name_max_length=256

csim_design
if {[info exists ::env(HLS_CSIM_ONLY)] &&
    $::env(HLS_CSIM_ONLY) ne "" &&
    $::env(HLS_CSIM_ONLY) ne "0"} {
    close_project
    exit
}

csynth_design
set cosim_cmd "cosim_design [llm_fpga_cosim_options]"
eval $cosim_cmd
close_project
exit
