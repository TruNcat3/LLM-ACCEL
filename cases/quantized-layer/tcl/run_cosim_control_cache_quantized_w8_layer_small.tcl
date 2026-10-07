set repo_root [file normalize [file dirname [file dirname [info script]]]]
cd $repo_root

set project_name qwen_hls_control_cache_quantized_w8_layer_small_cosim_prj
if {[info exists ::env(LLM_FPGA_HLS_PROJECT_NAME)] &&
    $::env(LLM_FPGA_HLS_PROJECT_NAME) ne ""} {
    set project_name $::env(LLM_FPGA_HLS_PROJECT_NAME)
}
if {[info exists ::env(LLM_FPGA_HLS_PROJECT_ROOT)] &&
    $::env(LLM_FPGA_HLS_PROJECT_ROOT) ne ""} {
    set project_root [file normalize $::env(LLM_FPGA_HLS_PROJECT_ROOT)]
    file mkdir $project_root
    cd $project_root
}

set cflags "-I[file normalize [file join $repo_root include]] -std=c++14 -pthread -DQWEN_TEST_SMALL -DMM_STREAM_QUANTIZED_NARROW_ACCUM"
open_project -reset $project_name
set_top control_cache_quantized_w8_layer
add_files [file join $repo_root kernel control_cache_quantized_w8_layer.cpp] -cflags $cflags
add_files -tb [file join $repo_root tests control_cache_quantized_w8_layer_small_tb.cpp] -cflags $cflags
open_solution -reset solution1 -flow_target vitis
set_part {xcu50-fsvh2104-2-e}
create_clock -period 5.000 -name default
config_interface -m_axi_alignment_byte_size 64 -m_axi_max_widen_bitwidth 512
config_compile -name_max_length 256

csim_design
if {[info exists ::env(HLS_CSIM_ONLY)] &&
    $::env(HLS_CSIM_ONLY) ne "" && $::env(HLS_CSIM_ONLY) ne "0"} {
    close_project
    exit
}

csynth_design
# Deadlock detection remains enabled by default. Full trace is requested so
# a future failure can be inspected in the generated xsim waveform database.
cosim_design -rtl verilog -tool xsim -trace_level all
close_project
exit
