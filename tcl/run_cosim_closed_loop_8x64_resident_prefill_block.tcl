cd [file dirname [file dirname [info script]]]
source tcl/common_hls_depth_config.tcl

# Keep the CSim, synthesis, and RTL CoSim stages on one immutable source
# snapshot.  A multi-token resident task is especially sensitive to a stale
# controller/testbench pairing because its exact task and packet counts depend
# on the configured resident-row count.
set fingerprint_files [concat [list \
    kernel/mm_controller.cpp \
    kernel/control_cache_8x64.cpp \
    kernel/mm_stream_8x64_fused_mac.cpp \
    kernel/compute_stream.cpp \
    kernel/compute_core_8x64_unified.cpp \
    kernel/cc8_status_sink.cpp \
    kernel/closed_loop_8x64_cosim.cpp \
    tests/closed_loop_8x64_cosim_tb.cpp] \
    [lsort [glob -nocomplain include/*.hpp]]]

proc source_fingerprint {files} {
    return [exec sha256sum -- {*}$files]
}

set fingerprint_before [source_fingerprint $fingerprint_files]
set project_name qwen_hls_closed_loop_resident_prefill_block_cosim_prj
if {[info exists ::env(HLS_PROJECT_NAME)] &&
    $::env(HLS_PROJECT_NAME) ne ""} {
    set project_name $::env(HLS_PROJECT_NAME)
}
append project_name [llm_fpga_depth_project_suffix]

open_project -reset $project_name
set_top cc8_closed_loop_inner_cosim
set cflags "-I./include -std=c++14 -DQWEN_TEST_SMALL -DCC8_CLOSED_LOOP_BLOCK_COMPOSED_LAYER_COSIM -DCC8_RESIDENT_LAYER_ONLY=1 -DCC8_RESIDENT_TOKEN_ROWS=8 -DCC8_RESIDENT_DUAL_VECTOR_PORTS=1[llm_fpga_depth_cflags]"
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
set fingerprint_after [source_fingerprint $fingerprint_files]
if {$fingerprint_after ne $fingerprint_before} {
    error "CoSim source fingerprint changed during synthesis; rerun from a fixed source snapshot"
}
set cosim_cmd "cosim_design [llm_fpga_cosim_options]"
eval $cosim_cmd
close_project
exit
