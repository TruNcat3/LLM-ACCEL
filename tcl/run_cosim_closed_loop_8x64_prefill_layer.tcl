cd [file dirname [file dirname [info script]]]
if {![info exists ::env(LLM_FPGA_MODEL_PROFILE)]} {
    set ::env(LLM_FPGA_MODEL_PROFILE) qwen-layer
}
source tcl/common_hls_model_profile.tcl
source tcl/common_hls_depth_config.tcl

set project_name [llm_fpga_profiled_project_name qwen_hls_closed_loop_8x64_prefill_layer_cosim_prj]
append project_name [llm_fpga_depth_project_suffix]
open_project -reset $project_name
set_top cc8_closed_loop_inner_cosim
set cflags "-I./include -std=c++14 -DCC8_RESIDENT_LAYER_ONLY=0 -DCC8_PREFILL_LAYER_ONLY -DCC8_CLOSED_LOOP_PREFILL_BLOCK_COSIM[llm_fpga_model_cflags][llm_fpga_depth_cflags]"
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
csynth_design
set cosim_cmd "cosim_design [llm_fpga_cosim_options]"
eval $cosim_cmd
close_project
exit
