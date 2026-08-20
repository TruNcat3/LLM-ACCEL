set repo_root [file normalize [file dirname [file dirname [info script]]]]
cd $repo_root
source tcl/common_hls_depth_config.tcl
source tcl/common_hls_model_profile.tcl

if {[info exists ::env(LLM_FPGA_HLS_PROJECT_NAME)] &&
    $::env(LLM_FPGA_HLS_PROJECT_NAME) ne ""} {
    set project_name $::env(LLM_FPGA_HLS_PROJECT_NAME)
} else {
    set project_name [llm_fpga_profiled_project_name qwen_hls_compute_core_8x64_nk_prj]
}
set kernel_name compute_core_8x64_unified_nk
set output_dir [llm_fpga_profiled_xo_dir]
set output_xo [file join $output_dir ${kernel_name}.xo]
set cflags "-I[file normalize include] -std=c++14[llm_fpga_model_cflags][llm_fpga_depth_cflags]"

puts "Model profile: [llm_fpga_model_profile]"

file mkdir $output_dir
if {[info exists ::env(LLM_FPGA_HLS_PROJECT_ROOT)] &&
    $::env(LLM_FPGA_HLS_PROJECT_ROOT) ne ""} {
    set project_root [file normalize $::env(LLM_FPGA_HLS_PROJECT_ROOT)]
    file mkdir $project_root
    cd $project_root
}
open_project -reset $project_name
set_top $kernel_name
add_files [file join $repo_root kernel/mm_stream_8x64_fused_mac.cpp] -cflags $cflags
add_files [file join $repo_root kernel/compute_stream.cpp] -cflags $cflags
add_files [file join $repo_root kernel/compute_core_8x64_unified.cpp] -cflags $cflags
add_files [file join $repo_root kernel/compute_core_8x64_nk.cpp] -cflags $cflags
open_solution -reset solution1 -flow_target vitis
set_part {xcu50-fsvh2104-2-e}
create_clock -period 3.333 -name default
config_compile -name_max_length=256
config_export -format xo -ipname $kernel_name
csynth_design
export_design -rtl verilog -format xo -output $output_xo
close_project

puts "Built $output_xo"
exit
