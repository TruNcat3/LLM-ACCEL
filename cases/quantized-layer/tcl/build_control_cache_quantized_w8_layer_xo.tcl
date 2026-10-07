set repo_root [file normalize [file dirname [file dirname [info script]]]]
cd $repo_root
source [file join $repo_root tcl quantized_layer_profile.tcl]

set project_name qwen_hls_control_cache_quantized_w8_layer_xo_prj
if {[info exists ::env(LLM_FPGA_HLS_PROJECT_NAME)] && $::env(LLM_FPGA_HLS_PROJECT_NAME) ne ""} {
    set project_name $::env(LLM_FPGA_HLS_PROJECT_NAME)
}
set output_dir [file normalize [expr {
    [info exists ::env(LLM_FPGA_XO_DIR)] && $::env(LLM_FPGA_XO_DIR) ne ""
        ? $::env(LLM_FPGA_XO_DIR) : [file join $repo_root .build w8 xo.hw_emu.f200]
}]]
set output_xo [file join $output_dir control_cache_quantized_w8_layer.xo]
set frequency 200
if {[info exists ::env(FREQUENCY)] && $::env(FREQUENCY) ne ""} {
    set frequency $::env(FREQUENCY)
}
set period [expr {1000.0 / double($frequency)}]
set cflags "-I[file normalize include] -std=c++14 -DMM_STREAM_QUANTIZED_NARROW_ACCUM"
append cflags " [quantized_layer_profile_cflags w8]"
set synth_only [quantized_layer_profile_env QUANTIZED_LAYER_SYNTH_ONLY 0]
if {$synth_only ni {0 1}} { error "QUANTIZED_LAYER_SYNTH_ONLY must be 0 or 1" }

file mkdir $output_dir
if {[info exists ::env(LLM_FPGA_HLS_PROJECT_ROOT)] && $::env(LLM_FPGA_HLS_PROJECT_ROOT) ne ""} {
    set project_root [file normalize $::env(LLM_FPGA_HLS_PROJECT_ROOT)]
    file mkdir $project_root
    cd $project_root
}
open_project -reset $project_name
set_top control_cache_quantized_w8_layer
add_files [file join $repo_root kernel control_cache_quantized_w8_layer.cpp] -cflags $cflags
open_solution -reset solution1 -flow_target vitis
set_part {xcu50-fsvh2104-2-e}
create_clock -period $period -name default
config_interface -m_axi_alignment_byte_size 64 -m_axi_max_widen_bitwidth 512
config_compile -name_max_length 256
config_export -format xo -ipname control_cache_quantized_w8_layer
csynth_design
quantized_layer_verify_weight_axi $repo_root $project_name w8
if {!$synth_only} {
    export_design -rtl verilog -format xo -ipname control_cache_quantized_w8_layer -output $output_xo
} else {
    puts "QUANTIZED_LAYER_SYNTH_ONLY_PASS precision=w8 component=controller"
}
close_project
puts "Built $output_xo at $frequency MHz"
exit
