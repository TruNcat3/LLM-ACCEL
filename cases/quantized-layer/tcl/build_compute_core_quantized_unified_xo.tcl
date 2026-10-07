set repo_root [file normalize [file dirname [file dirname [info script]]]]
source [file join $repo_root tcl quantized_layer_profile.tcl]
set precision w4
if {[info exists ::env(QUANTIZED_ALIGNMENT_PRECISION)]} {
    set precision $::env(QUANTIZED_ALIGNMENT_PRECISION)
}
if {$precision ni {w4 w8}} { error "precision must be w4 or w8" }
set frequency 200
if {[info exists ::env(FREQUENCY)]} { set frequency $::env(FREQUENCY) }
set top compute_core_quantized_${precision}_unified_nk
set project_name qwen_hls_compute_core_quantized_${precision}_unified_prj
if {[info exists ::env(LLM_FPGA_HLS_PROJECT_NAME)]} {
    set project_name $::env(LLM_FPGA_HLS_PROJECT_NAME)
}
set project_root [file join $repo_root .build hls quantized_unified]
if {[info exists ::env(LLM_FPGA_HLS_PROJECT_ROOT)]} {
    set project_root [file normalize $::env(LLM_FPGA_HLS_PROJECT_ROOT)]
}
set output_dir [file join $project_root xo]
if {[info exists ::env(LLM_FPGA_XO_DIR)]} {
    set output_dir [file normalize $::env(LLM_FPGA_XO_DIR)]
}
set cflags "-I$repo_root/include -std=c++14 -DMM_STREAM_QUANTIZED_NARROW_ACCUM"
set profile [quantized_layer_profile_name $precision]
append cflags " [quantized_layer_profile_cflags $precision]"
if {$precision eq "w4"} {
    append cflags " -DMM_STREAM_8X128_INT4X4_OUTPUTS_PER_WAVE_CONFIG=128 -DMM_STREAM_8X128_INT4X4_PACKED_ACCUM_CONFIG=0"
    set matrix_file mm_stream_8x128_int4x4_block.cpp
} else {
    append cflags " -DMM_STREAM_QUANTIZED_SINGLE_ACCUM_BANK"
    set matrix_file mm_stream_4x128_int8x8_block.cpp
}
set synth_only [quantized_layer_profile_env QUANTIZED_LAYER_SYNTH_ONLY 0]
if {$synth_only ni {0 1}} { error "QUANTIZED_LAYER_SYNTH_ONLY must be 0 or 1" }
file mkdir $project_root $output_dir
cd $project_root
open_project -reset $project_name
set_top $top
set sources [list compute_core_quantized_${precision}_unified.cpp compute_stream.cpp]
if {$profile eq "integrated"} {
    # The integrated profile owns one canonical decode-row matrix datapath;
    # adding the legacy matrix source here would instantiate a second array.
    lappend sources quantized_decode_rows.cpp
} else {
    lappend sources $matrix_file
}
foreach source $sources {
    add_files [file join $repo_root kernel $source] -cflags $cflags
}
open_solution -reset solution1 -flow_target vitis
set_part {xcu50-fsvh2104-2-e}
create_clock -period [expr {1000.0 / double($frequency)}] -name default
config_compile -name_max_length 256
config_export -format xo -ipname $top
csynth_design
if {!$synth_only} {
    export_design -rtl verilog -format xo -output [file join $output_dir $top.xo]
} else {
    puts "QUANTIZED_LAYER_SYNTH_ONLY_PASS precision=$precision component=compute profile=$profile"
}
close_project
puts "QUANTIZED_UNIFIED_XO_PASS precision=$precision frequency=$frequency"
exit
