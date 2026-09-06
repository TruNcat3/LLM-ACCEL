cd [file dirname [file dirname [info script]]]
set project_name qwen_hls_cosim_mm_stream_4x128_int8x8_block_prj
set top_name compute_mm_stream_4x128_int8x8_block_nk
set cflags "-I./include -I../../include -std=c++14"
set design_files {kernel/mm_stream_4x128_int8x8_block.cpp}
set tb_file tests/mm_stream_4x128_int8x8_block_tb.cpp
set shared_tcl_dir [file normalize [file join [pwd] ../../tcl]]
source [file join $shared_tcl_dir common_hls_depth_config.tcl]
source [file join $shared_tcl_dir common_separated_cosim_flow.tcl]
