cd [file dirname [file dirname [file dirname [file dirname [info script]]]]]
set project_name qwen_hls_cosim_mm_stream_4x128_int8x8_block_dspacc_narrow_prj
set top_name compute_mm_stream_4x128_int8x8_block_nk
set cflags "-I./include -I./cases/quantized-block/include -std=c++14"
set design_files {cases/quantized-block/kernel/mm_stream_4x128_int8x8_block.cpp}
set tb_file cases/quantized-block/tests/mm_stream_4x128_int8x8_block_tb.cpp
source tcl/common_separated_cosim_flow.tcl
