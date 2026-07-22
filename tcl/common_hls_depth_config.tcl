proc llm_fpga_depth_cflags {} {
    set depth_vars {
        CC8_NK_TASK_STREAM_DEPTH
        CC8_NK_DATA_STREAM_DEPTH
        CC8_NK_STATUS_STREAM_DEPTH
        CU8_NK_TASK_STREAM_DEPTH
        CU8_NK_DATA_STREAM_DEPTH
        CU8_UNIFIED_ACCUM_STREAM_DEPTH
        CU8_UNIFIED_CONVERTED_STREAM_DEPTH
        CC8_ENABLE_WEIGHT_TILE_PIPELINE
        CC8_WEIGHT_TILE_FIFO_DEPTH
        CC8_WEIGHT_TILE_LOAD_II
        CC8_ENABLE_MM_WAVE_REPEAT
        CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW
        CC8_MM_WAVE_RESULT_FIFO_DEPTH
    }

    set cflags ""
    foreach var $depth_vars {
        if {[info exists ::env($var)] && $::env($var) ne ""} {
            append cflags " -D${var}=$::env($var)"
        }
    }
    if {[info exists ::env(HLS_EXTRA_CFLAGS)] && $::env(HLS_EXTRA_CFLAGS) ne ""} {
        append cflags " $::env(HLS_EXTRA_CFLAGS)"
    }
    return $cflags
}

proc llm_fpga_depth_project_suffix {} {
    set suffix ""
    set suffix_vars {
        CC8_NK_TASK_STREAM_DEPTH
        CC8_NK_DATA_STREAM_DEPTH
        CC8_NK_STATUS_STREAM_DEPTH
        CU8_NK_TASK_STREAM_DEPTH
        CU8_NK_DATA_STREAM_DEPTH
        CU8_UNIFIED_ACCUM_STREAM_DEPTH
        CU8_UNIFIED_CONVERTED_STREAM_DEPTH
        CC8_ENABLE_WEIGHT_TILE_PIPELINE
        CC8_WEIGHT_TILE_FIFO_DEPTH
        CC8_WEIGHT_TILE_LOAD_II
        CC8_ENABLE_MM_WAVE_REPEAT
        CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW
        CC8_MM_WAVE_RESULT_FIFO_DEPTH
    }

    foreach var $suffix_vars {
        if {[info exists ::env($var)] && $::env($var) ne ""} {
            append suffix "_${var}$::env($var)"
        }
    }
    regsub -all {[^A-Za-z0-9_]} $suffix "_" suffix
    return $suffix
}

proc llm_fpga_cosim_options {} {
    set options "-rtl verilog -tool xsim -trace_level none"
    if {[info exists ::env(HLS_COSIM_DISABLE_DEADLOCK_DETECTION)] &&
        $::env(HLS_COSIM_DISABLE_DEADLOCK_DETECTION) ne "" &&
        $::env(HLS_COSIM_DISABLE_DEADLOCK_DETECTION) ne "0"} {
        append options " -disable_deadlock_detection"
    }
    return $options
}
