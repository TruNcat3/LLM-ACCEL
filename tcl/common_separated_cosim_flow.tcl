set prepare 0
set repo_root [pwd]
if {[info exists ::env(HLS_COSIM_PREPARE)] && $::env(HLS_COSIM_PREPARE) ne "0"} {
    set prepare 1
}

set csim_only 0
if {[info exists ::env(HLS_CSIM_ONLY)] && $::env(HLS_CSIM_ONLY) ne "0"} {
    set csim_only 1
    set prepare 1
    if {[regexp {_prj$} $project_name]} {
        regsub {_prj$} $project_name {_csim_prj} project_name
    } else {
        append project_name "_csim"
    }
}

set skip_csim 0
if {[info exists ::env(HLS_COSIM_SKIP_CSIM)] && $::env(HLS_COSIM_SKIP_CSIM) ne "0"} {
    set skip_csim 1
}

set prepare_only 0
if {[info exists ::env(HLS_COSIM_PREPARE_ONLY)] && $::env(HLS_COSIM_PREPARE_ONLY) ne "0"} {
    set prepare_only 1
}

if {$prepare} {
    open_project -reset $project_name
    set_top $top_name

    foreach design_file $design_files {
        add_files $design_file -cflags $cflags
    }
    add_files -tb $tb_file -cflags $cflags

    open_solution "solution1" -flow_target vitis
    set_part {xcu50-fsvh2104-2-e}
    create_clock -period 3.333 -name default
    config_interface -m_axi_alignment_byte_size 64 -m_axi_latency 64 -m_axi_max_widen_bitwidth 512
    config_compile -name_max_length 256

    if {$csim_only} {
        csim_design
        close_project
        exit
    }

    if {!$skip_csim} {
        csim_design
    }
    csynth_design

    if {$prepare_only} {
        # Avoid Vitis HLS spending hours in close_project/cleanup_all after
        # large RTL generation. At this point csynth reports and RTL are
        # already written, which is all prepare-only needs.
        exit
    }
} else {
    if {![file isdirectory $project_name]} {
        puts stderr "COSIM-only project '$project_name' does not exist."
        puts stderr "Run once with HLS_COSIM_PREPARE=1, or HLS_COSIM_PREPARE=1 HLS_COSIM_PREPARE_ONLY=1 to build RTL without launching cosim."
        exit 3
    }
    if {![file isdirectory "$project_name/solution1"]} {
        puts stderr "COSIM-only project '$project_name' has no solution1."
        puts stderr "Run once with HLS_COSIM_PREPARE=1, or HLS_COSIM_PREPARE=1 HLS_COSIM_PREPARE_ONLY=1 to build RTL without launching cosim."
        exit 3
    }

    open_project $project_name
    open_solution "solution1"
}

set cosim_rc [catch {
    cosim_design -rtl verilog -tool xsim -trace_level none -disable_deadlock_detection
} cosim_msg]
cd $repo_root
if {$cosim_rc != 0} {
    set cosim_log [file join $repo_root $project_name solution1 sim report cosim.log]
    puts stderr "cosim_design failed: $cosim_msg"
    puts stderr "Primary cosim log: $cosim_log"
    exit 1
}
close_project
exit
