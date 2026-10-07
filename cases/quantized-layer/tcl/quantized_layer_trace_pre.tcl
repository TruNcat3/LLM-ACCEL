# Trace actual RTL handshakes. Do not use the generated profile's fallback
# whole-simulation duration as kernel activity. Keep this independent of HLS
# function profiling, which may be unavailable in an exported XO.
if {![info exists ::env(QUANTIZED_LAYER_PRECISION)]} {
    puts stderr "QUANTIZED_LAYER_PRECISION is required"
    exit 2
}
set precision $::env(QUANTIZED_LAYER_PRECISION)
if {$precision ni {w4 w8}} { exit 2 }
set prefix q[string index $precision 1]_layer
set base /pfm_top_wrapper/pfm_top_i/pfm_dynamic_inst
set quantized_trace_signals {}
foreach cu {ctrl cu0 cu1 cu2 cu3} {
    foreach pin {ap_clk ap_rst_n ap_start ap_done ap_idle ap_ready} {
        set path $base/${prefix}_${cu}/inst/$pin
        if {[llength [get_objects -quiet $path]] != 1} {
            puts stderr "QUANTIZED_TRACE_MISSING $path"
            exit 3
        }
        log_wave -quiet $path
        lappend quantized_trace_signals $path
    }
}
puts "QUANTIZED_TRACE_READY precision=$precision signals=[llength $quantized_trace_signals]"

# The Prefill FFN experiment needs a small nested trace in addition to the
# direct-child function trace above.  Keep this opt-in and bounded: recursive
# wildcard logging of every generated HLS object can make the WDB unusable.
set quantized_prefill_silu_trace_enabled 0
set quantized_prefill_silu_trace_status disabled
set quantized_prefill_silu_trace_signals {}
set quantized_prefill_silu_trace_mode_signals {}
set quantized_prefill_silu_trace_fifo_signals {}
set quantized_prefill_silu_trace_roles {}
if {[info exists ::env(QUANTIZED_LAYER_PREFILL_FFN_OVERLAP)] &&
    $::env(QUANTIZED_LAYER_PREFILL_FFN_OVERLAP) eq "1"} {
    set quantized_prefill_silu_trace_enabled 1
    set quantized_prefill_silu_trace_status insufficient
    set nested_root "$base/${prefix}_ctrl/inst"

    # HLS emits one ap_ctrl_hs actor per non-inlined helper. Prefer semantic
    # actor names and ignore generated pipeline-loop descendants. The cap is
    # intentional: this evidence is for topology, not a full WDB.
    set nested_candidates [get_objects -quiet -r ap_idle]
    set nested_patterns [list \
        [list silu_issuer {emit_quantized_lookahead_operands emit_quantized_vector_operands}] \
        [list silu_collector {collect_quantized_vector_results collect_quantized_projection_with_lookahead}] \
        [list up_issuer {drive_quantized_projection_with_lookahead drive_quantized_projection_wave_range_banked}] \
        [list up_collector {commit_quantized_projection_wave_range}]]
    set nested_selected [dict create]
    set nested_role_counts [dict create]
    foreach candidate [lsort -unique $nested_candidates] {
        if {[string first "${nested_root}/" $candidate] != 0} { continue }
        set instance [file dirname $candidate]
        set instance_name [file tail $instance]
        if {[string match {*_Pipeline_*} $instance_name]} { continue }
        # Only accept actors below the new lookahead dataflow wrapper. This
        # prevents similarly named vector helpers elsewhere in the controller
        # from entering the bounded trace.
        set lookahead_ancestor 0
        foreach component [file split $instance] {
            if {[string first "run_quantized_projection_with_lookahead" $component] >= 0} {
                set lookahead_ancestor 1
                break
            }
        }
        if {!$lookahead_ancestor} { continue }
        set role ""
        # Test the specific vector names before their enclosing wrapper.
        foreach item $nested_patterns {
            lassign $item item_role patterns
            foreach pattern $patterns {
                if {[string first $pattern $instance_name] >= 0} {
                    set role $item_role
                    break
                }
            }
            if {$role ne ""} { break }
        }
        if {$role eq "" || [dict exists $nested_selected $instance]} { continue }
        set role_count 0
        if {[dict exists $nested_role_counts $role]} {
            set role_count [dict get $nested_role_counts $role]
        }
        if {$role_count >= 2} { continue }
        dict set nested_selected $instance $role
        dict set nested_role_counts $role [expr {$role_count + 1}]
        if {[dict size $nested_selected] >= 8} { break }
    }

    # Add a small handshake set for each selected actor. Missing generated
    # pins are expected across HLS versions and do not invalidate the run.
    foreach instance [lsort [dict keys $nested_selected]] {
        set role [dict get $nested_selected $instance]
        foreach pin {ap_start ap_done ap_ready ap_idle} {
            set path "$instance/$pin"
            if {[llength [get_objects -quiet $path]] != 1} { continue }
            if {[lsearch -exact $quantized_prefill_silu_trace_signals $path] >= 0} {
                continue
            }
            log_wave -quiet $path
            lappend quantized_prefill_silu_trace_signals $path
            dict set quantized_prefill_silu_trace_roles $path $role
        }
    }

    # The ordered result FIFO can backpressure Up while SiLU drains. Retain
    # only one matrix_results_U* FIFO owner per compute CU and its four direct
    # handshake pins. Do not recurse into arbitrary payload/storage signals.
    set fifo_pin_names {full_n write empty_n read}
    set fifo_instances_by_cu [dict create]
    foreach cu {cu0 cu1 cu2 cu3} {
        set cu_root "$base/${prefix}_${cu}/inst"
        set candidates [dict create]
        foreach local_name $fifo_pin_names {
            set fifo_objects {}
            if {[catch {set fifo_objects [get_objects -quiet -r $local_name]}]} {
                set fifo_objects {}
            }
            foreach path [lsort -unique $fifo_objects] {
                if {[string first "${cu_root}/" $path] != 0} { continue }
                set instance [file dirname $path]
                set instance_name [file tail $instance]
                if {[string first "matrix_results_U" $instance_name] < 0} { continue }
                dict set candidates $instance 1
            }
        }
        if {[dict size $candidates]} {
            dict set fifo_instances_by_cu $cu [lindex [lsort [dict keys $candidates]] 0]
        }
    }
    foreach cu [dict keys $fifo_instances_by_cu] {
        set instance [dict get $fifo_instances_by_cu $cu]
        foreach pin $fifo_pin_names {
            set path "$instance/$pin"
            if {[llength [get_objects -quiet $path]] != 1} { continue }
            if {[lsearch -exact $quantized_prefill_silu_trace_signals $path] >= 0} {
                continue
            }
            log_wave -quiet $path
            lappend quantized_prefill_silu_trace_signals $path
            lappend quantized_prefill_silu_trace_fifo_signals $path
            dict set quantized_prefill_silu_trace_roles $path matrix_results_fifo
        }
    }

    # Operation-selector pins are optional. Without enable_silu=1 the
    # reporter keeps the actor intervals as insufficient waits-included data.
    set nested_mode_count 0
    foreach local_name {enable_silu enable_silu_V kind kind_V} {
        set mode_objects {}
        if {[catch {set mode_objects [get_objects -quiet -r $local_name]}]} {
            set mode_objects {}
        }
        foreach path [lsort -unique $mode_objects] {
            set matched 0
            foreach instance [dict keys $nested_selected] {
                if {[file dirname $path] eq $instance} {
                    set matched 1
                    break
                }
            }
            if {!$matched || [lsearch -exact $quantized_prefill_silu_trace_signals $path] >= 0} {
                continue
            }
            set destination quantized_prefill_silu_trace_mode_signals
            log_wave -quiet $path
            lappend $destination $path
            lappend quantized_prefill_silu_trace_signals $path
            dict set quantized_prefill_silu_trace_roles $path $destination
            incr nested_mode_count
            if {$nested_mode_count >= 8} { break }
        }
        if {$nested_mode_count >= 8} { break }
    }
    if {[llength $quantized_prefill_silu_trace_signals]} {
        set quantized_prefill_silu_trace_status ready
    }
    puts "QUANTIZED_PREFILL_SILU_TRACE_READY enabled=1 status=$quantized_prefill_silu_trace_status signals=[llength $quantized_prefill_silu_trace_signals] mode_signals=[llength $quantized_prefill_silu_trace_mode_signals] fifo_signals=[llength $quantized_prefill_silu_trace_fifo_signals]"
}

# Record the four concurrent projection processes inside the controller.
# ap_idle intervals include backpressure waits; these establish process
# overlap, not multiply occupancy or useful transfers on every active cycle.
set quantized_stage_trace_signals {}
if {[info exists ::env(QUANTIZED_LAYER_STAGE_TRACE)] &&
    $::env(QUANTIZED_LAYER_STAGE_TRACE) eq "1"} {
    source [file join [file dirname [info script]] quantized_projection_trace_helpers.tcl]
    foreach stage {load_quantized_projection_weight_range drive_quantized_projection_wave_range collect_quantized_projection_wave_range commit_quantized_projection_wave_range} {
        set count 0
        foreach path [quantized_projection_stage_objects "$base/${prefix}_ctrl/inst" $stage] {
            log_wave -quiet $path
            lappend quantized_stage_trace_signals $path
            incr count
        }
        if {$count == 0} {
            puts stderr "QUANTIZED_TRACE_MISSING projection_stage=$stage"
            exit 5
        }
        puts "QUANTIZED_STAGE_TRACE_READY stage=$stage instances=$count"
    }
    set quantized_stage_trace_signals [lsort -unique $quantized_stage_trace_signals]
}

# Register all user and generated profiling signals at time zero before the
# clock probe. The helper removes unused startup WDB reads from the generated
# runtime script; the standard tool hook later repeats only registrations.
source [file join [file dirname [info script]] quantized_profile_init.tcl]
set quantized_function_trace_signals {}
if {[info exists ::env(QUANTIZED_LAYER_FUNCTION_TRACE)] &&
    $::env(QUANTIZED_LAYER_FUNCTION_TRACE) eq "1"} {
    set pattern "^${base}/${prefix}_"
    append pattern {(ctrl|cu[0-3])/inst/grp_[^/]+/ap_idle$}
    set controller_count 0
    foreach path [get_objects -quiet -r ap_idle] {
        if {![regexp $pattern $path]} { continue }
        log_wave -quiet $path
        lappend quantized_function_trace_signals $path
        if {[string first "${prefix}_ctrl/" $path] >= 0} { incr controller_count }
    }
    if {$controller_count == 0} { error "QUANTIZED_TRACE_MISSING controller_function_pins" }
    puts "QUANTIZED_FUNCTION_TRACE_READY signals=[llength $quantized_function_trace_signals]"
}
# --debug off snapshots expose names but return only a Blank transition.
# Detect this before spending hours on the workload.
run 1us
set clock_path $base/${prefix}_ctrl/inst/ap_clk
set clock_transitions [get_transitions $clock_path -start 0 -end 1us]
set clock_edges 0
foreach transition $clock_transitions {
    if {[lindex $transition 2] eq "1"} { incr clock_edges }
}
if {$clock_edges < 10} {
    puts stderr "QUANTIZED_TRACE_RECORDING_FAILED sample=$clock_transitions; relink HW Emu with v++ -g"
    exit 4
}
puts "QUANTIZED_TRACE_CLOCK_READY rising_edges=$clock_edges count=[llength $clock_transitions] sample=[lrange $clock_transitions 0 5]"
puts "QUANTIZED_TRACE_INIT_READY stage_signals=[llength $quantized_stage_trace_signals]"
