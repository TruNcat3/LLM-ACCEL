# Called before XSim closes. Save raw transitions so phase/cycle accounting
# can be independently rebuilt without the simulator or a binary WDB reader.
if {![info exists quantized_trace_signals]} { exit 3 }
set trace_sets [list [list quantized_layer_transitions.tsv $quantized_trace_signals]]
if {[info exists quantized_function_trace_signals] && [llength $quantized_function_trace_signals]} {
    lappend trace_sets [list quantized_function_transitions.tsv $quantized_function_trace_signals]
}
if {[info exists quantized_stage_trace_signals] && [llength $quantized_stage_trace_signals]} {
    lappend trace_sets [list quantized_projection_stage_transitions.tsv $quantized_stage_trace_signals]
}
if {[info exists quantized_prefill_silu_trace_enabled] &&
    $quantized_prefill_silu_trace_enabled} {
    # Keep an empty, header-only artifact when generated actor pins are absent.
    # The downstream reporter marks it insufficient instead of treating a
    # missing hierarchy as a successful overlap measurement.
    lappend trace_sets [list quantized_prefill_silu_nested_transitions.tsv $quantized_prefill_silu_trace_signals]
}
foreach trace_set $trace_sets {
lassign $trace_set output_path paths
set trace_file [open $output_path w]
puts $trace_file "signal\ttime_ps\tvalue"
foreach path $paths {
    # A short clock sample is sufficient to establish the actual RTL period.
    set trace_end [string map {" " "" "," ""} [now]]
    if {[file tail $path] eq "ap_clk"} { set trace_end 1us }
    foreach transition [get_transitions $path -start 0 -end $trace_end] {
        lassign $transition time unit value
        if {$value eq "Blank" && $time == 0} { set value x }
        switch -- [string trimright $unit ,] {
            fs { set factor 0.001 }
            ps { set factor 1 }
            ns { set factor 1000 }
            us { set factor 1000000 }
            ms { set factor 1000000000 }
            s { set factor 1000000000000 }
            default { close $trace_file; error "Unknown time unit: $unit" }
        }
        puts $trace_file "$path\t[format %.3f [expr {$time * $factor}]]\t$value"
    }
}
close $trace_file
puts "QUANTIZED_TRACE_SAVED file=[file normalize $output_path] end=[now]"
}
