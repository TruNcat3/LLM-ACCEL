# Vivado/XSim recursive object queries match the signal's local name.
# Filter full paths after discovery, then match the immediate owning module
# so descendants of a projection process are not counted as extra stages.
#
# The lookahead topology keeps the projection issue and result collection in
# ap_ctrl_hs wrapper functions.  Those wrappers also emit/collect the next
# block's RMSNorm work, so their ap_idle intervals are the logical issue and
# collect intervals.  Prefer them over their nested legacy wave helpers;
# otherwise the RMSNorm and its FIFO/backpressure waits disappear from the
# overlap trace.  A legacy banked or non-banked image has no wrappers and
# falls back to the old process names.
proc quantized_projection_stage_patterns {stage} {
    switch -- $stage {
        drive_quantized_projection_wave_range {
            return {
                drive_quantized_projection_with_lookahead
                drive_quantized_projection_wave_range_banked
                drive_quantized_projection_wave_range
            }
        }
        collect_quantized_projection_wave_range {
            return {
                collect_quantized_projection_with_lookahead
                collect_quantized_projection_wave_range
            }
        }
        default {
            return [list $stage]
        }
    }
}

proc quantized_projection_stage_objects {controller_root stage} {
    set paths [get_objects -quiet -r ap_idle]
    foreach pattern [quantized_projection_stage_patterns $stage] {
        set matches {}
        foreach path $paths {
            if {[string first "${controller_root}/" $path] != 0} { continue }
            set instance [file tail [file dirname $path]]
            if {[string first $pattern $instance] < 0} { continue }
            if {[string match {*_Pipeline_*} $instance]} { continue }
            lappend matches $path
        }
        # A topology must be represented by one semantic process family. Do
        # not mix the new wrapper and its nested legacy helper in one trace.
        if {[llength $matches]} {
            return [lsort -unique $matches]
        }
    }
    return {}
}
