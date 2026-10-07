# Vitis 2022.2 calls USER_PRE_SIM_SCRIPT before preprocess_profile.tcl.
# Register generated profiling signals before our clock probe advances time.
# The generated, standalone get_value_database calls discard their results.
# They can crash XSim's native WDB reader before any events are committed,
# even at time zero. Retain all registrations, but remove only these verified
# unused warmup reads from this run's generated copy. The normal tool hook
# then repeats registrations; post-simulation profiling remains unchanged.
proc quantized_prepare_profile_registration {} {
    set path preprocess_profile.tcl
    set input [open $path r]
    set text [read $input]
    close $input
    set registrations {}
    set pending ""
    set discarded 0
    foreach line [split $text \n] {
        set line [string trim $line]
        if {$line eq "" || [string index $line 0] eq "#"} { continue }
        if {[regexp {^log_wave -quiet (/[A-Za-z0-9_/]+)$} $line -> signal]} {
            # FIFO-pointer registrations have no following warmup read.
            set pending $signal
            lappend registrations [list log_wave -quiet $signal]
        } elseif {[regexp {^get_value_database (/[A-Za-z0-9_/]+) -time 0 -quiet$} $line -> signal]} {
            if {$pending ne $signal} { error "Unpaired generated warmup query: $line" }
            set pending ""
            incr discarded
        } else {
            error "Unrecognized generated profiling command: $line"
        }
    }
    if {[llength $registrations] == 0} { error "Missing generated profiling registrations" }
    set backup preprocess_profile.original.tcl
    if {[file exists $backup]} { error "Profiling backup already exists: $backup" }
    # Validate the entire input before replacing anything. Preserve the exact
    # generated input and publish the registration-only copy atomically.
    file copy $path $backup
    set temporary ${path}.pending.[pid]
    set output [open $temporary {WRONLY CREAT EXCL}]
    puts $output "# Registration-only startup; original retained in $backup"
    foreach command $registrations { puts $output $command }
    close $output
    file rename -force $temporary $path
    puts "QUANTIZED_PROFILE_REGISTRATION_ONLY signals=[llength $registrations] discarded_warmup_reads=$discarded backup=$backup"
}
if {[file exists preprocess_profile.tcl]} {
    if {[lindex [now] 0] != 0} {
        puts stderr "QUANTIZED_TRACE_RECORDING_FAILED profile initialization must precede time advance"
        exit 6
    }
    if {[catch {
        quantized_prepare_profile_registration
        source -notrace preprocess_profile.tcl
    } profile_error]} {
        puts stderr "QUANTIZED_TRACE_RECORDING_FAILED profile initialization: $profile_error"
        exit 6
    }
    puts "QUANTIZED_PROFILE_INIT_READY time=[now]"
}
