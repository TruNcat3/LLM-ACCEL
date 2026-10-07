# Shared Tcl profile resolver for quantized full-layer HLS builds.

proc quantized_layer_profile_env {name fallback} {
    if {[info exists ::env($name)] && $::env($name) ne ""} {
        return $::env($name)
    }
    return $fallback
}

proc quantized_layer_profile_optional_env {name requested_name} {
    set value [quantized_layer_profile_env $name ""]
    if {[info exists ::env($requested_name)] && $::env($requested_name) eq "0"} {
        return ""
    }
    return $value
}

proc quantized_layer_profile_resolve {precision} {
    if {$precision ni {w4 w8}} { error "precision must be w4 or w8" }
    set requested_profile [quantized_layer_profile_env QUANTIZED_LAYER_PROFILE ""]
    set requested_variant [quantized_layer_profile_env QUANTIZED_LAYER_ATTENTION_VARIANT ""]
    set requested_silu_lanes [quantized_layer_profile_env QUANTIZED_LAYER_SILU_LANES 1]
    set requested_rms_lanes [quantized_layer_profile_env QUANTIZED_LAYER_RMS_LANES 1]
    set requested_weight_outstanding [quantized_layer_profile_optional_env \
        QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING_REQUESTED]
    set requested_attention_wave [quantized_layer_profile_optional_env \
        QUANTIZED_LAYER_ATTENTION_WAVE QUANTIZED_LAYER_ATTENTION_WAVE_REQUESTED]
    set requested_prefill_ffn_overlap [quantized_layer_profile_optional_env \
        QUANTIZED_LAYER_PREFILL_FFN_OVERLAP QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED]
    if {$requested_silu_lanes ni {1 2 4}} {
        error "QUANTIZED_LAYER_SILU_LANES must be 1, 2, or 4"
    }
    if {[info exists ::env(CU_NL_LANES_CONFIG)] &&
        $::env(CU_NL_LANES_CONFIG) ne "" &&
        $::env(CU_NL_LANES_CONFIG) ne $requested_silu_lanes} {
        error "CU_NL_LANES_CONFIG=$::env(CU_NL_LANES_CONFIG) conflicts with QUANTIZED_LAYER_SILU_LANES=$requested_silu_lanes"
    }
    if {$requested_rms_lanes ni {1 2 4}} {
        error "QUANTIZED_LAYER_RMS_LANES must be 1, 2, or 4"
    }
    if {[info exists ::env(CU_RMS_LANES_CONFIG)] &&
        $::env(CU_RMS_LANES_CONFIG) ne "" &&
        $::env(CU_RMS_LANES_CONFIG) ne $requested_rms_lanes} {
        error "CU_RMS_LANES_CONFIG=$::env(CU_RMS_LANES_CONFIG) conflicts with QUANTIZED_LAYER_RMS_LANES=$requested_rms_lanes"
    }
    if {$requested_variant ni {"" baseline combined4}} {
        error "QUANTIZED_LAYER_ATTENTION_VARIANT must be baseline or combined4"
    }
    if {$requested_profile eq ""} {
        set profile legacy
        if {$requested_variant eq "combined4"} { set profile attention }
    } else {
        set profile $requested_profile
    }
    if {$profile ni {legacy attention attention_wave pipeline integrated}} {
        error "QUANTIZED_LAYER_PROFILE must be legacy, attention, attention_wave, pipeline, or integrated"
    }
    set variant baseline
    if {$profile ne "legacy"} { set variant combined4 }
    if {$requested_variant ne "" && $requested_variant ne $variant} {
        error "QUANTIZED_LAYER_PROFILE=$profile conflicts with QUANTIZED_LAYER_ATTENTION_VARIANT=$requested_variant"
    }
    if {$requested_weight_outstanding ne "" &&
        $requested_weight_outstanding ni {16 32}} {
        error "QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING must be 16 or 32"
    }
    if {$requested_attention_wave ne "" &&
        $requested_attention_wave ni {0 1}} {
        error "QUANTIZED_LAYER_ATTENTION_WAVE must be 0 or 1"
    }
    if {$requested_attention_wave ne "" && $profile eq "legacy"} {
        error "QUANTIZED_LAYER_ATTENTION_WAVE override is only valid with attention, attention_wave, pipeline, or integrated"
    }
    if {$requested_prefill_ffn_overlap ne "" &&
        $requested_prefill_ffn_overlap ni {0 1}} {
        error "QUANTIZED_LAYER_PREFILL_FFN_OVERLAP must be 0 or 1"
    }
    if {$requested_prefill_ffn_overlap eq "1" &&
        $profile ni {pipeline integrated}} {
        error "QUANTIZED_LAYER_PREFILL_FFN_OVERLAP=1 is only valid with pipeline or integrated profile"
    }

    set values [dict create \
        QUANTIZED_ATTENTION_VALID_TAIL 0 \
        QUANTIZED_ATTENTION_TILE_PREFETCH 0 \
        QUANTIZED_ATTENTION_PROBABILITY_FUSE 0 \
        QUANTIZED_ATTENTION_PACK_LANES 1 \
        QUANTIZED_ATTENTION_ACTIVE_STATE 0 \
        QUANTIZED_ATTENTION_HEAD_LANES 1 \
        QUANTIZED_ATTENTION_WAVE_PIPELINE 0 \
        QUANTIZED_BLOCK_PIPELINE 0 \
        QUANTIZED_ASYNC_COMPUTE 0 \
        QUANTIZED_LAYER_INTEGRATED_DECODE 0 \
        QUANTIZED_DECODE_ROWS_MERGED 0 \
        QDR_COMPACT 0 \
        QDR_INGRESS_BITS 4096 \
        QDR_SHARED_STATE 0 \
        QDR_DIRECT_WIDE 0 \
        QUANTIZED_DECODE_FFN_OVERLAP 0 \
        QUANTIZED_PREFILL_FFN_OVERLAP 0 \
        QUANTIZED_DECODE_SCALE_PREFETCH 0 \
        QDR_BLOCK_OPS 0 \
        QUANTIZED_DECODE_ROWS_ADAPTER_ACTIVATION_FIFO_DEPTH 2 \
        QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING 16 \
        QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST 16 \
        CU_NL_LANES_CONFIG $requested_silu_lanes \
        CU_RMS_LANES_CONFIG $requested_rms_lanes]
    if {$precision eq "w8"} { dict set values QUANTIZED_ALIGNMENT_W8 1 }
    if {$profile ne "legacy"} {
        foreach {name value} {
            QUANTIZED_ATTENTION_VALID_TAIL 1
            QUANTIZED_ATTENTION_TILE_PREFETCH 1
            QUANTIZED_ATTENTION_PROBABILITY_FUSE 1
            QUANTIZED_ATTENTION_PACK_LANES 4
            QUANTIZED_ATTENTION_ACTIVE_STATE 1
            QUANTIZED_ATTENTION_HEAD_LANES 4
            QUANTIZED_ATTENTION_WAVE_PIPELINE 0
        } { dict set values $name $value }
    }
    if {$profile eq "attention_wave"} {
        dict set values QUANTIZED_ATTENTION_WAVE_PIPELINE 1
    }
    if {$profile in {pipeline integrated}} {
        dict set values QUANTIZED_BLOCK_PIPELINE 1
        dict set values QUANTIZED_ASYNC_COMPUTE 1
    }
    if {$profile eq "integrated"} {
        foreach {name value} {
            AP_INT_MAX_W 4096
            QUANTIZED_LAYER_INTEGRATED_DECODE 1
            QUANTIZED_DECODE_ROWS_MERGED 1
            QDR_COMPACT 1
            QDR_INGRESS_BITS 4096
            QDR_SHARED_STATE 1
            QDR_DIRECT_WIDE 1
            QUANTIZED_DECODE_FFN_OVERLAP 1
            QUANTIZED_DECODE_SCALE_PREFETCH 1
            QUANTIZED_DECODE_ROWS_ADAPTER_ACTIVATION_FIFO_DEPTH 8
            QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING 32
            QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST 16
        } { dict set values $name $value }
        dict set values QDR_BLOCK_OPS [expr {$precision eq "w4" ? 1 : 0}]
    }
    if {$requested_weight_outstanding ne ""} {
        dict set values QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING $requested_weight_outstanding
    }
    if {$requested_attention_wave ne ""} {
        dict set values QUANTIZED_ATTENTION_WAVE_PIPELINE $requested_attention_wave
    }
    if {$requested_prefill_ffn_overlap ne ""} {
        dict set values QUANTIZED_PREFILL_FFN_OVERLAP $requested_prefill_ffn_overlap
    }

    # Reject a direct Tcl invocation that supplies a conflicting raw macro.
    # Shell callers may set the variables through the shared resolver; those
    # values are identical and therefore pass this check.
    foreach name [dict keys $values] {
        if {[info exists ::env($name)] && $::env($name) ne "" &&
            $::env($name) ne [dict get $values $name]} {
            error "$name=$::env($name) conflicts with profile value [dict get $values $name]"
        }
    }
    return [dict create profile $profile variant $variant values $values]
}

proc quantized_layer_profile_cflags {precision} {
    set resolved [quantized_layer_profile_resolve $precision]
    set flags {}
    foreach name [dict keys [dict get $resolved values]] {
        lappend flags "-D${name}=[dict get [dict get $resolved values] $name]"
    }
    return [join $flags " "]
}

proc quantized_layer_profile_name {precision} {
    return [dict get [quantized_layer_profile_resolve $precision] profile]
}

proc quantized_layer_profile_values {precision} {
    return [dict get [quantized_layer_profile_resolve $precision] values]
}

proc quantized_layer_verify_weight_axi {repo_root project_name precision} {
    set values [quantized_layer_profile_values $precision]
    set rtl [file join $project_name solution1 syn verilog control_cache_quantized_${precision}_layer.v]
    puts [exec python3 [file join $repo_root scripts check_quantized_weight_axi.py] \
        $rtl $precision \
        [dict get $values QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING] \
        [dict get $values QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST]]
}
