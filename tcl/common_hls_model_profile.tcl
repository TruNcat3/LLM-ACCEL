proc llm_fpga_model_profile {} {
    if {[info exists ::env(LLM_FPGA_MODEL_PROFILE)]} {
        set profile $::env(LLM_FPGA_MODEL_PROFILE)
    } else {
        set profile small
    }

    switch -- $profile {
        small - medium - weight-chunk - qwen-layer - qwen2.5-3b {
            return $profile
        }
        default {
            error "Unsupported LLM_FPGA_MODEL_PROFILE=$profile"
        }
    }
}

proc llm_fpga_model_tag {} {
    switch -- [llm_fpga_model_profile] {
        small { return small }
        medium { return medium }
        weight-chunk { return weight_chunk }
        qwen-layer { return qwen_layer }
        qwen2.5-3b { return qwen2_5_3b }
    }
}

proc llm_fpga_model_cflags {} {
    switch -- [llm_fpga_model_profile] {
        small { return " -DQWEN_TEST_SMALL" }
        medium { return " -DQWEN_TEST_COSIM_MEDIUM" }
        weight-chunk { return " -DQWEN_TEST_WEIGHT_CHUNK" }
        qwen-layer { return " -DQWEN_TEST_QWEN_LAYER" }
        qwen2.5-3b { return "" }
    }
}

proc llm_fpga_profiled_project_name {base_name} {
    if {[llm_fpga_model_profile] eq "small"} {
        return $base_name
    }
    return "${base_name}_[llm_fpga_model_tag]"
}

proc llm_fpga_profiled_xo_dir {} {
    if {[info exists ::env(LLM_FPGA_XO_DIR)]} {
        return [file normalize $::env(LLM_FPGA_XO_DIR)]
    }
    if {[llm_fpga_model_profile] eq "small"} {
        return [file normalize vitis_8x64/xo]
    }
    return [file normalize "vitis_8x64/xo.[llm_fpga_model_tag]"]
}
