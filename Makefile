TARGET ?= sw_emu
FREQUENCY ?= 300
THREADS ?= 16
OPT_LEVEL ?= 3
REPORT_LEVEL ?= 2
VITIS_8X64_MODEL_PROFILE ?= small
VITIS_8X64_VALID_MODEL_PROFILES := small medium qwen-layer qwen-layer-long qwen2.5-3b

ifeq ($(filter $(VITIS_8X64_MODEL_PROFILE),$(VITIS_8X64_VALID_MODEL_PROFILES)),)
$(error Unsupported VITIS_8X64_MODEL_PROFILE=$(VITIS_8X64_MODEL_PROFILE); expected one of $(VITIS_8X64_VALID_MODEL_PROFILES))
endif

DETECTED_VITIS := $(shell if command -v v++ >/dev/null 2>&1; then dirname $$(dirname $$(readlink -f $$(command -v v++))); else echo /tools/Xilinx/Vitis/2022.2; fi)
DETECTED_HLS := $(shell if command -v vitis_hls >/dev/null 2>&1; then dirname $$(dirname $$(readlink -f $$(command -v vitis_hls))); else echo /tools/Xilinx/Vitis_HLS/2022.2; fi)

XILINX_XRT ?= /opt/xilinx/xrt
XILINX_VITIS ?= $(DETECTED_VITIS)
XILINX_HLS ?= $(DETECTED_HLS)
DEVICE ?= xilinx_u50_gen3x16_xdma_5_202210_1

CUR_DIR := $(abspath .)
XPLATFORM := $(firstword \
	$(wildcard $(DEVICE)) \
	$(wildcard $(XILINX_VITIS)/platforms/$(DEVICE)/$(DEVICE).xpfm) \
	$(wildcard /opt/xilinx/platforms/$(DEVICE)/$(DEVICE).xpfm))

ifeq ($(XPLATFORM),)
$(warning $(DEVICE) was not found in the default paths; v++ will use the device name directly)
XPLATFORM := $(DEVICE)
endif

XDEVICE := $(notdir $(basename $(XPLATFORM)))
PROFILE_TAG := $(subst .,_,$(subst -,_,$(VITIS_8X64_MODEL_PROFILE)))
PROFILE_SUFFIX := $(if $(filter small,$(VITIS_8X64_MODEL_PROFILE)),,.$(PROFILE_TAG))
HLS_PROJECT_SUFFIX := $(if $(filter small,$(VITIS_8X64_MODEL_PROFILE)),,_$(PROFILE_TAG))
XO_DIR := vitis_8x64/xo$(PROFILE_SUFFIX)
# The specialized resident-layer build harness predates this publication
# Makefile and passes VITIS_8X64_BUILD_DIR.  Keep that interface as a fallback
# while allowing an explicit BUILD_DIR to remain the canonical override.
BUILD_DIR ?= $(if $(strip $(VITIS_8X64_BUILD_DIR)),$(VITIS_8X64_BUILD_DIR),vitis_8x64/build$(PROFILE_SUFFIX).$(TARGET).$(XDEVICE))
TEMP_DIR := vitis_8x64/_x$(PROFILE_SUFFIX).$(TARGET).$(XDEVICE)
REPORT_DIR := reports/vitis_8x64/$(PROFILE_TAG)/$(TARGET).$(XDEVICE)

COMPUTE_XO := $(XO_DIR)/compute_core_8x64_unified_nk.xo
CONTROL_XO := $(XO_DIR)/control_cache_8x64_dual_core_nk.xo
STATUS_XO := $(XO_DIR)/cc8_status_sink_nk.xo
XCLBIN := $(BUILD_DIR)/qwen_8x64_dual.xclbin
CONN_CFG ?= $(if $(filter qwen2.5-3b,$(VITIS_8X64_MODEL_PROFILE)),conn_u50_8x64_dual_full_resident.cfg,conn_u50_8x64_dual.cfg)

SMOKE_HOST := $(BUILD_DIR)/host_8x64.exe
QWEN_HOST := $(BUILD_DIR)/host_qwen_8x64.exe
EMCONFIG := $(BUILD_DIR)/emconfig.json

CXX := g++
CXXFLAGS += -std=c++14 -O3 -Wall -Wno-unknown-pragmas
CXXFLAGS += -Iinclude -Icommon/include
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_HLS)/include
LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -lpthread -lrt -ldl

RUN_TIMEOUT ?= 1800
QWEN_ARGS ?= --mode run --profile small --tokens 0 --zero-model
RANDOM_ARGS ?= --mode verify-random --profile small --seed 20260701

ifneq ($(filter $(TARGET),sw_emu hw_emu),)
RUN_ENV := XCL_EMULATION_MODE=$(TARGET)
RUN_EXTRA_DEPS := $(EMCONFIG)
else
RUN_ENV :=
RUN_EXTRA_DEPS :=
endif

.PHONY: help
.PHONY: hls_csim_compute hls_csynth_compute hls_cosim_compute
.PHONY: hls_csim_control hls_csynth_control hls_cosim_control
.PHONY: hls_csim_nk test_resident_attention_q214 test_q214_payload_golden test_qwen3b_e2e_plan test_qwen3b_e2e_launcher_contract test_coarse_task_residency_contract test_e2e_progress_contract test_e2e_performance_semantics test_result_installer_contract test_qwen3b_source_snapshot test_publication_tree test_publication_release verify_result_checksums regenerate_root_checksums verify_q214_pd_release verify_q214_resident_release hls_csim_closed_loop_8x64_resident_layer hls_cosim_closed_loop_8x64_resident_layer
.PHONY: hls_csim_closed_loop_8x64_composed_layer hls_cosim_closed_loop_8x64_composed_layer
.PHONY: hls_csim_closed_loop_8x64_resident_prefill_block hls_cosim_closed_loop_8x64_resident_prefill_block
.PHONY: hls_csynth_compute_xo hls_csynth_control_xo
.PHONY: hls_csynth_status_xo hls_csynth_xo
.PHONY: vitis_8x64_xo vitis_8x64_link vitis_8x64_hosts vitis_8x64_host vitis_8x64_qwen_host vitis_8x64_emconfig
.PHONY: vitis_8x64_run_smoke vitis_8x64_run_qwen vitis_8x64_run_random
.PHONY: clean

help:
	@echo "Dual-8x64 Qwen accelerator build entry points"
	@echo "  make hls_csim_compute"
	@echo "  make hls_csynth_compute"
	@echo "  make hls_cosim_compute"
	@echo "  make hls_csim_control"
	@echo "  make hls_csynth_control"
	@echo "  make hls_cosim_control"
	@echo "  make hls_csim_nk"
	@echo "  make test_resident_attention_q214"
	@echo "  make test_q214_payload_golden"
	@echo "  make test_qwen3b_e2e_plan"
	@echo "  make test_qwen3b_e2e_launcher_contract"
	@echo "  make test_coarse_task_residency_contract"
	@echo "  make test_e2e_progress_contract"
	@echo "  make test_e2e_performance_semantics"
	@echo "  make test_result_installer_contract"
	@echo "  make test_qwen3b_source_snapshot"
	@echo "  make test_publication_tree"
	@echo "  make test_publication_release  # all non-simulator release gates"
	@echo "  make verify_result_checksums"
	@echo "  make regenerate_root_checksums  # after git add -A"
	@echo "  make verify_q214_pd_release"
	@echo "  make verify_q214_resident_release"
	@echo "  make hls_cosim_closed_loop_8x64_resident_layer"
	@echo "  make hls_csim_closed_loop_8x64_composed_layer"
	@echo "  make hls_cosim_closed_loop_8x64_composed_layer"
	@echo "  make hls_csim_closed_loop_8x64_resident_prefill_block"
	@echo "  make hls_cosim_closed_loop_8x64_resident_prefill_block"
	@echo "  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-composed"
	@echo "  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-block"
	@echo "  scripts/build_vitis_8x64_resident_layer_hwemu.sh run-stack"
	@echo "  make vitis_8x64_xo"
	@echo "  make vitis_8x64_link TARGET=sw_emu|hw_emu|hw"
	@echo "  make vitis_8x64_run_smoke TARGET=sw_emu|hw_emu|hw"
	@echo "  make vitis_8x64_run_qwen TARGET=sw_emu|hw_emu|hw"
	@echo "  make vitis_8x64_run_random TARGET=sw_emu|hw_emu|hw"

hls_csim_compute:
	HLS_CSIM_ONLY=1 scripts/run_vitis_hls.sh tcl/run_cosim_compute_core_8x64_unified.tcl

hls_csynth_compute:
	HLS_COSIM_PREPARE=1 HLS_COSIM_PREPARE_ONLY=1 HLS_COSIM_SKIP_CSIM=1 \
		scripts/run_vitis_hls.sh tcl/run_cosim_compute_core_8x64_unified.tcl

hls_cosim_compute:
	scripts/run_vitis_hls.sh tcl/run_cosim_compute_core_8x64_unified.tcl

hls_csim_control:
	HLS_CSIM_ONLY=1 scripts/run_vitis_hls.sh tcl/run_cosim_control_cache_8x64_dual_core.tcl

hls_csynth_control:
	HLS_COSIM_PREPARE=1 HLS_COSIM_PREPARE_ONLY=1 HLS_COSIM_SKIP_CSIM=1 \
		scripts/run_vitis_hls.sh tcl/run_cosim_control_cache_8x64_dual_core.tcl

hls_cosim_control:
	scripts/run_vitis_hls.sh tcl/run_cosim_control_cache_8x64_dual_core.tcl

hls_csim_nk:
	scripts/run_vitis_hls.sh tcl/run_csim_compute_core_8x64_nk.tcl
	scripts/run_vitis_hls.sh tcl/run_csim_control_cache_8x64_nk.tcl

test_resident_attention_q214:
	$(CXX) -std=c++14 -O0 -Wno-unknown-pragmas \
		-Iinclude -I$(XILINX_HLS)/include \
		tests/resident_probability_buffer_tb.cpp kernel/mm_controller.cpp \
		-o /tmp/llm_accel_resident_probability_buffer_tb
	/tmp/llm_accel_resident_probability_buffer_tb

test_q214_payload_golden:
	scripts/test_q214_payload_golden.sh

test_qwen3b_e2e_plan:
	scripts/test_qwen3b_e2e_plan.sh

test_qwen3b_e2e_launcher_contract:
	tests/test_qwen3b_e2e_launcher_contract.sh

test_coarse_task_residency_contract:
	tests/test_coarse_task_residency_contract.sh

test_e2e_progress_contract:
	tests/test_e2e_progress_contract.sh

test_e2e_performance_semantics:
	tests/test_e2e_performance_semantics.sh

test_result_installer_contract:
	tests/test_result_installer_contract.sh

test_qwen3b_source_snapshot:
	tests/test_qwen3b_source_snapshot.sh

test_publication_tree:
	tests/test_publication_tree.sh

test_publication_release: test_qwen3b_e2e_plan \
		test_qwen3b_e2e_launcher_contract \
		test_coarse_task_residency_contract \
		test_e2e_progress_contract \
		test_e2e_performance_semantics \
		test_result_installer_contract \
		test_qwen3b_source_snapshot \
		verify_q214_pd_release \
		verify_q214_resident_release \
		test_publication_tree
	@echo "PUBLICATION RELEASE GATES PASS"

verify_result_checksums:
	scripts/verify_result_checksums.sh

regenerate_root_checksums:
	scripts/regenerate_root_checksums.sh

verify_q214_pd_release:
	scripts/verify_q214_pd_release.sh

verify_q214_resident_release:
	scripts/verify_q214_resident_release.sh

hls_csim_closed_loop_8x64_resident_layer:
	HLS_CSIM_ONLY=1 scripts/run_vitis_hls.sh tcl/run_cosim_closed_loop_8x64_resident_layer.tcl

hls_cosim_closed_loop_8x64_resident_layer:
	scripts/run_vitis_hls.sh tcl/run_cosim_closed_loop_8x64_resident_layer.tcl

hls_csim_closed_loop_8x64_composed_layer:
	HLS_CSIM_ONLY=1 scripts/run_vitis_hls.sh tcl/run_cosim_closed_loop_8x64_composed_layer.tcl

hls_cosim_closed_loop_8x64_composed_layer:
	scripts/run_vitis_hls.sh tcl/run_cosim_closed_loop_8x64_composed_layer.tcl

hls_csim_closed_loop_8x64_resident_prefill_block:
	HLS_CSIM_ONLY=1 scripts/run_vitis_hls.sh tcl/run_cosim_closed_loop_8x64_resident_prefill_block.tcl

hls_cosim_closed_loop_8x64_resident_prefill_block:
	scripts/run_vitis_hls.sh tcl/run_cosim_closed_loop_8x64_resident_prefill_block.tcl

$(COMPUTE_XO): \
	kernel/mm_stream_8x64_fused_mac.cpp \
	kernel/compute_stream.cpp \
	kernel/compute_core_8x64_unified.cpp \
	kernel/compute_core_8x64_nk.cpp \
	include/compute_core_8x64_unified.hpp \
	include/vitis_stream_8x64.hpp \
	include/stream_depth_config.hpp \
	tcl/common_hls_depth_config.tcl \
	tcl/common_hls_model_profile.tcl
	LLM_FPGA_MODEL_PROFILE='$(VITIS_8X64_MODEL_PROFILE)' \
	LLM_FPGA_XO_DIR='$(abspath $(XO_DIR))' \
	LLM_FPGA_HLS_PROJECT_NAME='qwen_hls_compute_core_8x64_nk_prj$(HLS_PROJECT_SUFFIX)' \
		scripts/run_vitis_hls.sh tcl/build_compute_core_8x64_nk_xo.tcl

$(CONTROL_XO): \
	kernel/mm_controller.cpp \
	kernel/control_cache_8x64.cpp \
	kernel/control_cache_8x64_nk.cpp \
	include/control_cache_8x64.hpp \
	include/vitis_stream_8x64.hpp \
	include/stream_depth_config.hpp \
	include/weight_pipeline_config.hpp \
	tcl/common_hls_depth_config.tcl \
	tcl/common_hls_model_profile.tcl
	LLM_FPGA_MODEL_PROFILE='$(VITIS_8X64_MODEL_PROFILE)' \
	LLM_FPGA_XO_DIR='$(abspath $(XO_DIR))' \
	LLM_FPGA_HLS_PROJECT_NAME='qwen_hls_control_cache_8x64_nk_prj$(HLS_PROJECT_SUFFIX)' \
		scripts/run_vitis_hls.sh tcl/build_control_cache_8x64_nk_xo.tcl

$(STATUS_XO): \
	kernel/cc8_status_sink.cpp \
	include/control_cache_8x64.hpp \
	include/vitis_stream_8x64.hpp \
	tcl/common_hls_model_profile.tcl
	LLM_FPGA_MODEL_PROFILE='$(VITIS_8X64_MODEL_PROFILE)' \
	LLM_FPGA_XO_DIR='$(abspath $(XO_DIR))' \
	LLM_FPGA_HLS_PROJECT_NAME='qwen_hls_cc8_status_sink_nk_prj$(HLS_PROJECT_SUFFIX)' \
		scripts/run_vitis_hls.sh tcl/build_cc8_status_sink_nk_xo.tcl

hls_csynth_compute_xo: $(COMPUTE_XO)
hls_csynth_control_xo: $(CONTROL_XO)
hls_csynth_status_xo: $(STATUS_XO)
hls_csynth_xo: $(COMPUTE_XO) $(CONTROL_XO) $(STATUS_XO)
vitis_8x64_xo: hls_csynth_xo

$(XCLBIN): $(COMPUTE_XO) $(CONTROL_XO) $(STATUS_XO) $(CONN_CFG)
	mkdir -p $(BUILD_DIR) $(TEMP_DIR) $(REPORT_DIR)
	v++ -l -t $(TARGET) --platform $(XPLATFORM) --save-temps \
		--optimize $(OPT_LEVEL) --report_level $(REPORT_LEVEL) \
		-I$(CUR_DIR)/include \
		--kernel_frequency $(FREQUENCY) --config $(CONN_CFG) \
		--vivado.synth.jobs $(THREADS) --vivado.impl.jobs $(THREADS) \
		--temp_dir $(TEMP_DIR) --report_dir $(REPORT_DIR) \
		-o $@ $(CONTROL_XO) $(COMPUTE_XO) $(STATUS_XO)

vitis_8x64_link: $(XCLBIN)

$(SMOKE_HOST): common/include/xcl2.cpp common/include/xcl2.hpp host/host_8x64.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ common/include/xcl2.cpp host/host_8x64.cpp \
		$(CXXFLAGS) $(LDFLAGS)

$(QWEN_HOST): common/include/xcl2.cpp common/include/xcl2.hpp host/host_qwen_8x64.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ common/include/xcl2.cpp host/host_qwen_8x64.cpp \
		$(CXXFLAGS) '-DLLM_FPGA_DEVICE_PROFILE="$(VITIS_8X64_MODEL_PROFILE)"' $(LDFLAGS)

vitis_8x64_hosts: $(SMOKE_HOST) $(QWEN_HOST)
vitis_8x64_host: $(SMOKE_HOST)
vitis_8x64_qwen_host: $(QWEN_HOST)

$(EMCONFIG):
	mkdir -p $(BUILD_DIR)
	emconfigutil --platform $(XPLATFORM) --od $(BUILD_DIR)

vitis_8x64_emconfig: $(EMCONFIG)

vitis_8x64_run_smoke: $(XCLBIN) $(SMOKE_HOST) $(RUN_EXTRA_DEPS)
	cd $(BUILD_DIR) && $(RUN_ENV) timeout $(RUN_TIMEOUT) \
		./host_8x64.exe --xclbin ./qwen_8x64_dual.xclbin --case all

vitis_8x64_run_qwen: $(XCLBIN) $(QWEN_HOST) $(RUN_EXTRA_DEPS)
	cd $(BUILD_DIR) && $(RUN_ENV) timeout $(RUN_TIMEOUT) \
		./host_qwen_8x64.exe --xclbin ./qwen_8x64_dual.xclbin $(QWEN_ARGS)

vitis_8x64_run_random: $(XCLBIN) $(QWEN_HOST) $(RUN_EXTRA_DEPS)
	cd $(BUILD_DIR) && $(RUN_ENV) timeout $(RUN_TIMEOUT) \
		./host_qwen_8x64.exe --xclbin ./qwen_8x64_dual.xclbin $(RANDOM_ARGS)

clean:
	rm -rf \
		qwen_hls_cosim_compute_core_8x64_unified_prj \
		qwen_hls_cosim_compute_core_8x64_unified_csim_prj \
		qwen_hls_cosim_control_cache_8x64_dual_core_prj \
		qwen_hls_cosim_control_cache_8x64_dual_core_csim_prj \
		qwen_hls_compute_core_8x64_nk_csim_prj \
		qwen_hls_control_cache_8x64_nk_csim_prj \
		qwen_hls_closed_loop_resident_prefill_block_cosim_prj* \
		qwen_hls_compute_core_8x64_nk_prj \
		qwen_hls_control_cache_8x64_nk_prj \
		qwen_hls_cc8_status_sink_nk_prj \
		vitis_8x64 reports logs
