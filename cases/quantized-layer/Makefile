SHELL := /usr/bin/env bash

CASE_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PRECISION ?= w4
PHASE ?= all
PROFILE ?= $(or $(QUANTIZED_LAYER_PROFILE),integrated)
SILU_LANES ?= $(or $(QUANTIZED_LAYER_SILU_LANES),4)
RMS_LANES ?= $(or $(QUANTIZED_LAYER_RMS_LANES),2)
WEIGHT_READ_OUTSTANDING ?= $(or $(QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING),32)
ATTENTION_WAVE ?= $(or $(QUANTIZED_LAYER_ATTENTION_WAVE),0)
PREFILL_FFN_OVERLAP ?= $(or $(QUANTIZED_LAYER_PREFILL_FFN_OVERLAP),1)
COSIM_STAGE ?= all

PROFILE_ENV = \
    QUANTIZED_LAYER_PROFILE='$(PROFILE)' \
    QUANTIZED_LAYER_SILU_LANES='$(SILU_LANES)' \
    QUANTIZED_LAYER_RMS_LANES='$(RMS_LANES)' \
    QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING='$(WEIGHT_READ_OUTSTANDING)' \
    QUANTIZED_LAYER_ATTENTION_WAVE='$(ATTENTION_WAVE)' \
    QUANTIZED_LAYER_PREFILL_FFN_OVERLAP='$(PREFILL_FFN_OVERLAP)'

.PHONY: help check dry-run test-native build host cosim hwemu reference performance

help:
	@printf '%s\n' \
	  'make check' \
	  'make dry-run PRECISION=w4|w8 [PHASE=all|controller-xo|compute-xo|link|host|emconfig|run]' \
	  'make test-native' \
	  'make build PRECISION=w4|w8 PHASE=all' \
	  'make host PRECISION=w4|w8' \
	  'make cosim PRECISION=w4|w8 COSIM_STAGE=prepare|cosim|all' \
	  'make hwemu PRECISION=w4|w8' \
	  'make reference'

check:
	@$(CASE_ROOT)/scripts/cowave_quantized_layer.sh check

dry-run:
	@$(PROFILE_ENV) $(CASE_ROOT)/scripts/cowave_quantized_layer.sh dry-run '$(PRECISION)' '$(PHASE)'

test-native:
	@$(CASE_ROOT)/scripts/cowave_quantized_layer.sh test-native

build:
	@$(PROFILE_ENV) $(CASE_ROOT)/scripts/cowave_quantized_layer.sh build '$(PRECISION)' '$(PHASE)'

host:
	@$(PROFILE_ENV) $(CASE_ROOT)/scripts/cowave_quantized_layer.sh host '$(PRECISION)'

cosim:
	@$(PROFILE_ENV) $(CASE_ROOT)/scripts/cowave_quantized_layer.sh cosim '$(PRECISION)' '$(COSIM_STAGE)'

hwemu:
	@$(PROFILE_ENV) $(CASE_ROOT)/scripts/cowave_quantized_layer.sh hwemu '$(PRECISION)'

reference:
	@$(PROFILE_ENV) $(CASE_ROOT)/scripts/cowave_quantized_layer.sh reference

performance:
	@$(PROFILE_ENV) $(CASE_ROOT)/scripts/cowave_quantized_layer.sh performance $(ARGS)
