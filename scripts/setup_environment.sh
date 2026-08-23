#!/usr/bin/env bash

# Source this file from the repository root:
#   source scripts/setup_environment.sh
#
# It intentionally does not change shell options in the caller.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "setup_environment.sh must be sourced so its exports remain active:" >&2
    echo "  source scripts/setup_environment.sh" >&2
    exit 2
fi

llm_accel_setup_environment() {
    local requested_version="${LLM_ACCEL_VITIS_VERSION:-2022.2}"
    local env_script="${VITIS_ENV_SCRIPT:-}"
    local candidate=""
    local inferred_root=""

    if [ -n "${env_script}" ] && [ ! -r "${env_script}" ]; then
        echo "LLM-ACCEL environment: VITIS_ENV_SCRIPT is not readable: ${env_script}" >&2
        return 66
    fi

    if [ -z "${env_script}" ] && command -v v++ >/dev/null 2>&1; then
        inferred_root="$(dirname "$(dirname "$(readlink -f "$(command -v v++)")")")"
        if [[ "${inferred_root}" == *"/${requested_version}" ]] &&
           [ -r "${inferred_root}/settings64.sh" ]; then
            env_script="${inferred_root}/settings64.sh"
        fi
    fi

    if [ -z "${env_script}" ]; then
        for candidate in \
            "${LLM_ACCEL_VITIS_ROOT:-}/settings64.sh" \
            "/opt/Xilinx/Vitis/${requested_version}/settings64.sh" \
            "/tools/Xilinx/Vitis/${requested_version}/settings64.sh" \
            "/tools/xilinx/Vitis/${requested_version}/settings64.sh"
        do
            if [ "${candidate}" != "/settings64.sh" ] && [ -r "${candidate}" ]; then
                env_script="${candidate}"
                break
            fi
        done
    fi

    if [ -z "${env_script}" ]; then
        echo "LLM-ACCEL environment: Vitis ${requested_version} settings64.sh was not found." >&2
        echo "Set VITIS_ENV_SCRIPT to the installed environment script." >&2
        return 66
    fi

    export VITIS_ENV_SCRIPT="${env_script}"
    # Vendor setup output is omitted so experiment logs start with resolved
    # identities rather than installation banners.
    source "${VITIS_ENV_SCRIPT}" >/dev/null 2>&1

    local xrt_root="${XILINX_XRT:-}"
    if [ -z "${xrt_root}" ] && [ -r /opt/xilinx/xrt/setup.sh ]; then
        xrt_root=/opt/xilinx/xrt
    fi
    if [ -n "${xrt_root}" ] && [ -r "${xrt_root}/setup.sh" ]; then
        source "${xrt_root}/setup.sh" >/dev/null 2>&1
        export XILINX_XRT="${xrt_root}"
    fi

    export DEVICE="${DEVICE:-xilinx_u50_gen3x16_xdma_5_202210_1}"
    if [ -n "${XPLATFORM:-}" ] && [ ! -r "${XPLATFORM}" ]; then
        echo "LLM-ACCEL environment: XPLATFORM is not readable: ${XPLATFORM}" >&2
        return 66
    fi

    if [ -z "${XPLATFORM:-}" ]; then
        local platform_root=""
        local platform_candidate=""
        for platform_root in \
            "${XILINX_VITIS:-}/platforms" \
            /opt/xilinx/platforms \
            /tools/Xilinx/Vitis/${requested_version}/platforms
        do
            platform_candidate="${platform_root}/${DEVICE}/${DEVICE}.xpfm"
            if [ -r "${platform_candidate}" ]; then
                export XPLATFORM="${platform_candidate}"
                break
            fi
        done
    fi

    if [ "${LLM_ACCEL_ENV_QUIET:-0}" != "1" ]; then
        echo "LLM-ACCEL environment ready"
        echo "  VITIS_ENV_SCRIPT=${VITIS_ENV_SCRIPT}"
        echo "  XILINX_VITIS=${XILINX_VITIS:-unresolved}"
        echo "  XILINX_HLS=${XILINX_HLS:-unresolved}"
        echo "  XILINX_XRT=${XILINX_XRT:-unresolved}"
        echo "  DEVICE=${DEVICE}"
        echo "  XPLATFORM=${XPLATFORM:-unresolved}"
    fi
}

llm_accel_setup_environment
unset -f llm_accel_setup_environment
