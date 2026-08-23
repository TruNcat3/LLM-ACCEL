#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-publication}"
case "${mode}" in
    publication|hls|hw-emu|board) ;;
    *)
        echo "usage: $0 [publication|hls|hw-emu|board]" >&2
        exit 2
        ;;
esac

failures=0
warnings=0

pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf 'FAIL  %s\n' "$*"; failures=$((failures + 1)); }

require_command() {
    local tool="$1"
    if command -v "${tool}" >/dev/null 2>&1; then
        pass "${tool}=$(readlink -f "$(command -v "${tool}")")"
    else
        fail "missing command: ${tool}"
    fi
}

require_file() {
    local label="$1"
    local path="$2"
    if [ -r "${path}" ]; then
        pass "${label}=${path}"
    else
        fail "missing ${label}: ${path}"
    fi
}

check_version() {
    local tool="$1"
    local expected="$2"
    local version_flag="$3"
    local output=""
    if ! command -v "${tool}" >/dev/null 2>&1; then
        return
    fi
    output="$(${tool} "${version_flag}" 2>&1 || true)"
    if grep -F -q "${expected}" <<<"${output}"; then
        pass "${tool} version contains ${expected}"
    else
        fail "${tool} is not the reference ${expected} release"
    fi
}

printf 'LLM-ACCEL environment preflight\n'
printf '  mode=%s\n' "${mode}"
printf '  repository=%s\n' "$PWD"

for tool in bash make g++ git awk sed perl sha256sum rg python3; do
    require_command "${tool}"
done

if [ -r /etc/os-release ]; then
    os_name="$(. /etc/os-release; printf '%s %s' "${ID:-unknown}" "${VERSION_ID:-unknown}")"
    if [ "${os_name}" = "ubuntu 20.04" ]; then
        pass "reference operating system: ${os_name}"
    else
        warn "reference evidence used Ubuntu 20.04; detected ${os_name}"
    fi
fi

case "${mode}" in
    hls|hw-emu)
        LLM_ACCEL_ENV_QUIET=1 source scripts/setup_environment.sh
        require_command vitis_hls
        require_command v++
        require_command vivado
        check_version vitis_hls 2022.2 -version
        check_version v++ 2022.2 --version
        check_version vivado 2022.2 -version
        require_file "HLS ap_int header" "${XILINX_HLS}/include/ap_int.h"
        require_file "XRT device header" "${XILINX_XRT}/include/xrt/xrt_device.h"
        if [ -r /usr/include/CL/cl2.hpp ] || [ -r "${XILINX_XRT}/include/CL/cl2.hpp" ]; then
            pass "OpenCL C++ headers"
        else
            fail "missing OpenCL C++ header CL/cl2.hpp"
        fi
        ;;
    board)
        xrt_root="${XILINX_XRT:-/opt/xilinx/xrt}"
        if [ -r "${xrt_root}/setup.sh" ]; then
            source "${xrt_root}/setup.sh" >/dev/null 2>&1
            export XILINX_XRT="${xrt_root}"
        fi
        require_command xbutil
        require_command xbmgmt
        require_file "XRT device header" "${xrt_root}/include/xrt/xrt_device.h"
        if compgen -G '/dev/dri/renderD*' >/dev/null; then
            pass "XRT render device node is present"
        else
            fail "no /dev/dri/renderD* device node; XRT user driver is unavailable"
        fi
        ;;
esac

if [ "${mode}" = "hw-emu" ]; then
    for tool in emconfigutil xclbinutil tmux; do
        require_command "${tool}"
    done
    if [ -n "${XPLATFORM:-}" ]; then
        require_file "Vitis platform" "${XPLATFORM}"
    else
        fail "XPLATFORM was not resolved for DEVICE=${DEVICE}"
    fi
fi

case "${mode}" in
    publication) default_mem_gib=2; default_tmp_gib=2 ;;
    hls) default_mem_gib=50; default_tmp_gib=20 ;;
    hw-emu) default_mem_gib=80; default_tmp_gib=100 ;;
    board) default_mem_gib=4; default_tmp_gib=2 ;;
esac

if [ "${LLM_ACCEL_SKIP_RESOURCE_CHECK:-0}" = "1" ]; then
    warn "memory and /tmp capacity checks were explicitly skipped"
else
    min_mem_gib="${LLM_ACCEL_MIN_AVAILABLE_GIB:-${default_mem_gib}}"
    min_tmp_gib="${LLM_ACCEL_MIN_TMP_GIB:-${default_tmp_gib}}"
    available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    tmp_kib="$(df -Pk /tmp | awk 'NR == 2 {print $4}')"
    if [ "${available_kib}" -ge $((min_mem_gib * 1024 * 1024)) ]; then
        pass "available memory meets ${min_mem_gib} GiB guard"
    else
        fail "available memory is below ${min_mem_gib} GiB guard"
    fi
    if [ "${tmp_kib}" -ge $((min_tmp_gib * 1024 * 1024)) ]; then
        pass "/tmp free space meets ${min_tmp_gib} GiB guard"
    else
        fail "/tmp free space is below ${min_tmp_gib} GiB guard"
    fi
fi

printf 'SUMMARY mode=%s failures=%d warnings=%d\n' "${mode}" "${failures}" "${warnings}"
if [ "${failures}" -ne 0 ]; then
    exit 1
fi
