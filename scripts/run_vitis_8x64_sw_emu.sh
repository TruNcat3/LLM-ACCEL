#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
vitis_env_script="${VITIS_ENV_SCRIPT:-/home/hepc/env/vitis_env_22.sh}"
source "$vitis_env_script" >/dev/null 2>&1

exec make vitis_8x64_run_smoke TARGET=sw_emu
