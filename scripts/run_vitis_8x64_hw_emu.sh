#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/hepc/env/vitis_env_22.sh >/dev/null 2>&1

exec make vitis_8x64_run_hw_emu TARGET=hw_emu
