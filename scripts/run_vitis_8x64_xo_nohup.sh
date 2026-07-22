#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/hepc/env/vitis_env_22.sh >/dev/null 2>&1
exec scripts/run_background.sh \
    vitis_8x64_xo \
    make vitis_8x64_xo
