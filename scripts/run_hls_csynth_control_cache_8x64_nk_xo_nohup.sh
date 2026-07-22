#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/hepc/env/vitis_env_22.sh >/dev/null 2>&1
exec scripts/run_background.sh \
    hls_csynth_control_cache_8x64_nk_xo \
    make hls_csynth_control_cache_8x64_nk_xo
