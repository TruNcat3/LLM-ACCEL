#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec scripts/run_background.sh \
    hls_cosim_control_cache_8x64_dual_core \
    make hls_cosim_control_cache_8x64_dual_core
