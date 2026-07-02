#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec scripts/run_background.sh \
    hls_cosim_compute_core_8x64_unified \
    make hls_cosim_compute
