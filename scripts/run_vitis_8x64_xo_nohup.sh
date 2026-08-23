#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_environment.sh >/dev/null
exec scripts/run_background.sh \
    vitis_8x64_xo \
    make vitis_8x64_xo
