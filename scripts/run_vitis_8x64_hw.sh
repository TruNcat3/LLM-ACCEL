#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_environment.sh >/dev/null

exec make vitis_8x64_run_smoke TARGET=hw
