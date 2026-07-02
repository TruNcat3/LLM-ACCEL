#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
vitis_env_script="${VITIS_ENV_SCRIPT:-/home/hepc/env/vitis_env_22.sh}"
source "$vitis_env_script" >/dev/null 2>&1
target="${TARGET:-hw_emu}"

min_available_gib="${VITIS_MIN_AVAILABLE_GIB:-50}"
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
required_kib=$((min_available_gib * 1024 * 1024))
if [ "$available_kib" -lt "$required_kib" ]; then
    available_gib="$(awk -v kib="$available_kib" 'BEGIN { printf "%.1f", kib / 1024 / 1024 }')"
    echo "Vitis memory guard: MemAvailable=${available_gib}GiB, need at least ${min_available_gib}GiB." >&2
    exit 75
fi

exec scripts/run_background.sh \
    "vitis_8x64_link_${target}" \
    make vitis_8x64_link TARGET="$target"
