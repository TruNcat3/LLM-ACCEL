#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 BUILD_DIR [FREQUENCY_MHZ] [PROFILE_KERNELS_CSV]" >&2
    exit 2
fi

build_dir="$(realpath "$1")"
frequency_mhz="${2:-300}"
if [ "$#" -eq 3 ]; then
    csv="$(realpath "$3")"
else
    csv="$({
        find "${build_dir}/.run" -type f -name profile_kernels.csv \
            -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi

if [ -z "${csv}" ] || [ ! -s "${csv}" ]; then
    echo "No profile_kernels.csv found below ${build_dir}/.run" >&2
    exit 66
fi

echo "profile=${csv}"
echo "frequency_mhz=${frequency_mhz}"
awk -F, -v frequency="${frequency_mhz}" '
    BEGIN { in_compute_units = 0 }
    /^Compute Units: Running Time and Stalls/ {
        in_compute_units = 1
        next
    }
    /^Functions: Running Time and Stalls/ { exit }
    in_compute_units && $1 ~ /^cc8_/ && $2 ~ /^[0-9.]+$/ {
        cycles = int($2 * frequency + 0.5)
        printf "compute_unit=%s active_us=%s cycles=%d\n", $1, $2, cycles
    }
' "${csv}"
