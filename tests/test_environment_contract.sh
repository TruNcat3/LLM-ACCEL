#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LLM_ACCEL_SKIP_RESOURCE_CHECK=1 scripts/check_environment.sh publication >/dev/null

if scripts/check_environment.sh invalid-mode >/dev/null 2>&1; then
    echo "Environment checker accepted an invalid mode" >&2
    exit 1
fi

if scripts/setup_environment.sh >/dev/null 2>&1; then
    echo "Environment setup incorrectly allowed execution instead of sourcing" >&2
    exit 1
fi

if rg -n '/home/(hepc|wt)/|/tools/Xilinx/Vitis/2022\.2/bin/v\+\+' \
    scripts docs README.md cases; then
    echo "Public environment flow contains a machine-private setup or tool path" >&2
    exit 1
fi

if ! rg -F -q 'source ../../scripts/setup_environment.sh' \
    cases/streaming-split/README.md \
    cases/streaming-split/tcl/run_v8_2x2_swemu.sh; then
    echo "Streaming-split case does not share the public environment resolver" >&2
    exit 1
fi

for mode in publication hls hw-emu board; do
    if ! rg -F -q "${mode}" docs/environment.md; then
        echo "Environment document is missing mode: ${mode}" >&2
        exit 1
    fi
done

echo "ENVIRONMENT CONTRACT PASS modes=publication,hls,hw-emu,board private_paths=0"
