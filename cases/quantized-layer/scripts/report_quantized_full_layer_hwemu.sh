#!/usr/bin/env bash
set -euo pipefail
# Requires RTL transitions, not a generated profile with no kernel events.
exec python3 "$(dirname "$0")/analyze_quantized_layer_trace.py" "$@"
