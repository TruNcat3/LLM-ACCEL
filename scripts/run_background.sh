#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <run-name> <command> [args...]" >&2
    exit 2
fi

cd "$(dirname "$0")/.."

run_name="$1"
shift

mkdir -p logs
timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="$PWD/logs/${run_name}_${timestamp}.log"
pid_path="$PWD/logs/${run_name}_${timestamp}.pid"

nohup "$@" > "$log_path" 2>&1 &
pid="$!"

printf "%s\n" "$pid" > "$pid_path"
printf "pid=%s\nlog=%s\npidfile=%s\n" "$pid" "$log_path" "$pid_path"
