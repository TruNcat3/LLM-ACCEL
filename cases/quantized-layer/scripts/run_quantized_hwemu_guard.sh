#!/usr/bin/env bash
# Detect a failed waveform hook before the Host spends hours polling XRT.
set -euo pipefail
[[ "$#" -gt 0 ]] || { echo "usage: $0 COMMAND [ARG...]" >&2; exit 2; }
if [[ "${QUANTIZED_LAYER_TRACE:-0}" != 1 ]]; then
    exec "$@"
fi

captured_log="$(mktemp "${TMPDIR:-/tmp}/quantized-hwemu-guard.XXXXXX")"
run_pid=
ready_seen=0
poll_before_ready="${QUANTIZED_HWEMU_GUARD_POLL_SECONDS:-5}"
poll_after_ready="${QUANTIZED_HWEMU_GUARD_READY_POLL_SECONDS:-60}"
if [[ -n "${QUANTIZED_HWEMU_GUARD_POLL_SECONDS:-}" ]]; then
    poll_after_ready="${QUANTIZED_HWEMU_GUARD_POLL_SECONDS}"
fi
stop_tree() {
    local pid="$1" child
    for child in $(pgrep -P "${pid}" || true); do
        stop_tree "${child}"
    done
    kill -TERM "${pid}" 2>/dev/null || true
}
cleanup() {
    if [[ -n "${run_pid}" ]] && kill -0 "${run_pid}" 2>/dev/null; then
        stop_tree "${run_pid}"
        wait "${run_pid}" 2>/dev/null || true
    fi
    rm -f "${captured_log}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
(
    set -o pipefail
    "$@" 2>&1 | tee "${captured_log}"
) &
run_pid=$!

while kill -0 "${run_pid}" 2>/dev/null; do
    sim_dir="$(sed -n 's/^INFO: \[HW-EMU 05\] Path of the simulation directory : //p' "${captured_log}" | tail -n 1)"
    if [[ -n "${sim_dir}" ]]; then
        for crash_report in "${sim_dir}"/hs_err_pid*.log; do
            if [[ -f "${crash_report}" ]]; then
                echo "HW Emu native XSim frontend crash detected; report=${crash_report}." >&2
                exit 66
            fi
        done
        if [[ -f "${sim_dir}/simulate.log" ]] && rg -q 'QUANTIZED_TRACE_(RECORDING_FAILED|MISSING)' "${sim_dir}/simulate.log"; then
            echo "HW Emu trace initialization failed; stopping this run. Relink with VITIS_QUANTIZED_LAYER_DEBUG=1." >&2
            exit 65
        fi
        # Keep watching after readiness: the native XSim frontend can fail while
        # the Host is already polling, and late stage-hook errors remain fatal.
        if [[ "${ready_seen}" -eq 0 && -f "${sim_dir}/simulate.log" ]] && rg -q '^QUANTIZED_TRACE_INIT_READY ' "${sim_dir}/simulate.log"; then
            ready_seen=1
            echo "trace_guard=all_hooks_ready"
        fi
    fi
    if [[ "${ready_seen}" -eq 1 ]]; then
        sleep "${poll_after_ready}"
    else
        sleep "${poll_before_ready}"
    fi
done
set +e
wait "${run_pid}"
status=$?
run_pid=
set -e
exit "${status}"
