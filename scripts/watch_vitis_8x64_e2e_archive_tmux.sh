#!/usr/bin/env bash
set -euo pipefail

script_path="$(realpath "$0")"
repo_root="$(dirname "$(dirname "${script_path}")")"
cd "${repo_root}"

usage() {
    echo "usage: $0 HOST_LOG BUILD_DIR OUTPUT_DIR [INTERVAL_SECONDS] [BUILD_LOG] [SOURCE_ROOT]" >&2
}

if [ "${1:-}" = "--worker" ]; then
    shift
    if [ "$#" -lt 4 ] || [ "$#" -gt 6 ]; then
        usage
        exit 2
    fi
    host_log="$(realpath "$1")"
    build_dir="$(realpath "$2")"
    output_dir="$(realpath -m "$3")"
    interval_seconds="$4"
    build_log="${5:-}"
    source_root="${6:-${repo_root}}"
    worker_pid_file="${host_log%.log}.pid"
    if [ ! -s "${worker_pid_file}" ]; then
        echo "Missing E2E worker PID file: ${worker_pid_file}" >&2
        exit 66
    fi
    worker_pid="$(awk 'NR == 1 { print $1 }' "${worker_pid_file}")"

    worker_is_running() {
        [ -r "/proc/${worker_pid}/cmdline" ] || return 1
        local command_line
        command_line="$(tr '\0' ' ' < "/proc/${worker_pid}/cmdline")"
        [[ "${command_line}" == *"run_vitis_8x64_qwen3b_e2e_hwemu_tmux.sh --worker"* ]]
    }

    echo "watch_started_at=$(date -Is)"
    echo "host_log=${host_log}"
    echo "worker_pid=${worker_pid}"
    echo "interval_seconds=${interval_seconds}"
    echo "archive_output=${output_dir}"
    while worker_is_running; do
        echo "status_sample_begin=$(date -Is)"
        "${repo_root}/scripts/status_vitis_8x64_qwen3b_e2e.sh" \
            "${host_log}"
        echo "status_sample_end=$(date -Is)"
        sleep "${interval_seconds}"
    done

    echo "worker_exit_observed_at=$(date -Is)"
    "${repo_root}/scripts/status_vitis_8x64_qwen3b_e2e.sh" "${host_log}"
    exit_status="$(
        awk -F '=' '$1 == "exit_status" { value = $2 } END { print value }' \
            "${host_log}"
    )"
    if [ "${exit_status}" != "0" ]; then
        echo "E2E run is not archivable: exit_status=${exit_status:-missing}" >&2
        exit 65
    fi
    archive_args=(
        "${host_log}" "${build_dir}" "${output_dir}" 200 200 1
    )
    if [ -n "${build_log}" ]; then
        archive_args+=("${build_log}")
        archive_args+=("${source_root}")
    fi
    "${repo_root}/scripts/archive_vitis_8x64_e2e_run.sh" \
        "${archive_args[@]}"
    "${repo_root}/scripts/verify_qwen3b_e2e_release.sh" \
        "${host_log}" "${build_dir}" "${output_dir}" "${source_root}"
    echo "archive_completed_at=$(date -Is)"
    echo "archive_output=${output_dir}"
    exit 0
fi

if [ "$#" -lt 3 ] || [ "$#" -gt 6 ]; then
    usage
    exit 2
fi
host_log="$(realpath "$1")"
build_dir="$(realpath "$2")"
output_dir="$(realpath -m "$3")"
interval_seconds="${4:-3600}"
build_log="${5:-}"
source_root="${6:-${repo_root}}"

if [ ! -s "${host_log}" ]; then
    echo "Missing or empty E2E Host log: ${host_log}" >&2
    exit 66
fi
if [ ! -d "${build_dir}" ]; then
    echo "Missing E2E build directory: ${build_dir}" >&2
    exit 66
fi
if [ -n "${build_log}" ] && [ ! -s "${build_log}" ]; then
    echo "Missing E2E build provenance log: ${build_log}" >&2
    exit 66
fi
if [ ! -d "${source_root}" ]; then
    echo "Missing E2E release source root: ${source_root}" >&2
    exit 66
fi
if [ -e "${output_dir}" ]; then
    echo "Archive output already exists: ${output_dir}" >&2
    exit 73
fi
if ! [[ "${interval_seconds}" =~ ^[0-9]+$ ]] ||
   [ "${interval_seconds}" -lt 60 ]; then
    echo "INTERVAL_SECONDS must be an integer of at least 60" >&2
    exit 2
fi

worker_pid_file="${host_log%.log}.pid"
if [ ! -s "${worker_pid_file}" ]; then
    echo "Missing E2E worker PID file: ${worker_pid_file}" >&2
    exit 66
fi
worker_pid="$(awk 'NR == 1 { print $1 }' "${worker_pid_file}")"
if [ ! -r "/proc/${worker_pid}/cmdline" ]; then
    echo "E2E worker is not running: ${worker_pid}" >&2
    exit 70
fi

mkdir -p logs
timestamp="$(date +%Y%m%d_%H%M%S)"
session_name="llm_e2e_archive_watch_${timestamp}"
watch_log="${repo_root}/logs/e2e_archive_watch_${timestamp}.log"
watch_pid_file="${watch_log%.log}.pid"
worker_argv=(
    "${script_path}" --worker
    "${host_log}" "${build_dir}" "${output_dir}" "${interval_seconds}"
)
if [ -n "${build_log}" ]; then
    worker_argv+=("$(realpath "${build_log}")")
    worker_argv+=("$(realpath "${source_root}")")
fi
printf -v worker_command '%q ' "${worker_argv[@]}"
printf -v quoted_watch_log '%q' "${watch_log}"
tmux new-session -d -s "${session_name}" \
    "exec ${worker_command} >>${quoted_watch_log} 2>&1"
watch_pid="$(tmux display-message -p -t "${session_name}:0.0" '#{pane_pid}')"
printf '%s\n' "${watch_pid}" > "${watch_pid_file}"
printf 'pid=%s\nsession=%s\nlog=%s\npidfile=%s\narchive_output=%s\n' \
    "${watch_pid}" "${session_name}" "${watch_log}" \
    "${watch_pid_file}" "${output_dir}"
