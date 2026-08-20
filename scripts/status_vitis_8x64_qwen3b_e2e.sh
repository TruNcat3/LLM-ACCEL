#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")/.."

if [ "$#" -gt 1 ]; then
    echo "usage: $0 [HOST_LOG]" >&2
    exit 2
fi

if [ "$#" -eq 1 ]; then
    host_log="$(realpath -m "$1")"
else
    host_log="$(
        find "$PWD/logs" -maxdepth 1 -type f \
            -name 'qwen3b_e2e_hwemu_p*_g*_l*.log' \
            -printf '%T@\t%p\n' 2>/dev/null |
        sort -n |
        awk -F '\t' 'NF >= 2 { latest = $2 } END { print latest }'
    )"
fi

if [ -z "${host_log}" ] || [ ! -f "${host_log}" ]; then
    echo "Cannot find a Qwen2.5-3B E2E Host log" >&2
    exit 66
fi

field_from_line() {
    local line="$1"
    local key="$2"
    awk -v key="${key}" '
        {
            for (i = 1; i <= NF; i++) {
                prefix = key "="
                if (index($i, prefix) == 1) {
                    print substr($i, length(prefix) + 1)
                    exit
                }
            }
        }
    ' <<<"${line}"
}

header_value() {
    local key="$1"
    awk -F '=' -v key="${key}" '
        $1 == key { value = substr($0, length(key) + 2) }
        END { print value }
    ' "${host_log}"
}

process_exists() {
    local pid="$1"
    [ -n "${pid}" ] && [ -d "/proc/${pid}" ]
}

process_snapshot() {
    local name="$1"
    local pid="$2"
    if process_exists "${pid}"; then
        ps -o pid=,stat=,etime=,%cpu=,rss= -p "${pid}" |
        awk -v name="${name}" '
            NF >= 5 {
                printf "%s_pid=%s %s_state=%s %s_elapsed=%s %s_cpu_percent=%s %s_rss_kib=%s\n",
                    name, $1, name, $2, name, $3, name, $4, name, $5
            }
        '
    else
        printf '%s_pid=NA %s_state=not_running\n' "${name}" "${name}"
    fi
}

started_at="$(header_value started_at)"
profile="$(header_value profile)"
prompt_tokens="$(header_value prompt_tokens)"
generated_tokens="$(header_value generated_tokens)"
layers="$(header_value layers)"
expected_tasks="$(header_value expected_coarse_tasks)"
expected_tasks="${expected_tasks:-0}"

worker_pid_file="${host_log%.log}.pid"
worker_pid=""
if [ -s "${worker_pid_file}" ]; then
    worker_pid="$(awk 'NR == 1 { print $1 }' "${worker_pid_file}")"
fi

simulation_dir="$(
    awk '
        /Path of the simulation directory[[:space:]]*:/ {
            sub(/^.*:[[:space:]]*/, "")
            path = $0
        }
        END { print path }
    ' "${host_log}"
)"
host_pid=""
if [[ "${simulation_dir}" =~ \.run/([0-9]+)/hw_em/ ]]; then
    host_pid="${BASH_REMATCH[1]}"
fi

xsim_pid=""
if [ -n "${simulation_dir}" ]; then
    while read -r candidate; do
        [ -n "${candidate}" ] || continue
        candidate_cwd="$(readlink -f "/proc/${candidate}/cwd" 2>/dev/null || true)"
        if [ "${candidate_cwd}" = "${simulation_dir}" ]; then
            xsim_pid="${candidate}"
            break
        fi
    done < <(pgrep -x xsimk 2>/dev/null || true)
fi

completed_tasks="$(
    awk '/^COARSE_TASK_PROGRESS / { count++ } END { print count + 0 }' \
        "${host_log}"
)"
numeric_steps="$(
    awk '/^QWEN_8X64_E2E_NUMERIC_STEP / { count++ } END { print count + 0 }' \
        "${host_log}"
)"
last_task="$(
    awk '/^COARSE_TASK_PROGRESS / { last = $0 } END { print last }' \
        "${host_log}"
)"
setup_line="$(
    awk '/^QWEN_8X64_SETUP_PROFILE / { last = $0 } END { print last }' \
        "${host_log}"
)"
profile_line="$(
    awk '/^QWEN_8X64_E2E_PROFILE / { last = $0 } END { print last }' \
        "${host_log}"
)"
if [ "${expected_tasks}" = "0" ] && [ -n "${profile_line}" ]; then
    expected_tasks="$(field_from_line "${profile_line}" coarse_task_count)"
    expected_tasks="${expected_tasks:-0}"
fi
exit_status="$(header_value exit_status)"

if [ -n "${exit_status}" ]; then
    if [ "${exit_status}" = "0" ]; then
        run_state="complete_pass"
    else
        run_state="complete_fail"
    fi
elif process_exists "${worker_pid}" && process_exists "${host_pid}"; then
    if [ -z "${setup_line}" ]; then
        run_state="running_setup"
    elif [ "${completed_tasks}" -lt "${expected_tasks}" ]; then
        run_state="running_hardware_tasks"
    else
        run_state="running_materialization_or_validation"
    fi
else
    run_state="stopped_incomplete"
fi

sim_log=""
sim_reported_ps="NA"
sim_log_age_seconds="NA"
simulation_bytes="NA"
if [ -n "${simulation_dir}" ] && [ -d "${simulation_dir}" ]; then
    sim_log="${simulation_dir}/simulate.log"
    simulation_bytes="$(du -sb "${simulation_dir}" | awk '{ print $1 }')"
    if [ -f "${sim_log}" ]; then
        sim_reported_ps="$(
            awk '
                {
                    for (i = 1; i + 2 <= NF; i++) {
                        if ($i == "@" && $(i + 2) == "ps" &&
                            $(i + 1) + 0 > maximum) {
                            maximum = $(i + 1) + 0
                        }
                    }
                }
                END { if (maximum > 0) print maximum; else print "NA" }
            ' "${sim_log}"
        )"
        sim_log_age_seconds=$((
            $(date +%s) - $(stat -c %Y "${sim_log}")
        ))
    fi
fi

memory_available_kib="$(awk '/MemAvailable:/ { print $2 }' /proc/meminfo)"
tmp_available_kib="$(df -Pk /tmp | awk 'NR == 2 { print $4 }')"
home_available_kib="$(df -Pk "$PWD" | awk 'NR == 2 { print $4 }')"

printf 'observed_at=%s\n' "$(date -Is)"
printf 'host_log=%s\n' "${host_log}"
printf 'run_state=%s started_at=%s profile=%s prompt_tokens=%s generated_tokens=%s layers=%s\n' \
    "${run_state}" "${started_at:-NA}" "${profile:-NA}" \
    "${prompt_tokens:-NA}" "${generated_tokens:-NA}" "${layers:-NA}"
printf 'coarse_tasks_completed=%s coarse_tasks_expected=%s numeric_steps=%s\n' \
    "${completed_tasks}" "${expected_tasks}" "${numeric_steps}"
if [ -n "${last_task}" ]; then
    printf 'last_task=%s\n' "${last_task}"
fi
if [ -n "${setup_line}" ]; then
    printf 'setup_total_ms=%s weight_preload=%s\n' \
        "$(field_from_line "${setup_line}" total_setup_ms)" \
        "$(field_from_line "${setup_line}" weight_preload)"
fi
if [ -n "${profile_line}" ]; then
    printf 'inference_host_ms=%s lm_head_host_ms=%s validation_ms=%s\n' \
        "$(field_from_line "${profile_line}" total_host_elapsed_ms)" \
        "$(field_from_line "${profile_line}" lm_head_host_ms)" \
        "$(field_from_line "${profile_line}" post_inference_validation_ms)"
fi
process_snapshot worker "${worker_pid}"
process_snapshot host "${host_pid}"
process_snapshot xsim "${xsim_pid}"
printf 'simulation_dir=%s\n' "${simulation_dir:-NA}"
printf 'simulation_bytes=%s sim_max_reported_ps=%s sim_log_age_seconds=%s\n' \
    "${simulation_bytes}" "${sim_reported_ps}" "${sim_log_age_seconds}"
printf 'memory_available_kib=%s tmp_available_kib=%s home_available_kib=%s\n' \
    "${memory_available_kib}" "${tmp_available_kib}" "${home_available_kib}"
