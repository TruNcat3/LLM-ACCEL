#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

status_script="$(realpath scripts/status_vitis_8x64_qwen3b_e2e.sh)"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/llm-e2e-status.XXXXXX")"
worker_pid=""
cleanup() {
    if [ -n "${worker_pid}" ]; then
        pkill -TERM -P "${worker_pid}" 2>/dev/null || true
        kill "${worker_pid}" 2>/dev/null || true
        wait "${worker_pid}" 2>/dev/null || true
    fi
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

fake_host="${tmp_dir}/host_qwen_8x64.exe"
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'sleep 60' > "${fake_host}"
chmod +x "${fake_host}"

host_log="${tmp_dir}/qwen3b_e2e_hwemu_p8_g2_l2_fixture.log"
printf '%s\n' \
    'started_at=2026-08-20T00:00:00+00:00' \
    'profile=qwen2.5-3b' \
    'prompt_tokens=8' \
    'generated_tokens=2' \
    'layers=2' \
    'prompt_blocks=1' \
    'decode_forwards=1' \
    'expected_coarse_tasks=10' > "${host_log}"

# Keep a distinct launcher parent alive while its Host child is in the
# pre-XRT setup window and no simulation-directory banner exists yet.
bash -c '"$1" --profile qwen2.5-3b & wait' _ "${fake_host}" &
worker_pid="$!"
printf '%s\n' "${worker_pid}" > "${host_log%.log}.pid"

for _ in $(seq 1 50); do
    if pgrep -P "${worker_pid}" -f 'host_qwen_8x64' >/dev/null 2>&1; then
        break
    fi
    sleep 0.02
done
if ! pgrep -P "${worker_pid}" -f 'host_qwen_8x64' >/dev/null 2>&1; then
    echo "Fixture Host did not start" >&2
    exit 70
fi

status_output="$(
    cd "${tmp_dir}"
    "${status_script}" "$(basename "${host_log}")"
)"
if ! grep -q '^run_state=running_setup ' <<<"${status_output}" ||
   ! grep -q '^progress_percent=0.000 workload_stage=prefill stage_tasks_completed=0 stage_tasks_expected=5$' <<<"${status_output}" ||
   ! grep -Eq '^host_pid=[0-9]+ host_state=' <<<"${status_output}" ||
   ! grep -q '^simulation_dir=NA$' <<<"${status_output}"; then
    echo "Status helper did not recognize a pre-XRT Host descendant" >&2
    printf '%s\n' "${status_output}" >&2
    exit 65
fi

for task in $(seq 1 5); do
    printf 'COARSE_TASK_PROGRESS completed=%s total=5 op=fixture\n' \
        "${task}" >> "${host_log}"
done
decode_status_output="$(
    cd "${tmp_dir}"
    "${status_script}" "$(basename "${host_log}")"
)"
if ! grep -q '^progress_percent=50.000 workload_stage=decode stage_tasks_completed=0 stage_tasks_expected=5$' \
    <<<"${decode_status_output}"; then
    echo "Status helper did not recognize the Prefill-to-Decode boundary" >&2
    printf '%s\n' "${decode_status_output}" >&2
    exit 65
fi

printf '%s\n' \
    'E2E STATUS CONTRACT PASS pre_xrt_host=recognized relative_log=caller_resolved prefill=0/5 decode_boundary=5/10 state=running_setup'
