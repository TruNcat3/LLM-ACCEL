#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")/.."

work_root="$(mktemp -d "${TMPDIR:-/tmp}/llm-accel-installer-test.XXXXXX")"
cleanup() {
    rm -rf "${work_root}"
}
trap cleanup EXIT
source_archive="${work_root}/source"
results_root="${work_root}/results"
mkdir -p "${source_archive}" "${results_root}"

printf '%s\n' \
    'model_source=random' \
    'tie_embeddings=1' \
    > "${source_archive}/host.raw.log"
printf '%s\t%s\n' \
    field value \
    profile small \
    trace_scope common_four_CU_running_time \
    physical_board_measurement 0 \
    host_inference_compute_scope embedding_plus_lm_head_argmax \
    accelerator_compute_scope decoder_layers_final_norm_rope_online_attention_kv \
    cpu_golden_scope post_inference_validation_only \
    > "${source_archive}/manifest.tsv"
printf '%s\n' \
    'Compute Units: Running Time and Stalls' \
    'Compute Unit, Running Time (us), Intra-Kernel Dataflow Stalls (%), External Memory Stalls (%), External Stream Stalls (%)' \
    'cc8_ctrl,270.842,0.000,0.000,0.000' \
    > "${source_archive}/profile_kernels.csv"

performance_header='evidence_source\ttimed_scope\thost_validation\tnumeric_validation\tnumeric_steps\tnumeric_checked_values\tnumeric_max_raw_error\tnumeric_tolerance\tnumeric_golden_host_ms\tnumeric_validation_schedule\thost_inference_ms\thost_process_ms\thost_lm_head_ms\thost_validation_ms\tartifact_identity\thost_exe_sha256\txclbin_sha256\temconfig_sha256\tprofile\tsequence_batch\tprompt_sequence_tokens\tsampled_output_tokens\tdecode_forwards\tprefill_blocks\tlayers\tconfigured_max_active_query_rows_per_prefill_block\trelease_nonfinal_blocks\texpected_coarse_tasks\txsim_cu_running_us\txsim_clock_mhz\txsim_cycles\ttarget_clock_mhz\tprojected_target_us\tuseful_mac\tuseful_gmac_s\tmodeled_interval_efficiency_percent\tquery_row_s\trequest_output_token_s'
fake_sha='0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'
performance_row="HW_Emu_CU_trace\tcommon_four_CU_running_time\tPASS\tPASS\t2\t128\t4\t64\t51.023\tpost_inference\t292523.000\t292574.000\t0.023\t51.123\tPASS\t${fake_sha}\t${fake_sha}\t${fake_sha}\tsmall\t1\t8\t2\t1\t1\t2\t8\t0\t10\t270.842\t200.000\t54168.4\t200.000\t270.8420\t675072\t2.492494\t1.217038\t33229.706\t7384.379"
printf '%b\n%b\n' "${performance_header}" "${performance_row}" \
    > "${source_archive}/performance.tsv"

(
    cd "${source_archive}"
    sha256sum host.raw.log manifest.tsv performance.tsv profile_kernels.csv \
        > checksums.sha256
)

install_output="$(
    LLM_ACCEL_RESULTS_ROOT="${results_root}" \
        scripts/install_vitis_8x64_e2e_result.sh \
        "${source_archive}" contract-result
)"
destination="${results_root}/contract-result"
if [ ! -s "${destination}/README.md" ] ||
   [ ! -s "${destination}/checksums.sha256" ] ||
   [ "$(find "${destination}" -type f | wc -l)" -ne 6 ] ||
   [[ "${install_output}" != *"result_package=${destination}"* ]] ||
   ! rg -q '^# small P8/G2/L2 End-to-End Hardware-Emulation Evidence$' \
        "${destination}/README.md" ||
   ! rg -q 'not batch 8 and not 8 generated tokens in parallel' \
        "${destination}/README.md"; then
    echo "Atomic result installer output contract failed" >&2
    exit 65
fi
scripts/verify_result_checksums.sh \
    "${destination}/checksums.sha256" >/dev/null

set +e
LLM_ACCEL_RESULTS_ROOT="${results_root}" \
    scripts/install_vitis_8x64_e2e_result.sh \
    "${source_archive}" contract-result >/dev/null 2>&1
overwrite_status="$?"
set -e
if [ "${overwrite_status}" -ne 73 ]; then
    echo "Atomic result installer did not refuse an existing destination" >&2
    exit 65
fi

echo 'RESULT INSTALLER CONTRACT PASS files=6 readme=rendered checksums=complete overwrite=refused'
