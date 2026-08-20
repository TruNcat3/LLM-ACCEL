#!/usr/bin/env bash
set -euo pipefail

script_path="$(realpath "$0")"
repo_root="$(dirname "$(dirname "${script_path}")")"
if [ "$#" -gt 1 ]; then
    echo "usage: $0 [SOURCE_ROOT]" >&2
    exit 2
fi
source_root="${1:-${repo_root}}"
if [ ! -d "${source_root}" ]; then
    echo "Missing Qwen2.5-3B source snapshot root: ${source_root}" >&2
    exit 66
fi
source_root="$(realpath "${source_root}")"
cd "${source_root}"

# This is deliberately an archive-time worktree snapshot.  The generated
# Host/XCLBIN/emconfig hashes remain the authoritative identity of the binary
# that ran.  Keeping the distinction explicit avoids claiming build-time
# provenance that an uncommitted development tree cannot prove retroactively.
entries=(
    'release_entrypoint:Makefile'
    'build_input:conn_u50_8x64_dual_full_resident.cfg'
    'build_input:common/include/xcl2.cpp'
    'build_input:common/include/xcl2.hpp'
    'build_input:host/host_qwen_8x64.cpp'
    'build_input:kernel/compute_core_8x64_nk.cpp'
    'build_input:kernel/compute_core_8x64_unified.cpp'
    'build_input:kernel/compute_stream.cpp'
    'build_input:kernel/mm_stream_8x64_fused_mac.cpp'
    'build_input:kernel/control_cache_8x64_nk.cpp'
    'build_input:kernel/control_cache_8x64.cpp'
    'build_input:kernel/mm_controller.cpp'
    'build_input:kernel/cc8_status_sink.cpp'
    'build_input:include/compute_core_8x64_unified.hpp'
    'build_input:include/compute_stream.hpp'
    'build_input:include/mm_stream_8x64.hpp'
    'build_input:include/vitis_stream_8x64.hpp'
    'build_input:include/stream_depth_config.hpp'
    'build_input:include/control_cache_8x64.hpp'
    'build_input:include/mm_controller.hpp'
    'build_input:include/linear.hpp'
    'build_input:include/hardware.hpp'
    'build_input:include/datatypes.hpp'
    'build_input:include/model_config.hpp'
    'build_input:include/util.hpp'
    'build_input:include/weight_pipeline_config.hpp'
    'build_input:tcl/build_compute_core_8x64_nk_xo.tcl'
    'build_input:tcl/build_control_cache_8x64_nk_xo.tcl'
    'build_input:tcl/build_cc8_status_sink_nk_xo.tcl'
    'build_input:tcl/common_hls_depth_config.tcl'
    'build_input:tcl/common_hls_model_profile.tcl'
    'build_harness:scripts/run_vitis_hls.sh'
    'build_harness:scripts/build_vitis_8x64_resident_layer_hwemu.sh'
    'build_harness:scripts/run_vitis_8x64_qwen3b_e2e_build_tmux.sh'
    'validation_harness:scripts/run_vitis_8x64_qwen3b_e2e_hwemu_tmux.sh'
    'validation_harness:scripts/run_vitis_8x64_small_e2e_numeric_tmux.sh'
    'validation_harness:scripts/report_vitis_8x64_e2e_trace.sh'
    'validation_harness:scripts/report_qwen3b_hls_resources.sh'
    'validation_harness:scripts/archive_vitis_8x64_e2e_run.sh'
    'validation_harness:scripts/install_vitis_8x64_e2e_result.sh'
    'validation_harness:scripts/status_vitis_8x64_qwen3b_e2e.sh'
    'validation_harness:scripts/watch_vitis_8x64_e2e_archive_tmux.sh'
    'validation_harness:scripts/verify_qwen3b_e2e_release.sh'
    'validation_harness:scripts/verify_vitis_8x64_e2e_progress.sh'
    'validation_harness:scripts/report_qwen3b_source_snapshot.sh'
    'validation_harness:scripts/report_qwen3b_build_source_equivalence.sh'
    'validation_harness:scripts/verify_q214_pd_release.sh'
    'validation_test:tests/test_qwen3b_e2e_launcher_contract.sh'
    'validation_test:tests/test_coarse_task_residency_contract.sh'
    'validation_test:tests/test_qwen3b_source_snapshot.sh'
    'validation_test:tests/test_e2e_progress_contract.sh'
    'validation_test:tests/test_e2e_performance_semantics.sh'
    'validation_test:tests/test_result_installer_contract.sh'
)

printf 'path\tsha256\tbytes\trole\n'
for entry in "${entries[@]}"; do
    role="${entry%%:*}"
    path="${entry#*:}"
    if [ ! -s "${path}" ]; then
        echo "Missing or empty Qwen2.5-3B release source: ${path}" >&2
        exit 66
    fi
    sha256="$(sha256sum "${path}" | awk '{ print $1 }')"
    bytes="$(stat -c '%s' "${path}")"
    printf '%s\t%s\t%s\t%s\n' \
        "${path}" "${sha256}" "${bytes}" "${role}"
done
