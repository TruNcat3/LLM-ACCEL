#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

env_script="${VITIS_ENV_SCRIPT:-/home/hepc/env/vitis_env_22.sh}"
build_dir="${Q214_PAYLOAD_GOLDEN_BUILD_DIR:-/tmp/llm_accel_q214_payload_golden}"
seed=20260718

if [ ! -r "${env_script}" ]; then
    echo "Missing Vitis environment script: ${env_script}" >&2
    exit 66
fi
source "${env_script}" >/dev/null 2>&1

make vitis_8x64_qwen_host \
    TARGET=hw_emu \
    VITIS_8X64_MODEL_PROFILE=qwen-layer \
    BUILD_DIR="${build_dir}"

host_exe="${build_dir}/host_qwen_8x64.exe"
if [ ! -x "${host_exe}" ]; then
    echo "Missing payload-golden Host: ${host_exe}" >&2
    exit 66
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

run_case() {
    local rows="$1"
    local log_path="${tmp_dir}/p${rows}.log"
    "${host_exe}" \
        --mode diagnose-q214-payload-golden \
        --profile qwen-layer \
        --random-model \
        --seed "${seed}" \
        --prefill-block-size "${rows}" | tee "${log_path}"
}

run_case 2
run_case 8

p2_log="${tmp_dir}/p2.log"
p8_log="${tmp_dir}/p8.log"
grep -Fq \
    'RANDOM q214_raw_payload_vs_direct_final_norm_tolerance32 PASS values=4096 max_raw_error=0 tolerance=32' \
    "${p2_log}"
grep -Fq \
    'RANDOM q214_raw_payload_vs_direct_final_norm_tolerance32 FAIL values=16384 max_raw_error=454 tolerance=32 mismatches=3615' \
    "${p8_log}"
grep -Fq \
    'Q214_PAYLOAD_GOLDEN_PROBE row=3 col=1 raw_payload=-11446 direct=-11398' \
    "${p8_log}"
grep -Fq \
    'Q214_PAYLOAD_GOLDEN_PROBE row=3 col=2 raw_payload=24082 direct=24240' \
    "${p8_log}"
grep -Fq \
    'Q214_PAYLOAD_GOLDEN_PROBE row=3 col=4 raw_payload=21908 direct=21751' \
    "${p8_log}"
grep -Fq \
    'Q214_PAYLOAD_GOLDEN_PROBE row=3 col=9 raw_payload=12575 direct=12716' \
    "${p8_log}"
grep -Fq \
    'Q214_PAYLOAD_GOLDEN_DIAGNOSTIC seed=20260718 active_query_rows=8 profile=qwen-layer PASS' \
    "${p8_log}"

echo "Q214 PAYLOAD GOLDEN REGRESSION PASS seed=${seed} p2=bit_identical p8=hardware_failure_fingerprint_exact"
