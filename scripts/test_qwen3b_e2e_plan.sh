#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

env_script="${VITIS_ENV_SCRIPT:-/home/hepc/env/vitis_env_22.sh}"
build_dir="${QWEN3B_PLAN_BUILD_DIR:-/tmp/llm_accel_qwen3b_plan}"
if [ ! -r "${env_script}" ]; then
    echo "Missing Vitis environment script: ${env_script}" >&2
    exit 66
fi
source "${env_script}" >/dev/null 2>&1

make vitis_8x64_qwen_host \
    TARGET=hw_emu \
    VITIS_8X64_MODEL_PROFILE=qwen2.5-3b \
    BUILD_DIR="${build_dir}"

host_exe="${build_dir}/host_qwen_8x64.exe"
if [ ! -x "${host_exe}" ]; then
    echo "Missing qwen2.5-3b plan Host: ${host_exe}" >&2
    exit 66
fi

plan="$(${host_exe} --mode plan --profile qwen2.5-3b)"
printf '%s\n' "${plan}"

grep -Fq 'profile=qwen2.5-3b' <<<"${plan}"
grep -Fq 'vocab=151936 hidden=2048 intermediate=11008 layers=36' <<<"${plan}"
grep -Fq 'head_dim=128 max_seq=2048' <<<"${plan}"
grep -Fq 'weight_shard_bytes=346816512 hbm_channels_per_shard=3 hbm_capacity_per_shard=805306368' <<<"${plan}"
grep -Fq 'paired_weight_kv_group_payload=769130496 paired_weight_kv_group_capacity=805306368 paired_weight_kv_group_headroom=36175872' <<<"${plan}"
grep -Fq 'repeat layer Task 18: RMS -> Q/K/V -> RoPE(controller)' <<<"${plan}"
grep -Fq 'repeat layer Task 19: RMS -> Gate/Up -> SiLU-Mul -> Down -> residual' <<<"${plan}"
grep -Fq 'final prompt/decode forward: Task 20 final RMS -> one hidden D2H' <<<"${plan}"

echo "QWEN2.5-3B E2E PLAN TEST PASS"
