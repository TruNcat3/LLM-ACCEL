#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

prefix="llm_pd_$(date +%Y%m%d_%H%M%S)"
build_dir="vitis_8x64/build.qwen_layer_long.prefill_layer.d2.ii2.wr33.cw1.scratch.hw_emu.xilinx_u50_gen3x16_xdma_5_202210_1"
profile="qwen-layer-long"
seed="20260722"

usage() {
    echo "Usage: $0 [--prefix NAME] [--build-dir DIR] [--profile NAME] [--seed N]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --prefix) prefix="$2"; shift 2 ;;
        --build-dir) build_dir="$2"; shift 2 ;;
        --profile) profile="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ ! -s "${build_dir}/qwen_8x64_dual.xclbin" ]; then
    echo "Missing xclbin: ${build_dir}/qwen_8x64_dual.xclbin" >&2
    exit 66
fi

runner="scripts/run_vitis_8x64_pd_length_sweep_hwemu.sh --build-dir ${build_dir} --profile ${profile} --seed ${seed}"

start_session() {
    local suffix="$1"
    local command="$2"
    local session="${prefix}_${suffix}"

    if tmux has-session -t "${session}" 2>/dev/null; then
        echo "tmux session already exists: ${session}" >&2
        return 70
    fi

    tmux new-session -d -s "${session}" -c "$PWD" "${command}"
    echo "session=${session}"
}

mkdir -p logs

for context_len in 64 256 512 1024; do
    for phase in p d; do
        suffix="${phase}${context_len}"
        start_session "${suffix}" \
            "${runner} --lengths ${context_len} --phase ${phase} 2>&1 | tee logs/${prefix}_${suffix}.log"
    done
done

echo "status_command=tmux ls | grep ${prefix}"
echo "attach_command=tmux attach -t ${prefix}_p64"
