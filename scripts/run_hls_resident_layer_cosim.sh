#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/hepc/env/vitis_env_22.sh >/dev/null 2>&1

export CC8_NK_TASK_STREAM_DEPTH="${CC8_NK_TASK_STREAM_DEPTH:-2}"
export CC8_NK_DATA_STREAM_DEPTH="${CC8_NK_DATA_STREAM_DEPTH:-2}"
export CC8_NK_STATUS_STREAM_DEPTH="${CC8_NK_STATUS_STREAM_DEPTH:-2}"
export CU8_NK_TASK_STREAM_DEPTH="${CU8_NK_TASK_STREAM_DEPTH:-2}"
export CU8_NK_DATA_STREAM_DEPTH="${CU8_NK_DATA_STREAM_DEPTH:-2}"
export CC8_WEIGHT_TILE_FIFO_DEPTH="${CC8_WEIGHT_TILE_FIFO_DEPTH:-2}"
export CC8_WEIGHT_TILE_LOAD_II="${CC8_WEIGHT_TILE_LOAD_II:-2}"
export CC8_ENABLE_MM_WAVE_REPEAT="${CC8_ENABLE_MM_WAVE_REPEAT:-0}"
export CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW="${CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW:-1}"
export CC8_MM_WAVE_RESULT_FIFO_DEPTH="${CC8_MM_WAVE_RESULT_FIFO_DEPTH:-33}"

echo "started_at=$(date -Is)"
echo "deadlock_detection=enabled"
echo "cc8_task_depth=${CC8_NK_TASK_STREAM_DEPTH}"
echo "cc8_data_depth=${CC8_NK_DATA_STREAM_DEPTH}"
echo "cu8_task_depth=${CU8_NK_TASK_STREAM_DEPTH}"
echo "cu8_data_depth=${CU8_NK_DATA_STREAM_DEPTH}"
echo "weight_fifo_depth=${CC8_WEIGHT_TILE_FIFO_DEPTH}"
echo "weight_load_ii=${CC8_WEIGHT_TILE_LOAD_II}"
echo "cross_wave_dataflow=${CC8_ENABLE_MM_CROSS_WAVE_DATAFLOW}"
echo "wave_result_fifo_depth=${CC8_MM_WAVE_RESULT_FIFO_DEPTH}"

set +e
make hls_cosim_closed_loop_8x64_resident_layer
status="$?"
set -e

echo "finished_at=$(date -Is)"
echo "exit_status=${status}"
exit "${status}"
