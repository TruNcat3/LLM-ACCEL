#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ] || [ "$#" -gt 10 ]; then
    echo "usage: $0 PROFILE_CSV MODEL_PROFILE PROMPT_TOKENS GENERATED_TOKENS LAYERS [BLOCK_SIZE] [TARGET_MHZ] [XSIM_MHZ] [RELEASE_NONFINAL_BLOCKS] [HOST_LOG]" >&2
    exit 2
fi

profile_csv="$1"
model_profile="$2"
prompt_tokens="$3"
generated_tokens="$4"
layers="$5"
block_size="${6:-8}"
target_mhz="${7:-200}"
xsim_mhz="${8:-300}"
release_nonfinal="${9:-1}"
host_log="${10:-}"

if [ ! -s "${profile_csv}" ]; then
    echo "Missing or empty HW-Emu CU profile: ${profile_csv}" >&2
    exit 66
fi

case "${model_profile}" in
    small)
        hidden=64
        intermediate=128
        kv_channels=32
        heads=4
        head_dim=16
        ;;
    qwen-layer|qwen-layer-long|qwen2.5-3b)
        hidden=2048
        intermediate=11008
        kv_channels=256
        heads=16
        head_dim=128
        ;;
    *)
        echo "Unsupported model profile: ${model_profile}" >&2
        exit 2
        ;;
esac

for value in \
    "${prompt_tokens}" "${generated_tokens}" "${layers}" "${block_size}" \
    "${target_mhz}" "${xsim_mhz}" "${release_nonfinal}"
do
    if ! [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Expected a non-negative numeric argument, got: ${value}" >&2
        exit 2
    fi
done
if [ "${prompt_tokens}" -lt 1 ] || [ "${layers}" -lt 1 ]; then
    echo "PROMPT_TOKENS and LAYERS must be positive" >&2
    exit 2
fi
if [ "${block_size}" -lt 1 ] || [ "${block_size}" -gt 8 ]; then
    echo "BLOCK_SIZE must be in 1..8" >&2
    exit 2
fi
if [ "${release_nonfinal}" != "0" ] && [ "${release_nonfinal}" != "1" ]; then
    echo "RELEASE_NONFINAL_BLOCKS must be 0 or 1" >&2
    exit 2
fi

host_validation="not_supplied"
if [ -n "${host_log}" ]; then
    if [ ! -s "${host_log}" ]; then
        echo "Missing or empty end-to-end Host log: ${host_log}" >&2
        exit 66
    fi
    e2e_line="$(rg '^QWEN_8X64_E2E_PROFILE ' "${host_log}" | tail -n 1 || true)"
    host_pass_line="$(rg '^QWEN_8X64_HOST PASS ' "${host_log}" | tail -n 1 || true)"
    setup_line="$(rg '^QWEN_8X64_SETUP_PROFILE ' "${host_log}" | tail -n 1 || true)"
    if [ -z "${e2e_line}" ] || [ -z "${host_pass_line}" ] || [ -z "${setup_line}" ]; then
        echo "Host log is missing setup, E2E, or final PASS evidence" >&2
        exit 65
    fi
    field() {
        local line="$1"
        local name="$2"
        awk -v name="${name}" '{
            for (i = 1; i <= NF; i++) {
                split($i, pair, "=")
                if (pair[1] == name) {
                    sub("^[^=]*=", "", $i)
                    print $i
                    exit
                }
            }
        }' <<<"${line}"
    }
    prompt_blocks=$(((prompt_tokens + block_size - 1) / block_size))
    decode_forwards=$((generated_tokens > 0 ? generated_tokens - 1 : 0))
    if [ "${release_nonfinal}" = "1" ]; then
        expected_tasks=$((prompt_blocks * 2 * layers + 1 + decode_forwards * (2 * layers + 1)))
    else
        expected_tasks=$((prompt_blocks * (2 * layers + 1) + decode_forwards * (2 * layers + 1)))
    fi
    expected_groups=$((prompt_blocks + decode_forwards))
    expected_attention_tasks=$((expected_groups * layers))
    expected_ffn_tasks=$((expected_groups * layers))
    expected_query_layer_rows=$(((prompt_tokens + decode_forwards) * layers))
    final_prompt_rows=$((prompt_tokens - (prompt_blocks - 1) * block_size))
    if [ "${release_nonfinal}" = "1" ]; then
        expected_final_norm_tasks=$((1 + decode_forwards))
        expected_final_norm_rows=$((final_prompt_rows + decode_forwards))
        expected_output_materializations=$((1 + decode_forwards))
        expected_released_prompt_blocks=$((prompt_blocks - 1))
    else
        expected_final_norm_tasks=$((prompt_blocks + decode_forwards))
        expected_final_norm_rows=$((prompt_tokens + decode_forwards))
        expected_output_materializations=$((prompt_blocks + decode_forwards))
        expected_released_prompt_blocks=0
    fi

    progress_summary="$(awk \
        -v layers="${layers}" \
        -v block_size="${block_size}" '
        /^COARSE_TASK_PROGRESS / {
            count++
            completed = total = op = query = -1
            phase = ""
            layer = -1
            controller_ms = -1
            for (i = 2; i <= NF; i++) {
                split($i, pair, "=")
                if (pair[1] == "completed") completed = pair[2] + 0
                else if (pair[1] == "total") total = pair[2] + 0
                else if (pair[1] == "op") op = pair[2] + 0
                else if (pair[1] == "phase") phase = pair[2]
                else if (pair[1] == "layer") layer = pair[2] + 0
                else if (pair[1] == "query_tokens") query = pair[2] + 0
                else if (pair[1] == "controller_ms") controller_ms = pair[2] + 0
            }
            if (completed == 1) group_starts++
            if (completed == total) group_ends++
            if (completed < 1 || completed > total ||
                (total != 2 * layers && total != 2 * layers + 1) ||
                query < 1 || query > block_size ||
                controller_ms <= 0) {
                bad++
            }
            if (phase == "attention") {
                attention++
                attention_rows += query
                if (op != 18 || layer < 0 || layer >= layers) bad++
            } else if (phase == "ffn") {
                ffn++
                ffn_rows += query
                if (op != 19 || layer < 0 || layer >= layers) bad++
            } else if (phase == "final_norm") {
                final_norm++
                final_norm_rows += query
                if (op != 20 || layer != 0) bad++
            } else {
                bad++
            }
        }
        END {
            print count + 0, attention + 0, ffn + 0, final_norm + 0, \
                  attention_rows + 0, ffn_rows + 0, final_norm_rows + 0, \
                  group_starts + 0, group_ends + 0, bad + 0
        }
    ' "${host_log}")"
    read -r \
        progress_count attention_count ffn_count final_norm_count \
        attention_rows ffn_rows final_norm_rows \
        group_starts group_ends malformed_progress \
        <<<"${progress_summary}"
    if [ "${progress_count}" -ne "${expected_tasks}" ] ||
       [ "${attention_count}" -ne "${expected_attention_tasks}" ] ||
       [ "${ffn_count}" -ne "${expected_ffn_tasks}" ] ||
       [ "${final_norm_count}" -ne "${expected_final_norm_tasks}" ] ||
       [ "${attention_rows}" -ne "${expected_query_layer_rows}" ] ||
       [ "${ffn_rows}" -ne "${expected_query_layer_rows}" ] ||
       [ "${final_norm_rows}" -ne "${expected_final_norm_rows}" ] ||
       [ "${group_starts}" -ne "${expected_groups}" ] ||
       [ "${group_ends}" -ne "${expected_groups}" ] ||
       [ "${malformed_progress}" -ne 0 ]; then
        echo "Host progress contract mismatch: got '${progress_summary}', expected tasks=${expected_tasks} attention=${expected_attention_tasks} ffn=${expected_ffn_tasks} final_norm=${expected_final_norm_tasks} query_layer_rows=${expected_query_layer_rows} final_norm_rows=${expected_final_norm_rows} groups=${expected_groups} malformed=0" >&2
        exit 65
    fi
    declare -A expected_fields=(
        [prompt_tokens]="${prompt_tokens}"
        [generated_tokens]="${generated_tokens}"
        [prompt_forward_passes]="${prompt_blocks}"
        [generated_forward_passes]="${decode_forwards}"
        [prefill_mode]="coarse_block"
        [prefill_block_size]="${block_size}"
        [coarse_task_count]="${expected_tasks}"
        [output_materializations]="${expected_output_materializations}"
        [released_prompt_blocks]="${expected_released_prompt_blocks}"
        [intermediate_host_copy]="0"
        [kv_cache_owner]="controller"
        [event_timing_domain]="hw_emu_host_wall_proxy"
    )
    for name in "${!expected_fields[@]}"; do
        actual="$(field "${e2e_line}" "${name}")"
        if [ "${actual}" != "${expected_fields[$name]}" ]; then
            echo "Host contract mismatch: ${name}=${actual:-missing}, expected ${expected_fields[$name]}" >&2
            exit 65
        fi
    done
    if [[ "${e2e_line}" != *" PASS" ]]; then
        echo "End-to-end Host profile did not finish with PASS" >&2
        exit 65
    fi
    if [ "$(field "${setup_line}" timing_domain)" != "host_wall" ] ||
       [ "$(field "${setup_line}" weight_preload)" != "1" ]; then
        echo "Host setup evidence does not prove a full weight preload" >&2
        exit 65
    fi
    host_validation="PASS"
fi

active_us="$(awk -F, '$1 == "cc8_ctrl" { print $2; exit }' "${profile_csv}")"
if [ -z "${active_us}" ]; then
    echo "No cc8_ctrl running-time row in ${profile_csv}" >&2
    exit 65
fi

awk \
    -v profile="${model_profile}" \
    -v prompt="${prompt_tokens}" \
    -v generated="${generated_tokens}" \
    -v layers="${layers}" \
    -v block="${block_size}" \
    -v target_mhz="${target_mhz}" \
    -v xsim_mhz="${xsim_mhz}" \
    -v active_us="${active_us}" \
    -v hidden="${hidden}" \
    -v intermediate="${intermediate}" \
    -v kv_channels="${kv_channels}" \
    -v heads="${heads}" \
    -v head_dim="${head_dim}" \
    -v release_nonfinal="${release_nonfinal}" \
    -v host_validation="${host_validation}" '
BEGIN {
    OFS="\t"
    decode_forwards = generated > 0 ? generated - 1 : 0
    query_rows = prompt + decode_forwards
    prompt_blocks = int((prompt + block - 1) / block)
    dense_per_row = \
        2 * hidden * hidden + 2 * hidden * kv_channels + \
        3 * hidden * intermediate
    prompt_context_sum = prompt * (prompt + 1) / 2
    decode_context_sum = \
        decode_forwards * ((prompt + 1) + (prompt + decode_forwards)) / 2
    attention_per_context = 2 * heads * head_dim
    useful_mac = layers * ( \
        query_rows * dense_per_row + \
        (prompt_context_sum + decode_context_sum) * attention_per_context \
    )
    prompt_tasks = release_nonfinal ? \
        prompt_blocks * 2 * layers + 1 : \
        prompt_blocks * (2 * layers + 1)
    decode_tasks = decode_forwards * (2 * layers + 1)
    tasks = prompt_tasks + decode_tasks
    cycles = active_us * xsim_mhz
    target_us = cycles / target_mhz
    throughput = target_mhz / 1000.0 * useful_mac / cycles
    efficiency = 100.0 * useful_mac / (cycles * 1024)
    query_row_s = query_rows * target_mhz * 1000000.0 / cycles
    generated_token_s = generated > 0 ? \
        generated * target_mhz * 1000000.0 / cycles : 0
    print "evidence_source", "timed_scope", "host_validation", "profile", "prompt_tokens", \
          "generated_tokens", "decode_forwards", "layers", "block_size", \
          "release_nonfinal_blocks", "expected_coarse_tasks", \
          "xsim_active_us", "xsim_clock_mhz", "xsim_cycles", \
          "target_clock_mhz", "projected_target_us", "useful_mac", \
          "useful_gmac_s", "physical_efficiency_percent", \
          "query_row_s", "generated_token_s"
    printf "HW_Emu_CU_trace\tkernel_active_only\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.1f\t%.3f\t%.4f\t%.0f\t%.6f\t%.6f\t%.3f\t%.3f\n", \
        host_validation, profile, prompt, generated, decode_forwards, layers, block, \
        release_nonfinal, tasks, active_us, xsim_mhz, cycles, target_mhz, \
        target_us, useful_mac, throughput, efficiency, query_row_s, \
        generated_token_s
}
'
