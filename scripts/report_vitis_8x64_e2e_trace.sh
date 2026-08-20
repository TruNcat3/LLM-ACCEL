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
xsim_mhz="${8:-${target_mhz}}"
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
numeric_validation="not_supplied"
numeric_steps=0
numeric_checked_values=0
numeric_max_raw_error=0
numeric_tolerance=0
numeric_golden_host_ms=0
numeric_validation_schedule="not_supplied"
host_inference_ms=0
host_process_ms=0
host_lm_head_ms=0
host_validation_ms=0
artifact_identity="not_recorded"
host_exe_sha256="NA"
xclbin_sha256="NA"
emconfig_sha256="NA"
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
    host_inference_ms="$(field "${e2e_line}" total_host_elapsed_ms)"
    host_process_ms="$(field "${e2e_line}" total_process_elapsed_ms)"
    host_lm_head_ms="$(field "${e2e_line}" lm_head_host_ms)"
    host_validation_ms="$(field "${e2e_line}" post_inference_validation_ms)"
    for host_time in \
        "${host_inference_ms}" "${host_process_ms}" \
        "${host_lm_head_ms}" "${host_validation_ms}"
    do
        if ! [[ "${host_time}" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
            echo "Malformed Host timing field: ${host_time:-missing}" >&2
            exit 65
        fi
    done
    if ! awk \
        -v inference="${host_inference_ms}" \
        -v process="${host_process_ms}" \
        'BEGIN { exit process + 0.001 >= inference ? 0 : 1 }'
    then
        echo "Host process time is shorter than inference-only time" >&2
        exit 65
    fi
    numeric_enabled="$(field "${e2e_line}" e2e_numeric_golden)"
    case "${numeric_enabled}" in
        0)
            numeric_validation_schedule="$(field "${e2e_line}" validation_schedule)"
            if [ "${numeric_validation_schedule}" != "disabled" ]; then
                echo "Host profile did not mark numerical validation disabled" >&2
                exit 65
            fi
            numeric_validation="not_requested"
            ;;
        1)
            numeric_line="$(rg '^QWEN_8X64_E2E_NUMERIC_VERIFY ' "${host_log}" | tail -n 1 || true)"
            if [ -z "${numeric_line}" ] ||
               [[ "${numeric_line}" != *" token_sequence_match=1 "*" PASS" ]]; then
                echo "Host log is missing a passing E2E numerical summary" >&2
                exit 65
            fi
            numeric_steps="$(field "${numeric_line}" steps)"
            numeric_checked_values="$(field "${numeric_line}" checked_values)"
            numeric_max_raw_error="$(field "${numeric_line}" max_raw_error)"
            numeric_tolerance="$(field "${numeric_line}" tolerance)"
            numeric_golden_host_ms="$(field "${numeric_line}" golden_host_ms)"
            numeric_validation_schedule="$(field "${numeric_line}" validation_schedule)"
            profile_validation_schedule="$(field "${e2e_line}" validation_schedule)"
            expected_numeric_values=$((generated_tokens * hidden))
            numeric_step_lines="$(
                rg -c '^QWEN_8X64_E2E_NUMERIC_STEP .* PASS$' \
                    "${host_log}" || true
            )"
            for numeric_value in \
                "${numeric_steps}" "${numeric_checked_values}" \
                "${numeric_max_raw_error}" "${numeric_tolerance}"
            do
                if ! [[ "${numeric_value}" =~ ^[0-9]+$ ]]; then
                    echo "Malformed E2E numerical field: ${numeric_value}" >&2
                    exit 65
                fi
            done
            if ! [[ "${numeric_golden_host_ms}" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] ||
               [ "${numeric_steps}" -ne "${generated_tokens}" ] ||
               [ "${numeric_step_lines}" -ne "${generated_tokens}" ] ||
               [ "${numeric_checked_values}" -ne "${expected_numeric_values}" ] ||
               [ "${numeric_max_raw_error}" -gt "${numeric_tolerance}" ] ||
               [ "${numeric_validation_schedule}" != "post_inference" ] ||
               [ "${profile_validation_schedule}" != "post_inference" ]; then
                echo "E2E numerical contract mismatch" >&2
                exit 65
            fi
            if ! awk \
                -v validation="${host_validation_ms}" \
                -v golden="${numeric_golden_host_ms}" \
                'BEGIN { exit validation + 0.001 >= golden ? 0 : 1 }'
            then
                echo "Post-inference Host timing does not contain the recorded oracle time" >&2
                exit 65
            fi
            numeric_validation="PASS"
            ;;
        *)
            echo "Host profile has invalid e2e_numeric_golden=${numeric_enabled:-missing}" >&2
            exit 65
            ;;
    esac
    if [ "$(field "${setup_line}" timing_domain)" != "host_wall" ] ||
       [ "$(field "${setup_line}" weight_preload)" != "1" ]; then
        echo "Host setup evidence does not prove a full weight preload" >&2
        exit 65
    fi
    artifact_sha() {
        local name="$1"
        awk -F= -v name="${name}" '
            $1 == name { value = $2 }
            END { print value }
        ' "${host_log}"
    }
    host_exe_sha256="$(artifact_sha host_exe_sha256)"
    xclbin_sha256="$(artifact_sha xclbin_sha256)"
    emconfig_sha256="$(artifact_sha emconfig_sha256)"
    recorded_artifacts=0
    for digest in \
        "${host_exe_sha256}" "${xclbin_sha256}" "${emconfig_sha256}"
    do
        if [ -n "${digest}" ]; then
            recorded_artifacts=$((recorded_artifacts + 1))
        fi
    done
    if [ "${recorded_artifacts}" -ne 0 ]; then
        if [ "${recorded_artifacts}" -ne 3 ]; then
            echo "Host log contains an incomplete generated-artifact identity set" >&2
            exit 65
        fi
        for digest in \
            "${host_exe_sha256}" "${xclbin_sha256}" "${emconfig_sha256}"
        do
            if ! [[ "${digest}" =~ ^[0-9a-fA-F]{64}$ ]]; then
                echo "Host log contains a malformed SHA-256 identity: ${digest}" >&2
                exit 65
            fi
        done
        artifact_identity="PASS"
    else
        host_exe_sha256="NA"
        xclbin_sha256="NA"
        emconfig_sha256="NA"
    fi
    host_validation="PASS"
fi

cu_time_summary="$(
    awk -F, '
        NF == 5 &&
        ($1 == "cc8_ctrl" || $1 == "cc8_cu0" ||
         $1 == "cc8_cu1" || $1 == "cc8_status") {
            count[$1]++
            time[$1] = $2
        }
        END {
            print count["cc8_ctrl"] + 0,
                  count["cc8_cu0"] + 0,
                  count["cc8_cu1"] + 0,
                  count["cc8_status"] + 0,
                  time["cc8_ctrl"], time["cc8_cu0"],
                  time["cc8_cu1"], time["cc8_status"]
        }
    ' "${profile_csv}"
)"
read -r ctrl_rows cu0_rows cu1_rows status_rows \
    active_us cu0_us cu1_us status_us <<<"${cu_time_summary}"
if [ "${ctrl_rows}" -ne 1 ] || [ "${cu0_rows}" -ne 1 ] ||
   [ "${cu1_rows}" -ne 1 ] || [ "${status_rows}" -ne 1 ]; then
    echo "HW-Emu CU profile must contain exactly one top-level running-time row for each deployed CU" >&2
    exit 65
fi
if ! awk \
    -v ctrl="${active_us}" -v cu0="${cu0_us}" \
    -v cu1="${cu1_us}" -v status="${status_us}" '
    function abs(value) { return value < 0 ? -value : value }
    BEGIN {
        exit abs(ctrl - cu0) <= 0.001 &&
             abs(ctrl - cu1) <= 0.001 &&
             abs(ctrl - status) <= 0.001 ? 0 : 1
    }
'; then
    echo "HW-Emu CU running times are not a common four-CU measurement: ${cu_time_summary}" >&2
    exit 65
fi

# A completed coarse task must at least write its hidden-state result.  This
# lower bound intentionally ignores every read, matrix operation, stream
# transfer, and control cycle: even an ideal 512-bit HBM write port can retire
# no more than 32 Fix16 values per cycle.  Reject shorter traces instead of
# turning a truncated XSim profile into an impossible efficiency result.
if [ "${host_validation}" = "PASS" ]; then
    minimum_output_cycles="$((
        (attention_rows + ffn_rows + final_norm_rows) * hidden / 32
    ))"
    trace_is_plausible="$(awk \
        -v active_us="${active_us}" \
        -v xsim_mhz="${xsim_mhz}" \
        -v minimum="${minimum_output_cycles}" \
        'BEGIN { print (active_us * xsim_mhz >= minimum) ? 1 : 0 }'
    )"
    if [ "${trace_is_plausible}" != "1" ]; then
        observed_cycles="$(awk \
            -v active_us="${active_us}" \
            -v xsim_mhz="${xsim_mhz}" \
            'BEGIN { printf "%.1f", active_us * xsim_mhz }'
        )"
        echo "Truncated or invalid HW-Emu CU trace: observed ${observed_cycles} cycles, below the output-only lower bound ${minimum_output_cycles}." >&2
        exit 65
    fi
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
    -v host_validation="${host_validation}" \
    -v numeric_validation="${numeric_validation}" \
    -v numeric_steps="${numeric_steps}" \
    -v numeric_checked_values="${numeric_checked_values}" \
    -v numeric_max_raw_error="${numeric_max_raw_error}" \
    -v numeric_tolerance="${numeric_tolerance}" \
    -v numeric_golden_host_ms="${numeric_golden_host_ms}" \
    -v numeric_validation_schedule="${numeric_validation_schedule}" \
    -v host_inference_ms="${host_inference_ms}" \
    -v host_process_ms="${host_process_ms}" \
    -v host_lm_head_ms="${host_lm_head_ms}" \
    -v host_validation_ms="${host_validation_ms}" \
    -v artifact_identity="${artifact_identity}" \
    -v host_exe_sha256="${host_exe_sha256}" \
    -v xclbin_sha256="${xclbin_sha256}" \
    -v emconfig_sha256="${emconfig_sha256}" '
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
    print "evidence_source", "timed_scope", "host_validation", \
          "numeric_validation", "numeric_steps", \
          "numeric_checked_values", "numeric_max_raw_error", \
          "numeric_tolerance", "numeric_golden_host_ms", \
          "numeric_validation_schedule", "host_inference_ms", \
          "host_process_ms", "host_lm_head_ms", "host_validation_ms", \
          "artifact_identity", "host_exe_sha256", "xclbin_sha256", \
          "emconfig_sha256", "profile", "sequence_batch", \
          "prompt_sequence_tokens", "sampled_output_tokens", \
          "decode_forwards", "prefill_blocks", "layers", \
          "configured_max_active_query_rows_per_prefill_block", \
          "release_nonfinal_blocks", "expected_coarse_tasks", \
          "xsim_cu_running_us", "xsim_clock_mhz", "xsim_cycles", \
          "target_clock_mhz", "projected_target_us", "useful_mac", \
          "useful_gmac_s", "modeled_interval_efficiency_percent", \
          "query_row_s", "request_output_token_s"
    printf "HW_Emu_CU_trace\tcommon_four_CU_running_time\t%s\t%s\t%d\t%d\t%d\t%d\t%.3f\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.1f\t%.3f\t%.4f\t%.0f\t%.6f\t%.6f\t%.3f\t%.3f\n", \
        host_validation, numeric_validation, numeric_steps, \
        numeric_checked_values, numeric_max_raw_error, numeric_tolerance, \
        numeric_golden_host_ms, numeric_validation_schedule, \
        host_inference_ms, host_process_ms, host_lm_head_ms, \
        host_validation_ms, \
        artifact_identity, host_exe_sha256, xclbin_sha256, \
        emconfig_sha256, profile, 1, prompt, generated, decode_forwards, \
        prompt_blocks, layers, block, \
        release_nonfinal, tasks, active_us, xsim_mhz, cycles, target_mhz, \
        target_us, useful_mac, throughput, efficiency, query_row_s, \
        generated_token_s
}
'
