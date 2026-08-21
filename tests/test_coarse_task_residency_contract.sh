#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")/.."

host_body="$({
    sed -n \
        '/composed_layer_result_t run_composed_decoder_stack(/,/decoded_status_t run_mm_wave_profile(/p' \
        host/host_qwen_8x64.cpp
})"
controller_body="$({
    sed -n \
        '/const bool resident_layer_task =/,/#if CC8_RESIDENT_LAYER_ONLY/p' \
        kernel/control_cache_8x64.cpp
})"
controller_wrapper="$(< kernel/control_cache_8x64_nk.cpp)"
initial_state_body="$({
    sed -n \
        '/void migrate_initial_model_state(/,/void ensure_persistent_auxiliary_state(/p' \
        host/host_qwen_8x64.cpp
})"

require_text() {
    local body="$1"
    local text="$2"
    local description="$3"
    if ! rg -F -q -- "${text}" <<<"${body}"; then
        echo "Missing coarse-task residency invariant: ${description}" >&2
        exit 65
    fi
}

count_text() {
    local body="$1"
    local text="$2"
    awk -v text="${text}" '
        {
            line = $0
            while ((position = index(line, text)) != 0) {
                count++
                line = substr(line, position + length(text))
            }
        }
        END { print count + 0 }
    ' <<<"${body}"
}

if [ -z "${host_body}" ] || [ -z "${controller_body}" ] ||
   [ -z "${controller_wrapper}" ] || [ -z "${initial_state_body}" ]; then
    echo "Cannot extract the coarse-task implementation bodies" >&2
    exit 66
fi

# The Host may migrate the initial embedding block to the device and the final
# materialized hidden state back.  It must not migrate a Task-18 intermediate
# before Task 19 consumes the same HBM-resident buffer pair.
if [ "$(count_text "${host_body}" 'CL_MIGRATE_MEM_OBJECT_HOST')" -ne 1 ] ||
   [ "$(count_text "${host_body}" 'execute_bound_resident_task(')" -ne 3 ] ||
   [ "$(rg -c '^[[:space:]]*pack_feature\(' <<<"${host_body}")" -ne 1 ] ||
   [ "$(count_text "${host_body}" 'unpack_feature(')" -ne 1 ]; then
    echo "Host coarse-task migration or execution count regressed" >&2
    exit 65
fi
if rg -q 'kv_cache_[kv]_(buffer|words)' <<<"${host_body}"; then
    echo "Host coarse-task path must not migrate or materialize KV state" >&2
    exit 65
fi
require_text "${host_body}" \
    'bind_controller_data_ports(0, 1, 2, 3);' \
    'Task 18 reads input pair 2/3 and writes pair 0/1'
require_text "${host_body}" \
    'bind_controller_data_ports(2, 3, 0, 1);' \
    'Task 19 consumes the Task-18 HBM result by buffer rebinding'
require_text "${host_body}" \
    'if (materialize_output) {' \
    'D2H is conditional on an explicitly materialized final output'
require_text "${host_body}" \
    'result.output_migration_ms =' \
    'only the final materialization is profiled as an output migration'

attention_line="$(rg -n -F 'kOpAttentionSublayer,' <<<"${host_body}" | head -n 1 | cut -d: -f1)"
ffn_line="$(rg -n -F 'kOpFfnSublayer,' <<<"${host_body}" | head -n 1 | cut -d: -f1)"
d2h_line="$(rg -n -F 'CL_MIGRATE_MEM_OBJECT_HOST' <<<"${host_body}" | cut -d: -f1)"
if [ -z "${attention_line}" ] || [ -z "${ffn_line}" ] ||
   [ -z "${d2h_line}" ] ||
   [ "${attention_line}" -ge "${ffn_line}" ] ||
   [ "${ffn_line}" -ge "${d2h_line}" ]; then
    echo "Host coarse-task execution/migration ordering regressed" >&2
    exit 65
fi

# Each Host-visible task is a controller-resident subgraph, not an operator
# callback loop.  These sites jointly cover on-chip hidden/wide storage,
# projection dispatch, RoPE, controller-owned KV/online attention, vector
# post-processing, and one HBM boundary store.
for invariant in \
    'op == CC8_OP_ATTENTION_SUBLAYER' \
    'op == CC8_OP_FFN_SUBLAYER' \
    'load_cc8_feature_gbuf(' \
    'run_cc8_projection_waves_banked(' \
    'apply_cc8_rope_from_aux(' \
    'run_cc8_resident_decode_attention(' \
    'run_cc8_vector_gbuf_task_inplace_binary(' \
    'store_cc8_gbuf_to_hbm('
do
    require_text "${controller_body}" "${invariant}" "${invariant}"
done

# KV is initialized in HBM once and then remains controller-owned throughout
# the coarse task sequence.  The task body above must never name these Host
# buffers, and the controller wrapper must retain two explicit AXI masters.
require_text "${initial_state_body}" \
    'buffers.push_back(kv_cache_k_buffer_);' \
    'initial K-cache migration'
require_text "${initial_state_body}" \
    'buffers.push_back(kv_cache_v_buffer_);' \
    'initial V-cache migration'
if [ "$(count_text "${initial_state_body}" 'enqueueMigrateMemObjects(')" -ne 1 ]; then
    echo "Persistent model state must use one initialization migration" >&2
    exit 65
fi
require_text "${controller_wrapper}" \
    '#pragma HLS interface m_axi port=kv_cache_k' \
    'controller K-cache HBM master'
require_text "${controller_wrapper}" \
    '#pragma HLS interface m_axi port=kv_cache_v' \
    'controller V-cache HBM master'

printf 'COARSE TASK RESIDENCY CONTRACT PASS '
printf 'host_d2h_sites=1 resident_tasks=3 task18_to_task19=HBM_rebind '
printf 'kv_task_migrations=0 kv_init_migrations=1 kv_axi_ports=2 '
printf 'controller_subgraphs=attention,ffn,final_norm\n'
