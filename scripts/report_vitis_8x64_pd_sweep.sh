#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 RESULTS_TSV [RESULTS_TSV ...]" >&2
    exit 2
fi

for results_tsv in "$@"; do
    if [ ! -s "${results_tsv}" ]; then
        echo "Missing or empty results TSV: ${results_tsv}" >&2
        exit 66
    fi
done

# Qwen2.5-3B layer dimensions used by qwen-layer-long:
#   Q/O: HxH, K/V: HxKV, Gate/Up/Down: HxI, HxI, IxH.
# The attention term counts useful QK and PV MACs for the causal contexts
# actually processed by the measured final P block or D token.
awk -F '\t' '
BEGIN {
    OFS="\t"
    hidden=2048
    intermediate=11008
    kv_channels=256
    heads=16
    head_dim=128
    projection_macs_per_token = \
        2 * hidden * hidden + 2 * hidden * kv_channels + \
        3 * hidden * intermediate
    attention_macs_per_context = 2 * heads * head_dim
    peak_macs_per_cycle = 2 * 8 * 64
    print "phase", "context_len", "active_query_rows_per_block", "cycles_200mhz", \
          "latency_ms", "useful_macs", "throughput_gmac_s", \
          "modeled_interval_efficiency_percent", \
          "target_block_query_row_s"
}
FNR == 1 { next }
$6 == 0 && $8 ~ /^[0-9]+$/ {
    phase=$1
    context=$2 + 0
    start=$3 + 0
    end=$4 + 0
    tokens=$5 + 0
    cycles=$8 + 0
    # Each query at zero-based position p attends to p+1 KV entries.
    first_context=start + 1
    last_context=end
    sum_contexts=tokens * (first_context + last_context) / 2
    useful_macs=tokens * projection_macs_per_token + \
                sum_contexts * attention_macs_per_context
    latency_ms=cycles / 200000.0
    throughput=0.2 * useful_macs / cycles
    efficiency=100.0 * useful_macs / (cycles * peak_macs_per_cycle)
    token_s=tokens * 200000000.0 / cycles
    printf "%s\t%d\t%d\t%d\t%.6f\t%.0f\t%.3f\t%.3f\t%.3f\n", \
        phase, context, tokens, cycles, latency_ms, useful_macs, \
        throughput, efficiency, token_s
}
' "$@"
