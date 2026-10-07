# Quantized full-layer development study

[Documentation](README.md) | [Implementation catalog](implementations.md) |
[Quantized design](designs/quantized.md) | [Experiments](experiments.md) |
[Archived evidence](../results/quantized-layer-w4-20260930/)

This September 30, 2026 snapshot reports a completed W4A4 full-layer comparison.
The archive includes RTL transitions, Host logs, numerical dumps, derived
metrics and the original analysis tools. The public full-layer source is now
available in [`cases/quantized-layer/`](../cases/quantized-layer/), with its
own README, source manifest, and build/evidence identity. That source is not
the frozen source snapshot recorded by this result package and does not inherit
these measurements. The executable [quantized matrix-block case](../cases/quantized-block/README.md)
remains a separate component study.

For current source and current evidence, start with the
[quantized design entry](designs/quantized.md), the
[public source README](../cases/quantized-layer/README.md), and the
[`quantized-layer-20261007` evidence package](../results/quantized-layer-20261007/).
The sections below intentionally retain the September 30 values and historical
feature labels for reproducibility; they are not a status report for the new
source tree.

The current package records complete-layer P66+D1 rows for both canonical
profiles and both named configurations. The base/wave rows are W4A4
2,288,540/187,844 and 2,284,511/187,557 Prefill/Decode cycles; W8A8
3,592,954/187,220 and 3,582,758/186,332. Read the package manifest for the
configuration flags and source identity. These are package measurements, not
board timing or a full-model projection.

## Design and configuration map

The development system retains CoWave's control/compute separation: a
model-aware controller manages HBM weights, hidden buffers and KV state, then
issues bounded matrix/vector tasks to four compute CUs. Transformer subgraphs
execute on the accelerator; the Host supplies the workload and checks outputs
after inference. W4A4 describes quantized matrix operands, not every stored
tensor or nonlinear operator's arithmetic.

| Configuration | Archived profile | Added mechanism |
| --- | --- | --- |
| Baseline | `legacy` | Existing full-layer controller and quantized compute path; reference for this comparison |
| Attention | `attention` | Valid-tail handling, packed K/V tile prefetch, probability fusion and parallel attention services |
| Attention + block pipeline | `pipeline` | Block overlap and asynchronous compute dispatch on the Attention configuration |
| Integrated decode | `integrated` | Merged decode row handling, shared/wide data paths and FFN overlap on the pipeline configuration |

Baseline is an established implementation with prior optimizations. It is not
an unoptimized mathematical reference. These names identify feature bundles,
not four independent arithmetic formats or software release numbers. The
separate Attention wave-pipeline option is disabled in all four profiles here.

## Matched workload and results

Each run executes one standard-shape decoder layer with a 66-token prompt
followed by one real single-row decode at position 66. There is one sequence,
eight physical rows per W4 block, nine prefill blocks including the two-row
tail, and a 67-entry decode KV context. Dimensions are H=2048, I=11008,
16 query heads, two KV heads and head dimension 128. Weights are deterministic
random fixtures.

| Configuration | Prefill cycles | Decode cycles | P + D cycles | Prefill / Decode useful-MAC efficiency |
| --- | ---: | ---: | ---: | --- |
| Baseline | 3,100,205 | 226,084 | 3,326,289 | 40.129% / 8.352% |
| Attention | 2,839,116 | 212,221 | 3,051,337 | 43.819% / 8.898% |
| Attention + block pipeline | 2,664,407 | 212,388 | 2,876,795 | 46.692% / 8.891% |
| Integrated decode | 2,647,038 | 204,385 | 2,851,423 | 46.998% / 9.239% |

Integrated reduces P+D cycles by **14.28%** against Baseline and **0.88%**
against Attention + block pipeline. At a modeled 200 MHz, its Prefill/Decode
latencies are 13.235 ms / 1.022 ms for this one layer. These are not full-model
TTFT or generation token rates. All four configurations compare 171,520 hidden
and retained K/V values exactly against their matching production C reference,
with zero raw tolerance and zero differing values.

Cycles come from the saved controller/four-compute-CU RTL `ap_idle` transitions
and the observed 3,334 ps clock. Each phase spans the earliest CU start through
the latest CU end; P+D sums the two intervals and excludes the Host gap.
The archived package declares **4,096 logical MAC/cycle** for this historical
profile. That denominator is a package contract; it must not be rederived by
counting a DSP packing factor twice or used as the canonical profile peak for
the newer source.
It includes time spent waiting and performing non-matrix work in the interval,
and is neither PE occupancy nor a physical resource-utilization measurement.
Host intermediate arithmetic is absent from these runs; the post-inference
C reference is not an independent arithmetic oracle or trained-model accuracy
test. Fix16's published P8 workloads and 1,024-MAC/cycle peak have a different
shape and denominator, so their percentages do not form a matched comparison.

## Follow-up experiments

The completed comparison motivates service-rate experiments rather than an
assumption that more logical MAC lanes alone increase end-to-end throughput.

| Experiment | Question | Status at this snapshot |
| --- | --- | --- |
| Same four profiles in W8A8 | How does the complete schedule scale with precision and row shape? | At this snapshot, baseline in full-layer HW Emu and matched matrix incomplete; current package records the W8 rows |
| AXI outstanding 16 to 32 | Can additional memory requests better sustain the pipeline? | W4 finite-FIFO CoSim passed with deadlock detection; full-layer HW Emu running |
| SiLU two/four lanes | Does nonlinear service rate limit matrix/vector overlap? | At this snapshot, matched full-layer comparisons pending; current package records the selected SiLU4 rows |
| RMS tree reduction, two/four lanes | Can parallel normalization reduce cycles within the HLS timing budget? | Standalone RTL checks completed; integrated reference running before lane comparisons |
| Attention wave pipeline with the combined changes | Do component gains survive their scheduling dependencies? | At this snapshot, comparison pending; current package records the selected `-wave` rows |

No completed full-layer speedup is assigned to these follow-ups. Resource
estimates remain HLS evidence; actual utilization and timing will be checked
with hardware compilation after the performance configuration is selected.
HW Emu alone does not supply routed timing or FPGA resource mapping.

## Reanalyze the evidence

```bash
bash results/quantized-layer-w4-20260930/verify.sh
```

Run this from the repository root. It checks package integrity and rebuilds
the performance/numerical reports from the archived inputs with Python's
standard library. It launches no simulator. The
[package README](../results/quantized-layer-w4-20260930/) records provenance and
the source-release boundary. Hardware rebuild recipes will accompany the
selected full-layer source release through the [release workflow](release-workflow.md).
