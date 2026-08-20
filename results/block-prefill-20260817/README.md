# Controller-Resident Block-Prefill Evidence

This artifact freezes the Vitis HLS 2022.2 and Vitis 2022.2 HW-Emu evidence
for the 1--8-row controller-resident coarse-task runtime.

`query_tokens=8` means eight consecutive positions from one sequence in one
prefill block. It is neither batch size eight nor eight decoded outputs, and it
does not require the complete prompt to contain only eight tokens. The Host
chunks a longer prompt into blocks of at most eight rows;
decode remains a one-row task.

The host-visible program is:

```text
for each prompt block and decoder layer:
  Task 18  Attention sublayer, block rows 1..8, HBM KV owned by controller
  Task 19  FFN sublayer, same block rows
Task 20    final RMSNorm
```

Task 18 expands inside the controller into RMSNorm, Q/K/V projections, per-row
RoPE, causal KV append/read, tiled QK, Q2.14 online softmax/PV, O projection,
and residual. Task 19 owns RMSNorm, Gate/Up, SiLU-Mul, Down, and residual.
Intermediate hidden tensors remain in two HBM feature-buffer pairs and are not
copied to the CPU between tasks or layers.

The evidence classes are intentionally separate:

- `validation.tsv` records CSim, finite-FIFO RTL CoSim, and HW-Emu correctness;
- `performance.tsv` contains only run-local HW-Emu CU intervals. Its
  `evidence_source`, `timed_scope`, and `active_query_rows_per_block` columns make
  the simulator source, Host-exclusion boundary, and block height explicit.
  Cycles are derived from the generated 300-MHz XSim clock and projected to a
  200-MHz implementation target;
- `resource.tsv` records local HLS synthesis estimates, not routed resources;
- `raw/` contains the frozen reports, host excerpts, and CU profiles used to
  derive the tables.

The historical `timed_scope=kernel_active_only` value in `performance.tsv` is
retained to preserve the published row. Its source profiles report the same
Running Time for the controller, both compute CUs, and the status sink, so the
current, narrower interpretation is a common four-CU modeled interval. The CSV
does not resolve separable per-CU occupancy or inter-task issue gaps.

The Small profile is a protocol/residency contract and not a Qwen throughput
proxy. Its low MAC efficiency is expected because 64-wide matrices do not
fill the standard 8x64/large-K datapath and the persistent interval includes
coarse-task boundaries. The pending standard `qwen-layer` numeric gate is the
relevant single-layer performance measurement and is not inferred from these
Small rows.

The two-layer P8 stack closes both dimensions of the residency contract at
once: one 8-row block traverses Task 18 and Task 19 for layers 0 and 1, followed
by one Task 20. All 512 final values pass the CPU fixed-point check with maximum
raw error 10 within tolerance 64. Its 185.655-us 300-MHz CU interval is 55,696.5
cycles, or 278.483 us when projected to 200 MHz. This is 11.5% fewer cycles than
two independent one-layer P8 runs because the stack retains state and amortizes
the final boundary.

Three generation contracts are frozen. P8/G2 executes one 8-row prompt forward
and one one-row decode forward. P16/G1 executes prompt blocks 0--7 and 8--15;
the second block consumes the controller-owned KV state left by the first and
therefore validates cross-block cache residency and forward progress. It is a
protocol check rather than a cross-block CPU-golden comparison. The final
sampled token in P16/G1 needs no additional decode forward. P11/G1 executes
an 8-row full block followed by a 3-row tail block, proving that `8` is the
configured maximum block height rather than a fixed token count.

The stronger P16 sequence gate performs a cross-block CPU-golden comparison,
not only a protocol check. It executes nine tasks: Task 18/19 for both layers
of the first block, Task 18/19 for both layers of the second block, and one
final Task 20. The first block is released after its KV updates with no D2H;
only the final 512-value tail is materialized. All values pass with maximum raw
error 10 within tolerance 64. The 357.259-us CU interval corresponds to
107,177.7 XSim cycles and 535.889 us when projected to 200 MHz.

Embedding and LM-head argmax/sampling remain Host operations in generation
mode. CPU fixed-point golden arithmetic is validation-only and is excluded
from useful hardware work. The decoder sublayers, final normalization,
intermediate residency, RoPE, online attention, and KV-cache updates execute
through the accelerator tasks.

All numbers are hardware-emulation or HLS estimates; none is a physical-board
measurement.
