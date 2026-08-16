# Coarse-Task Resident Runtime Evidence

This directory freezes the Vitis 2022.2 validation of the controller-resident
Task 18/19/20 runtime described in
[`docs/coarse-task-runtime.md`](../../docs/coarse-task-runtime.md).

The runtime invariants are:

- Task 18 owns Attention RMS/QKV/RoPE/KV/online-softmax/O/residual;
- Task 19 owns FFN RMS/Gate/Up/SiLU-Mul/Down/residual;
- Task 20 owns final RMSNorm;
- hidden state remains in HBM between tasks and decoder layers;
- Norm/RoPE tables are initialized once and tasks perform no auxiliary
  migration;
- the host reads only task status until the final hidden migration;
- `intermediate_host_copy=0` and the CPU fixed-point golden must pass.

`small` is a two-layer contract profile with 64 hidden values. `qwen-layer`
uses the Qwen2.5-3B layer dimensions (`hidden=2048`,
`intermediate=11008`, 16 query heads, 2 KV heads, head dimension 128).

The authoritative HW-Emu cycle source is the run-local
`profile_kernels.csv` combined with the generated XSim kernel clock. The U50
HW-Emu image uses 300 MHz (3.333 ns), even though the physical implementation
target is 200 MHz. `performance.tsv` records the XSim interval, derives cycles
at 300 MHz, and projects those cycles to 200 MHz. OpenCL event durations and
host elapsed time are simulator wall-time proxies under HW Emu.

The persistent-auxiliary revision passes a one-layer Task-18/19 run, a
two-layer/five-task stack, and a real `generate --coarse-tasks` composition.
The generation case uses a two-token serial prompt and three sampled tokens.
It performs four decoder forwards, 20 tasks, and reports
`kv_cache_owner=controller`, `auxiliary_migration_ms=0`, and
`intermediate_host_copy=0`. Embedding and LM-head/sampling remain host
operations; the result is a protocol/residency test, not full-size throughput.

Compared with the previous small-profile baseline, persistent state changes
the one-layer interval from 14,579.4 to 14,299.5 cycles (-1.92%) and the
two-layer interval from 29,017.2 to 28,742.7 cycles (-0.95%). The standard
controller HLS resource change is effectively neutral: +65 FF, -154 LUT, and
no BRAM/DSP or estimated-period change.

The full 36-layer host plan also passes its grouped-HBM capacity guard. Its
largest paired-shard-plus-KV allocation is 769,130,496 bytes in an
805,306,368-byte three-pseudo-channel group, leaving 36,175,872 bytes. The
exact plan output is frozen in `raw/qwen2.5_3b_execution_plan.txt`.

Files:

- `performance.tsv`: normalized cycle and correctness summary;
- `resource.tsv`: baseline, coarse-task, and persistent controller/system
  estimates;
- `raw/`: the authoritative RTL CoSim report, host validation excerpts, and
  CU profile tables, plus the persistent standard-controller HLS summary and
  full-profile execution plan;
- `checksums.sha256`: hashes for the frozen evidence in this directory.
