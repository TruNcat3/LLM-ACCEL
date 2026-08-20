# Q2.14 P/D length sweep artifact

This directory freezes the data behind the Q2.14 hardware-emulation table in
[`docs/q214-pd-length-hwemu.md`](../../docs/q214-pd-length-hwemu.md).

## Configuration

| Item | Value |
| --- | --- |
| Toolflow | Vitis/XRT 2022.2 hardware emulation |
| Modeled clock | 200 MHz |
| Layer shape | hidden 2048, intermediate 11008, 16 Q heads, 2 KV heads, head dimension 128 |
| Compute | two 8x64 CUs, 1024 MAC/cycle peak |
| Arithmetic | Fix16 features/weights and Q2.14 attention probabilities |
| Seed | 20260722 |

Prefill rows measure the final 8-row query block at each context. Decode rows
measure one active query row. The cycle count is the sum of hardware controller
active intervals for the operator calls composing the layer; it excludes host
scheduling, transfer gaps, CPU test-fixture work, and golden checks.

## Files

- `performance.tsv`: derived latency, useful work, throughput, and modeled
  useful-MAC efficiency against the two-CU peak. Release 0.6 renames the
  historical `physical_efficiency_pct` and `token_s` headers to
  `modeled_interval_efficiency_percent` and `target_block_query_row_s`;
  the frozen numeric values are unchanged and are not physical-board results.
- `precision.tsv`: Q2.14 bit-accurate and high-precision comparison counts.
- `raw/{p,d}{64,256,512,1024}_profile_kernels.csv`: XSim kernel profiles used
  to derive active cycles.
- `raw/{p,d}{64,256,512,1024}_host.txt`: functional and protocol checks.

The raw profiles are retained because hardware-emulation CPU wall time is not
accelerator latency. The `cc8_ctrl` running-time field, multiplied by 200 MHz,
is the authoritative cycle source for each case.

Reproduce both derived TSV files directly from the frozen Host logs and CU
profiles with `make verify_q214_pd_release`.
