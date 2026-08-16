# Coarse-Task Resident Runtime Evidence

This directory freezes the Vitis 2022.2 validation of the controller-resident
Task 18/19/20 runtime described in
[`docs/coarse-task-runtime.md`](../../docs/coarse-task-runtime.md).

The runtime invariants are:

- Task 18 owns Attention RMS/QKV/RoPE/KV/online-softmax/O/residual;
- Task 19 owns FFN RMS/Gate/Up/SiLU-Mul/Down/residual;
- Task 20 owns final RMSNorm;
- hidden state remains in HBM between tasks and decoder layers;
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

Files:

- `performance.tsv`: normalized cycle and correctness summary;
- `resource.tsv`: baseline/coarse-task controller and system estimates;
- `raw/`: the authoritative RTL CoSim report, host validation excerpts, and
  CU profile tables;
- `checksums.sha256`: hashes for the frozen evidence in this directory.
