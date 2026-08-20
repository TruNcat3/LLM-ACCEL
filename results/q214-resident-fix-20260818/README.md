# Controller-Resident Q2.14 Attention Fix

This artifact isolates the precision correction for controller-resident online
attention from the earlier block-prefill baseline. The probability and
old-accumulator scale state now use signed Q2.14. Their raw 16-bit payload is
carried through the existing activation stream, interpreted by the compute CU,
and compensated by a `1/64` task output scale during PV accumulation. The
external stream width and packet ABI are unchanged.

The workload vocabulary is shape based. `Active query rows = 8` means one
8-row prefill block from one sequence. It is neither batch size eight nor eight
decoded outputs, and it does not assert that the total prompt length is eight.
Decode activates one row.

## Evidence boundary

The evidence classes are intentionally separate:

- `validation.tsv` records the focused C simulation, finite-FIFO C/RTL CoSim,
  same-source HLS synthesis, and the P2 hardware-emulation numerical
  discriminator;
- `resource.tsv` contains local HLS estimates. Its system row is an arithmetic
  sum and is not a placed-and-routed utilization report;
- `performance.tsv` contains only run-local Vitis 2022.2 HW-Emu CU intervals.
  Host computation, simulator wall time, CPU golden-reference work, and
  initial model migration are excluded; the common four-CU field does not
  separately resolve inter-task issue gaps;
- `artifact_manifest.tsv` binds the HLS exports, linked XCLBIN, Host executable,
  and emulation configuration by SHA-256;
- `source_manifest.tsv` binds the release-critical source files;
- `raw/` contains the reports and run-local logs from which the tables are
  derived.

## Precision discriminator

The standard-dimension P2 diagnostic runs Task 18 over two consecutive query
rows. Both rows match the Q2.14 bit-accurate oracle with maximum raw error zero.
The first row also matches the legacy Q8.8 oracle because it has a single
causal key; the second row intentionally discriminates the implementations and
shows 550 legacy mismatches with maximum raw error one. This establishes that
the linked image executes the corrected Q2.14 resident path rather than the
old probability buffer.

The diagnostic migrates its Task-18 output to the Host for inspection, so its
`intermediate_host_copy` field is `diagnostic_snapshot`. Production Task
18/19/20 composition keeps hidden state in HBM and reports
`intermediate_host_copy=0`.

## Bit-accurate Host oracle

The production stream transports the raw Q2.14 probability bits through the
existing Q8.8 activation payload. The compute CU therefore sees a numerically
64x payload, accumulates four fixed-point banks, applies `output_scale=1/64`,
and then narrows to Q8.8. A direct CPU-side Q2.14-by-Q8.8 multiplication is
equivalent over real numbers, but it is not bit equivalent after these
quantizers.

The release Host now mirrors the hardware conversion order. The software-only
regression establishes two useful discriminators with seed 20260718:

- P2 is bit-identical under both conversion orders;
- at P8, raw-payload versus direct arithmetic differs by at most one raw unit
  immediately after attention, grows to 19 after FFN, and reaches 454 after
  final RMSNorm. The latter reproduces all 3,615 mismatches and four probe
  values from the obsolete-oracle HW-Emu failure exactly.

This fingerprint proves that the earlier P8 rejection was a Host-oracle error,
not a kernel error. An independent P8 Task-18 HW-Emu diagnostic also passes all
16,384 values with per-row Q2.14 error at most one raw unit and rejects the
legacy Q8.8 interpretation. The complete Task-18/19/20 P8 gate remains the
release criterion; it is not replaced by either focused discriminator.

The archived Task-18 log was produced before the launcher label was made
shape-aware, so its `gate` line says `p2`; the authoritative invocation fields
are `query_tokens=8` and `--prefill-block-size 8`. The corrected launcher emits
`gate=...p8` for the same workload.

## Finite-buffer gate

The closed controller-compute network retains deadlock detection and bounded
streams. Three Task-18/19/20 transactions pass RTL CoSim in 25,591 cycles, with
minimum/average/maximum task latency 2,354/8,533/19,057 cycles. Relative to the
previous Q8.8-buffer baseline (25,411 cycles), the precision correction adds
180 cycles, or 0.708%, without changing the cross-kernel ABI.

## HLS resource change

The controller estimates 1,212 BRAM18, 123 DSP, 468,481 FF, and 453,487 LUT at
3.746 ns. Relative to the immediately preceding resident controller, this is
+8 BRAM18, +12 DSP, -234 FF, and -260 LUT. One exact compute CU remains at 48
BRAM18, 677 DSP, 197,555 FF, and 117,643 LUT at 2.433 ns. The complete
controller + two compute CUs + status-sink arithmetic sum is 1,308 BRAM18,
1,477 DSP, 868,458 FF, and 697,305 LUT. These figures qualify synthesis only;
they do not prove routing or timing closure.

## Standard P8 release gate

The standard one-layer P8 Task-18/19/20 run completed with exit status zero.
All 16,384 final-hidden values match the bit-accurate fixed-point oracle with
maximum raw error zero at tolerance 32. The Host observed exactly three coarse
tasks in order, intermediate hidden state remained in HBM
(`intermediate_host_copy=0`), and the controller retained ownership of the KV
cache.

| Scope | Active query rows | Tasks | HW-Emu CU interval | Cycles at 200 MHz | Useful GMAC/s | Modeled useful-MAC efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Task 18 -> 19 -> 20 | 8 | 3 | 3,258.106 us | 651,621 | 189.285 | 92.424% |

The interval is the common run-local Vitis 2022.2 HW-Emu interval reported for
the controller, both compute CUs, and the status sink. The identical top-level
and per-function rows do not resolve separable CU occupancy or inter-task issue
gaps, so the interval is not labeled as pure kernel-active time. The CPU golden
reference runs after inference and is not part of the useful-work numerator.
The 616,710,144 useful-MAC numerator covers the standard Qwen layer-shape dense
work in the three-task sequence; efficiency is measured against the two-CU
peak of 1,024 MAC/cycle over the common modeled interval. These are not
physical-board measurements.
