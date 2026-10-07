# CoWave Quantized Full-Layer Evidence

This package is the compact public evidence bundle for the completed W4 and
W8 integrated full-layer configurations. It is derived from the frozen
follow-up archives listed in `provenance/provenance.json`; it does not modify
those archives or the older checksummed result package.

## Published cases

The machine-readable interface is [`summary.json`](summary.json). Each row is
one completed P66+D1 run: one synthetic decoder layer, sequence batch B1,
`H2048/I11008`, 16 query heads, 2 KV heads, and head dimension 128. The P and D
cycle fields are the RTL pin-handshake intervals, with the Host gap excluded.
The reported modeled milliseconds and efficiency use the declared 200 MHz
target clock; the source simulation clock is 3334 ps (about 299.94 MHz).

| Design | Configuration | P cycles | D cycles | P+D cycles | P+D efficiency | Peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cowave-int4-4-8-128` | `integrated-rms2-silu4-prefill-overlap` | 2,288,540 | 187,844 | 2,476,384 | 50.999764% | 4096 MAC/cycle |
| `cowave-int4-4-8-128` | `integrated-rms2-silu4-prefill-overlap-wave` | 2,284,511 | 187,557 | 2,472,068 | 51.088805% | 4096 MAC/cycle |
| `cowave-int8-4-4-128` | `integrated-rms2-silu4-prefill-overlap` | 3,592,954 | 187,220 | 3,780,174 | 66.819675% | 2048 MAC/cycle |
| `cowave-int8-4-4-128` | `integrated-rms2-silu4-prefill-overlap-wave` | 3,582,758 | 186,332 | 3,769,090 | 67.016176% | 2048 MAC/cycle |

The W4 pair uses the 8-row physical block and the W8 pair uses the 4-row
physical block. Both use four compute CUs, four SiLU lanes, two RMS lanes,
32 outstanding weight reads, and prefill FFN-overlap configuration. The
`-wave` rows set `QUANTIZED_ATTENTION_WAVE_PIPELINE=1`; the base rows set it
to zero. The complete flags are in `provenance/cflags-*.txt`.

## Evidence boundary

All four candidates completed numerical validation, C/RTL CoSim, HLS synthesis,
and HW-Emu evaluation. The numeric comparison covers 171,520 signed Fix16
values (prefill hidden, decode hidden, retained K, and retained V) with zero
differing values and zero maximum raw error. The reference is the production C
equivalent used by the run; it is not an independent checkpoint oracle and
does not establish trained-model or checkpoint accuracy.

`evidence/pin_cycles.tsv` retains the original handshake rows for every
`q[48]_layer_{ctrl,cu0..cu3}` `ap_clk`, `ap_idle`, `ap_done`, `ap_start`, and
`ap_rst_n` pin. It is the cycle-extraction input, not a generated cycle table.
`verify.py` recomputes the clock period, P/D intervals, total cycles, and
efficiency from those rows using only the Python standard library. The source
trace hashes in `evidence/pin_manifest.tsv` bind the retained rows to the
original traces.

The HLS rows are Vitis 2022.2 pre-platform estimates for the U50 target. All
scheduling budgets pass, but the four-CU controller-plus-compute estimate
exceeds device BRAM/LUT capacity before platform resources. This package makes
no routed timing, physical-board, deployability, PE-occupancy, power, or
36-layer performance claim. The archived SiLU/Up function report is retained
as a diagnostic; its mode evidence is insufficient for a PE-busy overlap claim.

The full-layer source is production C-equivalent quantized code. This release
does not claim exactness against a trained checkpoint. The complete frozen
source closure is published in [`cases/quantized-layer/`](../../cases/quantized-layer/),
including its portable profile, build, and verification scripts; its README and
`make check` are the public source/reproduction entry point. The W4 and W8
required hardware source files are byte-identical to the two frozen snapshots,
as recorded by the public source hashes. The old archive manifest also includes
development-only files, so its full manifest hash differs across precisions
without indicating a hardware-source mismatch. Rebuilding source reports is
distinct from reproducing the original binary identities and modeled
measurements; Vitis tools are required to rerun HLS or HW-Emu.

## Verification

From this directory, use the standard-library verifier and checksum manifest:

```bash
python3 verify.py
sha256sum -c checksums.sha256
```

The verifier does not launch experiments, require AMD/Xilinx tools, read a
private absolute path, or rewrite any archive file. Large WDB/xclbin/build
trees, simulator databases, and raw output dumps are intentionally excluded;
the numeric report keeps their exact source and output hashes.
