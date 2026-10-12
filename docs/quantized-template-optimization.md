# Quantized template optimization

The integrated W4 design currently carries a batch-capable decode wrapper even when the production layer workload is a single-request D1 operation. HLS evidence shows that the full `run_quantized_batch_decode_U0` wrapper is a major source of the integrated resource increase: compared with the pipeline family, the modeled delta is approximately +1,101 BRAM18K, +895 DSP, and +280k LUT. This is a template-storage and control-replication hypothesis, not proof that every integrated buffer is redundant.

## Diagnostic candidate

`quantized-layer-w4-integrated-b1` keeps the integrated attention, prefill overlap, RMS/SiLU lanes, KV ownership, and stream ABI. It changes only the decode descriptor capacity:

| Parameter | Production integrated | `integrated_b1` |
| --- | ---: | ---: |
| Decode descriptor capacity | 8 | 1 |
| D1 protocol | batch-capable | single-request |
| Compute geometry | 4 × 8 × 128 | 4 × 8 × 128 |
| BRAM/DSP admission | 80% raw-capacity budget | same |

The candidate is diagnostic-only until production CSim, deadlock-enabled RTL CoSim, and a matched HLS BRAM/DSP comparison pass. A lower descriptor capacity does not claim that KV, compute, or stream buffers disappear; any promotion must be justified by the HLS report and numerical/protocol evidence.

## Reproduction

```bash
python3 tools/designs/instantiate.py validate configs/designs/quantized-layer-w4-integrated-b1.json
python3 tools/designs/instantiate.py instantiate \
  configs/designs/quantized-layer-w4-integrated-b1.json \
  --output /tmp/cowave-w4-integrated-b1
```

The development queue runs an over-budget candidate in `diagnose` mode. It preserves the bounded functional evidence and writes an optimization-required marker, but does not link or start HW Emu until BRAM18K and DSP are both below their 80% budgets. Follow-up optimizations should first prove storage lifetime sharing or remove wrapper duplication; arbitrary CU/PE/port combinations remain outside the supported template boundary.
