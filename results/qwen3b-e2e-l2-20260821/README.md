# qwen2.5-3b P8/G2/L2 End-to-End Hardware-Emulation Evidence

This package freezes a completed, checksum-verified `HW_Emu_CU_trace` run. It is modeled RTL evidence, not a physical-board measurement.

## Workload contract

| Field | Value |
| --- | ---: |
| Sequence batch | 1 |
| Prompt sequence tokens | 8 |
| Maximum active prefill rows per block | 8 |
| Prefill blocks | 1 |
| Sampled output tokens | 2 |
| Real D1 decode forwards | 1 |
| Decoder layers exercised | 2 |
| Host-visible coarse tasks | 10 |

`P8` means 8 consecutive query rows from one sequence, not batch 8 and not 8 generated tokens in parallel. The first sampled token comes from the final prefill hidden state; the remaining 1 forward(s) are true one-row decode passes.

## Numerical result

| Validation | Steps | Checked values | Maximum raw error | Tolerance |
| --- | ---: | ---: | ---: | ---: |
| PASS | 2 | 4096 | 0 | 64 |

The CPU fixed-point oracle executes after production inference and is excluded from accelerator useful work. Model source is `random`; tied embeddings are `1`.

## Modeled performance

| Evidence source | Timed scope | CU running time | XSim clock | Cycles | Target clock | Target-equivalent latency | Useful MAC | Useful GMAC/s | Modeled useful-MAC efficiency | Query rows/s | Request output tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HW_Emu_CU_trace | common_four_CU_running_time | 11597.207 us | 200.000 MHz | 2319441.4 | 200.000 MHz | 11597.2070 us | 1387634688 | 119.652489 | 58.424067% | 776.049 | 172.455 |

The common four-CU profiler interval does not resolve separable per-CU occupancy or inter-task issue gaps. Request output-token throughput includes prefill and must not be read as steady-state D1 throughput.

Host wall-clock fields are retained only as HW-Emu simulator proxies:

| Inference | Whole process | LM head | Post-inference validation |
| ---: | ---: | ---: | ---: |
| 36156500.000 ms | 36193200.000 ms | 666.884 ms | 36730.300 ms |

## HLS resource estimates

| Scope | Instances | BRAM18 | DSP | FF | LUT | URAM | Estimated period |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| controller | 1 | 1212 | 126 | 429922 | 453437 | 0 | 3.746 ns |
| compute_two | 2 | 96 | 1354 | 395134 | 235298 | 0 | 2.433 ns |
| status | 1 | 0 | 0 | 4867 | 8532 | 0 | 2.433 ns |
| whole_system | 4 | 1308 | 1480 | 829923 | 697267 | 0 | 3.746 ns |

These are local CSynth estimates, not routed utilization or timing closure.

## Execution boundary

- Host inference compute: `embedding_plus_lm_head_argmax`.
- Accelerator compute: `decoder_layers_final_norm_rope_online_attention_kv`.
- CPU golden: `post_inference_validation_only`.
- Intermediate hidden copy: none; KV owner: controller/HBM.
- Trace scope: `common_four_CU_running_time`; physical-board measurement: `0`.

## Provenance

| Artifact | SHA-256 |
| --- | --- |
| Host executable | `99712b58f89958df18e8a0eb560236c771af501b19cec279f44035c6e622e0e3` |
| XCLBIN | `adf21bc4cc20d5bec514e69bfbdb28e12e9572a33f1071c090f5003e4d904076` |
| Emulation configuration | `696c863021ad86e4740c840916408d9fdae97a9bbaadaefe1e43a1f0f79116ac` |

The package contains the raw Host/build logs, CU profile, performance row, HLS resource row when applicable, archive-time source snapshot, build-to-release source-equivalence table, manifest, and checksums. Verify it from inside the package with:

```bash
sha256sum -c checksums.sha256
```
