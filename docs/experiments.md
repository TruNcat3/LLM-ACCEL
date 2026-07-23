# Experimental Results

## 1. Reporting policy

LLM-ACCEL separates evidence into four classes:

| Evidence | Supports | Does not support |
| --- | --- | --- |
| CSim | Algorithm and packet semantics | RTL timing or finite-buffer progress |
| RTL CoSim | RTL equivalence and deadlock-free completion for the tested bounds | Full-system memory/connectivity behavior |
| Hardware emulation | Multi-kernel ABI, connectivity, XRT execution, simulated active cycles | Physical timing, power, or board throughput |
| HLS synthesis | Local timing/II/resource estimates | Final place-and-route closure |

No result in this document is a physical-board measurement.

## 2. Reference configuration

| Parameter | Value |
| --- | ---: |
| hidden / intermediate | 2048 / 11008 |
| query heads / KV heads / head dimension | 16 / 2 / 128 |
| compute system | 2 x 8x64 MAC CUs |
| peak issue rate | 1024 MAC/cycle |
| weight block FIFO depth | 2 |
| weight load II | 2 |
| wave result FIFO depth | 33 |
| cross-wave dataflow | enabled |

## 3. Verification summary

| Test | Workload | Result |
| --- | --- | --- |
| Focused CSim | Packet panel, RoPE banking, probability state, vector paths | PASS |
| Controller CSim | 15 functional cases including multi-token paths | PASS |
| Controller C/RTL CoSim | 13 transactions, deadlock detection enabled | PASS |
| Closed controller-compute CoSim | Finite-depth feedback network | PASS, 6,129 cycles |
| Resident-layer hw_emu | Full standard layer, random fixed-point weights, position 0 | PASS, 2048 outputs bit exact |
| 2-token prefill hw_emu | Full standard projections, attention, and FFN | PASS |
| 8-token prefill hw_emu | Full standard projections, positions 0-7 causal attention, and FFN | PASS |

The 8-token run completes 48 attention MM tasks and 1536 result packets. Its
final hidden checksum is `0xb72a92cb5224f0c7`.

## 4. Single-token resident layer

The `.cw1` cross-wave dataflow version is compared with the same resident
layer before cross-wave overlap:

| Metric | Immediate-wave baseline | Cross-wave dataflow | Change |
| --- | ---: | ---: | ---: |
| Active cycles/layer | 714,495 | 683,601 | -4.32% |
| Latency projected to 200 MHz | 3.572 ms | 3.418 ms | -4.32% |
| Useful throughput at 200 MHz | 21.57 GMAC/s | 22.55 GMAC/s | +4.54% |
| Full two-CU utilization | 10.53% | 11.01% | +0.48 percentage points |

The layer contains approximately 77.07M useful MAC. M=1 decode activates only
one row of each 8-row CU. Relative to the 128-MAC/cycle one-row reference, the
cross-wave version reaches 88.09% utilization. The low full-array percentage
therefore mainly expresses a workload-shape limitation.

## 5. Projection steady state

The complete O projection (`K=N=2048`, 16 output waves) measures:

| Metric | Result |
| --- | ---: |
| Active cycles | 46,898 |
| Array-ideal cycles | 32,768 |
| Array efficiency | 69.87% |
| Later 15 waves, average | 2,210.4 cycles/wave |
| Per-wave array ideal | 2,048 cycles/wave |

The later waves are close to the compute ideal; the remaining gap is dominated
by first-wave setup and boundary work rather than steady-state weight delivery.

## 6. Attention scaling experiments

These experiments include QK, online normalization, PV, and KV-cache access,
but exclude Q/K/V/O projections:

| Context | Tiles | Cycles | Observation |
| ---: | ---: | ---: | --- |
| 64 | 1 | 38,097 | Full single tile |
| 65 | 2 | 74,203 | Second tile mostly padding |
| 96 | 2 | 73,984 | Same two-tile schedule with more valid entries |

Moving from one to two tiles increases cycles by 1.95x, close to the doubled
padded work. The low valid-MAC utilization at context 65 reflects one valid
entry in the second 64-entry tile, not a protocol stall.

## 7. Eight-token prefill

### Functional workload

The run executes full-size Q/K/V/O, Gate/Up/Down, both RMSNorms, RoPE,
positions 0-7 causal online attention, SiLU multiplication, and residual paths.
Maximum raw error is at most 1 for Q/K/V/O/Gate/Up and at most 4 for the
fixed-point attention/Down checks.

### Cycle calculation

The 200 MHz xclbin reports 3176.997 us of kernel-active time:

```text
active cycles = 3176.997 us * 200 MHz = 635,399.4 cycles
```

Useful work is counted as:

```text
Q/K/V/O + Gate/Up/Down = 616,562,688 MAC
valid causal QK + PV    =     147,456 MAC
total useful            = 616,710,144 MAC
```

| Metric | 2 tokens | 8 tokens | Scaling |
| --- | ---: | ---: | ---: |
| Active time | 2806.029 us | 3176.997 us | 1.132x |
| Active cycles | 561,206 | 635,399 | 1.132x |
| Useful MAC | 154.15M | 616.71M | 4.00x |
| Throughput | 54.94 GMAC/s | **194.12 GMAC/s** | **3.53x** |
| Two-CU utilization | 26.82% | **94.78%** | +67.96 points |
| Amortized time/token/layer | 1403.0 us | **397.1 us** | 3.53x faster |

Vector operations consume active cycles but are not added to useful MAC.
Causal attention counts only valid entries, not padded tile operations. The
efficiency is therefore not inflated by counting implementation padding as
useful work.

### Resource qualification

The prefill experiment uses a diagnostic controller compiled with all
per-operator profile paths. Its top-level HLS report is:

| Resource/timing | Estimate | Fraction of integration device |
| --- | ---: | ---: |
| Period / Fmax | 3.839 ns / 260.48 MHz | - |
| BRAM18 | 1,586 | 59% |
| DSP | 248 | 4% |
| FF | 2,405,836 | 137% |
| LUT | 1,021,326 | 117% |

This build is a functional and performance instrument, not a deployable
bitstream candidate. Its result establishes that the data path and M=8 schedule
can approach peak utilization. The controller must be specialized into a
resident static prefill task to remove mutually exclusive diagnostic paths and
duplicated GBUF state.

For comparison, the resource-pruned resident `.cw1` controller estimates
3.746 ns, 488 BRAM18, 90 DSP, 421,880 FF, and 390,283 LUT.

## 8. Interpretation

The combined results support three conclusions:

1. **The compute datapath is not intrinsically underutilized.** An 8-token
   block reaches 94.78% useful-MAC efficiency.
2. **Decode is shape limited.** Its one active row reaches 88.09% of the
   row-normalized reference while only using 11.01% of the physical array.
3. **Resource specialization is now the prefill bottleneck.** The diagnostic
   schedule is fast and correct but must be compiled into the pruned resident
   controller before physical implementation.

## 9. Current experimental boundaries

- Prefill covers one 8-token block and positions 0-7 in the first 64-entry
  attention tile.
- The fixed-point random model validates deterministic arithmetic and protocol
  behavior; it is not an end-to-end checkpoint accuracy result.
- A 36-layer value obtained by multiplying single-layer cycles would exclude
  embedding, final normalization, LM head, sampling, long-context growth, and
  inter-layer effects; it is therefore not reported as end-to-end throughput.
- Hardware-emulation CPU wall time is not accelerator latency.
- Final frequency, routing, power, and physical throughput require a completed
  implementation experiment.

## 10. Next experiments

1. Compile the verified 8-token schedule into a resource-pruned resident
   prefill layer and repeat synthesis, deadlock-on CoSim, and hw_emu.
2. Sweep multiple 8-token blocks and attention tile counts to quantify
   long-context behavior.
3. Replace the Flash-PV II=4 dependence with banked/interleaved accumulation.
4. Evaluate multi-request row batching for M=1 decode.
5. Run place-and-route and physical measurements for the pruned configuration.
