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

`P8` denotes one sequence with eight active prefill query rows in one block,
not eight sequences and not eight generated tokens. Current generation tests
use sequence batch one; decode remains one active row per forward (`D1`).

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
| Controller CSim | 21 functional cases including Tasks 18/19/20 | PASS |
| Controller C/RTL CoSim | 13 transactions, deadlock detection enabled | PASS |
| Closed controller-compute CoSim | Finite-depth feedback network | PASS, 6,129 cycles |
| Resident-layer hw_emu | Full standard layer, random fixed-point weights, position 0 | PASS, 2048 outputs bit exact |
| 2-row prefill HW Emu | Full standard projections, attention, and FFN | PASS |
| 8-row prefill HW Emu | Full standard projections, positions 0-7 causal attention, and FFN | PASS |
| Q2.14 P/D length sweep | Final 8-row P block and next-token D at contexts 64/256/512/1024 | 8/8 PASS |
| Coarse-task closed-loop RTL CoSim | Task 18 -> Task 19 -> Task 20, finite FIFOs | PASS, 7,983 cycles |
| Coarse-task stack HW Emu | Two layers, five tasks, final norm, no intermediate host copy | PASS, 64 outputs exact |
| 8-row coarse-task closed-loop RTL CoSim | Task 18 -> Task 19 -> Task 20, finite FIFOs | PASS, 25,591 cycles |
| 8-row coarse-task HW Emu | Small one-layer block plus final norm | PASS, 512 values, max raw error 10/32 |
| Two-layer 8-row stack HW Emu | Small P8 through five Task-18/19/20 commands | PASS, 512 values, max raw error 10/64 |
| Block prompt/decode HW Emu | Small two-layer P8 + G2 task composition | PASS, two forwards/ten tasks |
| Standard Qwen-layer coarse-task HW Emu | P8 Task 18 -> 19 -> 20, controller-owned KV and hidden state | PASS, 16,384 values exact, 651,621 cycles |
| Standard-shape generation-path HW Emu | Qwen2.5-3B P8/G2/L1, one prefill plus one real D1 forward | PASS, six tasks, two CPU-golden steps, 4,096 values exact |
| Standard-shape multi-layer generation HW Emu | Qwen2.5-3B P8/G2/L2, cross-layer prefill plus real D1 | PASS, ten tasks, two CPU-golden steps, 4,096 values exact |

The 8-row block run completes 48 attention MM tasks and 1536 result packets. Its
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

## 7. Eight-row prefill-block baseline

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

| Metric | 2 active query rows | 8 active query rows | Scaling |
| --- | ---: | ---: | ---: |
| Active time | 2806.029 us | 3176.997 us | 1.132x |
| Active cycles | 561,206 | 635,399 | 1.132x |
| Useful MAC | 154.15M | 616.71M | 4.00x |
| Throughput | 54.94 GMAC/s | **194.12 GMAC/s** | **3.53x** |
| Two-CU utilization | 26.82% | **94.78%** | +67.96 points |
| Amortized time/token/layer | 1403.0 us | **397.1 us** | 3.53x faster |

Both columns are single-sequence prefill blocks evaluated within one layer.
They are not request batch sizes and do not mean that autoregressive decode
produces two or eight tokens per forward. Decode uses one active query row.

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

For comparison, the preceding single-row resource-pruned `.cw1` controller
estimated 3.746 ns, 488 BRAM18, 90 DSP, 421,880 FF, and 390,283 LUT. It is a
historical baseline, not the current 8-row Q2.14 release controller summarized
in Section 10.

## 8. Q2.14 multi-length P/D sweep

The Q2.14 implementation was exercised with deterministic non-zero weights,
activations, and KV fixtures at four context lengths. Prefill measures the
final 8-row query block; Decode measures one new query row.

| Phase | Context | Active query rows / block | Cycles | Latency at 200 MHz | Useful GMAC/s | Modeled useful-MAC efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P | 64 | 8 | 637,103 | 3.1855 ms | 194.174 | 94.812% |
| P | 256 | 8 | 685,489 | 3.4274 ms | 182.304 | 89.016% |
| P | 512 | 8 | 749,407 | 3.7470 ms | 168.994 | 82.516% |
| P | 1024 | 8 | 877,846 | 4.3892 ms | 148.090 | 72.310% |
| D | 64 | 1 | 568,040 | 2.8402 ms | 27.229 | 13.296% |
| D | 256 | 1 | 590,794 | 2.9540 ms | 26.447 | 12.913% |
| D | 512 | 1 | 622,231 | 3.1112 ms | 25.448 | 12.426% |
| D | 1024 | 1 | 683,754 | 3.4188 ms | 23.771 | 11.607% |

Every Q2.14 bit-accurate comparison passed. The maximum raw error was 1;
comparisons against the higher-precision attention reference also passed with
a maximum raw error of 5. Task and result-packet counts matched in all cases.

The `Active query rows / block` column is the query-block height, not total prompt
length or request batch size.
P1024 covers positions 1016--1023; it is the final-block cost and not the sum
of all 128 blocks in a complete 1024-token prefill.

### Measurement boundary

The host currently invokes the hardware operators composing the layer. The
reported cycle count sums their controller-kernel active intervals. The main
dense, vector, and attention arithmetic is on the FPGA, but the profile host
still performs operator sequencing, RoPE/test-fixture packing, historical KV
preload, projection-to-cache fixture migration, and CPU golden checks. Those
host operations and transfer gaps are not timed.

Accordingly, these numbers measure accelerator datapath efficiency and
context scaling; they are not PCIe-inclusive end-to-end inference latency.
The efficiency column divides shape-counted useful MACs by the measured cycles
and the two-CU peak of 1,024 MAC/cycle. It is a model-based utilization metric,
not post-route physical utilization.
The raw `profile_kernels.csv` files and validation logs are published under
[`results/q214-pd-20260811/`](../results/q214-pd-20260811/), with the complete
interpretation in [Q2.14 P/D sweep](q214-pd-length-hwemu.md).

## 9. Coarse-task resident runtime

Tasks 18, 19, and 20 replace the operator-level host boundary with Attention,
FFN, and final-normalization subgraphs. Hidden state is ping-ponged between two
HBM feature pairs, while the KV cache remains controller-owned. Only task
status is returned until the final hidden migration.

| Gate | Result |
| --- | --- |
| Controller route CSim | PASS, 21 cases |
| Three-task closed-loop CSim | PASS |
| Three-task RTL CoSim | PASS, 7,983 cycles, deadlock detection enabled |
| Persistent Norm/RoPE/KV focused test | PASS, no row aliasing and banked cache layout exact |
| Small two-layer/five-task HW Emu | PASS, 64/64 exact, no intermediate host copy |
| Small prompt/decode composition HW Emu | PASS, 4 forwards/20 tasks, controller-owned KV |
| Standard Qwen-layer P8 Task-18/19/20 HW Emu | PASS, 16,384/16,384 exact, no intermediate Host copy |
| Standard-shape Qwen2.5-3B P8/G2/L1 HW Emu | PASS, six tasks, two forwards, 4,096/4,096 exact, controller-owned KV |
| Standard-shape Qwen2.5-3B P8/G2/L2 HW Emu | PASS, ten tasks, two forwards, 4,096/4,096 exact, cross-layer HBM residency |

The bounded standard-shape generation path is distinct from the one-forward
P8 layer gate:

| Evidence source | Timed boundary | Workload | Sequence batch | Layers | Tasks | Cycles at 200 MHz | Latency | Useful GMAC/s | Modeled useful-MAC efficiency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vitis 2022.2 HW Emu CU trace | Common four-CU interval; Host computation excluded | P8/G2: one prefill and one real D1 forward | 1 | 1 | 6 | 1,190,693 | 5.953 ms | 116.540 | 56.904% |
| Vitis 2022.2 HW Emu CU trace | Common four-CU interval; Host computation excluded | P8/G2: one prefill and one real D1 forward | 1 | 2 | 10 | 2,319,441.4 | 11.597 ms | 119.652 | 58.424% |

The standard-shape useful-work numerator is derived directly from the model
dimensions. With hidden width 2,048, KV width 256, FFN width 11,008, 16 query
heads, and head width 128, the seven dense matrices contribute 77,070,336 MAC
per query row. P8 plus one real D1 has nine rows, while valid causal QK and PV
contribute `2 * 16 * 128 * (1 + ... + 8 + 9) = 184,320` MAC per layer. The
result is therefore 693,817,344 useful MAC per layer, 1,387,634,688 for L2,
and 24,977,424,384 for L36. Vector operations and padded attention tiles are
deliberately excluded from this numerator.

`P8` is eight consecutive query rows from one sequence, not batch eight or
eight generated tokens. The prompt forward supplies the first sampled token;
the second comes from the D1 forward. Deterministic random Fix16 weights and
tied embeddings pass two post-inference CPU-golden steps in both L1 and L2,
with 4,096 checked values, maximum raw error zero at tolerances 32 and 64, and
identical sampled-token sequences. The L2 run uses the same Host, xclbin, and
emulation configuration as L1. It doubles useful MAC while increasing modeled
cycles by 1.948x: per-layer cycles fall 2.60%, useful throughput rises 2.67%,
and efficiency rises from 56.904% to 58.424%. The controller owns online
attention, KV, and intermediate hidden state across both layers. Host embedding
and LM-head/argmax are functionally part of the request but excluded from the
CU interval. Request output throughput includes prefill, so it is not a
steady-state decode rate. Complete evidence is archived under the
[L1](../results/qwen3b-e2e-20260820/) and
[L2](../results/qwen3b-e2e-l2-20260821/) packages.

The standard-dimension release gate is distinct from the small residency tests:

| Evidence source | Timed boundary | Scope | Active query rows / block | Layers | Tasks | Cycles at 200 MHz | Latency | Useful GMAC/s | Modeled useful-MAC efficiency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vitis 2022.2 HW Emu CU trace | Common four-CU interval; Host computation excluded | Task 18 -> 19 -> 20 | 8 | 1 | 3 | 651,621 | 3.258 ms | 189.285 | 92.424% |

The linked 200-MHz HW-Emu image reports a common profiler running time of
`3258.106 us` for all four CUs. The deterministic random-weight oracle checks 16,384 output
values with maximum raw error zero at tolerance 32. The controller owns the KV
cache, and hidden state is not copied to the Host between the three tasks.
Useful throughput uses 616,710,144 MAC and the nominal two-CU peak of
1,024 MAC/cycle. The raw log and trace are archived under
[`results/q214-resident-fix-20260818/`](../results/q214-resident-fix-20260818/).

The small profile validates cross-layer HBM residency rather than Qwen-layer
throughput:

| Scope | Layers | Tasks | XSim cycles | Projected latency at 200 MHz |
| --- | ---: | ---: | ---: | ---: |
| Attention+FFN layer | 1 | 2 | 14,300 | 71.498 us |
| Stack plus final norm | 2 | 5 | 28,743 | 143.714 us |
| Serial prompt/decode composition | 2 | 20 across 4 forwards | 109,574 | 547.872 us |
| 8-row block plus final norm | 1 | 3 | 31,453 | 157.265 us |
| Two-layer 8-row stack plus final norm | 2 | 5 | 55,697 | 278.483 us |
| Block prompt P8 plus G2 | 2 | 10 across 2 forwards | 77,551 | 387.755 us |
| Multi-block prompt P16 plus G1 | 2 | 10 across 2 forwards | 109,226 | 546.131 us |
| Exact multi-block P16 final-tail golden | 2 | 9 across 2 blocks | 107,178 | 535.889 us |
| Tail-block prompt P11 plus G1 (8+3) | 2 | 10 across 2 forwards | 89,087 | 445.434 us |

Cycles use the generated 300-MHz XSim clock and are projected to the 200-MHz
physical target. Host/OpenCL event durations under HW Emu are simulator
wall-time proxies. Standard-dimension data and raw profiles are maintained in
[`results/coarse-task-20260816/`](../results/coarse-task-20260816/); see the
[runtime report](coarse-task-runtime.md) for the task and measurement boundary.

The serial composition case uses two prompt tokens and three sampled tokens. Its four
forwards comprise two prompt forwards and two generated-token forwards; the
last sampled token is returned without a redundant decoder pass. It validates
cross-position KV state and the real host task-composition path, but its small
shape and serial-token prompt traversal are not a full-model performance
proxy.

The block composition case replaces eight serial prompt forwards with one
8-row forward. It then performs one one-row decode forward to produce two
sampled tokens. The controller owns per-row RoPE, causal KV append/read, online
softmax, and all intermediate hidden states. Embedding, LM-head argmax, and
coarse-task issue remain host responsibilities.

A second Small-profile run processes a 16-token prompt as blocks 0--7 and
8--15. The exact sequence gate omits Task 20 and D2H for the first block,
executes nine tasks total, retains controller-owned KV, and compares the final
512-value tail against a CPU fixed-point golden across both blocks. It passes
with maximum raw error 10 within tolerance 64.

The tail-block gate processes an 11-token prompt as rows 0--7 and 8--10. The
second controller invocation reports `query_tokens=3` and completes the same
two-layer task sequence without a host KV migration. This explicitly tests
the partial-block path that a non-multiple-of-eight prompt requires.

## 10. Interpretation

The combined results support three conclusions:

1. **The compute datapath is not intrinsically underutilized.** An 8-row
   block reaches 94.78% useful-MAC efficiency.
2. **Decode is shape limited.** Its one active row reaches 88.09% of the
   row-normalized reference while reaching only 11.01% full-array modeled
   utilization.
3. **The resident block specialization closes the functional architecture but
   raises the physical resource risk.** The current profile-matched
   Qwen2.5-3B controller plus two compute CUs and status sink estimates 1,308
   BRAM18, 1,480 DSP, 829,923 FF, and 697,267 LUT (48.661%, 24.866%, 47.605%,
   and 79.991% of the full U50). The controller alone exceeds one SLR's LUT
   estimate, so place-and-route remains a required gate. The earlier
   1,477-DSP/868,458-FF/697,305-LUT values remain preserved in the 2026-08-18
   historical release artifact rather than being rewritten in place.

## 11. Current experimental boundaries

- The coarse-task runtime now replays a prompt in blocks of up to eight
  consecutive query rows. A small two-layer prompt/decode HW-Emu contract is
  complete, and the standard-shape single-forward P8, P8/G2/L1, and
  P8/G2/L2 gates are complete. Standard-shape 36-layer, multi-block, and
  checkpoint-level runs remain open.
- The fixed-point random model validates deterministic arithmetic and protocol
  behavior; it is not an end-to-end checkpoint accuracy result.
- A 36-layer value obtained by multiplying single-layer cycles would exclude
  embedding, final normalization, LM head, sampling, long-context growth, and
  inter-layer effects; it is therefore not reported as end-to-end throughput.
- Hardware-emulation CPU wall time is not accelerator latency.
- Operator-level host scheduling and data-transfer gaps are excluded from the
  Q2.14 active-cycle totals. Coarse-task data is reported separately.
- Final frequency, routing, power, and physical throughput require a completed
  implementation experiment.

## 12. Next experiments

1. Extend the verified standard-shape P8/G2 gate from two to all 36 decoder
   layers, preserving the post-inference oracle and common four-CU interval.
2. Exercise standard-shape multi-block prompts and checkpoint-packed weights,
   including cross-block and cross-position KV state.
3. Reduce repeated Host task issue by packaging reusable task programs while
   preserving the explicit Task-18/19/20 boundary and controller-owned state.
4. Add accelerator-side vocabulary projection or quantify the Host LM-head
   boundary separately from the decoder-stack measurement.
5. After functional closure, evaluate multi-request row batching for M=1
   decode and repeat physical placement/resource checks.
