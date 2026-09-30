# CoWave implementation catalog

[Documentation](README.md) | [Source map](repository-map.md) |
[Reproduction](usage.md) | [Evidence](../results/README.md)

CoWave has three public implementation families. A diagnostic harness, model
dimension, arithmetic option or validation stage is not another accelerator
generation. This catalog applies to the whole repository.

## Implementation families

| Family | Architecture and arithmetic | Public source | Evidence boundary |
| --- | --- | --- | --- |
| **Fix16 resident** | Model-aware controller, two unified 8×64 matrix/vector CUs and status sink; signed fixed-point payloads | Root [kernel](../kernel/), [include](../include/), [host](../host/) | Resident layer, block prefill, coarse-task L1/L2 generation, protocol and operator diagnostics |
| **Streaming split** | Control/cache plus V8-2_s fixed compute; fixed-point arithmetic with a different tile organization | [Streaming split case](../cases/streaming-split/) | Linear-operation/accumulation tests and HLS studies; full-layer numbers are analytical |
| **Quantized matrix blocks** | W4A4 8×64 packed and W8A8 4×128 integer matrix kernels with configurable accumulators | [Quantized block case](../cases/quantized-block/) | Component CSim/CoSim, HLS resources and CU-capacity planning; no released full-layer runtime |

The common research theme is controller/compute separation. The families have
different task interfaces and integration boundaries; an xclbin, Host or packet
definition is not interchangeable merely because both designs use streams.

### Fix16 profiles and evaluation scopes

| Scope within Fix16 resident | What changes | Where to read / reproduce |
| --- | --- | --- |
| Resident layer | Controller owns one complete layer's intermediate computation and state | [Architecture](architecture.md), [usage](usage.md) |
| Coarse-task model runtime | Host composes Attention, FFN and final-norm descriptors; hidden/KV stay in HBM | [Runtime contract](coarse-task-runtime.md) |
| **Fix16 operator diagnostics** | Host sequences individual operators; the timed interval excludes additional Host fixture work | [Q2.14 P/D study](q214-pd-length-hwemu.md) |
| **Small-shape protocol tests** | Reduced model dimensions expose FIFO, task, tail and residency behavior | [Experiment summary](experiments.md), [usage](usage.md) |
| Checkpoint diagnostics | Intermediate Host readback locates the first numerical divergence | [Checkpoint evidence](../results/qwen3b-checkpoint-20260830/) |

The last three rows are test scopes, not separate hardware families. The
current Host marker `static_descriptor_v1` identifies an execution program.
Older `legacy_equivalent_sequence` evidence is retained with its own
provenance; updating a Host does not retroactively change the binary that
produced an older trace.

### Implementation options and historical progression

| Family | Examples of options or evolution | How to identify a specific result |
| --- | --- | --- |
| Fix16 resident | FIFO depths, load II, cross-wave DATAFLOW (`.cw1`), resident subgraphs, Q2.14 attention payloads, persistent Norm/RoPE, static descriptor execution | Model profile, feature flags, Host marker, source manifest and xclbin/Host hashes |
| Streaming split | V8-2_s sharing, control/cache DATAFLOW, weight-bank mapping, persistent reduction and finalization | Case-local source, control protocol, reduction length and recorded tool reports |
| Quantized matrix blocks | Rotating vs single accumulator bank, narrow vs wide accumulation, DSP accumulation alternative, CU count | Precision, tile, accumulator flags, script configuration and HLS/CoSim evidence |

This is a design lineage, not a claim that the last option is best for every
workload. For example, Q2.14 is an attention probability format, not a new
matrix precision family. Detailed tradeoffs belong in
[design space](design-space.md), and measured comparisons in
[experiments](experiments.md).

## Keep these axes separate

| Axis | Example | Definition |
| --- | --- | --- |
| Implementation family | Fix16 resident | Code and ownership boundary |
| Arithmetic | W4A4, W8A8, Fix16, Q2.14 probability | Exact operand/tensor format; Fix16 is not IEEE FP16 |
| Feature configuration | Cross-wave DATAFLOW, single accumulator | Selected mechanism and parameters |
| Model profile | small, qwen-layer, qwen2.5-3b | Compile-time dimensions/capacity; see model configuration |
| Workload | P8/G2/L2, context1024, B1 | Actual executed rows, layers, forwards and sequences |
| Evidence level | CSim, HLS, RTL CoSim, HW Emu | What was measured or checked |
| Source/artifact identity | Manifest SHA-256, Host/xclbin hashes | Which implementation snapshot produced the result |
| Software release | CITATION.cff version 0.10.0 | Public package metadata, not a hardware setting |

A model configured for 36 layers can be tested for only one or two layers.
A successful link is not a numerical test; a resource-planner output is not
a placed design. Record the executed workload separately from the compiled
capacity and the selected architecture.

## Workload notation

Use explicit fields in new tables: `prompt_tokens`, `query_rows`,
`sequence_batch`, `decode_forwards`, `sampled_outputs`, `layers`,
`context` and `block_rows`.

- In the resident generation contract, P8/G2/L2 means an eight-token prompt,
  two sampled outputs and two decoder layers. The prompt forward yields the
  first output; one real single-row D1 forward yields the second.
- B1/B4, when used for batch, mean one/four independent sequences.
  `block_rows` identifies physical query-block size separately.
- A standard-shape P8 gate executes one eight-row block. Small P16 and P11
  protocol tests cover multiple blocks and tails; their shape and evidence
  differ from standard-model full-prompt measurements.
- Historical [operator-sweep](q214-pd-length-hwemu.md) P1024 means only the final
  eight query rows at positions 1016–1023. D1024 means one query at position
  1024 against 1025 KV entries, not 1024 Decode forwards.
- Historical B8/B3 in some result prose mean block rows. Lowercase `b8` in
  development filenames may also mean block size; `p128` in a matrix symbol
  may mean output columns. Preserve raw labels and read their local schema.

Timing scope is another required field: full resident CU interval, sum of
operator calls, internal component interval, HLS latency estimate, or Host
wall time. Those quantities must not share one unlabeled performance column.

## Public versus development state

The public tree contains the three source families above and the packages in
[results](../results/README.md). It is a curated release, not a mirror of every
development work directory.

Development includes W4A4/W8A8 full-layer controller profiles such as Baseline,
Attention, Attention + pipeline and Integrated decode. They are **not** the
published quantized matrix-block case and cannot inherit its validation status.
The [September 30 progress report](quantized-layer-progress.md) publishes the
completed W4A4 four-profile measurement and its analysis inputs. The archive
preserves hardware source identities; it does not include a full-layer hardware
source/build closure or add a fourth executable public family. These development
profiles use four compute CUs, separately from the Fix16 resident CU topology.
The [release workflow](release-workflow.md) defines how source, reproduction
tools and evidence move together into the appropriate public family.

## Legacy Label Migration

Old labels remain in checksummed evidence and compatible filenames. New
narrative and figures use descriptive names.

| Legacy alias | Current interpretation | Kind |
| --- | --- | --- |
| `R1` | Fix16 resident | Implementation family |
| `D1` | Fix16 operator diagnostics when used as a former design label; single-row Decode in workload notation | Evidence scope / workload, determined by context |
| `P1` | Small-shape protocol tests | Evidence scope |
| `S1` | Streaming split | Implementation family |
| `Q1` | Quantized matrix blocks | Implementation family |
| Case 1 / default | Root Fix16 resident source | Former directory narrative |
| Case 2 / V8-2_s | Streaming split case | Former case numbering / kernel symbol |
| FP16 / FP version | Fix16 in these project notes | Historical informal name, not IEEE half |
| .cw1 / .wr33.scratch | Fix16 configuration/build suffixes | Feature snapshots |
| q214exp18 / Q2.14 | Diagnostic build suffix / probability format | Configuration / arithmetic |
| Q4.x / Q8.x | Development quantized experiment aliases | Not public release numbers |

Archived paths, ABI operation IDs, macro names and source hashes remain the
reproduction identifiers. Human-readable names supplement them rather than
reinterpreting their historical evidence.
