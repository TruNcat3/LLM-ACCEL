# Design Space and Alternatives

This document records the architectural alternatives explored by LLM-ACCEL.
The goal is to make the current implementation understandable as a sequence of
measured choices rather than as the only possible design.

## 1. Kernel partitioning

| Candidate | Advantages | Disadvantages | Status |
| --- | --- | --- | --- |
| Monolithic decoder kernel | Direct data reuse; no inter-kernel protocol | Large control cone, difficult synthesis/debug, weak reuse | Early baseline |
| Controller plus stream-only compute | Separates model semantics from arithmetic; reusable compute CUs | Requires explicit finite-depth protocol | **Selected** |
| Many operator-specific kernels | Natural software mapping | Repeated launch/data movement; large system interconnect | Not selected |

The selected split makes the controller the only stateful model component.
This avoids operator-by-operator host scheduling without turning every compute
island into a separate memory master.

## 2. Compute-array shape

Candidate shapes trade parallel rows, output columns, DSP count, and placement
regularity. The current pair of 8x64 CUs was selected because it provides:

- 1024 MAC/cycle system peak;
- a natural 8-token prefill block;
- two independently placeable compute islands;
- output-column partitioning with shared activation broadcast.

Its main limitation is explicit: M=1 decode uses one of eight rows. Enlarging
the array does not fix this shape mismatch. Better decode utilization requires
multiple independent rows of work, such as request batching or another
Split-M mapping.

## 3. Cross-kernel protocol granularity

| Candidate | Observation |
| --- | --- |
| Fine-grained/bit-level control | Creates many narrow pipelines and large control/fan-out networks |
| Custom structs across XO boundaries | Fragile ABI and tool-dependent packing |
| Fixed-width block packets | Regular interfaces, explicit compatibility, easier protocol checking |

The selected ABI uses fixed-width integer packets. Task and status are compact
encodings; payload channels move activation, weight, vector, and result blocks.
The block ABI improves the boundary, while timing-critical arithmetic inside a
kernel still requires separate optimization.

## 4. Operator implementation choices

| Operator class | Candidates | Current implementation |
| --- | --- | --- |
| Dense projection | Monolithic GEMM; operator-specific arrays; unified tiled MM | Unified 8x64 MM with output-wave Split-N across two CUs |
| RMSNorm | Host preprocessing; controller reduction; CU vector task | CU vector task so normalized features remain in the stream/resident path |
| RoPE | Host transform; CU vector opcode; controller-local banked transform | Controller-local transform using coefficients supplied through auxiliary memory |
| Exponential/normalization | Custom iterative exponential; lookup approximation; HLS math primitive | `hls::exp` on a bounded fixed-point input, followed by online normalization state |
| SiLU and gated product | Separate kernels; controller arithmetic; unified CU vector path | Unified CU vector task to avoid intermediate host/memory traffic |
| Residual addition | Host or external-memory round trip; controller; CU vector path | CU vector task with resident operands |
| KV cache | Host-managed cache; compute-local cache; controller-managed external cache | Controller-managed cache, keeping PCIe outside the token loop |

A custom iterative exponential was rejected because its recurrence created an
unfavorable timing path in the attention normalization stage. The selected HLS
math primitive gives the synthesis tool a recognized implementation while the
controller still clamps and represents its input explicitly. The remaining
attention bottleneck is the PV accumulator dependency, not the exponential
function itself.

## 5. Weight delivery pipeline

Several organizations were evaluated:

| Candidate | Outcome |
| --- | --- |
| Load a complete panel, then emit | Simple but serializes memory preparation and compute |
| Chunk ping-pong buffers | Good overlap, excessive BRAM for the explored configuration |
| Whole 4096-bit tile FIFO | II=1 loading, but high register/multiplexer cost |
| 512-bit block FIFO with dynamic selection | Lower storage, unfavorable selector path |
| 512-bit block FIFO plus row shift registers | **Selected:** rate matched and structurally regular |

The selected isolated weight path reduced the measured wave/loader latency
from roughly 6.8k to 1.1k HLS cycles in the exploration profile. Depths 1, 2,
and 4 produced the same cycle estimate; depth 2 was retained because it removes
the block-FIFO depth warning without paying the extra storage of depth 4.

## 6. Wave scheduling

### Sequential waves

The simplest controller completes load, drive, compute, collect, and commit for
one wave before starting the next. It is easy to verify but repeatedly pays
pipeline fill/drain and cannot hide memory preparation.

### Cross-wave dataflow

The selected path connects wave-local stages with bounded streams and separate
scratch state. It reduced the full single-token layer interval from 714,495 to
683,601 cycles in hardware emulation and brings later projection waves close
to the 2,048-cycle array ideal.

### Multi-wave repeat command

The compute ABI can describe repeated waves under one task. Closed-loop CoSim
demonstrated protocol correctness for a limited repeat case, but separated-
kernel hardware emulation did not meet the progress/performance threshold for
repeat counts above one during exploration. The release therefore keeps repeat
disabled and uses the better-understood cross-wave dataflow path.

## 7. Attention alternatives

| Candidate | Storage/traffic | Scheduling implication |
| --- | --- | --- |
| Materialize the score matrix | O(sequence length) score storage per query and additional traffic | Simple normalization pass |
| Tile scores, then make a second pass | Bounded temporary storage | Re-reads or retains tile data |
| Online normalization and PV accumulation | Bounded running state | Requires rescaling dependencies |

LLM-ACCEL selects online normalization. Its main HLS bottleneck is currently the
PV accumulator's carried dependency; the full-profile loop reaches II=4. This
is an internal attention problem and is not solved by widening the stream ABI.

Candidate improvements include banked partial accumulators, interleaved heads,
and a tree reduction followed by a less frequent state update. Any change must
preserve the fixed-point error envelope and cross-tile normalization semantics.

## 8. Prefill scheduling

The current diagnostic prefill host invokes individual operators to expose
each tensor to a golden checker. This produces strong functional evidence and
allows per-stage profiling, but compiling every diagnostic route together
duplicates controller state and exceeds the target resource budget.

The production candidate is a static resident prefill task:

1. load an 8-token feature block;
2. execute the complete attention sublayer while intermediates remain resident;
3. execute the complete FFN sublayer;
4. commit only the block output and updated KV state;
5. overlap the next block load through ping-pong GBUFs.

The 8-token hardware-emulation result shows that the arithmetic schedule can
reach 94.78% useful-MAC efficiency. The next research step is resource
specialization, not a new compute array.

## 9. Decode utilization candidates

Two-CU Split-N is already present: activation data is broadcast and output
columns are divided between CUs. It does not fill unused token rows for M=1.

Promising candidates are:

- **multi-request batching:** independent decode requests occupy different
  rows while sharing the same resident weights;
- **multi-token speculative blocks:** useful when the algorithm can provide
  more than one candidate token;
- **row remapping / Split-M:** partition independent row work across lanes when
  the workload exposes it.

Adding dynamic controller states without a corresponding independent row of
work cannot increase arithmetic utilization and is therefore not a priority.

## 10. Evaluation principles

Designs are compared using four independent dimensions:

1. functional error against deterministic fixed-point golden models;
2. finite-buffer progress with RTL deadlock detection enabled;
3. cycles and useful-MAC efficiency under matched shapes;
4. synthesis timing and resource cost.

An optimization is not accepted solely because it improves one dimension. For
example, the diagnostic prefill build is fast and correct in hw_emu but is not
considered deployable because its unpruned controller exceeds the resource
budget.
