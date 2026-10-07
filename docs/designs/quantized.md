# Quantized CoWave designs

[Design index](../README.md#designs) | [Architecture](../architecture.md) |
[Design space](../design-space.md) | [Reproduce](../usage.md#quantized-full-layer) |
[Evaluation](../experiments.md) | [Case index](../../cases/README.md)

The quantized mainline has two complete-layer designs. The isolated matrix
probes are a separate component family and are not a source shortcut for the
mainline:

| Design | Matrix arithmetic | CU geometry | Complete-layer source |
| --- | --- | --- | --- |
| `cowave-int4-4-8-128` | W4A4, signed INT4 activations and weights | 4 compute CUs, 8 logical rows, 128 logical output columns | [`cases/quantized-layer/`](../../cases/quantized-layer/) |
| `cowave-int8-4-4-128` | W8A8, signed INT8 activations and weights | 4 compute CUs, 4 logical rows, 128 logical output columns | [`cases/quantized-layer/`](../../cases/quantized-layer/) |

The numbers in a design name count compute CUs and logical matrix shape.
They do not count a controller, status sink, or other service kernel. DSP
packing is a physical implementation choice: it changes the DSP/resource
account, not the logical product count. In particular, do not multiply the
logical work by a W4A4 packing factor a second time.

Both designs have wave-enabled and wave-disabled selections in the
[configuration catalog](../implementations.md#named-configurations). That
table, generated from the machine-readable catalog, owns their exact parameters.

The matching W4 and W8 P66+D1 rows are in the
[`quantized-layer-20261007`](../../results/quantized-layer-20261007/) package.

## Arithmetic and service rates

W4 packs four matrix products per DSP and supplies eight logical query rows;
W8 uses one matrix product per DSP and four rows. Across four CUs, their
logical peaks are 4,096 and 2,048 MAC/cycle. This is capacity under sustained
supply, not a claim that every cycle performs that many useful products.
Both designs retain fixed-point vector, scaling and hidden/KV paths. Narrower
matrix operands therefore do not automatically accelerate normalization,
SiLU, softmax, data packing or state movement.

The selected controller schedule combines packed K/V tile prefetch,
asynchronous compute dispatch, block lookahead, Decode row reuse and FFN
overlap. In Prefill, SiLU on the Gate branch can overlap the Up projection;
the elementwise product and Down projection still depend on both branches.
Attention wave overlap is a separate option. These dependencies explain why
component speedups and a doubled matrix peak need not produce proportional
whole-layer gains. See the [scheduling choices](../design-space.md).

The production Host executes one sequence. Prefill consumes successive blocks
of up to eight W4 rows or four W8 rows, including a partial final block;
the prompt length is not limited to a single block. D1 is one new token against
the resident KV prefix. Reusing otherwise idle rows for Decode work does not
turn that request into a multi-sequence batch.

The checked-in `quantized-layer` source is the mainline full-layer closure. Its
README owns the controller, compute, configuration, build, and evidence
identity. The separate [`cowave-quantized-blocks`](../../cases/quantized-block/README.md)
case is deliberately component-level: its task word carries scale metadata,
but the probes do not apply scales or zero points. Its CSim, deadlock-enabled
RTL CoSim, HLS estimates, and CU planner support arithmetic, stream, and
resource claims only.

## Full-layer source boundary

The September 30, 2026 W4 package is a frozen historical source/evidence
snapshot. The public full-layer source is a newer source closure and does not
inherit that snapshot's measurements. The [dated progress report](../quantized-layer-progress.md)
and [experimental report](../experiments.md) explain the two identities.

## Reproduce

For either mainline design, use [`cases/quantized-layer/README.md`](../../cases/quantized-layer/README.md)
or the [quantized full-layer route](../usage.md#quantized-full-layer). For
isolated arithmetic/resource probes, use the
[quantized-block README](../../cases/quantized-block/README.md). Both routes
begin with the shared [environment preflight](../environment.md), but neither
route changes the root Fix16 build path or kernel ABI.

## Evidence boundary

The full-layer result package records the named profile, P66+D1 workload,
source identity, and its measured timing/numerical checks. It is not a board
timing or trained-model claim. The component result package records local II,
bounded stream completion, HLS/resource estimates, and replication arithmetic;
it does not establish controller integration, routed timing, board throughput,
scale application, or model accuracy. The full-layer source and frozen dated
result remain separate identities even when they use related W4A4 names.
