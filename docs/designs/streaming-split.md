# `cowave-streaming-split`

[Design index](../README.md#designs) | [Architecture](../architecture.md) |
[Design space](../design-space.md) | [Reproduce](../../cases/streaming-split/README.md) |
[Evaluation](../experiments.md) | [Case design](../../cases/streaming-split/docs/design.md)

`cowave-streaming-split` is an independent implementation family. It combines
the case-local `control_cache_core` (`cc`) with a fixed V8-2_s compute core;
it is not another configuration of the root resident controller.

## Boundary

`cc` reads hidden input, weights, and an `operator_program`, then routes packed
streams and stores packed output. V8-2_s is a fixed 16-input by 64-output
matmul service with two cores and one lane per core. The Host programs and
connectivity file live beside those kernels under
[`cases/streaming-split/`](../../cases/streaming-split/).

The case stores fixed-point payloads and accumulators (`ap_fixed<16,8>`,
`ap_fixed<16,4>`, `ap_fixed<32,16>`, and internal `<48,24>`), but normalization
uses `hls::recip` after a float conversion. It is therefore not an end-to-end
equivalent of `cowave-fix16-2-8-64` and must not inherit Fix16 numerical or
resident-KV claims.

## Reproduce and evaluate

The [case README](../../cases/streaming-split/README.md) is the two-click
reproduction entry from the [documentation index](../README.md): it contains
the `sw_emu` build, four weight-port connectivity, Host checks, and expected
scope. The [case design record](../../cases/streaming-split/docs/design.md)
holds detailed operator and optimization history.

CSim/sw_emu and HLS evidence covers the bounded stream path and local resource
estimates. Any full-layer number is an analytical composition, not a released
end-to-end HW-Emu or board measurement.
