# CoWave Summary And Example

[Documentation index](README.md) | [Repository](../README.md)

## Project summary

CoWave studies LLM acceleration with a model-aware controller and regular
streaming compute kernels. The controller schedules projection, attention,
normalization and FFN work while retaining hidden tensors and KV state in
accelerator memory.

Start with one of the three full-layer designs, or explore the separate
architecture and component studies:

| Design | Focus |
| --- | --- |
| `cowave-fix16-2-8-64` | Resident decoder execution with signed fixed-point arithmetic |
| `cowave-int4-4-8-128` | W4A4 full-layer controller and four packed matrix compute CUs |
| `cowave-int8-4-4-128` | W8A8 full-layer counterpart with four compute CUs |
| `cowave-streaming-split` | A separate compute and control/cache organization |
| `cowave-quantized-blocks` | Isolated W4A4/W8A8 matrix kernels and resource studies |

For the full-layer designs, names encode arithmetic, compute-CU count, rows and
columns. Configuration names identify optional mechanisms; workload dimensions
are recorded separately. The [implementation catalog](implementations.md) maps
each design to its source, configurations and released evidence.

## Example: inspect the published artifact

Use a Linux shell with the publication tools listed in
[environment setup](environment.md), including Make, a C++ compiler, Python,
Perl, Tcl and ripgrep. This example requires no FPGA card or vendor toolchain.

```bash
git clone https://github.com/TruNcat3/LLM-ACCEL.git
cd LLM-ACCEL
scripts/check_environment.sh publication
python3 scripts/cowave.py list
make test_design_catalog
make test_publication_tree
make verify_result_checksums
```

These commands check the host prerequisites, publication structure and archived
result checksums. A successful publication-tree check reports
`PUBLICATION TREE PASS`; each command should exit successfully before continuing.

For HLS or hardware-emulation examples, choose an implementation and validation
level in [Usage](usage.md). Read [Experiments](experiments.md) when interpreting
the workload and timing scope of a released result.
