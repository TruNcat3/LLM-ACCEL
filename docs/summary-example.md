# CoWave Summary And Example

[Documentation index](README.md) | [Repository](../README.md)

## Project summary

CoWave studies LLM acceleration with a model-aware controller and regular
streaming compute kernels. The controller schedules projection, attention,
normalization and FFN work while retaining hidden tensors and KV state in
accelerator memory.

The repository contains three implementation families:

| Family | Focus |
| --- | --- |
| Fix16 resident | Resident decoder execution with signed fixed-point arithmetic |
| Streaming split | A separate compute and control/cache organization |
| Quantized matrix blocks | W4A4/W8A8 matrix kernels and resource studies |

The [implementation catalog](implementations.md) maps each family to its
source, configurations and released evidence.

## Example: inspect the published artifact

Use a Linux shell with the publication tools listed in
[environment setup](environment.md), including Make, a C++ compiler, Python,
Perl and ripgrep. This example requires no FPGA card or vendor toolchain.

```bash
git clone https://github.com/TruNcat3/LLM-ACCEL.git
cd LLM-ACCEL
scripts/check_environment.sh publication
make test_publication_tree
make verify_result_checksums
```

These commands check the host prerequisites, publication structure and archived
result checksums. A successful publication-tree check reports
`PUBLICATION TREE PASS`; each command should exit successfully before continuing.

For HLS or hardware-emulation examples, choose an implementation and validation
level in [Usage](usage.md). Read [Experiments](experiments.md) when interpreting
the workload and timing scope of a released result.
