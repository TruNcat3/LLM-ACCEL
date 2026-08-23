# LLM-ACCEL Documentation

[Repository](../README.md) | [Setup](environment.md) |
[Evidence index](../results/README.md) | [License](../LICENSE)

The documentation is organized by question rather than by implementation
history. Start with the shortest path that matches your goal.

## Understand the research

1. [Architecture](architecture.md) explains the controller/compute split,
   memory hierarchy, packed stream ABI, five-stage pipeline, online attention,
   and coarse-task execution model.
2. [Design Space and Alternatives](design-space.md) compares kernel
   partitioning, array shapes, stream granularity, attention algorithms,
   prefill scheduling, and decode-utilization candidates.
3. [Experimental Results](experiments.md) defines the evidence ladder and
   consolidates performance, correctness, resource, and limitation tables.

## Reproduce the work

- [Environment Setup](environment.md) defines the supported toolchain, host
  dependencies, platform variables, resource guards, and four preflight modes.
- [Usage and Reproduction](usage.md) is the authoritative command reference
  after environment preflight: profiles, CSim, finite-buffer RTL CoSim, XO
  export, HW Emu, artifact inspection, and release checks.
- [Published Experimental Evidence](../results/README.md) indexes immutable
  result packages and the precise claim supported by each package.

## Study a subsystem

- [Controller-Resident Coarse-Task Runtime](coarse-task-runtime.md) documents
  Task 18/19/20, HBM hidden-state ping-pong, controller-owned KV, generation
  composition, numerical gates, and the current resource envelope.
- [Q2.14 Multi-Length P/D HW-Emu Results](q214-pd-length-hwemu.md) documents
  the operator-level context-length experiment, its precision checks, and its
  narrower timing boundary.

## Evidence boundaries

The repository uses the following vocabulary consistently:

- **P8** is up to eight consecutive query rows from one sequence; it is not
  batch eight.
- **G2** is two sampled output tokens. After prefill produces the first sample,
  one real D1 forward produces the second.
- **HW-Emu cycles** are modeled CU cycles, not simulator wall time and not
  physical-board timing.
- **Modeled useful-MAC efficiency** divides shape-counted useful MACs by the
  measured modeled cycles and the two-array 1,024-MAC/cycle peak. It is not a
  routed occupancy or power measurement.
- **CPU golden** is an out-of-band correctness oracle and is excluded from the
  accelerator timing boundary.

Raw measurements and source provenance belong in `results/`; interpretation
belongs in `docs/`; the root README only summarizes released evidence.
