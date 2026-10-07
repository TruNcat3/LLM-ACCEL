# Development-to-release workflow

[Documentation](README.md) | [Repository map](repository-map.md) |
[Implementation catalog](implementations.md) | [Evidence index](../results/README.md)

The development workspace and LLM-ACCEL have different purposes. Development
retains experiments and temporary builds. LLM-ACCEL publishes selected source
closures, portable reproduction tools, research explanations and traceable
evidence. Align them by implementation and artifact identity, rather than
copying an entire development directory.

## What moves together

| Development material | Public destination | Required context |
| --- | --- | --- |
| Selected resident source | Root kernel/include/host and build inputs | Source closure, profile, task/packet ABI and numerical contract |
| Alternative architecture or isolated component | Its family under cases/ | Case README, build/test entry and integration limits |
| Stable design decision | Architecture / design-space / runtime documents | Rationale, dependencies and implemented scope |
| Reproduction procedure | Environment / usage / scoped recipe | Tool release, parameters, workload and expected outputs |
| Completed measurement | New results package plus result index | Raw evidence, derived rows, source/binary identities and validation outcome |
| Failed or still-running experiment | Explicit discussion or external development record | Diagnostic status; no accepted-performance label |
| Build tree, model weights, waveform or binary | External artifact storage | Identity and retrieval details when available |

A documentation-only reorganization does not require a new hardware experiment.
A changed datapath, ABI or schedule needs its applicable C, finite-FIFO RTL,
resource and system checks before its previous numbers can be reused.

## Select a source snapshot

Identify the family, arithmetic, feature flags, model profile and executed
workload before copying files. Include local headers, wrappers, connectivity,
Host, Tcl settings and run scripts needed by that build.

For the resident E2E flow,
[`report_qwen3b_source_snapshot.sh`](../scripts/report_qwen3b_source_snapshot.sh)
enumerates the publication source closure. Source equivalence is checked by
[`report_qwen3b_build_source_equivalence.sh`](../scripts/report_qwen3b_build_source_equivalence.sh).
The publication Makefile is a separate release entry point; it must not be
represented as the historical build command unless it actually was one.

The canonical families remain separate: `cowave-fix16-2-8-64`,
`cowave-streaming-split`, `cowave-int4-4-8-128`, and
`cowave-int8-4-4-128`. The public full-layer quantized source now lives in
[`cases/quantized-layer/`](../cases/quantized-layer/) and has its own source
closure. It must not be represented as the component `quantized-block` case,
and it must not inherit the source identity of the September 30 W4 archive.

Historical evidence-only updates may predate a matching public source closure;
when they do, they must explicitly identify the missing source/build identity,
retain frozen source identities, and supply the inputs and tools needed to
reanalyze the result. The dated [W4 full-layer comparison](quantized-layer-progress.md)
is such a historical record; the newer public quantized-layer evidence has its
own package identity.

## Preserve and install evidence

Result directories are immutable. A corrected interpretation may update the
index or analysis page; changed raw evidence or a new run belongs in a new
package with its own identity.

For a completed resident E2E run, the existing workflow is:

1. Archive its Host log, build artifacts' identities, CU profile, resources and
   frozen sources with
   [archive_vitis_8x64_e2e_run.sh](../scripts/archive_vitis_8x64_e2e_run.sh).
2. Validate the archive against the run and source root using
   [verify_qwen3b_e2e_release.sh](../scripts/verify_qwen3b_e2e_release.sh).
3. Install the verified package with the installer, which refuses to replace
   an existing package:

```bash
scripts/install_vitis_8x64_e2e_result.sh /path/to/verified/archive new-result-name
```

The full argument contracts and retained E2E recipes are in [usage](usage.md).
Other families need an evidence package appropriate to their actual scope;
an HLS table is not required to pretend to be an E2E CU trace.

Document prompt/query rows, sequence batch, layers actually executed, context,
clock model, timing interval, useful-MAC denominator, numerical reference and
tolerance. Preserve distinctions between an independent oracle, production-C
equivalence, and checksum-only repeatability.

## Update the public narrative

Update [`designs/catalog.json`](../designs/catalog.json) and the generated
[implementation catalog](implementations.md), plus the source map, if
ownership or integration changes. Put reproducible commands in the family
recipe; put measured numbers in the result package and experimental report.
Add every package to [results/README.md](../results/README.md).

The root README shows selected strong or distinctive results with explicit
scope. It should link to the complete evidence matrix rather than accumulating
every experiment. Figures identify their implementation and workload.

Keep old names as aliases, existing cited paths as compatibility links, and
historical source/ABI names as reproduction identifiers. Do not renumber every
family after a FIFO or workload change. CITATION.cff versions describe a public
software release, not a benchmark configuration. Preserve author identity and
the repository's noncommercial license terms.

## Validate before publication

For documentation and existing evidence:

```bash
scripts/check_environment.sh publication
make test_publication_tree
make verify_result_checksums
```

For the full non-simulator release gate, prepare the HLS/XRT headers as described
in [environment setup](environment.md), then run `make test_publication_release`.
This gate checks Host planning, task semantics, source provenance and archived
results; it launches neither synthesis nor XSim.

Review and stage the exact release candidate before generating its root
manifest. The generator rejects untracked release files and unstaged changes:

```bash
git status --short
# Review changes, then stage the intended files with git add.
scripts/regenerate_root_checksums.sh
git add CHECKSUMS.sha256
VERIFY_ROOT_CHECKSUMS=1 make test_publication_release
make test_publication_snapshot SNAPSHOT_ARGS=--index
git diff --cached --check
```

Commit and publish the reviewed candidate through the repository's Git
workflow. Running a watcher, completing a simulator job or installing a local
archive does not itself publish a GitHub change. After publication, verify the
remote commit against the local commit being cited.

The snapshot check exports the Git index into a temporary checkout, compiles
the Host plan there, and runs the release checks using only staged files. It
cannot borrow ignored logs or stale local binaries. With no `SNAPSHOT_ARGS`,
the command checks `HEAD` instead. Evidence files must be tracked even when
local checksums already pass; `verify_tracked_evidence.py` enforces that rule.

Historical source verifiers check the archived identities by default.
`Q214_VERIFY_CURRENT_SOURCE=1 make verify_q214_resident_release` additionally
requires today's checkout to match that historical source; that stricter check
is appropriate only when reproducing that exact snapshot.
