# Quantized W4 Full-Layer HW-Emu Evidence

This package publishes the completed development measurement for one
integrated quantized decoder layer.  It is public analysis/evidence, not a
complete hardware-source reproduction package.

## Workload and boundary

The frozen contract is `P66+D1`, sequence batch `B1`, one measured decoder
layer, and block size 8.  The layer shape is `H2048/I11008`, 16 query heads,
2 KV heads, and head dimension 128.  It uses `W4A4` (4-bit weights and
4-bit activation operands in matrix kernels) with four compute CUs.  Stored
outputs remain signed Fix16 words.  The declared peak is
`4096 MAC/cycle` (`4 CUs * 8 rows * 128 lanes`).

The transition traces are XSim RTL-handshake evidence.  Their parsed clock
period is 3.334 ns (`3334 ps`, approximately 299.94 MHz in simulation).
The analysis converts the measured P and D cycle counts to the declared
200 MHz target clock for modeled latency and throughput; this is not a
physical-board timing result.  The reported P+D count is the sum of the two
phase intervals and explicitly excludes the Host gap between them.

The Host did not perform intermediate arithmetic (`host_intermediate_compute`
is false).  The CPU path is the production C reference used after inference;
it is not an independent oracle and the comparison is not a model-accuracy
claim.  Every profile compares the same four output groups, totaling 171,520
raw Fix16 values per profile (prefill hidden, decode hidden, retained K and
retained V), bit-for-bit: zero differing values and maximum raw error zero.

## Profile measurements

`legacy` is the baseline for the relative column.  P+D is the
`p_plus_d_cycles_excluding_host_gap` field from each frozen `performance.json`.

| Profile | P cycles | D cycles | P+D cycles (no Host gap) | Reduction vs. legacy | P+D modeled useful-MAC efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy | 3,100,205 | 226,084 | 3,326,289 | 0.000000% | 37.968739% |
| attention | 2,839,116 | 212,221 | 3,051,337 | 8.266029% | 41.390053% |
| pipeline | 2,664,407 | 212,388 | 2,876,795 | 13.513378% | 43.901286% |
| integrated | 2,647,038 | 204,385 | 2,851,423 | 14.276150% | 44.291920% |

These are completed `candidate=pass` HW-Emu measurements from the frozen
study, accepted for this measured scope.  Resource closure, routed timing,
and physical-board validation remain deferred until hardware performance and
profile selection converges.  This package contains no resource report and
makes no routed, physical-board, trained-checkpoint, or 36-layer
measured-performance claim.

## What is public

`profiles/<name>/raw/` retains the original performance and numerical JSON,
RTL transition trace, Host log, reference/actual output dumps, and reference
log for each profile.  `profiles/<name>/profile_identity.txt` retains the
original small profile flag snapshot (`reference/w4_profile.txt`) so the
legacy/attention/pipeline/integrated mapping is auditable.  The Host log also
binds each run to its profile-source and profile-CFLAGS hashes.  These are
identity metadata only, not a complete source closure.  `analysis/` contains
the two frozen analysis scripts, copied byte-for-byte from the study.
`hardware_source.sha256` is the frozen hardware source identity manifest only;
it is not this package's checksum manifest.

The complete hardware source closure is not published here, and the current
`cases/quantized-block` sources cannot reconstruct these full-layer HW-Emu
images.  Full hardware-source reproduction remains pending.  Large build
trees, binaries, XSim output, `steps.log`, and `simulate.log` are deliberately
excluded.

## Verification

From this directory, a standard-Python environment can reproduce both frozen
JSON reports from every retained trace/log/dump, compare them with the
archived JSON by parsed JSON semantics, and verify the package manifest:

```bash
cd results/quantized-layer-w4-20260930
./verify.sh
sha256sum -c checksums.sha256
```

The verifier writes all regenerated reports below a temporary directory and
never overwrites archive data.  The package checksum manifest covers every
file in this directory except
`checksums.sha256` itself; it also checks each profile identity snapshot
against the Host profile/CFLAGS fields and the helper hashes in
`hardware_source.sha256`.
