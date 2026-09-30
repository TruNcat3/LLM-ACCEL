#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
profiles=(legacy attention pipeline integrated)
tmp=$(mktemp -d "${TMPDIR:-/tmp}/quantized-layer-w4-verify.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

fail() {
    printf 'verify.sh: %s\n' "$*" >&2
    exit 1
}

cd "$ROOT"

[[ -f checksums.sha256 ]] || fail "missing checksums.sha256"
sha256sum -c checksums.sha256 >/dev/null || fail "package checksum mismatch"

"$PYTHON" - "$ROOT" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = root / "checksums.sha256"
manifest = {}
for line_number, line in enumerate(manifest_path.read_text().splitlines(), 1):
    fields = line.split(maxsplit=1)
    if len(fields) != 2 or len(fields[0]) != 64:
        raise SystemExit(f"malformed checksum line {line_number}")
    digest, name = fields
    if name.startswith("*"):
        name = name[1:]
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"unsafe checksum path: {name}")
    key = path.as_posix()
    if key in manifest:
        raise SystemExit(f"duplicate checksum path: {name}")
    manifest[key] = digest

actual = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path.name != "checksums.sha256"
}
if set(manifest) != actual:
    missing = sorted(actual - set(manifest))
    extra = sorted(set(manifest) - actual)
    raise SystemExit(f"checksum coverage mismatch: missing={missing} extra={extra}")
for name, expected in manifest.items():
    observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"checksum mismatch: {name}")
PY

for profile in "${profiles[@]}"; do
    raw="profiles/$profile/raw"
    [[ -f "$raw/performance.json" ]] || fail "$profile: missing performance.json"
    [[ -f "$raw/numerical.json" ]] || fail "$profile: missing numerical.json"
    [[ -f "$raw/quantized_layer_transitions.tsv" ]] || fail "$profile: missing trace"
    [[ -f "$raw/host.log" ]] || fail "$profile: missing host.log"
    [[ -f "$raw/reference_outputs.tsv" ]] || fail "$profile: missing reference dump"
    [[ -f "$raw/actual_outputs.tsv" ]] || fail "$profile: missing actual dump"
    [[ -f "profiles/$profile/profile_identity.txt" ]] || \
        fail "$profile: missing profile identity metadata"
    grep -Fqx "profile=$profile" "profiles/$profile/profile_identity.txt" || \
        fail "$profile: profile identity metadata mismatch"
    grep -Eq "^profile=$profile( |$)" "$raw/host.log" || \
        fail "$profile: Host log profile binding mismatch"
    shell_source=$(awk '$2 == "scripts/quantized_layer_profiles.sh" { print $1 }' \
        hardware_source.sha256)
    tcl_source=$(awk '$2 == "tcl/quantized_layer_profile.tcl" { print $1 }' \
        hardware_source.sha256)
    [[ -n "$shell_source" && -n "$tcl_source" ]] || \
        fail "$profile: profile helper hashes missing from hardware source identity"
    source_identity=$(printf 'shell=%s\ntcl=%s\n' "$shell_source" "$tcl_source" |
        sha256sum | awk '{ print $1 }')
    profile_source=$(sed -n -E \
        's/^profile_source_sha256=([^ ]+) profile_cflags_sha256=([^ ]+)$/\1/p' \
        "$raw/host.log")
    profile_flags=$(sed -n -E \
        's/^profile_source_sha256=([^ ]+) profile_cflags_sha256=([^ ]+)$/\2/p' \
        "$raw/host.log")
    [[ "$profile_source" == "$source_identity" ]] || \
        fail "$profile: Host profile helper identity mismatch"
    flags_identity=$(tail -n +2 "profiles/$profile/profile_identity.txt" |
        sha256sum | awk '{ print $1 }')
    [[ "$profile_flags" == "$flags_identity" ]] || \
        fail "$profile: Host profile CFLAGS identity mismatch"

    "$PYTHON" analysis/analyze_quantized_layer_trace.py \
        "$raw/quantized_layer_transitions.tsv" w4 66 8 200 \
        "$raw/host.log" \
        --numerical-reference "$raw/reference_outputs.tsv" \
        --numerical-output "$raw/actual_outputs.tsv" \
        >"$tmp/$profile.performance.json"
    "$PYTHON" analysis/compare_quantized_layer_outputs.py \
        "$raw/reference_outputs.tsv" "$raw/actual_outputs.tsv" \
        >"$tmp/$profile.numerical.json"

    "$PYTHON" - "$raw/performance.json" "$tmp/$profile.performance.json" \
        "$raw/numerical.json" "$tmp/$profile.numerical.json" "$profile" <<'PY'
import json
import sys

for expected_path, observed_path, label in (
    (sys.argv[1], sys.argv[2], "performance"),
    (sys.argv[3], sys.argv[4], "numerical"),
):
    with open(expected_path) as stream:
        expected = json.load(stream)
    with open(observed_path) as stream:
        observed = json.load(stream)
    if expected != observed:
        raise SystemExit(f"{sys.argv[5]}: {label}.json semantic mismatch")
print(f"{sys.argv[5]}: regenerated performance.json and numerical.json match")
PY
done

printf 'quantized-layer-w4-20260930: PASS (%s profiles, checksum coverage complete)\n' "${#profiles[@]}"
