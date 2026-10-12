#!/usr/bin/env python3
"""Materialize a reproducible CoWave public-case build instance."""
from __future__ import annotations
import argparse, hashlib, json, shutil, stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "templates/designs/registry.json"
SOURCE_SUFFIXES = {".h", ".hpp", ".cpp", ".c", ".py", ".sh", ".tcl"}
PARAM_KEYS = {"profile", "rms_lanes", "silu_lanes", "weight_read_outstanding", "attention_wave", "prefill_ffn_overlap"}

def canonical(v): return (json.dumps(v, sort_keys=True, separators=(",", ":")) + "\n").encode()
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def write_json(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
def fail(msg): raise ValueError(msg)

def validate(raw):
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "id", "template", "parameters", "workload", "target", "frequency_mhz"} or raw["schema_version"] != 1 or not isinstance(raw["id"], str) or not raw["id"]: fail("invalid configuration keys or schema")
    registry = json.loads(REGISTRY.read_text())
    template = registry["templates"].get(raw["template"])
    if template is None: fail("unsupported template")
    p, w = raw["parameters"], raw["workload"]
    if not isinstance(p, dict) or not isinstance(w, dict) or set(p) != PARAM_KEYS or set(w) != {"prefill_tokens", "block_size", "weights"}: fail("unknown or missing parameter/workload key")
    if p["profile"] not in {"legacy", "attention", "attention_wave", "pipeline", "integrated"}: fail("unsupported profile")
    if p["rms_lanes"] not in (1, 2, 4) or p["silu_lanes"] not in (1, 2, 4): fail("invalid lane count")
    if p["weight_read_outstanding"] not in (16, 32) or p["attention_wave"] not in (0, 1) or p["prefill_ffn_overlap"] not in (0, 1): fail("invalid profile control")
    if p["profile"] == "legacy" and (p["attention_wave"] or p["prefill_ffn_overlap"]): fail("legacy cannot enable overlap controls")
    if p["attention_wave"] and p["profile"] == "legacy": fail("attention_wave requires a non-legacy profile")
    if p["prefill_ffn_overlap"] and p["profile"] not in {"pipeline", "integrated"}: fail("prefill_ffn_overlap requires pipeline or integrated profile")
    if not 1 <= w["prefill_tokens"] <= 2047 or w["weights"] not in {"zero", "random"}: fail("invalid workload")
    if w["block_size"] != (8 if template["precision"] == "w4" else 4): fail("block size does not match template")
    if raw["target"] != "hw_emu" or raw["frequency_mhz"] != 200: fail("only hw_emu at 200 MHz is supported")
    return registry, template

def materialize(config_path, output):
    raw = json.loads(config_path.read_text()); registry, template = validate(raw)
    if output.exists(): fail(f"refusing to overwrite {output}")
    output.mkdir(parents=True); source = output / "source"; source.mkdir()
    case = ROOT / template["case_root"]
    for path in sorted(case.rglob("*")):
        if path.is_file() and path.suffix in SOURCE_SUFFIXES:
            dst = source / template["case_root"] / path.relative_to(case)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dst)
    cfg = ROOT / template["case_root"] / template["connection"]
    dst_cfg = source / template["case_root"] / template["connection"]
    dst_cfg.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(cfg, dst_cfg)
    p, w = raw["parameters"], raw["workload"]; precision = template["precision"]
    env = {"TARGET": raw["target"], "FREQUENCY": str(raw["frequency_mhz"]), "QUANTIZED_ALIGNMENT_PRECISION": precision,
           "QUANTIZED_LAYER_PROFILE": p["profile"], "QUANTIZED_LAYER_RMS_LANES": str(p["rms_lanes"]), "QUANTIZED_LAYER_SILU_LANES": str(p["silu_lanes"]),
           "QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING": str(p["weight_read_outstanding"]), "QUANTIZED_LAYER_ATTENTION_WAVE": str(p["attention_wave"]),
           "QUANTIZED_LAYER_PREFILL_FFN_OVERLAP": str(p["prefill_ffn_overlap"]), "COWAVE_QUANTIZED_LAYER_OUTPUT_DIR": str(output / "build"),
           "COWAVE_QUANTIZED_LAYER_CONFIG_DIR": str(source / template["case_root"] / "config"),
           "QUANTIZED_W4_LAYER_PREFILL": str(w["prefill_tokens"]), "QUANTIZED_W4_LAYER_BLOCK_SIZE": str(w["block_size"]), "QUANTIZED_W4_LAYER_WEIGHTS": w["weights"]}
    records = [f"{sha(x)}  {x.relative_to(source).as_posix()}" for x in sorted(source.rglob("*")) if x.is_file()]
    manifest = "\n".join(records) + "\n"; (output / "source.sha256").write_text(manifest)
    identity = {"config_sha256": hashlib.sha256(canonical(raw)).hexdigest(), "generator_sha256": sha(Path(__file__)), "source_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest()}
    resolved = dict(raw, template_spec=template, identity=identity, resolved_environment=env); write_json(output / "resolved.json", resolved)
    (output / "build.env").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(f"export {k}={json.dumps(v)}" for k, v in sorted(env.items())) + f"\nexport INSTANCE_ROOT={json.dumps(str(output))}\n")
    build = output / "build"; build.write_text("#!/usr/bin/env bash\nset -euo pipefail\nroot=$(cd \"$(dirname \"$0\")\" && pwd)\nsource \"$root/build.env\"\nexec bash \"$root/source/cases/quantized-layer/scripts/build_vitis_quantized_${QUANTIZED_ALIGNMENT_PRECISION}_layer.sh\" \"${1:-all}\"\n"); build.chmod(build.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    write_json(output / "manifest.json", {"schema_version": 1, "identity": identity, "source_file_count": len(records), "build": "build", "verification": "source.sha256"})

def verify(instance):
    resolved = json.loads((instance / "resolved.json").read_text()); manifest = (instance / "source.sha256").read_text()
    for line in manifest.splitlines():
        digest, rel = line.split("  ", 1); path = instance / "source" / rel
        if not path.is_file() or sha(path) != digest: fail(f"source verification failed: {rel}")
    if hashlib.sha256(manifest.encode()).hexdigest() != resolved["identity"]["source_manifest_sha256"]: fail("source identity mismatch")

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("validate",): sub.add_parser(n).add_argument("config", type=Path)
    i = sub.add_parser("instantiate"); i.add_argument("config", type=Path); i.add_argument("--output", required=True, type=Path)
    sub.add_parser("verify").add_argument("instance", type=Path); a = ap.parse_args()
    try:
        if a.cmd == "validate":
            _, t = validate(json.loads(a.config.read_text())); print(json.dumps({"status": "ok", "template": a.config.stem, "geometry": t["geometry"]}, sort_keys=True))
        elif a.cmd == "instantiate": materialize(a.config.resolve(), a.output.resolve()); print(json.dumps({"status": "ok", "output": str(a.output.resolve())}, sort_keys=True))
        else: verify(a.instance.resolve()); print('{"status": "ok"}')
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as e: print(f"instantiate: {e}", file=__import__("sys").stderr); return 2
if __name__ == "__main__": raise SystemExit(main())
