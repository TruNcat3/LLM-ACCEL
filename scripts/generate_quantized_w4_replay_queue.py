#!/usr/bin/env python3
"""Write a dependency-aware W4 replay manifest; never starts a worker."""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/queues/quantized-w4-replay.json"
BASES = {
    "pipeline": ROOT / "configs/designs/quantized-layer-w4-pipeline.json",
    "integrated-rms2-ooo16": ROOT / "configs/designs/quantized-layer-w4-integrated-rms2-ooo16.json",
}
EXPERIMENTS = (
    ("layer", ("base", "rms1", "rms2", "rms4", "silu1", "silu2", "silu4", "wave0", "wave1", "ooo16", "ooo32", "overlap0", "overlap1"), "queued"),
    ("batch", ("b1", "b2", "b4", "b8"), "blocked-on-layer-reference"),
    ("request", ("l1-p8-d4", "l2-p8-d4", "l2-p128-d4"), "blocked-on-batch-reference"),
    ("attention", ("loadmax", "head8-merge4", "prepare", "pair", "pair-prepare"), "blocked-on-production-attention-cosim"),
)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", choices=sorted(BASES), required=True); ap.add_argument("--output", type=Path, required=True); a = ap.parse_args()
    if a.output.exists(): raise SystemExit(f"refusing to overwrite {a.output}")
    spec = json.loads(SPEC.read_text()); config = json.loads(BASES[a.base].read_text())
    rows = [{"base": a.base, "family": family, "experiment": experiment, "status": status} for family, experiments, status in EXPERIMENTS for experiment in experiments]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"schema_version": 1, "queue_id": spec["queue_id"], "base": a.base, "base_config": config, "selection_status": "selected_two_track_pending_fresh_replay", "rows": rows, "gates": spec["gates"]}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "rows": len(rows), "output": str(a.output.resolve())}, sort_keys=True))

if __name__ == "__main__": main()
