#!/usr/bin/env python3
"""Render the compact README figure from published, scoped evidence tables."""
import argparse
import csv
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tsv(path):
    with (ROOT / path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def render():
    l2 = tsv("results/qwen3b-e2e-l2-20260821/performance.tsv")[0]
    p8 = next(r for r in tsv("results/q214-resident-fix-20260818/performance.tsv") if r["scope"] == "Task18_Task19_Task20")
    study = json.loads((ROOT / "results/quantized-layer-20261007/summary.json").read_text())
    rows = [next(c for c in study["cases"] if c["precision"] == p and c["attention_wave"]) for p in ("w4", "w8")]
    assert study["workload"]["prefill_tokens"] == 66
    assert study["workload"]["sequence_batch"] == study["workload"]["layers"] == 1
    assert all(c["config_name"] == "integrated-rms2-silu4-prefill-overlap-wave" for c in rows)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="716" viewBox="0 0 1200 716" role="img" aria-labelledby="title desc" font-family="DejaVu Sans">',
           '<title id="title">CoWave: selected HW-Emu evidence by design and workload</title>',
           '<desc id="desc">Separate Fix16 multi-layer and resident-P8 evidence panels, followed by a matched INT4 and INT8 full-layer P66 plus Decode comparison. Latencies are modeled at 200 MHz; efficiencies use each design\'s logical matrix peak. Different panel workloads must not be interpreted as a matched Fix16 versus integer speedup.</desc>',
           '<rect width="1200" height="716" rx="16" fill="#0d1117"/>',
           '<style>.title{font-size:24px;font-weight:700;fill:#f0f6fc}.head{font-size:18px;font-weight:700;fill:#f0f6fc}.label{font-size:14px;fill:#c9d1d9}.note{font-size:12px;fill:#a2abb7}.metric{font-size:30px;font-weight:700;fill:#f0f6fc}.card{fill:#161b22;stroke:#30363d}.bar4{fill:#58a6ff}.bar8{fill:#d2a8ff}.value{font-size:14px;font-weight:700;fill:#f0f6fc}</style>']

    def text(x, y, content, style="label"):
        size, color, weight = {
            "title": (24, "#f0f6fc", 700), "head": (18, "#f0f6fc", 700),
            "label": (14, "#c9d1d9", 400), "note": (12, "#a2abb7", 400),
            "metric": (30, "#f0f6fc", 700), "value": (14, "#f0f6fc", 700),
        }[style]
        svg.append(f'<text x="{x}" y="{y}" fill="{color}" stroke="none" font-size="{size}" font-weight="{weight}">{escape(str(content))}</text>')

    def rect(x, y, width, height, style="card"):
        svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" class="{style}"/>')

    text(28, 38, "CoWave · representative results", "title")
    text(28, 62, "HW Emu evidence · modeled at 200 MHz · each panel identifies its own workload and timing boundary", "note")
    for x, data, title, workload, ms in (
        (28, l2, "Multi-layer generation-path gate", "P8 / G2 / L2 · one Prefill + one real D1", float(l2["projected_target_us"]) / 1000),
        (614, p8, "Bounded resident forward", "P8 / L1 · Attention + FFN + final norm", float(p8["latency_us_at_200mhz"]) / 1000),
    ):
        rect(x, 84, 558, 202)
        text(x + 20, 112, "cowave-fix16-2-8-64 / resident-q214", "note")
        text(x + 20, 142, title, "head")
        text(x + 20, 169, workload)
        text(x + 20, 211, f"{float(data['useful_gmac_s']):.3f}", "metric")
        text(x + 185, 210, "useful GMAC/s")
        text(x + 360, 211, f"{float(data['modeled_interval_efficiency_percent']):.3f}%", "metric")
        text(x + 20, 246, f"{ms:.3f} ms · {float(data['xsim_cycles']):,.1f} cycles")
        text(x + 20, 270, "Archived CU interval · Host computation excluded · separate source identities", "note")

    rect(28, 306, 1144, 342)
    text(48, 338, "Quantized full-layer comparison · matched P66 + D1 / context67 / B1 / L1", "head")
    text(48, 362, "Configuration: integrated-rms2-silu4-prefill-overlap-wave · outstanding32 · deterministic random weights", "note")
    text(390, 398, "Prefill latency (0–20 ms)", "value")
    text(800, 398, "Decode latency (0–1.1 ms)", "value")
    for i, row in enumerate(rows):
        y = 445 + 94 * i
        text(48, y, row["design"], "value")
        text(48, y + 23, f"{row['peak_mac_per_cycle']:,} logical MAC/cycle", "note")
        rect(390, y - 20, row["prefill_modeled_ms"] / 20 * 295, 24, "bar" + row["precision"][1])
        rect(800, y - 20, row["decode_modeled_ms"] / 1.1 * 225, 24, "bar" + row["precision"][1])
        text(690, y - 2, f"{row['prefill_modeled_ms']:.3f} ms", "value")
        text(1036, y - 2, f"{row['decode_modeled_ms']:.3f} ms", "value")
        text(390, y + 25, f"{row['prefill_efficiency_percent']:.3f}% useful-MAC efficiency", "note")
        text(800, y + 25, f"{row['decode_efficiency_percent']:.3f}% efficiency", "note")
    text(48, 600, "Each quantized run: 171,520 hidden/KV values exact against its production-C reference; wave-disabled controls archived.", "note")
    text(48, 626, "W4 has twice the logical peak of W8. Efficiency percentages alone do not measure latency or physical PE occupancy.", "note")
    text(28, 677, "Sources: qwen3b-e2e-l2-20260821 · q214-resident-fix-20260818 · quantized-layer-20261007", "note")
    text(28, 699, "Not physical-board measurements or trained-model accuracy. The Fix16 and quantized workloads differ; no cross-panel speedup is claimed.", "note")
    return "\n".join(svg + ["</svg>", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "docs/assets/results-overview.svg"
    content = render()
    if args.check:
        if target.read_text() != content:
            raise SystemExit("Figure is stale: run scripts/render_cowave_results.py")
        print("COWAVE FIGURE EVIDENCE PASS")
    else:
        target.write_text(content)
