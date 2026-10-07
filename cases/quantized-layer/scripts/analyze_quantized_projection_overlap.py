#!/usr/bin/env python3
"""Measure projection process overlap, including waits, from nested RTL pins."""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

STAGES = {
    "load": "load_quantized_projection_weight_range",
    "issue": "drive_quantized_projection_wave_range",
    "collect": "collect_quantized_projection_wave_range",
    "commit": "commit_quantized_projection_wave_range",
}
STAGE_ALIASES = {
    stage: (function,) + ({
        "issue": ("drive_quantized_projection_with_lookahead",),
        "collect": ("collect_quantized_projection_with_lookahead",),
    }.get(stage, ()))
    for stage, function in STAGES.items()
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_intervals(path, precision):
    prefix = f"/pfm_top_wrapper/pfm_top_i/pfm_dynamic_inst/q{precision[1]}_layer_ctrl/inst/"
    signals = {}
    with path.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == ["signal", "time_ps", "value"], "Expected stage transition TSV")
        for row in reader:
            name, value, time = row["signal"], row["value"].lower(), float(row["time_ps"])
            require(name.startswith(prefix) and name.endswith("/ap_idle"), "Unexpected stage hierarchy")
            module = name.rsplit("/", 2)[-2]
            matches = [stage for stage, functions in STAGE_ALIASES.items()
                       if any(function in module for function in functions)]
            require(len(matches) == 1 and "_Pipeline_" not in module, "Expected a projection process pin")
            require(math.isfinite(time) and time >= 0 and value in ("0", "1", "x", "z"),
                    "Invalid stage transition")
            events = signals.setdefault(name, (matches[0], []))[1]
            require(not events or time >= events[-1][0], "Out-of-order stage transitions")
            events.append((time, value))
    require({stage for stage, _ in signals.values()} == set(STAGES), "Missing projection stage")
    intervals = {}
    for name, (stage, events) in signals.items():
        armed, begin, spans = False, None, []
        for time, value in events:
            if value == "1":
                if begin is not None:
                    require(time > begin, "Empty stage interval")
                    spans.append((begin, time))
                    begin = None
                armed = True
            elif value == "0" and armed and begin is None:
                begin = time
            elif value not in ("0", "1") and armed:
                raise ValueError(f"Unknown stage pin after initialization: {name}")
        require(armed, f"Stage pin never initialized: {name}")
        require(begin is None, f"Incomplete stage interval: {name}")
        intervals[name] = (stage, spans)
    return intervals


def summarize(spans, begin, end, period):
    # Count a stage once even when multiple specialized instances overlap.
    changes = {begin: [], end: []}
    calls = dict.fromkeys(STAGES, 0)
    for stage, intervals in spans.values():
        for start, finish in intervals:
            start, finish = max(begin, start), min(end, finish)
            if start < finish:
                calls[stage] += 1
                changes.setdefault(start, []).append((stage, 1))
                changes.setdefault(finish, []).append((stage, -1))
    require(all(calls.values()), "A P/D phase has no activity for a projection stage")
    active = dict.fromkeys(STAGES, 0)
    histogram = dict.fromkeys(range(5), 0.0)
    totals = dict.fromkeys(STAGES, 0.0)
    exclusive = dict.fromkeys(STAGES, 0.0)
    pairs = {f"{a}+{b}": 0.0 for a, b in itertools.combinations(STAGES, 2)}
    previous = begin
    for time in sorted(changes):
        running = [stage for stage, count in active.items() if count > 0]
        duration = (time - previous) / period
        histogram[len(running)] += duration
        for stage in running:
            totals[stage] += duration
        if len(running) == 1:
            exclusive[running[0]] += duration
        for a, b in itertools.combinations(running, 2):
            pairs[f"{a}+{b}"] += duration
        for stage, delta in changes[time]:
            active[stage] += delta
        require(all(count >= 0 for count in active.values()), "Unbalanced stage events")
        previous = time
    require(not any(active.values()), "Stage events did not close")
    total_cycles = (end - begin) / period
    union = sum(histogram[n] for n in range(1, 5))
    concurrent = sum(histogram[n] for n in range(2, 5))
    require(abs(sum(histogram.values()) - total_cycles) < 1e-6, "Phase accounting mismatch")
    return {
        "phase_cycles": total_cycles,
        "projection_process_union_cycles": union,
        "outside_projection_processes_cycles": histogram[0],
        "at_least_two_stages_active_cycles": concurrent,
        "at_least_two_stages_percent_of_projection_union": 100 * concurrent / union,
        "all_four_stages_active_cycles": histogram[4],
        "cycles_by_simultaneously_active_stage_count": histogram,
        "stages": {stage: {"calls": calls[stage], "active_cycles_including_waits": totals[stage],
                           "exclusive_active_cycles": exclusive[stage]} for stage in STAGES},
        "pairwise_overlap_cycles_including_waits": pairs,
    }


def analyze(stage_trace, phase_trace, host_log, precision, prefill, block, target_mhz):
    spec = importlib.util.spec_from_file_location(
        "quantized_phase_report", Path(__file__).with_name("analyze_quantized_layer_trace.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    performance = module.analyze(phase_trace, host_log, precision, prefill, block, target_mhz)
    intervals = read_intervals(stage_trace, precision)
    period = performance["actual_clock_period_ps"]
    return {
        "schema": 1, "precision": precision, "prefill": prefill, "block_size": block,
        "evidence": "hw_emu_nested_projection_process_ap_idle",
        "scope": "process activity includes FIFO and memory waits; lookahead issue/collect wrappers also include next-block RMSNorm; not PE occupancy or successful-transfer overlap",
        "actual_clock_period_ps": period,
        "phases": {name: summarize(intervals, phase["start_ps"], phase["end_ps"], period)
                   for name, phase in performance["phases"].items()},
        "signals": {name: stage for name, (stage, _) in intervals.items()},
        "stage_trace_sha256": hashlib.sha256(stage_trace.read_bytes()).hexdigest(),
        **{key: performance[key] for key in
           ("trace_sha256", "host_log_sha256", "host_sha256", "xclbin_sha256", "emconfig_sha256")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_transitions", type=Path)
    parser.add_argument("phase_transitions", type=Path)
    parser.add_argument("host_log", type=Path)
    parser.add_argument("precision", choices=("w4", "w8"))
    parser.add_argument("prefill", type=int)
    parser.add_argument("block_size", type=int)
    parser.add_argument("target_mhz", type=float)
    args = parser.parse_args()
    try:
        result = analyze(args.stage_transitions, args.phase_transitions, args.host_log,
                         args.precision, args.prefill, args.block_size, args.target_mhz)
    except (ValueError, OSError, KeyError) as error:
        print(f"Projection overlap validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
