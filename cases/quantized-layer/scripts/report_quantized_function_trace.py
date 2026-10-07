#!/usr/bin/env python3
"""Attribute controller activity using recorded function pins, clipped to P/D."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def category(name):
    if "run_quantized_vector_exchange" in name or "run_quantized_vector_inplace" in name:
        # Operand issue and result waits belong to the shared vector path.
        # Splitting this into RMS/SiLU/add requires task-mode pins as well.
        return "compute_vector_exchange"
    if "silu_multiply" in name:
        return "silu_gating"
    if "_scale_" in name:
        return "dynamic_quantization_scales"
    if "rmsnorm" in name:
        return "rmsnorm"
    if "run_quantized_" in name and "_projection_" in name:
        return "linear_projections"
    if "quantized_attention" in name or "online_attention" in name:
        return "attention"
    if "apply_quantized_" in name and "_rope" in name:
        return "rope"
    return "attention_and_io"


def union_duration(intervals):
    total, end = 0, -1
    for begin, finish in sorted(intervals):
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return total


def exclusive_group_durations(spans):
    events, active, durations = [], {}, {}
    for begin, end, group in spans:
        events.extend(((begin, 1, group), (end, -1, group)))
    previous = 0
    for time, delta, group in sorted(events):
        groups = [name for name, count in active.items() if count]
        if groups and time > previous:
            name = groups[0] if len(groups) == 1 else "overlapping_functions"
            durations[name] = durations.get(name, 0) + time - previous
        active[group] = active.get(group, 0) + delta
        previous = time
    return durations


def analyze(trace_path, performance_path):
    performance = json.loads(performance_path.read_text())
    require(performance["evidence"] == "hw_emu_rtl_handshakes", "Missing RTL phase evidence")
    precision = performance["precision"]
    prefix = f"/pfm_top_wrapper/pfm_top_i/pfm_dynamic_inst/q{precision[1]}_layer_"
    events = {}
    with trace_path.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == ["signal", "time_ps", "value"], "Expected function TSV")
        for row in reader:
            path, value, time = row["signal"], row["value"], float(row["time_ps"])
            require(path.startswith(prefix), "Unexpected precision/hierarchy")
            name = path[len(prefix):]
            require(re.fullmatch(r"(ctrl|cu[0-3])/inst/grp_[^/]+/ap_idle", name),
                    "Expected direct-child function ap_idle")
            require(value in ("0", "1") and math.isfinite(time) and time >= 0,
                    "Function pin is unknown or has an invalid timestamp")
            signal_events = events.setdefault(name, [])
            require(not signal_events or time >= signal_events[-1][0], "Out-of-order events")
            signal_events.append((time, value))
    require(events and any(name.startswith("ctrl/") for name in events), "No controller pins")
    intervals = {}
    for name, signal_events in events.items():
        armed, begin, spans = False, None, []
        for time, value in signal_events:
            if value == "1":
                if begin is not None:
                    require(time > begin, "Empty function interval")
                    spans.append((begin, time))
                    begin = None
                armed = True
            elif armed and begin is None:
                begin = time
        require(begin is None, f"Incomplete function interval: {name}")
        intervals[name] = spans

    period = performance["actual_clock_period_ps"]
    phases = {}
    for phase, window in performance["phases"].items():
        functions, controller_spans, categorized_spans = {}, [], []
        for name, spans in intervals.items():
            clipped = [(max(begin, window["start_ps"]), min(end, window["end_ps"]))
                       for begin, end in spans
                       if begin < window["end_ps"] and end > window["start_ps"]]
            cycles = sum(end - begin for begin, end in clipped) / period
            functions[name] = {"calls": len(clipped), "active_cycles_including_waits": cycles}
            if name.startswith("ctrl/"):
                group = category(name)
                controller_spans.extend(clipped)
                categorized_spans.extend((begin, end, group) for begin, end in clipped)
        covered = union_duration(controller_spans) / period
        groups = {name: time / period for name, time in
                  exclusive_group_durations(categorized_spans).items()}
        controller_cycles = window["controller_cycles"]
        require(covered <= controller_cycles + 2, "Function activity exceeds controller window")
        # Handshake boundaries can overlap by a cycle. Use disjoint groups
        # and expose overlap explicitly rather than double-counting time.
        require(abs(sum(groups.values()) - covered) <= 2, "Group accounting differs from union")
        phases[phase] = {
            "controller_cycles": controller_cycles,
            "groups": {name: {"cycles": cycles,
                               "percent_of_controller": 100 * cycles / controller_cycles}
                       for name, cycles in groups.items()},
            "unattributed_controller_cycles": controller_cycles - covered,
            "functions": functions,
        }
    return {
        "schema": 1, "precision": precision,
        "evidence": "saved_wdb_direct_child_ap_idle_transitions",
        "scope": "activity includes waits; no PE occupancy or stall-counter claim",
        "actual_clock_period_ps": period, "phases": phases,
        "function_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "performance_sha256": hashlib.sha256(performance_path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("function_transitions", type=Path)
    parser.add_argument("performance", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.function_transitions, args.performance), indent=2, allow_nan=False))
