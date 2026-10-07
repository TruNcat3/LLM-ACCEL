#!/usr/bin/env python3
"""Report bounded Prefill SiLU/Up nested-wave evidence.

The trace intentionally contains function-level handshakes, not a complete
HLS WDB.  ``ap_idle`` low intervals include FIFO, memory, and backpressure
waits.  Only actor overlap inside an observed ``enable_silu=1`` window is
reported, and this report never upgrades that interval to a PE-busy or
arithmetic-overlap claim.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


VALUE_SET = {"0", "1", "x", "z"}
MODE_NAMES = {"kind", "kind_v", "enable_silu", "enable_silu_v"}
FIFO_PIN_NAMES = {"full_n", "write", "empty_n", "read"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _basename(signal):
    return signal.rsplit("/", 1)[-1].lower()


def _actor_role(signal):
    """Classify only the selected semantic actors; unknown pins are ignored."""
    name = signal.lower()
    basename = _basename(signal)
    if basename in FIFO_PIN_NAMES and "matrix_results_U".lower() in signal.lower():
        return "matrix_results_fifo"
    if basename in MODE_NAMES:
        return "mode"
    if basename not in {"ap_idle", "ap_start", "ap_done", "ap_ready"}:
        return None
    # Specific vector actors precede their enclosing lookahead wrappers.
    if "emit_quantized_vector_operands" in name:
        return "silu_issuer"
    if "emit_quantized_lookahead_operands" in name:
        return "silu_issuer"
    if "collect_quantized_vector_results" in name:
        return "silu_collector"
    if "collect_quantized_projection_with_lookahead" in name:
        return "silu_collector"
    if "drive_quantized_projection_wave_range_banked" in name:
        return "up_issuer"
    if "drive_quantized_projection_with_lookahead" in name:
        return "up_issuer"
    if "commit_quantized_projection_wave_range" in name:
        return "up_collector"
    return None


def read_trace(path):
    pins = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == ["signal", "time_ps", "value"],
                "Expected signal/time_ps/value TSV")
        for row in reader:
            signal = row["signal"].strip()
            require(signal, "Empty signal name")
            try:
                time = float(row["time_ps"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid timestamp for {signal}") from error
            value = row["value"].strip().lower()
            require(math.isfinite(time) and time >= 0,
                    f"Invalid timestamp for {signal}")
            require(value, f"Empty transition value for {signal}")
            events = pins.setdefault(signal, [])
            require(not events or time >= events[-1][0],
                    f"Out-of-order transition: {signal}")
            require(not events or time != events[-1][0] or value != events[-1][1],
                    f"Duplicate transition: {signal} at {time}")
            events.append((time, value))
    return pins


def _low_intervals(events):
    """Return ap_idle-low spans after the first known idle-high value."""
    armed = False
    begin = None
    spans = []
    unknown = False
    for time, value in events:
        if value == "1":
            if begin is not None:
                if time > begin:
                    spans.append((begin, time))
                begin = None
            armed = True
        elif value == "0":
            if armed and begin is None:
                begin = time
        else:
            unknown = True
    if begin is not None:
        unknown = True
    return spans, unknown


def _high_intervals(events):
    """Return explicit 1-valued windows, requiring a closing 0 transition."""
    begin = None
    spans = []
    unknown = False
    for time, value in events:
        if value == "1":
            if begin is None:
                begin = time
        elif value == "0":
            if begin is not None:
                if time > begin:
                    spans.append((begin, time))
                begin = None
        else:
            unknown = True
    if begin is not None:
        unknown = True
    return spans, unknown


def _union(spans):
    result = []
    for begin, end in sorted(spans):
        if end <= begin:
            continue
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((begin, end))
    return result


def _overlap(first, second):
    total = 0.0
    left = right = 0
    while left < len(first) and right < len(second):
        total += max(0.0, min(first[left][1], second[right][1]) -
                     max(first[left][0], second[right][0]))
        if first[left][1] <= second[right][1]:
            left += 1
        else:
            right += 1
    return total


def _clip(spans, begin, end):
    return [(max(start, begin), min(finish, end))
            for start, finish in spans
            if start < end and finish > begin]


def _phase_window(performance):
    if not performance:
        return None, None
    try:
        phase = performance["phases"]["prefill"]
        return float(phase["start_ps"]), float(phase["end_ps"])
    except (KeyError, TypeError, ValueError):
        return None, None


def analyze(trace_path, performance_path=None):
    pins = read_trace(Path(trace_path))
    performance = None
    if performance_path is not None:
        performance = json.loads(Path(performance_path).read_text())
    period = None
    if performance is not None:
        try:
            period = float(performance["actual_clock_period_ps"])
        except (KeyError, TypeError, ValueError):
            period = None
    start, end = _phase_window(performance)
    actor_spans = {}
    actor_signals = {}
    unknown_idle = []
    mode_signals = []
    mode_values = set()
    enable_silu_windows = []
    unknown_mode = []
    raw_fifo_signals = []
    for signal, events in pins.items():
        role = _actor_role(signal)
        if role == "matrix_results_fifo":
            raw_fifo_signals.append(signal)
            continue
        if role == "mode":
            mode_signals.append(signal)
            mode_values.update(value for _, value in events if value not in {"x", "z"})
            if _basename(signal) in {"enable_silu", "enable_silu_v"}:
                spans, unknown = _high_intervals(events)
                enable_silu_windows.extend(spans)
                if unknown:
                    unknown_mode.append(signal)
            continue
        if role is None:
            continue
        actor_signals.setdefault(role, []).append(signal)
        if _basename(signal) != "ap_idle":
            continue
        spans, unknown = _low_intervals(events)
        if unknown:
            unknown_idle.append(signal)
        actor_spans.setdefault(role, []).extend(spans)

    if start is not None and end is not None and start < end:
        for role in list(actor_spans):
            actor_spans[role] = _clip(actor_spans[role], start, end)
        enable_silu_windows = _clip(enable_silu_windows, start, end)
        window_scope = "prefill_phase"
    else:
        window_scope = "trace"
    for role in list(actor_spans):
        actor_spans[role] = _union(actor_spans[role])
    enable_silu_windows = _union(enable_silu_windows)
    silu = _union(actor_spans.get("silu_issuer", []) +
                  actor_spans.get("silu_collector", []))
    up = _union(actor_spans.get("up_issuer", []) +
                actor_spans.get("up_collector", []))
    # The enable_silu argument is the only recorded operation selector. Do
    # not call actor overlap SiLU/Up evidence when this window is absent.
    if enable_silu_windows:
        silu = _union([(max(a, c), min(b, d))
                       for a, b in silu for c, d in enable_silu_windows
                       if max(a, c) < min(b, d)])
        up = _union([(max(a, c), min(b, d))
                     for a, b in up for c, d in enable_silu_windows
                     if max(a, c) < min(b, d)])
    else:
        silu = up = []
    function_overlap_ps = _overlap(silu, up)
    mode_available = bool(mode_signals and mode_values)
    actor_evidence = bool(silu and up and enable_silu_windows)
    status = "available" if actor_evidence else "insufficient_evidence"
    if unknown_idle or unknown_mode:
        status = "insufficient_evidence"
    scale = 1.0 / period if period and period > 0 else None
    return {
        "schema": 1,
        "evidence": "hw_emu_nested_prefill_silu_actor_handshakes",
        "status": status,
        "scope": window_scope,
        "function_intervals_include_waits": True,
        "pe_busy_evidence_available": False,
        "mode_evidence_available": mode_available,
        "enable_silu_windows_ps": sum(end - begin for begin, end in enable_silu_windows),
        "mode_signals": sorted(mode_signals),
        "mode_values": sorted(mode_values),
        "raw_fifo_signals": sorted(raw_fifo_signals),
        "actor_signals": {role: sorted(names) for role, names in actor_signals.items()},
        "unknown_idle_signals": sorted(unknown_idle),
        "unknown_mode_signals": sorted(unknown_mode),
        "function_overlap_ps_including_waits": function_overlap_ps,
        "function_overlap_cycles_including_waits":
            function_overlap_ps * scale if scale is not None else None,
        # This experiment has no validated PE occupancy semantics. Actor
        # overlap remains a waits-included interval only.
        "silu_up_compute_overlap_claim": False,
        "trace_sha256": hashlib.sha256(Path(trace_path).read_bytes()).hexdigest(),
        "performance_sha256": (hashlib.sha256(Path(performance_path).read_bytes()).hexdigest()
                                if performance_path is not None else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nested_trace", type=Path)
    parser.add_argument("performance", type=Path, nargs="?")
    parser.add_argument("--performance", dest="performance_option", type=Path)
    args = parser.parse_args()
    try:
        performance = args.performance_option or args.performance
        result = analyze(args.nested_trace, performance)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
