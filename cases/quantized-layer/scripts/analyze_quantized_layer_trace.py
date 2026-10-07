#!/usr/bin/env python3
"""Rebuild standard-shape P/D layer metrics from saved XSim pin transitions."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_trace(path, precision):
    prefix = f"/pfm_top_wrapper/pfm_top_i/pfm_dynamic_inst/q{precision[1]}_layer_"
    signals = {}
    with path.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == ["signal", "time_ps", "value"],
                "Expected RTL transition TSV; profile CSV cannot resolve P/D")
        for row in reader:
            name = row["signal"]
            require(name.startswith(prefix), "Unexpected signal hierarchy")
            short = name[len(prefix):].replace("/inst/", "/")
            require(re.fullmatch(r"(ctrl|cu[0-3])/ap_(clk|rst_n|start|done|idle|ready)", short),
                    f"Unexpected signal: {name}")
            time = float(row["time_ps"])
            require(math.isfinite(time) and time >= 0, "Invalid transition time")
            value = row["value"].lower()
            require(value in ("0", "1", "x", "z"), f"Invalid bit value: {value}")
            events = signals.setdefault(short, [])
            require(not events or time >= events[-1][0], "Out-of-order transitions")
            events.append((time, value))
    return signals


def clock_period(events):
    rising = [time for time, value in events if value == "1"]
    require(len(rising) >= 10, "Clock must have at least ten rising edges")
    periods = [b - a for a, b in zip(rising, rising[1:])]
    period = sum(periods) / len(periods)
    require(period > 0 and max(abs(p - period) for p in periods) <= 2,
            "Clock is not stable (2 ps tolerance)")
    return period


def active_intervals(events):
    # Ignore initial reset busy; arm only after first idle=1.
    armed, begin, intervals = False, None, []
    for time, value in events:
        if value == "1":
            if begin is not None:
                require(time > begin, "Empty active interval")
                intervals.append((begin, time))
                begin = None
            armed = True
        elif value == "0" and armed and begin is None:
            begin = time
        elif value not in ("0", "1") and armed:
            raise ValueError("Unknown ap_idle after initialization")
    require(begin is None, "Incomplete active interval; simulation has not finished")
    require(len(intervals) == 2, f"Expected Prefill and Decode, found {len(intervals)} intervals")
    return intervals


def verify_host(path, precision, prefill, block):
    log = path.read_text()
    status = re.search(r"^exit_status=([0-9]+)$", log, re.M)
    require(status is not None, "Host worker did not record an exit status")
    # Archived failed-wrapper runs may be inspected diagnostically, but only
    # a clean worker exit is acceptable for new performance evidence.
    require(status[1] == "0", "Host worker did not exit successfully")
    markers = (f"Q{precision[1]} FULL-LAYER HW EMU {status} prefill={prefill} decode=1 block_size={block}"
               for status in ("EXECUTION PASS", "PASS"))
    # Legacy archives used PASS for execution completion as well. Numerical
    # acceptance still requires the separate reference/output comparison.
    require(any(marker in log for marker in markers),
            "Host workload or completion marker does not match")
    expected = {"prefill": (prefill, 0, 0, math.ceil(prefill / block)),
                "decode": (1, prefill, prefill, 1)}
    for phase, (seq, pos, context, blocks) in expected.items():
        marker = (f"phase={phase} sequence_length={seq} position={pos} "
                  f"kv_context_length={context} block_size={block} blocks={blocks} tasks=")
        require(log.count(marker) == 1, f"Missing/duplicate {phase} completion")
    mode = re.search(r"^weight_mode=(zero|random)$", log, re.M)
    require(mode is not None, "Missing weight mode")
    identities = {}
    for field in ("host_sha256", "xclbin_sha256", "emconfig_sha256"):
        match = re.search(rf"^{field}=([0-9a-f]{{64}})$", log, re.M)
        require(match is not None, f"Missing {field}")
        identities[field] = match[1]
    return mode[1], identities


def analyze(trace, host, precision, prefill, block, target_mhz):
    physical_rows = 8 if precision == "w4" else 4
    require(1 <= prefill <= 2047 and 1 <= block <= physical_rows, "Invalid P/block size")
    require(math.isfinite(target_mhz) and target_mhz > 0, "Invalid target clock")
    signals = read_trace(trace, precision)
    intervals, periods = {}, {}
    for cu in ("ctrl", "cu0", "cu1", "cu2", "cu3"):
        for pin in ("clk", "rst_n", "start", "done", "idle", "ready"):
            require(f"{cu}/ap_{pin}" in signals, f"Missing {cu}/ap_{pin}")
        periods[cu] = clock_period(signals[f"{cu}/ap_clk"])
        intervals[cu] = active_intervals(signals[f"{cu}/ap_idle"])
        done = [t for t, v in signals[f"{cu}/ap_done"] if v == "1"]
        for begin, end in intervals[cu]:
            require(any(begin < t <= end + 2 * periods[cu] for t in done),
                    f"{cu} interval lacks an ap_done assertion")
    period = periods["ctrl"]
    require(max(abs(p - period) for p in periods.values()) < 2, "CU clock domains differ")
    require(max(intervals[cu][0][1] for cu in intervals) <
            min(intervals[cu][1][0] for cu in intervals), "P/D phases overlap")
    mode, identities = verify_host(host, precision, prefill, block)
    linear_mac_per_row = 2 * 2048 * 2048 + 2 * 2048 * 256 + 3 * 2048 * 11008
    peak_mac_per_cycle = 4 * physical_rows * 128
    phases = {}
    for index, phase in enumerate(("prefill", "decode")):
        begin = min(times[index][0] for times in intervals.values())
        end = max(times[index][1] for times in intervals.values())
        require(max(times[index][0] for times in intervals.values()) <
                min(times[index][1] for times in intervals.values()), "CUs have no common phase overlap")
        cycles = (end - begin) / period
        rows = prefill if phase == "prefill" else 1
        pairs = prefill * (prefill + 1) // 2 if phase == "prefill" else prefill + 1
        # Useful MAC excludes masked/padded positions and is block invariant.
        useful_mac = rows * linear_mac_per_row + 2 * 16 * 128 * pairs
        latency_ms = cycles / (target_mhz * 1000)
        phases[phase] = {
            "query_rows": rows, "causal_attention_pairs": pairs,
            "start_ps": begin, "end_ps": end, "cycles": cycles,
            "controller_cycles": (intervals["ctrl"][index][1] - intervals["ctrl"][index][0]) / period,
            "modeled_layer_ms": latency_ms, "useful_mac": useful_mac,
            "useful_gmac_s": useful_mac / (latency_ms * 1e6),
            "modeled_useful_mac_efficiency_percent": 100 * useful_mac / (cycles * peak_mac_per_cycle),
            "projected_36_layer_ms": 36 * latency_ms,
            "cu_active_cycles_including_waits": {
                cu: (times[index][1] - times[index][0]) / period
                for cu, times in intervals.items()},
        }
    total_cycles = sum(p["cycles"] for p in phases.values())
    total_mac = sum(p["useful_mac"] for p in phases.values())
    return {
        "schema": 1, "precision": precision, "evidence": "hw_emu_rtl_handshakes",
        "model_shape": {"hidden": 2048, "ffn": 11008, "q_heads": 16, "kv_heads": 2,
                        "head_dim": 128, "measured_layers": 1, "sequence_batch": 1},
        "prefill": prefill, "block_size": block, "weight_mode": mode,
        "numerical_validation": "zero_residual_and_kv" if mode == "zero" else "not_checked",
        "host_intermediate_compute": False, "actual_clock_period_ps": period,
        "actual_simulation_mhz": 1e6 / period, "target_mhz": target_mhz,
        "logical_peak_gmac_s": peak_mac_per_cycle * target_mhz / 1000,
        "phases": phases, "p_plus_d_cycles_excluding_host_gap": total_cycles,
        "p_plus_d_useful_mac": total_mac,
        "p_plus_d_modeled_efficiency_percent": 100 * total_mac / (total_cycles * peak_mac_per_cycle),
        "projected_36_layer_decode_tokens_s": 1000 / phases["decode"]["projected_36_layer_ms"],
        "projection_scope": "linear decoder-layer scaling; excludes embedding, final norm, LM head, sampling and PCIe; only sampled context",
        "limitations": "synthetic weights; no trained-checkpoint accuracy, physical-board timing, or PE occupancy claim; emulated memory/interconnect",
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "host_log_sha256": hashlib.sha256(host.read_bytes()).hexdigest(), **identities,
    }


def attach_numerical_validation(result, host, reference, actual):
    spec = importlib.util.spec_from_file_location(
        "quantized_output_comparison", Path(__file__).with_name("compare_quantized_layer_outputs.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    numerical = module.compare(reference, actual)
    require(numerical["numerical_validation"] == "PASS", "Full-layer numerical comparison failed")
    metadata = numerical["metadata"]
    expected = {"bits": int(result["precision"][1]), "prefill": result["prefill"],
                "block": result["block_size"], "weights": result["weight_mode"],
                "layer": 0}
    expected.update({k: result["model_shape"][k] for k in
                     ("hidden", "ffn", "q_heads", "kv_heads", "head_dim")})
    require(all(metadata[k] == v for k, v in expected.items()),
            "Numerical workload does not match the timed workload")
    hashes = re.findall(r"^numerical_dump_sha256=([0-9a-f]{64})$", host.read_text(), re.M)
    require(hashes == [numerical["actual_sha256"]],
            "Numerical output is not bound to this Host run")
    result["numerical_validation"] = "c_model_hidden_and_kv"
    result["numerical_evidence"] = numerical
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transitions", type=Path)
    parser.add_argument("precision", choices=("w4", "w8"))
    parser.add_argument("prefill", type=int)
    parser.add_argument("block_size", type=int)
    parser.add_argument("target_mhz", type=float)
    parser.add_argument("host_log", type=Path)
    parser.add_argument("--numerical-reference", type=Path)
    parser.add_argument("--numerical-output", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.transitions, args.host_log, args.precision,
                         args.prefill, args.block_size, args.target_mhz)
        require(bool(args.numerical_reference) == bool(args.numerical_output),
                "Both numerical reference and actual output are required")
        if args.numerical_reference:
            result = attach_numerical_validation(result, args.host_log,
                args.numerical_reference, args.numerical_output)
    except (ValueError, OSError, KeyError) as error:
        print(f"Trace validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
