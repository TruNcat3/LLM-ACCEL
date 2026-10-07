#!/usr/bin/env python3
"""Verify the compact CoWave evidence package using Python's standard library."""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def fail(message):
    raise ValueError(message)


def close(left, right, tolerance=1e-6):
    if not math.isfinite(left) or abs(left - right) > tolerance:
        fail(f"value mismatch: {left} != {right}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pin_rows():
    path = ROOT / "evidence" / "pin_cycles.tsv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = ["case_id", "signal", "time_ps", "value"]
        if reader.fieldnames != expected:
            fail(f"unexpected pin header: {reader.fieldnames}")
        rows = {}
        for row in reader:
            case = row["case_id"]
            signal = row["signal"]
            try:
                time = float(row["time_ps"])
            except ValueError as error:
                fail(f"invalid pin time: {error}")
            if not math.isfinite(time) or time < 0 or row["value"] not in ("0", "1"):
                fail(f"invalid pin row: {row}")
            rows.setdefault(case, {}).setdefault(signal, []).append((time, row["value"]))
        return rows


def clock_period(events):
    rising = [time for time, value in events if value == "1"]
    if len(rising) < 10:
        fail("clock has fewer than ten rising edges")
    periods = [right - left for left, right in zip(rising, rising[1:])]
    period = sum(periods) / len(periods)
    if period <= 0 or max(abs(item - period) for item in periods) > 2:
        fail("clock period is not stable")
    return period


def idle_intervals(events):
    values = [value for _, value in events]
    if values != ["1", "0", "1", "0", "1"]:
        fail(f"unexpected ap_idle sequence: {values}")
    return [(events[1][0], events[2][0]), (events[3][0], events[4][0])]


def check_done(done, intervals, period):
    assertions = [time for time, value in done if value == "1"]
    if len(assertions) != 2:
        fail("expected two ap_done assertions")
    for begin, end in intervals:
        if not any(begin < time <= end + 2 * period for time in assertions):
            fail("an active interval has no ap_done assertion")


def read_manifest():
    path = ROOT / "evidence" / "pin_manifest.tsv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return {row["case_id"]: row for row in reader}


def verify_case(case, pins, pin_manifest, target_clock_mhz):
    case_id = f"{case['precision']}-{'wave' if case['attention_wave'] else 'ref'}"
    if case_id not in pins or case_id not in pin_manifest:
        fail(f"missing pin case: {case_id}")
    prefix = f"/pfm_top_wrapper/pfm_top_i/pfm_dynamic_inst/q{case['precision'][1]}_layer_"
    units = ["ctrl", "cu0", "cu1", "cu2", "cu3"]
    signals = pins[case_id]
    for unit in units:
        required = {
            f"{prefix}{unit}/inst/ap_{pin}": pin
            for pin in ("clk", "idle", "done", "start", "rst_n")
        }
        if set(required) - set(signals):
            fail(f"missing selected pins for {case_id}/{unit}")
        if signals[next(path for path in required if path.endswith("/ap_rst_n"))] != [(0.0, "0"), (518537.0, "1")]:
            fail(f"reset pin changed for {case_id}/{unit}")
    period = clock_period(signals[f"{prefix}ctrl/inst/ap_clk"])
    close(period, 3334.0, 1e-9)
    intervals = {}
    for unit in units:
        intervals[unit] = idle_intervals(signals[f"{prefix}{unit}/inst/ap_idle"])
        check_done(signals[f"{prefix}{unit}/inst/ap_done"], intervals[unit], period)
    for phase_index, field in enumerate(("prefill_cycles", "decode_cycles")):
        begins = [intervals[unit][phase_index][0] for unit in units]
        ends = [intervals[unit][phase_index][1] for unit in units]
        begin, end = min(begins), max(ends)
        if max(begins) >= min(ends):
            fail(f"no common phase interval for {case_id}")
        cycles = (end - begin) / period
        close(cycles, case[field], 1e-6)
    total = case["prefill_cycles"] + case["decode_cycles"]
    close(total, case["p_plus_d_cycles"], 1e-6)
    linear = 2 * 2048 * 2048 + 2 * 2048 * 256 + 3 * 2048 * 11008
    prefill_pairs = 66 * 67 // 2
    decode_pairs = 67
    prefill_mac = 66 * linear + 2 * 16 * 128 * prefill_pairs
    decode_mac = linear + 2 * 16 * 128 * decode_pairs
    if case["prefill_useful_mac"] != prefill_mac or case["decode_useful_mac"] != decode_mac:
        fail(f"useful-MAC count mismatch for {case_id}")
    if case["useful_mac"] != prefill_mac + decode_mac:
        fail(f"total useful-MAC count mismatch for {case_id}")
    peak = 4 * (8 if case["precision"] == "w4" else 4) * 128
    if case["peak_mac_per_cycle"] != peak:
        fail(f"peak MAC count mismatch for {case_id}")
    close(case["prefill_efficiency_percent"], 100 * prefill_mac / (case["prefill_cycles"] * peak), 1e-9)
    close(case["decode_efficiency_percent"], 100 * decode_mac / (case["decode_cycles"] * peak), 1e-9)
    close(case["efficiency_percent"], 100 * (prefill_mac + decode_mac) / (total * peak), 1e-9)
    close(case["prefill_modeled_ms"], case["prefill_cycles"] / (target_clock_mhz * 1000), 1e-9)
    close(case["decode_modeled_ms"], case["decode_cycles"] / (target_clock_mhz * 1000), 1e-9)
    close(case["peak_gmac_s"], peak * target_clock_mhz / 1000, 1e-9)
    if pin_manifest[case_id]["selected_rows"] != str(sum(len(v) for v in signals.values())):
        fail(f"selected pin row count mismatch for {case_id}")
    if pin_manifest[case_id]["source_trace_sha256"] != case["trace_sha256"]:
        fail(f"source trace hash mismatch for {case_id}")


def verify_flags(summary):
    flags = {
        "w4-ref": "cflags-w4-ref.txt",
        "w4-wave": "cflags-w4-wave.txt",
        "w8-ref": "cflags-w8-ref.txt",
        "w8-wave": "cflags-w8-wave.txt",
    }
    for case in summary["cases"]:
        key = f"{case['precision']}-{'wave' if case['attention_wave'] else 'ref'}"
        path = ROOT / "provenance" / flags[key]
        if sha256(path) != case["cflags_sha256"]:
            fail(f"cflags hash mismatch for {key}")


def verify_report_evidence(summary, hls):
    expected_keys = {
        f"{case['precision']}-{'wave' if case['attention_wave'] else 'ref'}"
        for case in summary["cases"]
    }
    if set(hls.get("cases", {})) != expected_keys:
        fail("HLS evidence case set mismatch")
    if hls.get("compute_cus") != 4:
        fail("HLS evidence compute-CU count mismatch")
    target_ns = hls.get("target_ns")
    if not isinstance(target_ns, (int, float)) or target_ns <= 0:
        fail("HLS evidence has no positive target period")
    for key, item in hls["cases"].items():
        if not item.get("all_hls_scheduling_budgets_met"):
            fail(f"HLS scheduling gate did not pass for {key}")
        if not item.get("device_capacity_exceeded"):
            fail(f"HLS capacity caveat missing for {key}")
        for stage in ("controller", "compute"):
            estimate = item.get(stage, {}).get("estimated_ns")
            if not isinstance(estimate, (int, float)) or estimate > target_ns:
                fail(f"HLS {stage} estimate exceeds target for {key}")

    numeric_lines = (ROOT / "evidence" / "numeric_comparison.log").read_text().splitlines()
    cosim_lines = (ROOT / "evidence" / "cosim.log").read_text().splitlines()
    for case in summary["cases"]:
        key = f"{case['precision']}-{'wave' if case['attention_wave'] else 'ref'}"
        numeric_prefix = f"{case['precision']} {case['candidate']}:"
        numeric = [line for line in numeric_lines if line.startswith(numeric_prefix)]
        if len(numeric) != 1:
            fail(f"numeric evidence line missing or duplicated for {key}")
        numeric_line = numeric[0]
        numeric_tokens = (
            "numerical_validation=PASS",
            f"checked_values={case['numeric_checked_values']}",
            "max_raw_error=0",
            "different_values=0",
        )
        if any(token not in numeric_line for token in numeric_tokens):
            fail(f"numeric evidence does not match summary for {key}")
        cosim_prefix = f"{case['precision']} {case['candidate']}:"
        cosim = [line for line in cosim_lines if line.startswith(cosim_prefix)]
        if len(cosim) != 1 or "C/RTL co-simulation PASS" not in cosim[0]:
            fail(f"CoSim evidence missing for {key}")
        if case["attention_wave"] and "attention CoSim PASS" not in cosim[0]:
            fail(f"wave CoSim evidence missing for {key}")


def main():
    try:
        summary = json.loads((ROOT / "summary.json").read_text())
        if summary["schema"] != 1 or len(summary["cases"]) != 4:
            fail("unexpected summary schema")
        target_clock_mhz = summary["workload"]["clock_target_mhz"]
        if not isinstance(target_clock_mhz, (int, float)) or target_clock_mhz <= 0:
            fail("invalid target clock")
        pins = read_pin_rows()
        manifest = read_manifest()
        for case in summary["cases"]:
            verify_case(case, pins, manifest, target_clock_mhz)
        verify_flags(summary)
        hls = json.loads((ROOT / "evidence" / "hls_metrics.json").read_text())
        verify_report_evidence(summary, hls)
        print("COWAVE EVIDENCE PASS cases=4 pin_rows=12340 cycles=verified numeric=PASS cosim=PASS hls=PASS")
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"COWAVE EVIDENCE FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
