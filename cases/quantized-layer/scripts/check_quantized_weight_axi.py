#!/usr/bin/env python3
"""Verify the generated weight AXI adapters, not just profile compiler flags."""
import argparse
import json
import re
from pathlib import Path


def inspect_rtl(text, precision, outstanding, burst):
    ports = [f"weight{cu}{lane}" for cu in range(4)
             for lane in ("ab" if precision == "w4" else "abcd")]
    found = {}
    for match in re.finditer(
            r"\b\w+_(weight[0-3][a-d])_m_axi\s*#\s*\((.*?)\)\s*"
            r"\1_m_axi_U\s*\(", text, re.S):
        port, parameters = match.groups()
        if port in found:
            raise ValueError("Duplicate weight adapter: " + port)
        values = {}
        for name in ("NUM_READ_OUTSTANDING", "MAX_READ_BURST_LENGTH", "USER_DW"):
            value = re.search(r"\." + name + r"\s*\(\s*(\d+)\s*\)", parameters)
            if not value:
                raise ValueError(f"Missing numeric {name} on {port}")
            values[name] = int(value[1])
        found[port] = values
    if set(found) != set(ports):
        raise ValueError(f"Weight adapter coverage differs: expected {ports}, found {sorted(found)}")
    for port, values in found.items():
        expected = dict(NUM_READ_OUTSTANDING=outstanding,
                        MAX_READ_BURST_LENGTH=burst, USER_DW=256)
        if values != expected:
            raise ValueError(f"{port}: generated {values}, expected {expected}")
    return found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rtl", type=Path)
    parser.add_argument("precision", choices=("w4", "w8"))
    parser.add_argument("outstanding", type=int)
    parser.add_argument("burst", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = inspect_rtl(args.rtl.read_text(), args.precision,
                             args.outstanding, args.burst)
    except ValueError as error:
        parser.exit(1, f"QUANTIZED_WEIGHT_AXI_FAIL {error}\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"QUANTIZED_WEIGHT_AXI_PASS precision={args.precision} ports={len(result)} "
              f"outstanding={args.outstanding} burst={args.burst} bits=256")
