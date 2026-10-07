#!/usr/bin/env python3
"""Compare complete P/D hidden and retained KV dumps after inference."""
import argparse
import hashlib
import json
from pathlib import Path


def read_dump(path):
    lines = path.read_text().splitlines()
    if len(lines) < 2 or lines[0] != "quantized_layer_dump_v1":
        raise ValueError("Unsupported or incomplete numerical dump")
    pairs = [field.split("=", 1) for field in lines[1].split()]
    if any(len(pair) != 2 for pair in pairs) or len({p[0] for p in pairs}) != len(pairs):
        raise ValueError("Malformed/duplicate workload metadata")
    metadata = dict(pairs)
    integer_keys = ("bits", "hidden", "ffn", "q_heads", "kv_heads", "head_dim", "prefill", "block", "layer")
    if set(metadata) != set(integer_keys) | {"weights", "fixture"}:
        raise ValueError("Missing/unexpected workload metadata")
    for key in integer_keys:
        metadata[key] = int(metadata[key])
    if (metadata["bits"] not in (4, 8) or metadata["layer"] != 0 or
            any(metadata[key] <= 0 for key in integer_keys if key != "layer") or
            metadata["block"] > (8 if metadata["bits"] == 4 else 4) or
            metadata["q_heads"] * metadata["head_dim"] != metadata["hidden"] or
            metadata["q_heads"] % metadata["kv_heads"] or
            metadata["weights"] not in ("zero", "random") or
            metadata["fixture"] != "deterministic_v1"):
        raise ValueError("Invalid workload shape/fixture")
    hidden_blocks = (metadata["hidden"] + 15) // 16
    kv_blocks = (metadata["kv_heads"] * metadata["head_dim"] + 15) // 16
    counts = {"prefill_hidden": metadata["prefill"] * hidden_blocks,
              "decode_hidden": hidden_blocks,
              "key_cache": (metadata["prefill"] + 1) * kv_blocks,
              "value_cache": (metadata["prefill"] + 1) * kv_blocks}
    tensors = {key: [] for key in counts}
    for line in lines[2:]:
        fields = line.split("\t")
        if len(fields) != 3 or fields[0] not in tensors:
            raise ValueError("Invalid tensor record")
        tensor, index, raw = fields
        if int(index) != len(tensors[tensor]):
            raise ValueError("Duplicate, missing or reordered tensor word")
        word = int(raw, 16)
        if not 0 <= word < 1 << 256:
            raise ValueError("Tensor word outside 256-bit ABI")
        tensors[tensor].append(word)
    if any(len(tensors[key]) != count for key, count in counts.items()):
        raise ValueError("Incomplete hidden/KV dump")
    return metadata, tensors


def compare(reference, actual, tolerance=0):
    if not 0 <= tolerance <= 65535:
        raise ValueError("Invalid raw Fix16 tolerance")
    metadata, expected = read_dump(reference)
    actual_metadata, observed = read_dump(actual)
    if metadata != actual_metadata:
        raise ValueError("Reference and actual workload metadata differ")
    reports = {}
    for tensor, words in expected.items():
        errors, first_mismatches = [], []
        for index, (left, right) in enumerate(zip(words, observed[tensor])):
            for lane in range(16):
                a, b = (left >> (16 * lane)) & 65535, (right >> (16 * lane)) & 65535
                a = a if a < 32768 else a - 65536
                b = b if b < 32768 else b - 65536
                error = abs(a - b)
                errors.append(error)
                if error > tolerance and len(first_mismatches) < 8:
                    first_mismatches.append({"word": index, "lane": lane,
                                             "expected_raw": a, "actual_raw": b})
        reports[tensor] = {"checked_values": len(errors), "max_raw_error": max(errors),
                           "different_values": sum(e != 0 for e in errors),
                           "outside_tolerance": sum(e > tolerance for e in errors),
                           "first_mismatches": first_mismatches}
    passed = all(r["outside_tolerance"] == 0 for r in reports.values())
    return {"schema": 1, "numerical_validation": "PASS" if passed else "FAIL",
            "reference_kind": "production_controller_and_unified_compute_C_model",
            "scope": "P hidden, D hidden and retained K/V; post-inference; independent arithmetic and RTL timing gates are separate",
            "metadata": metadata, "tolerance_raw": tolerance, "tensors": reports,
            "checked_values": sum(r["checked_values"] for r in reports.values()),
            "max_raw_error": max(r["max_raw_error"] for r in reports.values()),
            "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "actual_sha256": hashlib.sha256(actual.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--tolerance-raw", type=int, default=0)
    args = parser.parse_args()
    try:
        report = compare(args.reference, args.actual, args.tolerance_raw)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Numerical evidence error: {error}\n")
    print(json.dumps(report, indent=2))
    return 0 if report["numerical_validation"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
