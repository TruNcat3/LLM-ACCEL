#!/usr/bin/env python3
"""Keep execution completion distinct from numerical acceptance."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "cases/quantized-layer/scripts/analyze_quantized_layer_trace.py"
spec = importlib.util.spec_from_file_location("trace", SCRIPT)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)
RAW = ROOT / "results/quantized-layer-w4-20260930/profiles/integrated/raw"


class ExecutionStatusTest(unittest.TestCase):
    def test_legacy_and_new_execution_markers_need_numerical_comparison(self):
        legacy = (RAW / "host.log").read_text()
        current = legacy.replace("FULL-LAYER HW EMU PASS", "FULL-LAYER HW EMU EXECUTION PASS")
        current = current.replace("weight_mode=random controller_owned_kv=1",
                                  "weight_mode=random numerical_validation=NOT_RUN controller_owned_kv=1")
        with tempfile.TemporaryDirectory(prefix="cowave-status-") as directory:
            host = Path(directory) / "host.log"
            for log in (legacy, current):
                with self.subTest(marker="new" if log == current else "legacy"):
                    host.write_text(log)
                    result = trace.analyze(RAW / "quantized_layer_transitions.tsv", host, "w4", 66, 8, 200)
                    self.assertEqual(result["numerical_validation"], "not_checked")
                    result = trace.attach_numerical_validation(result, host,
                               RAW / "reference_outputs.tsv", RAW / "actual_outputs.tsv")
                    self.assertEqual(result["numerical_validation"], "c_model_hidden_and_kv")
            host.write_text(current.replace("exit_status=0", "exit_status=1"))
            with self.assertRaisesRegex(ValueError, "exit successfully"):
                trace.verify_host(host, "w4", 66, 8)


if __name__ == "__main__":
    unittest.main()
