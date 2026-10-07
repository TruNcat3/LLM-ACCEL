#!/usr/bin/env python3
"""Exercise the release omission that local checksum checks cannot detect."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tracking", ROOT / "scripts/verify_tracked_evidence.py")
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)


class TrackingTest(unittest.TestCase):
    def test_ignored_evidence_must_be_staged(self):
        with tempfile.TemporaryDirectory(prefix="cowave-tracking-") as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text("*.log\n")
            evidence = root / "results/example/evidence"
            evidence.mkdir(parents=True)
            (evidence / "numeric.log").write_text("numerical_validation=PASS\n")
            subprocess.run(["git", "add", ".gitignore"], cwd=root, check=True)
            with self.assertRaisesRegex(ValueError, "results/example/evidence/numeric.log"):
                tracking.verify(root)
            subprocess.run(["git", "add", "-f", "results"], cwd=root, check=True)
            self.assertEqual(tracking.verify(root), 1)


if __name__ == "__main__":
    unittest.main()
