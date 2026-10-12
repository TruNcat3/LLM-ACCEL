import json, subprocess, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/designs/instantiate.py"
W4 = ROOT / "configs/designs/quantized-layer-w4-pipeline.json"

class DesignInstanceTest(unittest.TestCase):
    def test_public_case_materializes_and_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "instance"
            run = lambda *args: subprocess.run(["python3", "-B", str(TOOL), *map(str, args)], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(run("validate", W4).returncode, 0)
            self.assertEqual(run("instantiate", W4, "--output", out).returncode, 0)
            self.assertTrue((out / "build").is_file())
            self.assertEqual(run("verify", out).returncode, 0)
            self.assertEqual(run("instantiate", W4, "--output", out).returncode, 2)

if __name__ == "__main__": unittest.main()
