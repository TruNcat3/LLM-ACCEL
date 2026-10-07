#!/usr/bin/env python3
"""Validate public name resolution, geometry and configuration isolation."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cowave", ROOT / "scripts/cowave.py")
cowave = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cowave)


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.catalog = cowave.read_catalog()

    def test_unique_identity_and_geometry(self):
        seen = set()
        for design in self.catalog["designs"]:
            self.assertNotIn(design["id"], seen)
            seen.add(design["id"])
            for key in ("source", "guide", "recipe"):
                self.assertTrue((ROOT / design[key]).exists(), design[key])
            if "compute_cus" in design:
                self.assertEqual(design["peak_mac_per_cycle"], design["compute_cus"] * design["rows"] * design["columns"])
                self.assertTrue(design["id"].endswith("-{}-{}-{}".format(design["compute_cus"], design["rows"], design["columns"])))
            names = set()
            hashes = set()
            for config in design["configurations"]:
                self.assertNotIn(config["name"], names)
                names.add(config["name"])
                digest = cowave.configuration_hash(design, config)
                self.assertNotIn(digest, hashes)
                hashes.add(digest)
                for evidence in config["evidence"]:
                    self.assertTrue((ROOT / evidence / "checksums.sha256").is_file(), evidence)

    def test_geometry_does_not_multiply_dsp_packing_twice(self):
        w4, _ = cowave.select(self.catalog, "cowave-int4-4-8-128")
        w8, _ = cowave.select(self.catalog, "cowave-int8-4-4-128")
        self.assertEqual(w4["peak_mac_per_cycle"] / w8["peak_mac_per_cycle"], 2)
        self.assertEqual(w4["matrix_products_per_dsp"], 4)

    def test_configuration_and_output_isolation(self):
        design, _ = cowave.select(self.catalog, "cowave-int4-4-8-128")
        paths = set()
        for config in design["configurations"]:
            cwd, env, cmd = cowave.resolve(design, config, "host", "/tmp/cowave test", 128, "random")
            self.assertTrue((cwd / cmd[1]).is_file())
            self.assertEqual(env["QUANTIZED_W4_LAYER_PREFILL"], "128")
            self.assertEqual(env["QUANTIZED_LAYER_PREFILL_FFN_OVERLAP"], "1")
            self.assertNotIn(env["VITIS_QUANTIZED_W4_LAYER_ROOT"], paths)
            paths.add(env["VITIS_QUANTIZED_W4_LAYER_ROOT"])

    def test_dry_run_and_conflicts(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("QUANTIZED_", "VITIS_QUANTIZED_")) and k not in ("TARGET", "FREQUENCY")}
        cmd = [sys.executable, str(ROOT / "scripts/cowave.py"), "build", "cowave-int8-4-4-128", "integrated-rms2-silu4-prefill-overlap", "--dry-run"]
        run = subprocess.run(cmd, env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("QUANTIZED_LAYER_RMS_LANES=2", run.stdout)
        self.assertIn("QUANTIZED_W8_LAYER_BLOCK_SIZE=4", run.stdout)
        for key, value in (("QUANTIZED_LAYER_ATTENTION_WAVE", "1"), ("QUANTIZED_LAYER_PROFILE_MANAGED_NAMES", "stale"), ("FREQUENCY", "300")):
            failed = subprocess.run(cmd, env={**env, key: value}, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Conflicting inherited settings", failed.stderr)
        failed = subprocess.run(cmd + ["--prefill", "2048"], env=env, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)

    def test_unknown_names_do_not_fall_back(self):
        with self.assertRaises(ValueError):
            cowave.select(self.catalog, "R1")
        with self.assertRaises(ValueError):
            cowave.select(self.catalog, "cowave-int4-4-8-128", "latest")

    def test_generated_catalog_is_current(self):
        self.assertEqual((ROOT / "docs/implementations.md").read_text(), cowave.render(self.catalog))

    def test_selected_configs_match_archived_parameters(self):
        study = json.loads((ROOT / "results/quantized-layer-20261007/summary.json").read_text())
        for design in self.catalog["designs"]:
            if "precision_selector" not in design:
                continue
            self.assertEqual(len(cowave.source_identity(design)), 64)
            for config in design["configurations"]:
                matches = [c for c in study["cases"] if c["design"] == design["id"] and c["config_name"] == config["name"]]
                self.assertEqual(len(matches), 1)
                row = matches[0]
                for field in ("profile", "rms_lanes", "silu_lanes", "attention_wave", "prefill_ffn_overlap"):
                    self.assertEqual(row[field], config[field])
                self.assertEqual(row["outstanding"], config["weight_read_outstanding"])
                self.assertEqual(row["peak_mac_per_cycle"], design["peak_mac_per_cycle"])
                self.assertEqual(row["numeric"], "PASS")
                self.assertEqual(row["hwemu"], "PASS")

    def test_reproduction_resolvers_emit_archived_flags(self):
        clean = {k: v for k, v in os.environ.items()
                 if not k.startswith(("QUANTIZED_", "VITIS_QUANTIZED_", "QDR_", "CU_NL_", "CU_RMS_"))}
        for design in self.catalog["designs"]:
            if "precision_selector" not in design:
                continue
            precision = design["precision_selector"]
            for config in design["configurations"]:
                cwd, selected, _ = cowave.resolve(design, config, "host", "/tmp/cowave-test", 66, "random")
                env = {**clean, **selected}
                suffix = "wave" if config["attention_wave"] else "ref"
                expected = (ROOT / "results/quantized-layer-20261007/provenance" / ("cflags-" + precision + "-" + suffix + ".txt")).read_text().splitlines()
                shell = subprocess.run(["bash", "-c", 'source scripts/quantized_layer_profiles.sh; quantized_layer_profile_apply "$1" || exit; printf "%s\\n" "${QUANTIZED_LAYER_PROFILE_CFLAGS[@]}"', "_", precision], cwd=cwd, env=env, text=True, capture_output=True)
                self.assertEqual(shell.returncode, 0, shell.stderr)
                self.assertEqual(shell.stdout.splitlines(), expected)
                tcl = subprocess.run(["tclsh"], input="source tcl/quantized_layer_profile.tcl\nputs [quantized_layer_profile_cflags " + precision + "]\n", cwd=cwd, env=env, text=True, capture_output=True)
                self.assertEqual(tcl.returncode, 0, tcl.stderr)
                # The Tcl resolver emits the same distinct -D switches in a
                # slightly different order for W8; no redefinition is allowed.
                actual = shlex.split(tcl.stdout)
                self.assertEqual(len({flag.split("=", 1)[0] for flag in actual}), len(actual))
                self.assertCountEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
