#!/usr/bin/env python3
"""Validate public name resolution, geometry and configuration isolation."""
import importlib.util
import json
import os
import re
from pathlib import Path
import shlex
import shutil
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

    def test_point_and_build_share_resolution(self):
        design, config = cowave.select(self.catalog, "cowave-int4-4-8-128", "integrated-rms2-silu4-prefill-overlap-wave")
        point = cowave.resolve_point(
            self.catalog, design["id"], config["name"], phase="compute-xo",
            output="/tmp/cowave point", prefill=127, weights="zero"
        )
        cwd, env, command = cowave.resolve(design, config, "compute-xo", "/tmp/cowave point", 127, "zero")
        self.assertEqual(point["build"]["cwd"], str(cwd))
        self.assertEqual(point["build"]["environment"], env)
        self.assertEqual(point["build"]["command"], command)
        self.assertIn("QUANTIZED_W4_LAYER_PREFILL=127", point["build"]["shell_command"])
        self.assertIn("QUANTIZED_W4_LAYER_WEIGHTS=zero", point["build"]["shell_command"])
        self.assertEqual(point["geometry"]["rows"], 8)

        clean = {k: v for k, v in os.environ.items()
                 if not k.startswith(("QUANTIZED_", "VITIS_QUANTIZED_")) and k not in ("TARGET", "FREQUENCY")}
        cli = [sys.executable, str(ROOT / "scripts/cowave.py"), "point", design["id"], config["name"],
               "--phase", "compute-xo", "--prefill", "127", "--weights", "zero",
               "--output", "/tmp/cowave point"]
        run = subprocess.run(cli, env=clean, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        emitted = json.loads(run.stdout)
        self.assertEqual(emitted["build"]["environment"], env)
        self.assertEqual(emitted["build"]["command"], command)
        self.assertEqual(emitted["workload"]["selected"]["prefill_tokens"], 127)

    def test_point_links_evidence_and_recipe_only_family(self):
        for design in self.catalog["designs"]:
            configs = design["configurations"] or [None]
            for config in configs:
                point = cowave.resolve_point(self.catalog, design["id"], config and config["name"])
                records = []
                for value in point["paths"].values():
                    records.extend(value if isinstance(value, list) else [value])
                for record in records:
                    self.assertTrue((ROOT / record["path"]).exists() if record["path"] != "." else ROOT.exists())
                    self.assertFalse(record["href"].startswith("/"))
                if config and config.get("evidence_case"):
                    self.assertEqual(point["evidence"]["case_id"], config["evidence_case"])
                    row = point["evidence"]["row"]
                    self.assertEqual(row["design"], design["id"])
                    self.assertEqual(row["config_name"], config["name"])
                if design["id"] == "cowave-streaming-split":
                    self.assertFalse(point["build"]["supported"])
                    self.assertIsNone(point["build"]["command"])
                    self.assertEqual(point["build"]["recipe_link"], point["links"]["recipe"])

    def test_point_workload_defaults_bounds_and_invalid_selection(self):
        point = cowave.resolve_point(self.catalog, "cowave-int8-4-4-128", "integrated-rms2-silu4-prefill-overlap")
        selected = point["workload"]["selected"]
        self.assertEqual(selected["prefill_tokens"], 66)
        self.assertEqual(selected["weights"], "random")
        self.assertEqual(point["workload"]["bounds"]["prefill_tokens"], {"default": 66, "min": 1, "max": 2047})
        self.assertEqual(point["build"]["environment"]["QUANTIZED_W8_LAYER_BLOCK_SIZE"], "4")
        with self.assertRaises(ValueError):
            cowave.resolve_point(self.catalog, "cowave-int8-4-4-128", "integrated-rms2-silu4-prefill-overlap", prefill=2048)
        with self.assertRaises(ValueError):
            cowave.resolve_point(self.catalog, "cowave-int8-4-4-128", "latest")

    def test_catalog_rejects_invalid_paths_and_evidence_case_ids(self):
        broken_path = json.loads(json.dumps(self.catalog))
        broken_path["designs"][0]["recipe"] = "../README.md"
        with self.assertRaises(ValueError):
            cowave.validate_catalog(broken_path)
        broken_case = json.loads(json.dumps(self.catalog))
        broken_case["designs"][1]["configurations"][0]["evidence_case"] = "w4-no-such-case"
        with self.assertRaises(ValueError):
            cowave.validate_catalog(broken_case)

    def test_generated_point_selector_is_current_and_self_contained(self):
        page = (ROOT / "docs/design-point-generator.html").read_text()
        self.assertIn('name="catalog-sha256" content="' + cowave.catalog_identity(self.catalog), page)
        self.assertIn("const DEFAULT_POINTS", page)
        self.assertIn("window.COWAVE", page)
        self.assertNotIn("https://", page)
        self.assertEqual(page, cowave.render_html(self.catalog))
        self.assertEqual((ROOT / "docs/implementations.md").read_text(), cowave.render(self.catalog))

    @unittest.skipUnless(shutil.which("node"), "Optional browser JavaScript check requires Node.js")
    def test_browser_commands_match_cli_for_all_phases(self):
        html = cowave.render_html(self.catalog)
        javascript = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
        cases, expected = [], []
        for design in self.catalog["designs"]:
            if not design["capabilities"]["build"]:
                continue
            for config in design["configurations"]:
                for phase in design["build"]["phases"]:
                    options = dict(phase=phase, output="/tmp/cowave browser's build", prefill=128, weights="random")
                    cases.append(dict(design=design["id"], config=config["name"], options=options))
                    expected.append(cowave.resolve_point(self.catalog, design["id"], config["name"],
                                                         **options)["build"])
        runner = '''const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = {window: {}};
vm.runInNewContext(input.script, context);
const results = input.cases.map(c => context.window.COWAVE.pointFor(c.design, c.config, c.options).build);
process.stdout.write(JSON.stringify(results));'''
        result = subprocess.run([shutil.which("node"), "-e", runner],
                                input=json.dumps(dict(script=javascript, cases=cases)),
                                capture_output=True, text=True, check=True)
        for actual, reference in zip(json.loads(result.stdout), expected):
            self.assertEqual(actual["environment"], reference["environment"])
            self.assertEqual(actual["command"], reference["command"])
            self.assertEqual(shlex.split(actual["terminal_command"]), shlex.split(reference["terminal_command"]))


if __name__ == "__main__":
    unittest.main()
