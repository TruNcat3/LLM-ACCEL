#!/usr/bin/env python3
"""Vendor-free regression tests for quantized-layer artifact identities."""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = REPO_ROOT / "cases" / "quantized-layer"
PROFILE_SCRIPT = "scripts/quantized_layer_profiles.sh"


class QuantizedArtifactIdentityTest(unittest.TestCase):
    def _copy_case(self, destination):
        def ignore(directory, names):
            return {name for name in names if name in {".build", "__pycache__"}}

        shutil.copytree(CASE_ROOT, destination, ignore=ignore)

    def _environment(self, case_root, precision, artifact_dir):
        env = os.environ.copy()
        # Do not let a developer's active profile/toolchain settings change a
        # deterministic identity fixture.  The test does not invoke HLS/XRT.
        for name in list(env):
            if name.startswith("QUANTIZED_") or name.startswith("QDR_"):
                env.pop(name, None)
        env.update(
            {
                "QUANTIZED_LAYER_PROFILE": "integrated",
                "QUANTIZED_LAYER_SILU_LANES": "4",
                "QUANTIZED_LAYER_RMS_LANES": "2",
                "QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING": "32",
                "QUANTIZED_LAYER_ATTENTION_WAVE": "0",
                "QUANTIZED_LAYER_PREFILL_FFN_OVERLAP": "1",
                "TARGET": "hw_emu",
                "DEVICE": "stub-device",
                "FREQUENCY": "200",
                "THREADS": "2",
                "QUANTIZED_LAYER_ARTIFACT_TARGET": "hw_emu",
                "QUANTIZED_LAYER_ARTIFACT_DEVICE": "stub-device",
                "QUANTIZED_LAYER_ARTIFACT_FREQUENCY": "200",
                "QUANTIZED_LAYER_ARTIFACT_PLATFORM": str(artifact_dir / "platform.xpfm"),
                "QUANTIZED_LAYER_ARTIFACT_CONN_CFG": str(artifact_dir / "conn.cfg"),
                "PRECISION": precision,
                "ARTIFACT_DIR": str(artifact_dir),
                "CASE_ROOT": str(case_root),
            }
        )
        return env

    def _run(self, case_root, precision, artifact_dir, command):
        shell = f"""
set -e
cd "$CASE_ROOT"
source {PROFILE_SCRIPT}
quantized_layer_profile_apply "$PRECISION"
{command}
"""
        return subprocess.run(
            ["bash", "-c", shell],
            cwd=case_root,
            env=self._environment(case_root, precision, artifact_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def _assert_ok(self, result):
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def _assert_rejected(self, result, *needles):
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"expected rejection, stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        diagnostic = f"{result.stdout}\n{result.stderr}".lower()
        self.assertTrue(
            any(needle.lower() in diagnostic for needle in needles),
            msg=f"missing actionable diagnostic ({needles}):\n{diagnostic}",
        )

    @contextmanager
    def _fixture(self, precision):
        with tempfile.TemporaryDirectory(prefix="quantized-artifact-") as temporary:
            case_root = Path(temporary) / "quantized-layer"
            self._copy_case(case_root)
            artifact_dir = case_root / "artifacts"
            artifact_dir.mkdir()
            files = {
                "controller": artifact_dir / "controller.xo",
                "compute": artifact_dir / "compute.xo",
                "xclbin": artifact_dir / "layer.xclbin",
                "host": artifact_dir / "host.exe",
                "platform": artifact_dir / "platform.xpfm",
                "conn": artifact_dir / "conn.cfg",
            }
            files["controller"].write_bytes(b"controller-v1")
            files["compute"].write_bytes(b"compute-v1")
            files["xclbin"].write_bytes(b"linked-image-v1")
            files["host"].write_bytes(b"host-v1")
            files["platform"].write_bytes(b"stub-platform-v1")
            files["conn"].write_bytes(b"[connectivity]\nsp=stub\n")

            artifact_dir_env = {
                "controller": '"$ARTIFACT_DIR/controller.xo"',
                "compute": '"$ARTIFACT_DIR/compute.xo"',
                "xclbin": '"$ARTIFACT_DIR/layer.xclbin"',
                "host": '"$ARTIFACT_DIR/host.exe"',
                "platform": '"$ARTIFACT_DIR/platform.xpfm"',
                "conn": '"$ARTIFACT_DIR/conn.cfg"',
            }
            self._assert_ok(
                self._run(
                    case_root,
                    precision,
                    artifact_dir,
                    f"""
quantized_layer_profile_write_xo_manifest "$PRECISION" {artifact_dir_env['controller']}
quantized_layer_profile_write_xo_manifest "$PRECISION" {artifact_dir_env['compute']}
quantized_layer_profile_write_xclbin_manifest "$PRECISION" {artifact_dir_env['xclbin']} \
    {artifact_dir_env['controller']} {artifact_dir_env['compute']} \
    {artifact_dir_env['platform']} {artifact_dir_env['conn']}
quantized_layer_profile_write_host_manifest "$PRECISION" {artifact_dir_env['host']}
""",
                )
            )
            yield case_root, artifact_dir, files

    def _check_all(self, case_root, artifact_dir, precision):
        return self._run(
            case_root,
            precision,
            artifact_dir,
            """
quantized_layer_profile_check_xo_reuse "$PRECISION" "$ARTIFACT_DIR/controller.xo"
quantized_layer_profile_check_xo_reuse "$PRECISION" "$ARTIFACT_DIR/compute.xo"
quantized_layer_profile_check_xclbin_reuse "$PRECISION" "$ARTIFACT_DIR/layer.xclbin" \
    "$ARTIFACT_DIR/controller.xo" "$ARTIFACT_DIR/compute.xo" \
    "$ARTIFACT_DIR/platform.xpfm" "$ARTIFACT_DIR/conn.cfg"
quantized_layer_profile_check_host_reuse "$PRECISION" "$ARTIFACT_DIR/host.exe"
""",
        )

    def test_matching_w4_and_w8_pairs_are_accepted(self):
        for precision in ("w4", "w8"):
            with self.subTest(precision=precision), self._fixture(precision) as (
                case_root,
                artifact_dir,
                _,
            ):
                self._assert_ok(self._check_all(case_root, artifact_dir, precision))

    def test_changed_kernel_or_header_rejects_xo_reuse(self):
        for relative_path in ("kernel/compute_stream.cpp", "include/model_config.hpp"):
            with self.subTest(relative_path=relative_path), self._fixture("w4") as (
                case_root,
                artifact_dir,
                _,
            ):
                source = case_root / relative_path
                source.write_bytes(source.read_bytes() + b"\n// identity regression mutation\n")
                result = self._run(
                    case_root,
                    "w4",
                    artifact_dir,
                    'quantized_layer_profile_check_xo_reuse "$PRECISION" "$ARTIFACT_DIR/controller.xo"',
                )
                self._assert_rejected(result, "source inputs changed", "rebuild")

    def test_missing_xo_manifest_is_rejected(self):
        with self._fixture("w4") as (case_root, artifact_dir, files):
            Path(f"{files['controller']}.profile.manifest").unlink()
            result = self._run(
                case_root,
                "w4",
                artifact_dir,
                'quantized_layer_profile_check_xo_reuse "$PRECISION" "$ARTIFACT_DIR/controller.xo"',
            )
            self._assert_rejected(result, "manifest", "rebuild")

    def test_rebuilt_xo_invalidates_prior_xclbin(self):
        with self._fixture("w4") as (case_root, artifact_dir, files):
            files["controller"].write_bytes(b"controller-rebuilt-v2")
            self._assert_ok(
                self._run(
                    case_root,
                    "w4",
                    artifact_dir,
                    'quantized_layer_profile_write_xo_manifest "$PRECISION" "$ARTIFACT_DIR/controller.xo"',
                )
            )
            result = self._run(
                case_root,
                "w4",
                artifact_dir,
                """
quantized_layer_profile_check_xclbin_reuse "$PRECISION" "$ARTIFACT_DIR/layer.xclbin" \
    "$ARTIFACT_DIR/controller.xo" "$ARTIFACT_DIR/compute.xo" \
    "$ARTIFACT_DIR/platform.xpfm" "$ARTIFACT_DIR/conn.cfg"
""",
            )
            self._assert_rejected(result, "xo", "relink")

    def test_changed_platform_or_connection_rejects_xclbin(self):
        for changed in ("platform", "conn"):
            with self.subTest(changed=changed), self._fixture("w8") as (
                case_root,
                artifact_dir,
                files,
            ):
                files[changed].write_bytes(files[changed].read_bytes() + b"\nmutation\n")
                result = self._run(
                    case_root,
                    "w8",
                    artifact_dir,
                    """
quantized_layer_profile_check_xclbin_reuse "$PRECISION" "$ARTIFACT_DIR/layer.xclbin" \
    "$ARTIFACT_DIR/controller.xo" "$ARTIFACT_DIR/compute.xo" \
    "$ARTIFACT_DIR/platform.xpfm" "$ARTIFACT_DIR/conn.cfg"
""",
                )
                self._assert_rejected(result, "changed", "relink")

    def test_host_only_edit_keeps_kernel_pair_valid_but_rejects_host(self):
        with self._fixture("w4") as (case_root, artifact_dir, files):
            host_source = case_root / "host" / "host_control_cache_quantized_w4_layer_hwemu.cpp"
            host_source.write_bytes(host_source.read_bytes() + b"\n// host-only identity mutation\n")

            self._assert_ok(
                self._run(
                    case_root,
                    "w4",
                    artifact_dir,
                    'quantized_layer_profile_check_xo_reuse "$PRECISION" "$ARTIFACT_DIR/controller.xo"',
                )
            )
            self._assert_ok(
                self._run(
                    case_root,
                    "w4",
                    artifact_dir,
                    """
quantized_layer_profile_check_xclbin_reuse "$PRECISION" "$ARTIFACT_DIR/layer.xclbin" \
    "$ARTIFACT_DIR/controller.xo" "$ARTIFACT_DIR/compute.xo" \
    "$ARTIFACT_DIR/platform.xpfm" "$ARTIFACT_DIR/conn.cfg"
""",
                )
            )
            result = self._run(
                case_root,
                "w4",
                artifact_dir,
                'quantized_layer_profile_check_host_reuse "$PRECISION" "$ARTIFACT_DIR/host.exe"',
            )
            self._assert_rejected(result, "host source inputs changed", "rebuild")


if __name__ == "__main__":
    unittest.main()
