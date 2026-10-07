#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LLM_ACCEL_SKIP_RESOURCE_CHECK=1 scripts/check_environment.sh publication >/dev/null

if scripts/check_environment.sh invalid-mode >/dev/null 2>&1; then
    echo "Environment checker accepted an invalid mode" >&2
    exit 1
fi

if scripts/setup_environment.sh >/dev/null 2>&1; then
    echo "Environment setup incorrectly allowed execution instead of sourcing" >&2
    exit 1
fi

if rg -n '/home/(hepc|wt)/|/tools/Xilinx/Vitis/2022\.2/bin/v\+\+' \
    scripts docs README.md cases; then
    echo "Public environment flow contains a machine-private setup or tool path" >&2
    exit 1
fi

if ! rg -F -q 'source ../../scripts/setup_environment.sh' \
    cases/streaming-split/README.md \
    cases/streaming-split/tcl/run_v8_2x2_swemu.sh; then
    echo "Streaming-split case does not share the public environment resolver" >&2
    exit 1
fi

for mode in publication hls hw-emu board; do
    if ! rg -F -q "${mode}" docs/environment.md; then
        echo "Environment document is missing mode: ${mode}" >&2
        exit 1
    fi
done

python3 - <<'PY'
import os
from pathlib import Path
import subprocess
import tempfile

root = Path.cwd()
with tempfile.TemporaryDirectory(prefix="cowave-env-") as directory:
    scratch = Path(directory)
    good, bad = scratch / "good.sh", scratch / "bad.sh"
    good.write_text("return 0\n")
    bad.write_text("return 42\n")
    platform = scratch / "custom.xpfm"
    platform.write_text("fake platform for environment test\n")
    env = dict(os.environ, XILINX_XRT=str(scratch / "xrt"),
               XPLATFORM=str(platform), LLM_ACCEL_ENV_QUIET="1")
    for settings, expected in ((good, 0), (bad, 42), (scratch / "missing.sh", 66)):
        for conditional in (False, True):
            command = ('if source scripts/setup_environment.sh; then rc=0; else rc=$?; fi'
                       if conditional else 'source scripts/setup_environment.sh; rc=$?')
            command += '; declare -F llm_accel_setup_environment >/dev/null && exit 99; exit "$rc"'
            result = subprocess.run(["bash", "--noprofile", "--norc", "-c", command],
                                    env=dict(env, VITIS_ENV_SCRIPT=str(settings)), capture_output=True)
            assert result.returncode == expected, (settings, conditional, result.returncode, result.stderr)
    result = subprocess.run(["bash", "-c", "source scripts/setup_environment.sh"],
                            env=dict(env, VITIS_ENV_SCRIPT=str(good), XPLATFORM=str(scratch / "absent.xpfm")),
                            capture_output=True)
    assert result.returncode == 66, result.stderr
    # Query make's resolved variable without executing any recipe/toolchain.
    query = '$(info REVIEW_PLATFORM=$(XPLATFORM))\n.PHONY: review_platform\nreview_platform:;@:\n'
    for command_line in ([], ["XPLATFORM=" + str(platform)]):
        result = subprocess.run(["make", "--no-print-directory", "-f", "Makefile", "-f", "-",
                                 "review_platform", *command_line], input=query, text=True,
                                env=dict(env, XPLATFORM=str(platform)), capture_output=True)
        assert result.returncode == 0, result.stderr
        assert "REVIEW_PLATFORM=" + str(platform) in result.stdout, result.stdout
print("ENVIRONMENT ERROR PROPAGATION AND PLATFORM OVERRIDE PASS")
PY

echo "ENVIRONMENT CONTRACT PASS modes=publication,hls,hw-emu,board private_paths=0"
