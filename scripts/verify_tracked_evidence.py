#!/usr/bin/env python3
"""Reject result packages that pass locally using files absent from the Git index."""
from pathlib import Path
import subprocess
import sys


def verify(root):
    result = subprocess.run(
        ["git", "ls-files", "--cached", "-z", "--", "results"],
        cwd=root, check=True, capture_output=True,
    )
    tracked = set(result.stdout.decode().split("\0"))
    present = {p.relative_to(root).as_posix()
               for p in (root / "results").rglob("*") if p.is_file()}
    missing = sorted(present - tracked)
    if missing:
        raise ValueError("Result files are absent from the Git index; review and stage them:\n"
                         + "\n".join(missing))
    return len(present)


if __name__ == "__main__":
    try:
        count = verify(Path(__file__).resolve().parents[1])
        print(f"TRACKED EVIDENCE PASS files={count}")
    except (ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
