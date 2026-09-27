#!/usr/bin/env python3
"""Hard format gate CLI: python scripts/validate_submission.py file.tif"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gems10 import raster  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_submission.py FILE.tif", file=sys.stderr)
        return 2
    try:
        report = raster.assert_submittable(
            sys.argv[1], REPO_ROOT / "data" / "sample_submission.tif")
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    for c in report.checks:
        print(f"PASS: {c.name}")
    print("SUBMITTABLE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
