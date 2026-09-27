#!/usr/bin/env python3
"""Verify (and optionally assemble) the official competition rasters.

    python scripts/fetch_data.py --data-dir data [--bridge-dir PATH]

The three official files and their sha256 pins live in src/gems10/spec.py
(PINS + BRIDGE_PARTS), measured from the official bytes:

  training_features.tif  418,912,844 B  4371c82e...  (NOT committed: >100 MB)
  labels.tif                   425,830 B  7ba308cc...  (committed)
  sample_submission.tif      1,599,597 B  2176d08e...  (committed)

Because GitHub rejects blobs >= 100 MB, the 419 MB feature stack travels as
five concatenated parts (see the manifest in 6GEMSDOE's data/bridge/ for the
same layout). With --bridge-dir pointing at a directory holding
gems-geodawn-numerical-features.tif.part-000..004, this script verifies each
part, concatenates into --data-dir/training_features.tif, and re-verifies the
whole-file hash before declaring success. Without --bridge-dir it only
verifies whatever is already in --data-dir (the CI path: labels + sample only).

Official source (login-gated): https://www.drivendata.org/competitions/306/competition-doe-gems/data/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gems10 import spec  # noqa: E402
from gems10.raster import sha256_file  # noqa: E402

PART_PREFIX = "gems-geodawn-numerical-features.tif.part-"


def verify_file(path: Path, expect_bytes: int, expect_sha: str, label: str) -> bool:
    if not path.exists():
        print(f"[MISSING] {label}: {path} not present")
        return False
    size = path.stat().st_size
    if size != expect_bytes:
        print(f"[SIZE] {label}: {size} B, expected {expect_bytes} B")
        return False
    got = sha256_file(path)
    if got != expect_sha:
        print(f"[SHA] {label}: {got}, expected {expect_sha}")
        return False
    print(f"[OK] {label}: {size} B sha256 {got[:16]}...")
    return True


def assemble_bridge(bridge_dir: Path, out_path: Path) -> bool:
    parts = []
    for name, nbytes, sha in spec.BRIDGE_PARTS:
        p = bridge_dir / f"gems-geodawn-numerical-features.tif.{name}"
        if not p.exists():  # also accept bare "<name>" filenames
            p = bridge_dir / name
        if not verify_file(p, nbytes, sha, f"bridge {name}"):
            return False
        parts.append(p)
    print(f"concatenating {len(parts)} parts -> {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        for p in parts:
            with open(p, "rb") as src:
                while True:
                    block = src.read(1 << 22)
                    if not block:
                        break
                    fh.write(block)
    pin = spec.PINS["training_features.tif"]
    return verify_file(out_path, pin["bytes"], pin["sha256"], "training_features.tif")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    ap.add_argument("--bridge-dir", default=None)
    ap.add_argument("--no-features", action="store_true",
                    help="skip training_features.tif (verify small files only)")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    ok = True
    pin = spec.PINS["training_features.tif"]
    feat = data_dir / "training_features.tif"
    if not args.no_features:
        if args.bridge_dir and not (
            feat.exists() and feat.stat().st_size == pin["bytes"]
        ):
            ok &= assemble_bridge(Path(args.bridge_dir), feat)
        else:
            ok &= verify_file(feat, pin["bytes"], pin["sha256"], "training_features.tif")
    for key, fname in (("labels.tif", "labels.tif"),
                       ("sample_submission.tif", "sample_submission.tif")):
        p = spec.PINS[key]
        ok &= verify_file(data_dir / fname, p["bytes"], p["sha256"], fname)
    print("ALL OK" if ok else "VERIFICATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
