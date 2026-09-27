#!/usr/bin/env python3
"""Emit the gated submission GeoTIFF (+ .zip + meta) from a probability map.

    python scripts/build_submission.py --prob /tmp/gems10final/prob_final.npy \
        --policy topk03_envelope --name gems10-discovery-v1

Steps: apply policy -> write_submission (official grid, NaN outside footprint)
-> assert_submittable hard gate -> sha256 + zip -> meta JSON with the suggested
submission note. Refuses to publish a byte-duplicate of any known sibling
artifact (the 0.1563 lesson: 5GEMSDOE == GEMSDOE1 bit-for-bit).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gems10 import modeling, raster, spec  # noqa: E402
from gems10.raster import sha256_file  # noqa: E402

# file-sha256 of sibling submissions we must never re-publish (byte duplicates)
KNOWN_DUPLICATES = {
    "7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15",  # GEMSDOE1/2/5 0.1563
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", required=True)
    ap.add_argument("--policy", default="topk03_envelope")
    ap.add_argument("--name", default="gems10-submission")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "docs" / "downloads"))
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    with rasterio.open(data_dir / "sample_submission.tif") as s:
        footprint = np.isfinite(s.read(1))
    prob = np.load(args.prob)
    assert prob.shape == (spec.HEIGHT, spec.WIDTH), prob.shape
    emis = modeling.apply_policy(prob, footprint, args.policy)

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / f"{args.name}-{stamp}.tif"
    raster.write_submission(emis, tif, data_dir / "sample_submission.tif",
                            footprint=footprint)
    report = raster.assert_submittable(tif, data_dir / "sample_submission.tif")
    sha = sha256_file(tif)
    if sha in KNOWN_DUPLICATES:
        tif.unlink()
        print("REFUSED: byte-duplicate of a known sibling artifact", file=sys.stderr)
        return 1
    zpath = tif.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tif, arcname=tif.name)
    note = args.note or f"{args.name} | {args.policy} | sha {sha[:10]}"
    meta = {
        "name": args.name, "policy": args.policy, "generated_utc": stamp,
        "file": tif.name, "sha256": sha, "bytes": tif.stat().st_size,
        "zip": zpath.name, "zip_sha256": sha256_file(zpath),
        "suggested_note": note,
        "footprint_px": int(footprint.sum()),
        "positive_px": int((np.nan_to_num(emis, nan=0.0) > 0).sum()),
        "mass": float(np.nan_to_num(emis, nan=0.0).sum()),
        "gate": [{"name": c.name, "ok": bool(c.ok), "detail": str(c.detail),
                    "hard": bool(c.hard)} for c in report.checks],
    }
    tif.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "gate"}, indent=1))
    print("GATE: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
