#!/usr/bin/env python3
"""Assemble restored H20 footprint vectors into a full-grid static channel stack.

    python scripts/fetch_external.py --tag ext/dem10-<run_id>   # -> data/external/dem10/
    python scripts/build_dem10_grid.py                          # -> data/external/dem10/dem10_channels.f32.npy

Input:  data/external/dem10/<channel>.f32.npy — one float32 vector per channel in
        np.nonzero(np.isfinite(sample_submission)) row-major order (the builder's
        documented `vector_order`), NaN where the 10 m DEM had no valid cells.
Output: (3730, 3292, C) float32 .npy (NaN outside the footprint) + a .meta.json
        with channel names, provenance (tag, run id, product, tiles) and hashes,
        consumed by `validate_candidate.py --hypothesis H20 --extra ...` and by
        `train_final.py`. Label-free by construction (no labels are read here).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems10.raster import sha256_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data/external/dem10")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.src / "dem10_channels.f32.npy"
    manifest = json.loads((args.src / "manifest.json").read_text())
    restored = json.loads((args.src / "RESTORED_FROM.json").read_text()) \
        if (args.src / "RESTORED_FROM.json").exists() else {}
    with rasterio.open(ROOT / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    template_sha = sha256_file(ROOT / "data/sample_submission.tif")
    if manifest.get("template_sha256") != template_sha:
        raise SystemExit("template sha256 differs from the builder's template — refuse to assemble")
    if manifest.get("footprint_pixels") != int(fp.sum()):
        raise SystemExit("footprint pixel count differs from manifest")
    if manifest.get("blocks_processed", 0) <= 0:
        raise SystemExit("manifest reports zero processed blocks")
    channels = list(manifest["channels"])
    H, W = fp.shape
    grid = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(H, W, len(channels)))
    grid[:] = np.nan
    rows, cols = np.nonzero(fp)
    stats = {}
    for i, name in enumerate(channels):
        vec_path = args.src / f"{name}.f32.npy"
        if sha256_file(vec_path) != manifest["channel_stats"][name]["sha256"]:
            raise SystemExit(f"{vec_path.name}: sha256 differs from manifest")
        v = np.load(vec_path, allow_pickle=False)
        if v.shape != rows.shape:
            raise SystemExit(f"{name}: vector length {v.shape} != footprint {rows.shape}")
        grid[rows, cols, i] = v
        finite = np.isfinite(v)
        stats[name] = {"finite_fraction": float(finite.mean()),
                       "p50": float(np.nanmedian(v)) if finite.any() else None}
    grid.flush()
    del grid
    meta = {
        "hypothesis": "H20",
        "channels": channels,
        "shape": [H, W, len(channels)],
        "dtype": "float32",
        "nan_outside_footprint": True,
        "label_free": True,
        "source": {
            "product": manifest.get("product"),
            "product_catalog": manifest.get("product_catalog"),
            "tag": restored.get("tag"), "commit": restored.get("commit"),
            "github_run_id": manifest.get("github_run_id"),
            "builder_code_sha256": manifest.get("code_sha256"),
            "tiles": [t.get("tile") or t.get("url") for t in manifest.get("tiles", [])]
            if isinstance(manifest.get("tiles"), list) else manifest.get("tiles"),
            "blocks_processed": manifest.get("blocks_processed"),
            "grid": manifest.get("grid"),
        },
        "manifest_sha256": sha256_file(args.src / "manifest.json"),
        "stats": stats,
        "sha256": sha256_file(out),
    }
    meta_path = out.with_suffix(".meta.json")  # dem10_channels.f32.meta.json (repo convention)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "stats"}, indent=1))
    print("stats:", json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
