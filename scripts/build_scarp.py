#!/usr/bin/env python3
"""Build preregistered H12 features in bounded memory; no external data needed."""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems10 import scarp, spec
from gems10.raster import sha256_file


def main():
    source = ROOT / "data/training_features.tif"
    dest = ROOT / "data/features_scarp.npy"
    if sha256_file(source) != spec.PINS["training_features.tif"]["sha256"]:
        raise ValueError("feature input hash mismatch")
    with rasterio.open(source) as ds:
        out = np.lib.format.open_memmap(dest, mode="w+", dtype="float32",
                                       shape=(ds.height, ds.width, len(scarp.CHANNELS)))
        for row in range(0, ds.height, 128):
            end = min(row + 128, ds.height)
            start_h, end_h = max(0, row - 10), min(ds.height, end + 10)
            a = ds.read(spec.BAND_INDEX["det_elev"],
                        window=Window(0, start_h, ds.width, end_h - start_h))
            a[a < spec.FEATURE_INVALID_BELOW] = np.nan
            out[row:end] = scarp.features(a)[row - start_h:end - start_h]
            print(f"H12 rows {row}:{end}", flush=True)
        out.flush()
    dest.with_suffix(".meta.json").write_text(json.dumps({
        "channels": scarp.CHANNELS, "input_sha256": sha256_file(source),
        "module_sha256": sha256_file(ROOT / "src/gems10/scarp.py"),
        "output_sha256": sha256_file(dest), "hypothesis": "H12"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
