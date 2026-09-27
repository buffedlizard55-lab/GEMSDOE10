#!/usr/bin/env python3
"""Build preregistered H13 strip-NCC features (39 channels, label-free)."""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems10 import alignment, spec
from gems10.raster import sha256_file

BANDS = ("rtp", "tmi", "iso_grav_anom")
ANGLES = (0, 45, 90, 135)


def channel_names():
    names = []
    for b in BANDS:
        for ang in ANGLES:
            names += [f"off_{b}_ncc0_{ang}", f"off_{b}_best_{ang}", f"off_{b}_adv_{ang}"]
        names.append(f"off_{b}_advmax")
    return names


def main():
    source = ROOT / "data/training_features.tif"
    dest = ROOT / "data/features_offset.npy"
    channels = channel_names()
    if sha256_file(source) != spec.PINS["training_features.tif"]["sha256"]:
        raise ValueError("feature input hash mismatch")
    with rasterio.open(source) as ds:
        out = np.lib.format.open_memmap(dest, mode="w+", dtype="float32",
                                        shape=(ds.height, ds.width, len(channels)))
        k = 0
        for b in BANDS:
            band = ds.read(spec.BAND_INDEX[b]).astype(np.float32)
            band[band < spec.FEATURE_INVALID_BELOW] = np.nan
            # writes in place into the memmap slice (verified bit-identical
            # to the full-grid path by tests/test_alignment.py)
            alignment.offset_channels_tiled(band, angles=ANGLES,
                                            out=out[:, :, k:k + 13])
            k += 13
            del band
            out.flush()
            print(f"H13 band {b}: 13 channels done", flush=True)
        out.flush()
    assert k == len(channels)
    dest.with_suffix(".meta.json").write_text(json.dumps({
        "channels": channels, "hypothesis": "H13",
        "input_sha256": sha256_file(source),
        "module_sha256": sha256_file(ROOT / "src/gems10/alignment.py"),
        "output_sha256": sha256_file(dest),
        "window": alignment.WINDOW, "seps": list(alignment.SEPS),
        "lags": list(alignment.LAGS), "angles": list(ANGLES)}, indent=2) + "\n")
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
