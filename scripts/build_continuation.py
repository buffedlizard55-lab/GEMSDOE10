#!/usr/bin/env python3
"""Build H16 endpoint-continuation features for ALL systems (final-training use).

For spatial CV the runner rebuilds these features per fold from TRAIN systems
only (see scripts/validate_candidate.py); this script is for the full-data
final model and for diagnostics.
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems10 import alignment, discovery, features, spec
from gems10.raster import sha256_file


def read_band(path: Path, name: str) -> np.ndarray:
    with rasterio.open(path) as ds:
        a = ds.read(spec.BAND_INDEX[name]).astype(np.float32)
    a[a < spec.FEATURE_INVALID_BELOW] = np.nan
    return a.astype(np.float32)


def continuation_grid(systems: list, bands: dict, lineaments: dict,
                      shape: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    out, names = alignment.ray_continuation_channels(
        bands["tmi"], lineaments, systems, ncc_band=bands["rtp"])
    return out, names


def main():
    data = ROOT / "data"
    feat = data / "training_features.tif"
    if sha256_file(feat) != spec.PINS["training_features.tif"]["sha256"]:
        raise ValueError("feature input hash mismatch")
    dest = data / "features_continuation.npy"
    with rasterio.open(data / "labels.tif") as ds:
        labels = ds.read(1)
    from gems10 import systems as S
    sys_id, n, _ = S.label_systems(labels)
    strikes = discovery.system_strikes(sys_id, np.arange(1, n + 1))
    print(f"systems: {n}, strikes: {len(strikes)}")
    bands = {b: read_band(feat, b) for b in ("tmi", "det_elev", "rtp")}
    tmi_invalid = ~np.isfinite(bands["tmi"])
    lineaments = {}
    for b in ("tmi", "det_elev"):
        lin = features.structure_tensor(bands[b], sigma=1.5,
                                        integration_sigma=4.0)
        # float32, matching the per-fold construction in validate_candidate
        lineaments[b] = type(lin)(energy=lin.energy.astype(np.float32),
                                  coherence=lin.coherence.astype(np.float32),
                                  orientation=lin.orientation.astype(np.float32))
        del bands[b]
        print(f"lineaments {b} done", flush=True)
    # ray engine needs the tmi validity pattern only (0/NaN mask, as in the
    # per-fold path); scores come from the lineaments, not the band values.
    bands["tmi"] = np.where(tmi_invalid, np.nan, 0.0).astype(np.float32)
    out, names = continuation_grid(strikes, bands, lineaments, labels.shape)
    np.save(dest, out)
    dest.with_suffix(".meta.json").write_text(json.dumps({
        "channels": names, "hypothesis": "H16", "n_systems": n,
        "input_sha256": sha256_file(feat),
        "labels_sha256": sha256_file(data / "labels.tif"),
        "module_sha256": sha256_file(ROOT / "src/gems10/alignment.py"),
        "output_sha256": sha256_file(dest),
        "radii": list(alignment.RAY_RADII), "ncc_rays": list(alignment.NCC_RADII)},
        indent=2) + "\n")
    print(dest, out.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
