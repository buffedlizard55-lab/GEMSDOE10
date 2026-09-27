"""Canonical prediction identity, independent of TIFF container or file name."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import rasterio


def field_hash(values, footprint, *, excluded=None, geometry=None):
    """Hash finite float32 values on canonical scored support and its geometry.

    Optional catalogue exclusion detects catalogue-only changes. This flags
    equivalence on the evaluated pixel field, not guaranteed identical private
    scores (organizer kernel masking order is not public).
    """
    a = np.asarray(values, dtype="<f4")
    mask = np.asarray(footprint, dtype=bool).copy()
    if a.shape != mask.shape:
        raise ValueError("prediction/footprint shape mismatch")
    if not np.isfinite(a[mask]).all() or np.any((a[mask] < 0) | (a[mask] > 1)):
        raise ValueError("invalid prediction values inside footprint")
    if excluded is not None:
        excluded = np.asarray(excluded, dtype=bool)
        if excluded.shape != mask.shape:
            raise ValueError("exclusion shape mismatch")
        mask &= ~excluded
    if not mask.any():
        raise ValueError("empty score support")
    vals = a[mask].copy()
    vals[vals == 0] = 0  # normalize -0.0
    h = hashlib.sha256()
    h.update(json.dumps({"schema": 1, "shape": a.shape, "geometry": geometry},
                        sort_keys=True).encode())
    h.update(np.packbits(mask).tobytes())
    h.update(vals.tobytes())
    return h.hexdigest()


def raster_identity(path, template, labels):
    with rasterio.open(template) as ds:
        fp = np.isfinite(ds.read(1))
        shape, crs, transform = ds.shape, ds.crs, ds.transform
    with rasterio.open(path) as ds:
        if ds.count != 1 or ds.shape != shape or ds.crs != crs or ds.transform != transform:
            raise ValueError("prediction geometry differs from template")
        a = ds.read(1)
    with rasterio.open(labels) as ds:
        if ds.shape != shape or ds.crs != crs or ds.transform != transform:
            raise ValueError("label geometry differs from template")
        known = ds.read(1) == 1
    geom = {"crs": str(crs), "transform": list(transform)}
    return {"field_sha256": field_hash(a, fp, geometry=geom),
            "noncatalogue_sha256": field_hash(a, fp, excluded=known, geometry=geom),
            "positive_pixels": int((a[fp] > 0).sum()),
            "noncatalogue_positive_pixels": int((a[fp & ~known] > 0).sum())}


def refuse_duplicate(identity, registry):
    for entry in registry:
        for key in ("field_sha256", "noncatalogue_sha256"):
            if identity.get(key) and identity[key] == entry.get(key):
                raise ValueError(f"duplicate {key}: {entry.get('id', entry.get('file', 'prior artifact'))}")
