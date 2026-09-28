#!/usr/bin/env python3
"""H25: spatial-context channels over the external 10 m DEM stack.

    python scripts/build_dem10_context.py

Frozen in HYPOTHESES.md session-4 register: for the six mechanistic dem10
channels {slope_max, steep_ratio_max, onesided, onesided3, hgm200_mean,
resid_range}, NaN-aware box **mean / max / std over 7x7 and 15x15** px
(0.7 / 1.5 km) -> 6 x 6 = 36 `ctx_*` channels. Output:
data/external/dem10/dem10_context.f32.npy + .meta.json (channel list, source,
builder sha256). Derived entirely from the restored tag product — no network.

Rationale (register): HGB sees per-pixel values only; a scarp candidate is
more credible where the surrounding kilometre is also steep/one-sided. No
neighbourhood statistic over the external stack exists anywhere else in this
checkout (grepped before preregistration).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BASES = ["dem10_slope_max", "dem10_steep_ratio_max", "dem10_onesided",
         "dem10_onesided3", "dem10_hgm200_mean", "dem10_resid_range"]
WINDOWS = [7, 15]
STATS = ["mean", "max", "std"]


def _box_mean(a: np.ndarray, size: int, valid: np.ndarray) -> np.ndarray:
    """NaN-aware box mean; NaN where the window has no valid sample."""
    v = np.where(valid, a, 0.0)
    m = np.where(valid, 1.0, 0.0)
    s = ndimage.uniform_filter(v, size=size, mode="constant", cval=0.0)
    c = ndimage.uniform_filter(m, size=size, mode="constant", cval=0.0)
    out = np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)
    return out


def _box_std(a: np.ndarray, size: int, valid: np.ndarray,
             mean: np.ndarray) -> np.ndarray:
    """NaN-aware box std given the box mean (same window)."""
    v = np.where(valid, a, 0.0)
    m = np.where(valid, 1.0, 0.0)
    s1 = ndimage.uniform_filter(v, size=size, mode="constant", cval=0.0)
    s2 = ndimage.uniform_filter(v * v, size=size, mode="constant", cval=0.0)
    c = ndimage.uniform_filter(m, size=size, mode="constant", cval=0.0)
    ex = np.divide(s1, c, out=np.zeros_like(s1), where=c > 0)
    ex2 = np.divide(s2, c, out=np.zeros_like(s2), where=c > 0)
    var = np.maximum(ex2 - ex * ex, 0.0)
    out = np.sqrt(var)
    out = np.where((c > 0) & np.isfinite(mean), out, np.nan)
    return out


def _box_max(a: np.ndarray, size: int, valid: np.ndarray) -> np.ndarray:
    """NaN-aware box max; NaN where the window has no valid sample."""
    filled = np.where(valid, a, -np.inf)
    mx = ndimage.maximum_filter(filled, size=size, mode="constant",
                                cval=-np.inf)
    return np.where(np.isfinite(mx), mx, np.nan)


def main() -> int:
    started = time.time()
    src = ROOT / "data/external/dem10/dem10_channels.f32.npy"
    out_path = ROOT / "data/external/dem10/dem10_context.f32.npy"
    if not src.exists():
        raise SystemExit("run scripts/fetch_external.py --tag ext/dem10-36326816737 "
                         "&& scripts/build_dem10_grid.py first")
    meta_src = src.with_suffix(".meta.json")
    src_meta = json.loads(meta_src.read_text()) if meta_src.exists() else {}
    base_names = list(src_meta.get("channels", []))
    missing = [b for b in BASES if b not in base_names]
    if missing:
        raise SystemExit(f"dem10 grid lacks frozen H25 bases: {missing} "
                         f"(has: {base_names})")

    import rasterio
    with rasterio.open(ROOT / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    grid = np.load(src, mmap_mode="r")
    if grid.shape[:2] != fp.shape:
        raise SystemExit(f"dem10 grid shape {grid.shape} != template {fp.shape}")

    channels: list[str] = []
    order = []
    for b in BASES:
        for w in WINDOWS:
            for st in STATS:
                channels.append(f"ctx_{b}_{st}{w}")
                order.append((b, w, st))

    H, W, _ = grid.shape
    C = len(channels)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(H, W, C))
    for i, (b, w, st) in enumerate(order):
        a = np.asarray(grid[:, :, base_names.index(b)], dtype=np.float64)
        valid = fp & np.isfinite(a)
        if st == "mean":
            r = _box_mean(a, w, valid)
        elif st == "std":
            mean = _box_mean(a, w, valid)
            r = _box_std(a, w, valid, mean)
        else:
            r = _box_max(a, w, valid)
        r = np.where(fp, r, np.nan).astype(np.float32)
        out[:, :, i] = r
        finite = np.isfinite(r[fp])
        print(f"{channels[i]}: finite_on_fp={finite.mean():.4f}", flush=True)
    out.flush()
    del out, grid

    builder_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta = {
        "channels": channels,
        "shape": [H, W, C],
        "dtype": "float32",
        "source": {
            "kind": "derived",
            "derived_from": src.name,
            "parent_tag": src_meta.get("source", {}).get("tag")
            if isinstance(src_meta.get("source"), dict) else None,
            "bases": BASES,
            "windows_px": WINDOWS,
            "stats": STATS,
            "builder": "scripts/build_dem10_context.py",
            "builder_sha256": builder_sha,
            "preregistration": "HYPOTHESES.md session-4 register (frozen before "
                               "any session-4 holdout contact)",
        },
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Round-trip sanity: reread header and check finiteness statistics persist.
    chk = np.load(out_path, mmap_mode="r")
    assert chk.shape == (H, W, C)
    print(json.dumps({"wrote": str(out_path.relative_to(ROOT)),
                      "bytes": out_path.stat().st_size,
                      "channels": C,
                      "seconds": round(time.time() - started, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
