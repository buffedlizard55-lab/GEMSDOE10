#!/usr/bin/env python3
"""H29 (session 5): km-scale oriented line support of fault-sensitive edge maps.

    MALLOC_ARENA_MAX=2 .venv/bin/python scripts/build_linesupport.py

Physical signature. A fault that was never catalogued is, almost by definition,
a FAINT one: a degraded piedmont scarp, a short strand, or a buried contact
whose edge is weak at any single 100 m pixel. What survives is linear
continuity over kilometres (Basin and Range normal faults and Walker Lane
strands are 5-60 km long and nearly straight). Every channel in this repo is
local (structure tensor sigma <= 4 px, Frangi/Gabor <= 11 px, H25 isotropic
context boxes) — none integrates evidence ALONG a straight line. Here each
edge map is rank-normalised and correlated with zero-sum oriented kernels:
the mean along a centred L-pixel segment minus the mean along two parallel
flanking segments 4 px (400 m) away, at 12 orientations (15 deg steps), for
L = 21 px (2.1 km) and 51 px (5.1 km). Averaging along L raises the
signal-to-noise of a weak straight edge by ~sqrt(L) (the principle behind
Hough/Radon lineament detection), while the flank subtraction makes broad
bright areas (a whole steep range) score zero.

Inputs (label-free; nothing here sees the catalogue):
  dem10_onesided, dem10_slope_max          (10 m 3DEP DEM aggregates, H20 tag)
  hgm_tmi_s1.5, hgm_iso_grav_anom_s1.5     (provided magnetics/gravity edges)
Output: data/external/linesupport/linesupport.f32.npy  (3730, 3292, 16) float32,
NaN outside the footprint; channels ls_<input>_L<21|51>_<max|aniso>, where
max = max over orientations and aniso = max - mean over orientations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import scipy.fft as sfft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems10.raster import sha256_file  # noqa: E402

INPUTS = [("dem10", "dem10_onesided"), ("dem10", "dem10_slope_max"),
          ("feat", "hgm_tmi_s1.5"), ("feat", "hgm_iso_grav_anom_s1.5")]
LENGTHS = (21, 51)
FLANK_PX = 4
N_THETA = 12


def _splat(K: np.ndarray, xs: np.ndarray, ys: np.ndarray, w: float) -> None:
    """Bilinear splat of points (x = column, y = row offsets from centre)."""
    c = (K.shape[0] - 1) // 2
    X = xs + c
    Y = ys + c
    x0 = np.floor(X).astype(int)
    y0 = np.floor(Y).astype(int)
    fx = X - x0
    fy = Y - y0
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            np.add.at(K, (y0 + dy, x0 + dx), w * wy * wx)


def line_kernel(length: int, theta: float, flank: int = FLANK_PX) -> np.ndarray:
    """Zero-sum oriented line-minus-flanks kernel (odd square)."""
    half = (length - 1) / 2.0
    R = int(np.ceil(half + flank + 1))
    K = np.zeros((2 * R + 1, 2 * R + 1), dtype=np.float64)
    t = np.arange(length, dtype=np.float64) - half
    ux, uy = np.cos(theta), np.sin(theta)
    nx, ny = -uy, ux
    _splat(K, t * ux, t * uy, 1.0 / length)
    _splat(K, t * ux + flank * nx, t * uy + flank * ny, -0.5 / length)
    _splat(K, t * ux - flank * nx, t * uy - flank * ny, -0.5 / length)
    return K


def rank_normalise(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Percentile rank in [-0.5, 0.5] over finite footprint pixels; 0 elsewhere."""
    out = np.zeros(x.shape, dtype=np.float64)
    ok = valid & np.isfinite(x)
    v = x[ok]
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(v.size, dtype=np.float64)
    out[ok] = ranks / max(v.size - 1, 1) - 0.5
    return out


def oriented_support(u: np.ndarray, lengths=LENGTHS, n_theta=N_THETA):
    """{L: (max_theta, aniso)} responses of `u` to the oriented kernels."""
    H, W = u.shape
    kernels = {L: [line_kernel(L, np.pi * i / n_theta) for i in range(n_theta)]
               for L in lengths}
    ks = max(k.shape[0] for ks_ in kernels.values() for k in ks_)
    shape = (sfft.next_fast_len(H + ks - 1, real=True),
             sfft.next_fast_len(W + ks - 1, real=True))
    U = sfft.rfft2(u, s=shape, workers=2)
    out = {}
    for L, klist in kernels.items():
        rmax = np.full((H, W), -np.inf, dtype=np.float32)
        rsum = np.zeros((H, W), dtype=np.float32)
        for K in klist:
            c = (K.shape[0] - 1) // 2
            full = sfft.irfft2(U * sfft.rfft2(K, s=shape, workers=2), s=shape,
                               workers=2)
            r = full[c:c + H, c:c + W].astype(np.float32)
            del full
            np.maximum(rmax, r, out=rmax)
            rsum += r
            del r
        out[L] = (rmax, rmax - rsum / len(klist))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/external/linesupport/linesupport.f32.npy")
    args = ap.parse_args()
    t0 = time.time()
    with rasterio.open(ROOT / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    feat_p = ROOT / "data/features107.f32.npy"
    dem_p = ROOT / "data/external/dem10/dem10_channels.f32.npy"
    fmeta = json.loads(feat_p.with_suffix(".meta.json").read_text())["channels"]
    dmeta = json.loads(dem_p.with_suffix(".meta.json").read_text())["channels"]
    feat = np.load(feat_p, mmap_mode="r")
    dem = np.load(dem_p, mmap_mode="r")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_out = len(INPUTS) * len(LENGTHS) * 2
    grid = np.lib.format.open_memmap(str(args.out), mode="w+", dtype=np.float32,
                                     shape=fp.shape + (n_out,))
    names = []
    j = 0
    for src, name in INPUTS:
        arr = (dem[:, :, dmeta.index(name)] if src == "dem10"
               else feat[:, :, fmeta.index(name)]).astype(np.float64)
        arr[arr < -1e38] = np.nan
        u = rank_normalise(arr, fp)
        del arr
        res = oriented_support(u)
        del u
        for L in LENGTHS:
            for stat, a in zip(("max", "aniso"), res[L]):
                grid[:, :, j] = np.where(fp, a, np.nan)
                names.append(f"ls_{name}_L{L}_{stat}")
                j += 1
        del res
        print(f"[{time.time() - t0:6.1f}s] {name} done", flush=True)
    grid.flush()
    del grid
    kern_sha = hashlib.sha256(b"".join(
        line_kernel(L, np.pi * i / N_THETA).tobytes()
        for L in LENGTHS for i in range(N_THETA))).hexdigest()
    meta = {"channels": names, "shape": list(fp.shape) + [n_out],
            "inputs": {"features107.f32.npy": sha256_file(feat_p),
                       "dem10_channels.f32.npy": sha256_file(dem_p)},
            "lengths_px": list(LENGTHS), "flank_px": FLANK_PX, "n_theta": N_THETA,
            "kernel_sha256": kern_sha, "label_free": True,
            "source": "derived: provided 107-channel stack + 3DEP 10 m DEM aggregates "
                      "(tag ext/dem10-36326816737)",
            "seconds": round(time.time() - t0, 1)}
    meta["sha256"] = sha256_file(args.out)
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("sha256", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
