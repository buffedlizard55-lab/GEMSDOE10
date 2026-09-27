#!/usr/bin/env python3
"""H20 — 3DEP 1/3 arc-second (~10 m) DEM scarp channels, aggregated to the 100 m grid.

Runs on a GitHub-hosted runner (unrestricted network, 16 GB RAM); the sandbox
cannot reach prd-tnm.s3.amazonaws.com (TLS blocked, measured 2026-09-27).

Source (official, public domain, U.S. Geological Survey 3D Elevation Program):
  https://www.sciencebase.gov/catalog/item/4f70aa9fe4b058caae3f8de5
  Tiles: https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/<tile>/USGS_13_<tile>.tif
  (1 deg x 1 deg, EPSG:4269, ~10812 px square, float32 metres NAVD88, nodata -999999)

Physical rationale (why the 100 m bands cannot carry this signal): a 2-10 m high
Quaternary fault scarp on an alluvial fan is a step 20-50 m wide. At 100 m
resolution it is one pixel's worth of elevation change against a fan slope of
2-5 m per 100 m and is therefore invisible in `det_elev`; at 10 m it is a 2-5 px
steepening of 10-25 degrees against a 1-3 degree background, one-sided (unlike a
channel, whose two banks face each other), and laterally continuous.

Channels (each reduced 10x10 -> one 100 m cell; halo-safe; NaN outside footprint):
  dem10_slope_max       max 10 m slope (deg) in the cell
  dem10_slope_mean      mean 10 m slope (deg)
  dem10_slope_std       std of 10 m slope (deg)                 (roughness)
  dem10_hgm20_max       max |grad G_sigma=20m z| (m/m)          (scarp-scale gradient)
  dem10_hgm50_max       max |grad G_sigma=50m z| (m/m)
  dem10_hgm200_mean     mean |grad G_sigma=200m z| (m/m)        (background slope)
  dem10_steep_ratio_max max hgm20 / (hgm200 + 0.01)             (local steepening)
  dem10_resid_std       std of (z - G_sigma=100m z) (m)         (short-wavelength relief)
  dem10_resid_range     max-min of the same residual (m)
  dem10_curv_absmax     max |Laplacian(G_sigma=30m z)| (1/m)    (crest/toe curvature)
  dem10_onesided        |sum r| / (sum |r| + eps), r = grad G20 - grad G200 (residual gradient;
                        1 = one-sided step such as a scarp, ~0 = symmetric valley/ridge or noise)
  dem10_onesided3       the same ratio over a 3x3-cell (300 m) window
  dem10_valid_frac      fraction of the 100 valid 10 m samples

Everything is label-free. Output: per-channel float32 vectors over the official
footprint (order = np.nonzero(footprint), row-major) plus a manifest with tile
provenance (URL, bytes, ETag/LastModified from S3, sha256) and code hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform as warp_transform
from scipy import ndimage

S3 = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"
CHANNELS = ["dem10_slope_max", "dem10_slope_mean", "dem10_slope_std",
            "dem10_hgm20_max", "dem10_hgm50_max", "dem10_hgm200_mean",
            "dem10_steep_ratio_max", "dem10_resid_std", "dem10_resid_range",
            "dem10_curv_absmax", "dem10_onesided", "dem10_onesided3", "dem10_valid_frac"]
FACTOR = 10          # 100 m / 10 m
BLOCK = 400          # 100 m cells per processing block side (40 km)
HALO = 12            # 100 m cells of halo (1.2 km) > 4 sigma of the widest filter (200 m)
DX = 10.0            # metres per 10 m pixel (destination grid is exactly 10 m UTM)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def tiles_for_footprint(template: Path) -> list[str]:
    """Every 1x1 degree tile touched by any footprint pixel (exact, all pixels)."""
    with rasterio.open(template) as ds:
        fp = np.isfinite(ds.read(1))
        rows, cols = np.nonzero(fp)
        names = set()
        # Pixel corners, in chunks (5.2 M pixels).
        for i in range(0, rows.size, 500_000):
            r = rows[i:i + 500_000]
            c = cols[i:i + 500_000]
            for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
                xs, ys = rasterio.transform.xy(ds.transform, r + dr, c + dc, offset="ul")
                lons, lats = warp_transform(ds.crs, "EPSG:4326", xs, ys)
                lons = np.asarray(lons)
                lats = np.asarray(lats)
                for la, lo in zip(np.floor(lats).astype(int), np.floor(lons).astype(int)):
                    names.add(f"n{la + 1:02d}w{-lo:03d}")
    return sorted(names)


def s3_head(url: str) -> dict:
    out = subprocess.run(["curl", "-sSI", "-L", "--retry", "5", "--retry-delay", "5", url],
                         capture_output=True, text=True, timeout=120)
    meta = {"http": None, "bytes": None, "etag": None, "last_modified": None}
    for line in out.stdout.splitlines():
        low = line.lower()
        if low.startswith("http/"):
            meta["http"] = line.strip()
        elif low.startswith("content-length:"):
            meta["bytes"] = int(line.split(":", 1)[1].strip())
        elif low.startswith("etag:"):
            meta["etag"] = line.split(":", 1)[1].strip()
        elif low.startswith("last-modified:"):
            meta["last_modified"] = line.split(":", 1)[1].strip()
    return meta


def download(url: str, dest: Path, expected_bytes: int | None) -> None:
    for attempt in range(4):
        subprocess.run(["curl", "-sS", "-L", "--retry", "5", "--retry-delay", "10",
                        "-C", "-", "-o", str(dest), url], check=False, timeout=3600)
        if dest.exists() and (expected_bytes is None or dest.stat().st_size == expected_bytes):
            return
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"download failed or size mismatch: {url}")


def block_reduce(a: np.ndarray, fn: str) -> np.ndarray:
    """(n*10, m*10) -> (n, m) with a NaN-aware reduction."""
    n, m = a.shape[0] // FACTOR, a.shape[1] // FACTOR
    v = a[: n * FACTOR, : m * FACTOR].reshape(n, FACTOR, m, FACTOR)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if fn == "max":
                return np.nanmax(v, axis=(1, 3))
            if fn == "min":
                return np.nanmin(v, axis=(1, 3))
            if fn == "mean":
                return np.nanmean(v, axis=(1, 3))
            if fn == "std":
                return np.nanstd(v, axis=(1, 3))
            if fn == "sum":
                return np.nansum(v, axis=(1, 3))
            if fn == "count":
                return np.isfinite(v).sum(axis=(1, 3)).astype(np.float32)
    raise ValueError(fn)


def channels_from_block(z: np.ndarray) -> dict[str, np.ndarray]:
    """All twelve channels for one 10 m block (already includes the halo)."""
    valid = np.isfinite(z)
    if not valid.any():
        return {}
    fill = float(np.nanmean(z))
    zf = np.where(valid, z, fill).astype(np.float32)
    # Filters: reflect padding is fine because the halo (>4 sigma) is trimmed later.
    gy, gx = np.gradient(zf, DX)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    del gx, gy
    g20 = ndimage.gaussian_filter(zf, 2.0)
    g20y, g20x = np.gradient(g20, DX)
    hgm20 = np.hypot(g20x, g20y).astype(np.float32)
    del g20
    g50 = ndimage.gaussian_filter(zf, 5.0)
    a, b = np.gradient(g50, DX)
    hgm50 = np.hypot(a, b).astype(np.float32)
    del g50, a, b
    g200 = ndimage.gaussian_filter(zf, 20.0)
    g200y, g200x = np.gradient(g200, DX)
    hgm200 = np.hypot(g200x, g200y).astype(np.float32)
    del g200
    # Residual (high-pass) gradient: scarp-scale gradient minus the regional
    # fan/hillslope gradient, so one-sidedness is not dominated by the fan.
    rx = (g20x - g200x).astype(np.float32)
    ry = (g20y - g200y).astype(np.float32)
    del g200x, g200y
    steep = (hgm20 / (hgm200 + 0.01)).astype(np.float32)
    resid = (zf - ndimage.gaussian_filter(zf, 10.0)).astype(np.float32)
    curv = (np.abs(ndimage.laplace(ndimage.gaussian_filter(zf, 3.0))) / (DX * DX)).astype(np.float32)
    inv = ~valid
    for arr in (slope, hgm20, hgm50, hgm200, steep, resid, curv):
        arr[inv] = np.nan
    rmag = np.hypot(rx, ry).astype(np.float32)
    rx[inv] = np.nan
    ry[inv] = np.nan
    rmag[inv] = np.nan
    out = {
        "dem10_slope_max": block_reduce(slope, "max"),
        "dem10_slope_mean": block_reduce(slope, "mean"),
        "dem10_slope_std": block_reduce(slope, "std"),
        "dem10_hgm20_max": block_reduce(hgm20, "max"),
        "dem10_hgm50_max": block_reduce(hgm50, "max"),
        "dem10_hgm200_mean": block_reduce(hgm200, "mean"),
        "dem10_steep_ratio_max": block_reduce(steep, "max"),
        "dem10_resid_std": block_reduce(resid, "std"),
        "dem10_resid_range": block_reduce(resid, "max") - block_reduce(resid, "min"),
        "dem10_curv_absmax": block_reduce(curv, "max"),
    }
    sx = block_reduce(rx, "sum")
    sy = block_reduce(ry, "sum")
    tot = block_reduce(rmag, "sum")
    out["dem10_onesided"] = (np.hypot(sx, sy) / (tot + 1e-6)).astype(np.float32)
    # Same ratio over a 3x3-cell (300 m) window: a 30-50 m wide channel that
    # straddles a 100 m cell boundary shows one bank per cell (one-sided per
    # cell) but both banks within 300 m (two-sided); a scarp stays one-sided.
    k3 = np.ones((3, 3), dtype=np.float32)
    sx3 = ndimage.convolve(np.nan_to_num(sx), k3, mode="nearest")
    sy3 = ndimage.convolve(np.nan_to_num(sy), k3, mode="nearest")
    tot3 = ndimage.convolve(np.nan_to_num(tot), k3, mode="nearest")
    out["dem10_onesided3"] = (np.hypot(sx3, sy3) / (tot3 + 1e-6)).astype(np.float32)
    out["dem10_valid_frac"] = block_reduce(np.where(valid, 1.0, np.nan).astype(np.float32),
                                           "count") / (FACTOR * FACTOR)
    return {k: v.astype(np.float32) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, default=Path("data/sample_submission.tif"))
    ap.add_argument("--work", type=Path, default=Path("work/dem10"))
    ap.add_argument("--out", type=Path, default=Path("out/dem10"))
    ap.add_argument("--max-blocks", type=int, default=0, help="debug: stop after N blocks")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with rasterio.open(args.template) as ds:
        fp = np.isfinite(ds.read(1))
        T = ds.transform
        crs = ds.crs
        H, W = ds.shape
        template_sha = sha256_file(args.template)
    print(f"template {H}x{W} footprint={int(fp.sum())} crs={crs}", flush=True)

    tiles = tiles_for_footprint(args.template)
    print("tiles touched by footprint:", tiles, flush=True)
    provenance = {}
    for t in tiles:
        url = f"{S3}/{t}/USGS_13_{t}.tif"
        meta = s3_head(url)
        ok = meta["http"] is not None and " 200" in meta["http"]
        provenance[t] = {"url": url, "available": ok, **meta}
        print(t, meta, flush=True)
    sources = []
    for t in tiles:
        p = provenance[t]
        if not p["available"]:
            continue
        dest = args.work / f"USGS_13_{t}.tif"
        if not (dest.exists() and dest.stat().st_size == p["bytes"]):
            download(p["url"], dest, p["bytes"])
        p["sha256"] = sha256_file(dest)
        src = rasterio.open(dest)
        p["src_crs"] = str(src.crs)
        p["src_nodata"] = src.nodata
        p["src_shape"] = list(src.shape)
        b = src.bounds
        # Tile bounds in the template CRS (densified) for intersection tests.
        from rasterio.warp import transform_bounds
        p["bounds_template_crs"] = list(transform_bounds(src.crs, crs, b.left, b.bottom,
                                                         b.right, b.top, densify_pts=21))
        sources.append((t, src))
        print(f"  {t}: {p['bytes']} B sha256 {p['sha256'][:16]} nodata={src.nodata}", flush=True)

    full = {c: np.full((H, W), np.nan, dtype=np.float32) for c in CHANNELS}
    n_blocks = 0
    n_rows = math.ceil(H / BLOCK)
    n_cols = math.ceil(W / BLOCK)
    for bi in range(n_rows):
        for bj in range(n_cols):
            r0, c0 = bi * BLOCK, bj * BLOCK
            r1, c1 = min(H, r0 + BLOCK), min(W, c0 + BLOCK)
            if not fp[r0:r1, c0:c1].any():
                continue
            # Destination 10 m grid with halo (may extend past the template edge).
            rr0, cc0 = r0 - HALO, c0 - HALO
            hh, ww = (r1 - r0 + 2 * HALO) * FACTOR, (c1 - c0 + 2 * HALO) * FACTOR
            x0, y0 = T * (cc0, rr0)
            dst_T = Affine(DX, 0.0, x0, 0.0, -DX, y0)
            xmin, ymax = x0, y0
            xmax, ymin = x0 + ww * DX, y0 - hh * DX
            dst = np.full((hh, ww), np.nan, dtype=np.float32)
            for t, src in sources:
                bx0, by0, bx1, by1 = provenance[t]["bounds_template_crs"]
                if bx1 < xmin or bx0 > xmax or by1 < ymin or by0 > ymax:
                    continue
                tmp = np.full((hh, ww), np.nan, dtype=np.float32)
                reproject(source=rasterio.band(src, 1), destination=tmp,
                          dst_transform=dst_T, dst_crs=crs,
                          src_nodata=src.nodata if src.nodata is not None else -999999.0,
                          dst_nodata=np.nan, resampling=Resampling.bilinear, num_threads=2)
                need = np.isnan(dst)
                dst[need] = tmp[need]
                del tmp, need
            ch = channels_from_block(dst)
            del dst
            if ch:
                for name, arr in ch.items():
                    core = arr[HALO:HALO + (r1 - r0), HALO:HALO + (c1 - c0)]
                    full[name][r0:r1, c0:c1] = core
            n_blocks += 1
            print(f"block ({bi},{bj}) done  {n_blocks} blocks  {time.time() - started:.0f}s",
                  flush=True)
            if args.max_blocks and n_blocks >= args.max_blocks:
                break
        if args.max_blocks and n_blocks >= args.max_blocks:
            break

    stats = {}
    for name in CHANNELS:
        v = full[name][fp]
        finite = np.isfinite(v)
        q = np.nanpercentile(v[finite], [1, 50, 99]).tolist() if finite.any() else [None] * 3
        stats[name] = {"finite": int(finite.sum()), "p01": q[0], "p50": q[1], "p99": q[2],
                       "max": float(np.nanmax(v)) if finite.any() else None}
        outp = args.out / f"{name}.f32.npy"
        np.save(outp, v.astype(np.float32), allow_pickle=False)
        stats[name]["sha256"] = sha256_file(outp)
        stats[name]["bytes"] = outp.stat().st_size
        del full[name]
    manifest = {
        "hypothesis": "H20",
        "product": "USGS 3DEP 1/3 arc-second seamless DEM (current), public domain",
        "product_catalog": "https://www.sciencebase.gov/catalog/item/4f70aa9fe4b058caae3f8de5",
        "template_sha256": template_sha,
        "footprint_pixels": int(fp.sum()),
        "vector_order": "np.nonzero(np.isfinite(sample_submission)) row-major",
        "grid": {"rows": H, "cols": W, "block_cells": BLOCK, "halo_cells": HALO,
                 "dst_pixel_m": DX, "resampling": "bilinear"},
        "channels": CHANNELS,
        "channel_stats": stats,
        "tiles": provenance,
        "blocks_processed": n_blocks,
        "code_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": round(time.time() - started, 1),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_sha": os.environ.get("GITHUB_SHA"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "tiles"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
