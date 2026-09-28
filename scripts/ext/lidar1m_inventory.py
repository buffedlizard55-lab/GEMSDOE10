#!/usr/bin/env python3
"""H31 step 1 — inventory of USGS 3DEP 1 m DEM tiles over the competition footprint.

Runs on a GitHub-hosted runner (the research sandbox cannot reach AWS S3).
Nothing is trained and no tile pixels are downloaded: this answers, before any
heavy job is designed, "which 1 m lidar projects cover the footprint, how much
of it, and how many bytes would a full derivation read?"

Source (official, public domain, USGS 3D Elevation Program), public bucket:
  https://prd-tnm.s3.amazonaws.com/?list-type=2&prefix=StagedProducts/Elevation/1m/Projects/
GeoDAWN lidar = 3DEP projects NV_WestCentral_EarthMRI_2020_D20 and
NV_NorthWestElko_2020_D20 (data.gov "GeoDAWN West Central Nevada EarthMRI Data").

Tile naming USGS_1M_<utm zone>_x<E10km>y<N10km>_<project>.tif: 10 km x 10 km
tiles whose upper-left corner is (E10km*10000, N10km*10000) in NAD83 / UTM
<zone> (EPSG:269<zone>). The convention is VERIFIED here by reading the COG
header (bounds + CRS) of a sample of tiles per project via /vsicurl/; any
mismatch makes the job fail loudly rather than report a wrong coverage.

Output (out/lidar1m/): inventory.json (projects, tiles, bytes, coverage of the
official footprint per project and union) and coverage_union.npz (uint8 mask).
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import transform as warp_transform

ROOT = Path(__file__).resolve().parents[2]
BUCKET = "https://prd-tnm.s3.amazonaws.com"
PREFIX = "StagedProducts/Elevation/1m/Projects/"
STATE_PREFIXES = ("NV_", "CA_")
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
TILE_RE = re.compile(r"USGS_1M_(\d{1,2})_x(\d+)y(\d+)_")
OUT = ROOT / "out/lidar1m"


def s3_list(prefix: str, delimiter: str | None = None):
    """Yield ('prefix', p) and ('object', key, size, etag) over all pages."""
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        url = f"{BUCKET}/?{urllib.parse.urlencode(q)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for cp in root.findall("s3:CommonPrefixes", NS):
            yield ("prefix", cp.find("s3:Prefix", NS).text)
        for c in root.findall("s3:Contents", NS):
            yield ("object", c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text),
                   (c.find("s3:ETag", NS).text or "").strip('"'))
        if (root.findtext("s3:IsTruncated", default="false", namespaces=NS) or "").lower() != "true":
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)


def parse_tile(key: str):
    m = TILE_RE.search(key.rsplit("/", 1)[-1])
    if not m:
        return None
    zone, ex, ny = (int(g) for g in m.groups())
    x0, y1 = ex * 10000.0, ny * 10000.0
    return {"zone": zone, "epsg": 26900 + zone,
            "bounds": [x0, y1 - 10000.0, x0 + 10000.0, y1]}  # left, bottom, right, top


def tile_polygon_32611(t):
    """Tile outline densified along its edges, reprojected to EPSG:32611."""
    l, b, r, tp = t["bounds"]
    s = np.linspace(0, 1, 21)
    xs = np.r_[l + (r - l) * s, np.full(21, r), r - (r - l) * s, np.full(21, l)]
    ys = np.r_[np.full(21, tp), tp - (tp - b) * s, np.full(21, b), b + (tp - b) * s]
    X, Y = warp_transform(f"EPSG:{t['epsg']}", "EPSG:32611", xs.tolist(), ys.tolist())
    return {"type": "Polygon", "coordinates": [list(zip(X, Y))]}


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ROOT / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
        transform, shape = ds.transform, ds.shape
    left, top = transform.c, transform.f
    right, bottom = left + shape[1] * transform.a, top + shape[0] * transform.e
    projects = [p for kind, p in s3_list(PREFIX, "/") if kind == "prefix"
                and p[len(PREFIX):].startswith(STATE_PREFIXES)]
    print(f"[{time.time()-t0:5.0f}s] {len(projects)} NV_/CA_ 1 m projects", flush=True)
    union = np.zeros(shape, dtype=np.uint8)
    inv = {"source": f"{BUCKET}/?list-type=2&prefix={PREFIX}", "grid_epsg": 32611,
           "footprint_cells": int(fp.sum()), "projects": {}, "verified_headers": []}
    for proj in projects:
        name = proj[len(PREFIX):].strip("/")
        tiles = []
        for item in s3_list(proj + "TIFF/"):
            if item[0] != "object" or not item[1].lower().endswith(".tif"):
                continue
            t = parse_tile(item[1])
            if t is None:
                continue
            t.update({"key": item[1], "bytes": item[2], "etag": item[3]})
            tiles.append(t)
        if not tiles:
            continue
        # cheap reject: tile lon/lat box far from the grid (reproject centres)
        cx = [(t["bounds"][0] + t["bounds"][2]) / 2 for t in tiles]
        cy = [(t["bounds"][1] + t["bounds"][3]) / 2 for t in tiles]
        near = []
        for epsg in sorted({t["epsg"] for t in tiles}):
            idx = [i for i, t in enumerate(tiles) if t["epsg"] == epsg]
            X, Y = warp_transform(f"EPSG:{epsg}", "EPSG:32611",
                                  [cx[i] for i in idx], [cy[i] for i in idx])
            for i, x, y in zip(idx, X, Y):
                if left - 15000 <= x <= right + 15000 and bottom - 15000 <= y <= top + 15000:
                    near.append(tiles[i])
        if not near:
            continue
        mask = np.zeros(shape, dtype=np.uint8)
        kept = []
        for t in near:
            m = features.rasterize([(tile_polygon_32611(t), 1)], out_shape=shape,
                                   transform=transform, fill=0, dtype="uint8")
            cells = int((m.astype(bool) & fp).sum())
            if cells:
                t["footprint_cells"] = cells
                kept.append(t)
                mask |= m
        if not kept:
            continue
        # verify the naming convention on up to 3 tiles (COG header only)
        for t in kept[:3]:
            url = f"/vsicurl/{BUCKET}/{t['key']}"
            with rasterio.open(url) as ds:
                b = list(ds.bounds)
                epsg = ds.crs.to_epsg() if ds.crs else None
            ok = max(abs(u - v) for u, v in zip(b, t["bounds"])) <= 20.0
            inv["verified_headers"].append({"key": t["key"], "header_bounds": b,
                                            "header_epsg": epsg, "name_bounds": t["bounds"],
                                            "match": bool(ok)})
            if not ok:
                raise SystemExit(f"tile naming convention mismatch for {t['key']}: {b} vs {t['bounds']}")
        cov = int((mask.astype(bool) & fp).sum())
        union |= mask
        inv["projects"][name] = {"tiles_intersecting": len(kept),
                                 "bytes": int(sum(t["bytes"] for t in kept)),
                                 "footprint_cells_covered": cov,
                                 "footprint_fraction": round(cov / int(fp.sum()), 5),
                                 "tiles": [{k: t[k] for k in ("key", "bytes", "etag", "epsg",
                                                              "bounds", "footprint_cells")}
                                           for t in kept]}
        print(f"[{time.time()-t0:5.0f}s] {name}: {len(kept)} tiles, "
              f"{inv['projects'][name]['bytes']/1e9:.1f} GB, "
              f"{inv['projects'][name]['footprint_fraction']:.1%} of footprint", flush=True)
    covered = union.astype(bool) & fp
    inv["union_footprint_cells_covered"] = int(covered.sum())
    inv["union_footprint_fraction"] = round(float(covered.sum()) / int(fp.sum()), 5)
    inv["total_tiles"] = int(sum(p["tiles_intersecting"] for p in inv["projects"].values()))
    inv["total_bytes"] = int(sum(p["bytes"] for p in inv["projects"].values()))
    inv["seconds"] = round(time.time() - t0, 1)
    (OUT / "inventory.json").write_text(json.dumps(inv, indent=1) + "\n")
    np.savez_compressed(OUT / "coverage_union.npz", mask=union)
    print(json.dumps({k: inv[k] for k in ("union_footprint_fraction", "total_tiles",
                                          "total_bytes", "seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
