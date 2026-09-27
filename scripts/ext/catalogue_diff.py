#!/usr/bin/env python3
"""H21 — catalogue-version differencing (runs on a GitHub-hosted runner).

Fetches the *current* official fault catalogues, rasterises them onto the
official 100 m template, and measures which traces are NOT in the provided
label raster (`existing_faults.tif`). Official sources (verified 2026-09-27):

  USGS Quaternary Fault and Fold Database (public domain), GIS files (16 MB):
      https://earthquake.usgs.gov/static/lfs/nshm/qfaults/Qfaults_GIS.zip
      cite: U.S. Geological Survey, 2020, Quaternary Fault and Fold Database for
      the Nation, https://doi.org/10.5066/P9BCVRCK
  INGENIOUS Great Basin Regional Dataset Compilation, GDR 1391 (CC BY 4.0),
  DOI 10.15121/1881483 — Quaternary Faults v1 and v2 (v2 supersedes v1):
      https://gdr.openei.org/files/1391/faults_quaternary_INGENIOUS_regional_data.zip
      https://gdr.openei.org/files/1391/qfaults_ingenious_nad83conus117_2023-06-27.zip
  Same GDR entry, fetched and hashed here for H15 (paleo-discharge), not modelled:
      https://gdr.openei.org/files/1391/paleo_geothermal_regional.zip
      https://gdr.openei.org/files/1391/2m_temperature_probe_INGENIOUS_regional_data.zip
      https://gdr.openei.org/files/1391/wellspringdata.gdb.zip
      https://gdr.openei.org/files/1391/study_area_boundary_INGENIOUS_regional_data.zip

Outputs (out/catalogue): packed-bit masks on the full 3730x3292 grid for each
catalogue and rasterisation convention, a JSON report with pixel counts, overlap
with the provided labels, and the count/geometry of trace pixels farther than
300 m from any provided label pixel ("catalogue-new" candidates), plus sha256
provenance of every archive. Nothing here is a prediction; it is evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from scipy import ndimage

ARCHIVES = {
    "usgs_qffd": ("https://earthquake.usgs.gov/static/lfs/nshm/qfaults/Qfaults_GIS.zip", "Qfaults_GIS.zip"),
    "ingenious_v1": ("https://gdr.openei.org/files/1391/faults_quaternary_INGENIOUS_regional_data.zip",
                     "faults_quaternary_INGENIOUS_regional_data.zip"),
    "ingenious_v2": ("https://gdr.openei.org/files/1391/qfaults_ingenious_nad83conus117_2023-06-27.zip",
                     "qfaults_ingenious_nad83conus117_2023-06-27.zip"),
    "paleo_geothermal": ("https://gdr.openei.org/files/1391/paleo_geothermal_regional.zip",
                         "paleo_geothermal_regional.zip"),
    "probes_2m": ("https://gdr.openei.org/files/1391/2m_temperature_probe_INGENIOUS_regional_data.zip",
                  "2m_temperature_probe_INGENIOUS_regional_data.zip"),
    "wellspring": ("https://gdr.openei.org/files/1391/wellspringdata.gdb.zip", "wellspringdata.gdb.zip"),
    "study_area": ("https://gdr.openei.org/files/1391/study_area_boundary_INGENIOUS_regional_data.zip",
                   "study_area_boundary_INGENIOUS_regional_data.zip"),
}
FAULT_SETS = ("usgs_qffd", "ingenious_v1", "ingenious_v2")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> dict:
    if not dest.exists() or dest.stat().st_size == 0:
        subprocess.run(["curl", "-sS", "-L", "--retry", "5", "--retry-delay", "10",
                        "-o", str(dest), url], check=True, timeout=1800)
    return {"url": url, "bytes": dest.stat().st_size, "sha256": sha256_file(dest)}


def vector_layers(root: Path) -> list[tuple[str, str | None]]:
    """(path, layer) pairs for every shapefile / file geodatabase under root."""
    out = []
    for p in sorted(root.rglob("*.shp")):
        out.append((str(p), None))
    for p in sorted(root.rglob("*.gdb")):
        try:
            import pyogrio
            for lyr in pyogrio.list_layers(str(p))[:, 0]:
                out.append((str(p), str(lyr)))
        except Exception as exc:  # pragma: no cover
            print("gdb listing failed", p, exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, default=Path("data/sample_submission.tif"))
    ap.add_argument("--labels", type=Path, default=Path("data/labels.tif"))
    ap.add_argument("--work", type=Path, default=Path("work/catalogue"))
    ap.add_argument("--out", type=Path, default=Path("out/catalogue"))
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    import geopandas as gpd
    import pyogrio  # noqa: F401
    from shapely.geometry import box

    started = time.time()
    with rasterio.open(args.template) as ds:
        fp = np.isfinite(ds.read(1))
        T, crs, shape = ds.transform, ds.crs, ds.shape
        bounds = ds.bounds
    with rasterio.open(args.labels) as ds:
        labels = ds.read(1) == 1
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "template_sha256": sha256_file(args.template),
              "labels_sha256": sha256_file(args.labels),
              "label_pixels": int(labels.sum()), "footprint_pixels": int(fp.sum()),
              "archives": {}, "fault_sets": {}, "point_sets": {},
              "github_run_id": os.environ.get("GITHUB_RUN_ID"),
              "code_sha256": sha256_file(Path(__file__))}
    d_label = ndimage.distance_transform_edt(~labels)  # px to nearest label pixel
    clip = box(bounds.left - 2000, bounds.bottom - 2000, bounds.right + 2000, bounds.top + 2000)

    for key, (url, fname) in ARCHIVES.items():
        dest = args.work / fname
        try:
            report["archives"][key] = fetch(url, dest)
        except Exception as exc:
            report["archives"][key] = {"url": url, "error": str(exc)}
            print("FETCH FAILED", key, exc, flush=True)
            continue
        ex = args.work / key
        ex.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(ex)
            report["archives"][key]["members"] = sorted(
                str(p.relative_to(ex)) for p in ex.rglob("*") if p.is_file())[:200]
        except zipfile.BadZipFile as exc:
            report["archives"][key]["error"] = f"bad zip: {exc}"
            continue

    for key in FAULT_SETS:
        if "error" in report["archives"].get(key, {"error": "missing"}):
            continue
        ex = args.work / key
        entry = {"layers": []}
        frames = []
        for path, layer in vector_layers(ex):
            try:
                gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
            except Exception as exc:
                entry["layers"].append({"path": path, "layer": layer, "error": str(exc)})
                continue
            geom_types = sorted(set(gdf.geom_type.dropna().unique().tolist()))
            info = {"path": os.path.relpath(path, ex), "layer": layer, "rows": int(len(gdf)),
                    "crs": str(gdf.crs), "geom_types": geom_types,
                    "columns": [str(c) for c in gdf.columns][:60]}
            entry["layers"].append(info)
            if not any(t.endswith("LineString") for t in geom_types) or gdf.crs is None:
                continue
            g = gdf[gdf.geom_type.str.endswith("LineString")].to_crs(crs)
            g = g[g.intersects(clip)]
            info["rows_in_template_bbox"] = int(len(g))
            if len(g):
                frames.append(g)
        if not frames:
            report["fault_sets"][key] = entry
            continue
        import pandas as pd
        g = pd.concat(frames, ignore_index=True)
        entry["traces_in_bbox"] = int(len(g))
        for col in ("age", "AGE", "Age", "slip_rate", "SLIP_RATE", "fault_type", "mapped_sca",
                    "MAPPED_SCA", "num_sectio", "FaultAge", "fault_age", "Age_Cat"):
            if col in g.columns:
                vc = g[col].astype(str).value_counts().head(20)
                entry.setdefault("attribute_counts", {})[col] = {str(k): int(v) for k, v in vc.items()}
        shapes = [(geom, 1) for geom in g.geometry if geom is not None and not geom.is_empty]
        for touched in (False, True):
            mask = features.rasterize(shapes, out_shape=shape, transform=T, fill=0,
                                      all_touched=touched, dtype="uint8").astype(bool)
            mask &= fp
            tag = "all_touched" if touched else "center"
            far = mask & (d_label > 3.0)
            near = mask & (d_label <= 3.0) & ~labels
            comp_far, n_far = ndimage.label(far, np.ones((3, 3)))
            sizes = np.bincount(comp_far.ravel())[1:] if n_far else np.array([], dtype=int)
            entry[tag] = {
                "pixels": int(mask.sum()),
                "on_label_pixels": int((mask & labels).sum()),
                "within_300m_not_on_label": int(near.sum()),
                "beyond_300m_of_labels": int(far.sum()),
                "beyond_300m_components": int(n_far),
                "beyond_300m_components_ge8px": int((sizes >= 8).sum()) if n_far else 0,
                "label_pixels_beyond_300m_of_this_set": int(
                    (labels & (ndimage.distance_transform_edt(~mask) > 3.0)).sum()),
                "iou_with_labels": float((mask & labels).sum() / max(1, (mask | labels).sum())),
            }
            np.save(args.out / f"{key}_{tag}.packbits.npy", np.packbits(mask), allow_pickle=False)
            np.save(args.out / f"{key}_{tag}_beyond300m.packbits.npy", np.packbits(far),
                    allow_pickle=False)
        report["fault_sets"][key] = entry
        print(key, json.dumps({k: v for k, v in entry.items() if k != "layers"}, indent=1), flush=True)

    # Point sets for H15 (counts only; no modelling here).
    for key in ("paleo_geothermal", "probes_2m", "wellspring", "study_area"):
        if "error" in report["archives"].get(key, {"error": "missing"}):
            continue
        ex = args.work / key
        entry = {"layers": []}
        for path, layer in vector_layers(ex):
            try:
                gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
            except Exception as exc:
                entry["layers"].append({"path": path, "layer": layer, "error": str(exc)})
                continue
            info = {"path": os.path.relpath(path, ex), "layer": layer, "rows": int(len(gdf)),
                    "crs": str(gdf.crs), "columns": [str(c) for c in gdf.columns][:80],
                    "geom_types": sorted(set(gdf.geom_type.dropna().unique().tolist()))}
            if gdf.crs is not None and len(gdf):
                try:
                    gg = gdf.to_crs(crs)
                    inside = gg[gg.intersects(box(*bounds))]
                    info["rows_in_template_bbox"] = int(len(inside))
                    pts = inside[inside.geom_type == "Point"]
                    if len(pts):
                        rows, cols = rasterio.transform.rowcol(T, pts.geometry.x.values,
                                                               pts.geometry.y.values)
                        rows = np.asarray(rows)
                        cols = np.asarray(cols)
                        ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
                        info["points_in_footprint"] = int(fp[rows[ok], cols[ok]].sum())
                        csv = ex / f"{Path(path).stem}_{layer or 'layer'}_points_utm.csv"
                        pd_out = pts.drop(columns="geometry").copy()
                        pd_out["utm_x"] = pts.geometry.x.values
                        pd_out["utm_y"] = pts.geometry.y.values
                        pd_out.to_csv(csv, index=False)
                        target = args.out / csv.name
                        target.write_bytes(csv.read_bytes())
                        info["points_csv"] = target.name
                        info["points_csv_sha256"] = sha256_file(target)
                except Exception as exc:
                    info["error"] = str(exc)
            entry["layers"].append(info)
        report["point_sets"][key] = entry

    for p in sorted(args.out.glob("*.npy")):
        report.setdefault("files", {})[p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    report["elapsed_seconds"] = round(time.time() - started, 1)
    (args.out / "catalogue_diff.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("archives", "point_sets")},
                     indent=1, default=str)[:6000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
