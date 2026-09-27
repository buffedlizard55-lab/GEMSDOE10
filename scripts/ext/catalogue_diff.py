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


def ogr_layers(path: Path) -> list[str]:
    """Layer names of a vector datasource via `ogrinfo -so -q`."""
    try:
        res = subprocess.run(["ogrinfo", "-so", "-q", str(path)], capture_output=True,
                             text=True, timeout=600)
    except Exception as exc:  # pragma: no cover
        print("ogrinfo failed", path, exc)
        return []
    layers = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if line[:1].isdigit() and ":" in line:
            name = line.split(":", 1)[1].strip()
            name = name.split(" (")[0].strip()
            layers.append(name)
    return layers


def vector_sources(root: Path) -> list[tuple[Path, str | None]]:
    """(datasource, layer) pairs for every shapefile / file geodatabase under root."""
    out = [(p, None) for p in sorted(root.rglob("*.shp"))]
    for gdb in sorted(d for d in root.rglob("*.gdb") if d.is_dir()):
        for lyr in ogr_layers(gdb):
            out.append((gdb, lyr))
    return out


def to_geojson(src: Path, layer: str | None, dest: Path, epsg: int, clip_bounds) -> dict:
    """Reproject (+clip to the template bbox) with system GDAL; returns parsed GeoJSON."""
    cmd = ["ogr2ogr", "-f", "GeoJSON", "-t_srs", f"EPSG:{epsg}", "-skipfailures",
           "-lco", "RFC7946=NO", "-lco", "COORDINATE_PRECISION=3"]
    if clip_bounds is not None:
        x0, y0, x1, y1 = clip_bounds
        cmd += ["-clipdst", str(x0), str(y0), str(x1), str(y1)]
    cmd += [str(dest), str(src)] + ([layer] if layer else [])
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0 or not dest.exists():
        raise RuntimeError(f"ogr2ogr rc={res.returncode}: {res.stderr[-800:]}")
    with dest.open() as fh:
        return json.load(fh)


def line_geoms(gj: dict) -> list[dict]:
    out = []
    for f in gj.get("features", []):
        g = f.get("geometry")
        if g and g.get("type") in ("LineString", "MultiLineString"):
            out.append(g)
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
    import traceback

    started = time.time()
    with rasterio.open(args.template) as ds:
        fp = np.isfinite(ds.read(1))
        T, crs, shape = ds.transform, ds.crs, ds.shape
        bounds = ds.bounds
    epsg = crs.to_epsg()
    with rasterio.open(args.labels) as ds:
        labels = ds.read(1) == 1
    gdal_version = subprocess.run(["ogrinfo", "--version"], capture_output=True,
                                  text=True).stdout.strip()
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "template_sha256": sha256_file(args.template),
              "labels_sha256": sha256_file(args.labels),
              "template_epsg": epsg, "gdal": gdal_version,
              "label_pixels": int(labels.sum()), "footprint_pixels": int(fp.sum()),
              "archives": {}, "fault_sets": {}, "point_sets": {},
              "github_run_id": os.environ.get("GITHUB_RUN_ID"),
              "code_sha256": sha256_file(Path(__file__))}
    d_label = ndimage.distance_transform_edt(~labels)  # px to nearest label pixel
    clip_bounds = (bounds.left - 2000, bounds.bottom - 2000, bounds.right + 2000, bounds.top + 2000)

    def save():
        report["elapsed_seconds"] = round(time.time() - started, 1)
        (args.out / "catalogue_diff.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n")
    save()

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
        save()

    def fault_set(key: str):
        ex = args.work / key
        entry = {"layers": []}
        geoms, props = [], []
        for src, layer in vector_sources(ex):
            info = {"path": os.path.relpath(src, ex), "layer": layer}
            entry["layers"].append(info)
            dest = ex / f"__{src.stem}_{layer or 'layer'}.geojson"
            try:
                gj = to_geojson(src, layer, dest, epsg, clip_bounds)
            except Exception as exc:
                info["error"] = str(exc)
                continue
            feats = gj.get("features", [])
            info["rows_in_template_bbox"] = len(feats)
            info["geom_types"] = sorted({(f.get("geometry") or {}).get("type", "None")
                                         for f in feats})
            if feats:
                info["columns"] = sorted(feats[0].get("properties", {}).keys())[:60]
            lines = line_geoms(gj)
            info["line_features"] = len(lines)
            geoms.extend(lines)
            props.extend(f.get("properties", {}) for f in feats
                         if (f.get("geometry") or {}).get("type") in ("LineString", "MultiLineString"))
        entry["traces_in_bbox"] = len(geoms)
        if not geoms:
            report["fault_sets"][key] = entry
            return
        for col in ("age", "AGE", "Age", "slip_rate", "SLIP_RATE", "fault_type", "mapped_sca",
                    "MAPPED_SCA", "num_sectio", "FaultAge", "fault_age", "Age_Cat", "sliprate",
                    "slipsense", "fault_class", "confidence", "Confidence"):
            vals = [str(pr.get(col)) for pr in props if col in pr]
            if vals:
                counts = {}
                for v in vals:
                    counts[v] = counts.get(v, 0) + 1
                top = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:20])
                entry.setdefault("attribute_counts", {})[col] = top
        shapes = [(g, 1) for g in geoms]
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

    for key in FAULT_SETS:
        if "error" in report["archives"].get(key, {"error": "missing"}):
            continue
        try:
            fault_set(key)
        except Exception:
            tb = traceback.format_exc()
            print("FAULT SET FAILED", key, tb, flush=True)
            report["fault_sets"].setdefault(key, {})["error"] = tb
        save()

    # Point sets for H15 (counts only; no modelling here).
    def point_set(key: str):
        ex = args.work / key
        entry = {"layers": []}
        for src, layer in vector_sources(ex):
            info = {"path": os.path.relpath(src, ex), "layer": layer}
            entry["layers"].append(info)
            dest = ex / f"__{src.stem}_{layer or 'layer'}.geojson"
            try:
                gj = to_geojson(src, layer, dest, epsg, None)
            except Exception as exc:
                info["error"] = str(exc)
                continue
            feats = gj.get("features", [])
            info["rows"] = len(feats)
            info["geom_types"] = sorted({(f.get("geometry") or {}).get("type", "None")
                                         for f in feats})
            if feats:
                info["columns"] = sorted(feats[0].get("properties", {}).keys())[:80]
            pts = [f for f in feats if (f.get("geometry") or {}).get("type") == "Point"]
            if not pts:
                continue
            xs = np.array([f["geometry"]["coordinates"][0] for f in pts], dtype=float)
            ys = np.array([f["geometry"]["coordinates"][1] for f in pts], dtype=float)
            rows, cols = rasterio.transform.rowcol(T, xs, ys)
            rows = np.asarray(rows)
            cols = np.asarray(cols)
            ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
            info["points_in_template"] = int(ok.sum())
            info["points_in_footprint"] = int(fp[rows[ok], cols[ok]].sum())
            keys = sorted({k for f in pts for k in f.get("properties", {}).keys()})
            target = args.out / f"{key}_{src.stem}_{layer or 'layer'}_points_utm.csv"
            import csv
            with target.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["utm_x", "utm_y", "row", "col", "in_footprint"] + keys)
                for f, x, y, r, c, o in zip(pts, xs, ys, rows, cols, ok):
                    pr = f.get("properties", {})
                    w.writerow([f"{x:.3f}", f"{y:.3f}", int(r), int(c),
                                int(bool(o and fp[r, c]))] + [pr.get(k) for k in keys])
            info["points_csv"] = target.name
            info["points_csv_sha256"] = sha256_file(target)
        report["point_sets"][key] = entry

    for key in ("paleo_geothermal", "probes_2m", "wellspring", "study_area"):
        if "error" in report["archives"].get(key, {"error": "missing"}):
            continue
        try:
            point_set(key)
        except Exception:
            tb = traceback.format_exc()
            print("POINT SET FAILED", key, tb, flush=True)
            report["point_sets"].setdefault(key, {})["error"] = tb
        save()

    for p in sorted(args.out.glob("*.npy")) + sorted(args.out.glob("*.csv")):
        report.setdefault("files", {})[p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    save()
    print(json.dumps({k: v for k, v in report.items() if k not in ("archives", "point_sets")},
                     indent=1, default=str)[:6000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
