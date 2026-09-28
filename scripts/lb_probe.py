#!/usr/bin/env python3
"""Leaderboard-probe analysis of the group's scored artifacts (session 5).

    .venv/bin/python scripts/audit_submissions.py      # fetch + hash the artifacts
    .venv/bin/python scripts/lb_probe.py               # -> reports/lb_probe.json

Seven DISTINCT fields carry user-reported public scores. Scores cannot be
reproduced locally (the private truth is hidden), but two things can be measured
exactly and related to them:

1. STRUCTURE — emitted mass by distance to the catalogue, clumpiness, and the
   kernel "reach" (expected recall if the truth were uniformly random pixels:
   mean over scored pixels of M(x) = max p*k within 300 m).
2. DENSITY BOUNDS — every artifact is binary, so with the metric identity
   DTI = r / (0.8 + 0.2 r + 0.2 f), r = TP/N, f = FP/N and FP ~= phi * mass
   (phi = public share of the region, N = public new-fault pixels), each score
   fixes r_i as a function of the hidden density rho = N / (phi * A):
       r_i(rho) = s_i (0.8 + 0.2 mass_i / (rho A)) / (1 - 0.2 s_i).
   r_i <= 1 for every artifact bounds rho from below; the implied placement
   skill r_i / reach_i (1 = random placement) shows which densities are
   plausible. This is a bound on a proxy, not a measurement of the truth:
   FP ~= phi * mass ignores the small near-truth credit, and scores are
   user-reported (account/file association not independently verified).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems10 import metric  # noqa: E402
from gems10.raster import sha256_file  # noqa: E402

DISTINCT = ["GEMSDOE1", "GEMSDOE2-union", "GEMSDOE3-nodes", "GEMSDOE3-ridge",
            "GEMSDOE3-gap", "GEMSDOE4", "6GEMSDOE"]
BANDS = [(0.5, 1.01, "d=1px"), (1.01, 1.5, "d=sqrt2"), (1.5, 2.01, "d=2px"),
         (2.01, 3.01, "2-3px"), (3.01, 10.01, "3-10px"), (10.01, 30.01, "10-30px"),
         (30.01, 1e9, ">30px")]
DENSITIES = [0.0012, 0.0015, 0.002, 0.003, 0.004, 0.006, 0.008, 0.0118]


def structure(x, scored, known_d):
    xm = np.where(scored, x, 0.0)
    mass = float(xm.sum())
    b = xm > 0
    nb = ndimage.convolve(b.astype(np.int8), np.ones((3, 3), np.int8), mode="constant") - b
    M = metric.best_weighted_prediction(xm)
    return {"mass": mass, "n_pos": int(b.sum()),
            "binary": bool(np.all(xm[b] == 1.0)),
            "mass_share_by_distance_to_catalogue": {
                name: round(float(xm[(known_d > lo) & (known_d <= hi)].sum() / mass), 4)
                for lo, hi, name in BANDS},
            "share_within_300m": round(float(xm[(known_d > 0) & (known_d <= 3.01)].sum()
                                             / mass), 4),
            "share_beyond_1km": round(float(xm[known_d > 10.01].sum() / mass), 4),
            "frac_pos_with_ge5_neighbours": round(float((nb[b] >= 5).mean()), 4),
            "components_8conn": int(ndimage.label(b, np.ones((3, 3)))[1]),
            "reach_uniform_truth_recall": round(float(M[scored].mean()), 4)}


def main() -> int:
    audit = json.loads((ROOT / "reports/submission_audit.json").read_text())
    by_id = {e["id"]: e for e in audit["artifacts"]}
    with rasterio.open(ROOT / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    with rasterio.open(ROOT / "data/labels.tif") as ds:
        known = ds.read(1) == 1
    scored = fp & ~known
    A = int(scored.sum())
    known_d = ndimage.distance_transform_edt(~known)
    rows = {}
    for ident in DISTINCT:
        e = by_id[ident]
        path = ROOT / "scratch/artifact_audit" / f"{ident}.tif"
        if sha256_file(path) != e["sha256"]:
            raise SystemExit(f"{ident}: file differs from the audited sha256")
        with rasterio.open(path) as ds:
            x = np.where(fp, np.nan_to_num(ds.read(1).astype(np.float64), nan=0.0), 0.0)
        rows[ident] = {"user_reported_score": e["user_reported_score"],
                       "sha256": e["sha256"], "source_url": e.get("source_url"),
                       **structure(x, scored, known_d)}
        print(ident, json.dumps(rows[ident]["mass_share_by_distance_to_catalogue"]), flush=True)
    ours = {}
    for f in sorted((ROOT / "docs/downloads").glob("*.tif")):
        with rasterio.open(f) as ds:
            x = np.where(fp, np.nan_to_num(ds.read(1).astype(np.float64), nan=0.0), 0.0)
        ours[f.name] = {"sha256": sha256_file(f), **structure(x, scored, known_d)}
    table = {}
    for rho in DENSITIES:
        lam = 1.0 / (rho * A)
        entry = {}
        for ident, r in rows.items():
            s = r["user_reported_score"]
            rec = s * (0.8 + 0.2 * r["mass"] * lam) / (1 - 0.2 * s)
            entry[ident] = {"implied_recall": round(rec, 4),
                            "implied_skill_vs_random": round(rec / r["reach_uniform_truth_recall"], 3)}
        entry["feasible_all_recall_le_1"] = all(v["implied_recall"] <= 1 for v in entry.values()
                                                if isinstance(v, dict))
        entry["min_skill"] = min(v["implied_skill_vs_random"] for v in entry.values()
                                 if isinstance(v, dict))
        table[f"{rho:.4f}"] = entry
    out = {"scored_area_px": A, "artifacts": rows, "ours_unscored": ours,
           "density_inversion": table,
           "catalogue_density_of_footprint": 0.0118,
           "reading": [
               "All seven scored fields are binary.",
               "rho below ~0.12% is infeasible (the best artifact would need recall > 1).",
               "At catalogue density (~1.2%) both isolated-pixel artifacts would have to be "
               "placed WORSE than random (skill ~0.4-0.6); skills of all artifacts are >= ~0.9 "
               "only for rho <~0.4%. Plausible hidden density: ~0.13-0.6% (2-8x sparser than "
               "the catalogue proxy).",
               "6GEMSDOE (worst, 0.0286) put 68% of its mass within 300 m of catalogue faults; "
               "the best (0.1563) put 18.5% there and 59% beyond 1 km. Blobby emission "
               "(GEMSDOE4, 6GEMSDOE) scored ~0.03 regardless.",
               "The local catalogue proxy never scores emission next to KNOWN faults (score "
               "blocks contain no training faults) — a blind spot this probe exposes."],
           "caveats": ["Scores are user-reported, not authenticated submission history",
                       "FP ~= phi * mass ignores near-truth credit (a few % at most)",
                       "Seven heterogeneous models; bounds, not estimates"]}
    (ROOT / "reports/lb_probe.json").write_text(json.dumps(out, indent=2) + "\n")
    for rho, e in table.items():
        print(rho, "feasible" if e["feasible_all_recall_le_1"] else "INFEASIBLE",
              "min_skill", e["min_skill"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
