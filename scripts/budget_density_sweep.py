#!/usr/bin/env python3
"""How does the optimal emission budget move when the truth gets sparser?

    python scripts/budget_density_sweep.py --report reports/h19_blocked.json --arm H16 \
        --out reports/budget_density_sweep.json

Motivation (session 3): the hidden test truth is *new* faults only — almost
certainly far sparser than the catalogue used by every local fold — and the
metric's FP term scales with emitted pixels / truth pixels. On the saved OOF
grids of a completed blocked report this script simulates sparser truth
honestly: within each held-out score region it labels catalogue systems
(8-connected), keeps a random fraction f of the systems as "new" truth, treats
the remaining systems as "known" (pixel-exact fp_ignore mask, exactly the
staff-described masking), and scores a budget ladder. It reports, per f, the
mean DTI by policy and the development-mean argmax. Nothing here trains,
selects a release, or estimates the hidden density; it maps the sensitivity.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems10 import cv, metric, modeling  # noqa: E402
from gems10.raster import sha256_file  # noqa: E402

POLICIES = ["topk01_binary", "topk02_binary", "topk03_binary", "topk04_binary",
            "topk06_binary", "topk08_binary", "thin04_binary", "thin06_binary",
            "thin08_binary", "thin12_binary"]
FRACTIONS = [1.0, 0.5, 0.25, 0.1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--arm", default="H16")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    report = json.loads(args.report.read_text())
    with rasterio.open(ROOT / "data/labels.tif") as ds:
        labels = ds.read(1) == 1
    with rasterio.open(ROOT / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    folds = cv.make_folds(n_blocks=4, n_folds=4, buffer_px=report["config"]["buffer_px"])
    out = {"source_report": str(args.report.resolve().relative_to(ROOT)), "arm": args.arm,
           "policies": POLICIES, "fractions": FRACTIONS, "draws": args.draws,
           "known_systems_masked": "pixel-exact fp_ignore_mask (staff-described masking)",
           "folds": []}
    for row in sorted(report["folds"], key=lambda r: r["fold"]):
        k = row["fold"]
        _, score = folds.train_test_masks(fp.shape, k)
        score &= fp
        pf = row["prediction_files"][args.arm]
        path = ROOT / pf["file"]
        if sha256_file(path) != pf["sha256"]:
            raise SystemExit(f"fold {k}: OOF grid hash mismatch")
        prob = np.load(path, allow_pickle=False)
        gt_all = labels & score
        sys_id, n_sys = ndimage.label(gt_all, np.ones((3, 3)))
        res = {"fold": k, "role": row["role"], "systems": int(n_sys), "gt_pixels": int(gt_all.sum()),
               "by_fraction": {}}
        emitted = {pol: np.nan_to_num(modeling.apply_policy(prob, score, pol), nan=0)
                   for pol in POLICIES}
        for f in FRACTIONS:
            draws = []
            for d in range(1 if f == 1.0 else args.draws):
                rng = np.random.default_rng(100 * k + d)
                keep = rng.random(n_sys + 1) < f
                keep[0] = False
                truth = keep[sys_id]
                known = gt_all & ~truth
                if not truth.any():
                    continue
                scores = {pol: metric.components(emitted[pol], truth, fp_ignore_mask=known).as_dict()
                          for pol in POLICIES}
                draws.append({"truth_pixels": int(truth.sum()), "known_pixels": int(known.sum()),
                              "dti": {pol: scores[pol]["dti"] for pol in POLICIES},
                              "n_pos_pred": {pol: scores[pol]["n_pos_pred"] for pol in POLICIES}})
            mean = {pol: float(np.mean([dr["dti"][pol] for dr in draws])) for pol in POLICIES}
            res["by_fraction"][str(f)] = {"draws": draws, "mean_dti": mean,
                                          "best_policy": max(POLICIES, key=lambda p: mean[p]),
                                          "truth_density": float(np.mean(
                                              [dr["truth_pixels"] for dr in draws]) / score.sum())}
            print(f"fold {k} f={f}: best {res['by_fraction'][str(f)]['best_policy']} "
                  + ", ".join(f"{p.split('_')[0]}={mean[p]:.4f}" for p in POLICIES), flush=True)
        out["folds"].append(res)
        del emitted, prob
    summary = {}
    for f in FRACTIONS:
        dev = [r for r in out["folds"] if r["role"] == "development" and str(f) in r["by_fraction"]]
        conf = [r for r in out["folds"] if r["role"] == "confirmation" and str(f) in r["by_fraction"]]
        dev_mean = {p: float(np.mean([r["by_fraction"][str(f)]["mean_dti"][p] for r in dev]))
                    for p in POLICIES}
        best = max(POLICIES, key=lambda p: dev_mean[p])
        summary[str(f)] = {"development_mean_dti": dev_mean, "development_best_policy": best,
                           "confirmation_dti_at_development_best": (
                               conf[0]["by_fraction"][str(f)]["mean_dti"][best] if conf else None),
                           "confirmation_best_policy": (
                               conf[0]["by_fraction"][str(f)]["best_policy"] if conf else None),
                           "mean_truth_density": float(np.mean(
                               [r["by_fraction"][str(f)]["truth_density"] for r in dev + conf]))}
    out["summary"] = summary
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
