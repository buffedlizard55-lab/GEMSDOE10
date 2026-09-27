#!/usr/bin/env python3
"""Second-look rescoring of the OOF grids saved by validate_candidate.py.

    python scripts/rescore_blocked.py --report reports/h19_blocked.json \
        --policies thin25_binary,thin30_binary,thin40_binary,thin50_binary \
        --out reports/h19b_rescore.json

Loads the completed blocked report, rebuilds the spatial folds, verifies that
the rebuilt score masks hash to the values recorded in the report, loads each
arm's saved OOF probability grid (sha256-checked) and scores extra policies on
the SAME held-out geography. No retraining, no new predictions.

This is explicitly a POST-HOC extension ("second look"): the policies were not
in the preregistered union, so the output is labelled `post_hoc: true` and the
release gate does not accept it by itself. Its purpose is to find out whether
the preregistered sweep stopped short of the optimum; any policy chosen here
still has to beat the incumbent on the development mean AND the untouched
confirmation fold, and the fact that it was a second look is recorded.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems10 import cv, metric, modeling  # noqa: E402
from gems10.raster import sha256_file  # noqa: E402


def mask_hash(mask) -> str:
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--policies", required=True, help="comma list of extra policies")
    ap.add_argument("--arms", default=None, help="comma list; default: all arms with saved grids")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    started = time.time()
    report = json.loads(args.report.read_text())
    if report.get("status") != "completed":
        raise SystemExit("report is not completed")
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for p in policies:
        modeling.parse_policy(p)  # validate names up front
    with rasterio.open(ROOT / "data/labels.tif") as ds:
        labels = ds.read(1)
    with rasterio.open(ROOT / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    inputs = report["inputs"]
    for name in ("labels.tif", "sample_submission.tif"):
        if sha256_file(ROOT / "data" / name) != inputs[name]:
            raise SystemExit(f"{name} differs from the report's input")
    cfg = report["config"]
    folds = cv.make_folds(n_blocks=4, n_folds=4, buffer_px=cfg["buffer_px"])
    arms = (args.arms.split(",") if args.arms else
            [a for a in report["folds"][0].get("prediction_files", {})])
    out = {"source_report": str(args.report.resolve().relative_to(ROOT)),
           "source_protocol": report["protocol"], "post_hoc": True,
           "note": ("second look on saved OOF grids; policies were NOT in the preregistered "
                    "union of the source report — selection here is exploratory and any use "
                    "must be declared as such"),
           "policies": policies, "arms": arms, "folds": []}
    for row in sorted(report["folds"], key=lambda r: r["fold"]):
        k = row["fold"]
        train, score = folds.train_test_masks(fp.shape, k)
        train &= fp
        score &= fp
        if mask_hash(train) != row["train_mask_sha256"] or mask_hash(score) != row["score_mask_sha256"]:
            raise SystemExit(f"fold {k}: rebuilt masks differ from the report")
        gt = (labels == 1) & score
        res = {"fold": k, "role": row["role"], "scores": {}}
        for arm in arms:
            pf = row["prediction_files"][arm]
            path = ROOT / pf["file"]
            if sha256_file(path) != pf["sha256"]:
                raise SystemExit(f"fold {k} {arm}: saved OOF grid hash differs from the report")
            prob = np.load(path, allow_pickle=False)
            if np.isfinite(prob[~score]).any() or not np.isfinite(prob[score]).all():
                raise SystemExit(f"fold {k} {arm}: OOF footprint differs from score geography")
            res["scores"][arm] = {}
            for pol in policies:
                e = np.nan_to_num(modeling.apply_policy(prob, score, pol), nan=0)
                res["scores"][arm][pol] = metric.components(e, gt).as_dict()
            # carry the preregistered selection for side-by-side reading
            sel = report["policy_selection"].get(arm)
            if sel:
                res["scores"][arm][f"preregistered:{sel}"] = row["scores"][arm][sel]
            print(f"fold {k} {arm}: " + ", ".join(
                f"{p}={res['scores'][arm][p]['dti']:.5f}" for p in policies), flush=True)
        out["folds"].append(res)
    # Development-mean argmax per arm over the extra policies, confirmation shown separately.
    summary = {}
    for arm in arms:
        rows = out["folds"]
        means = {p: float(np.mean([r["scores"][arm][p]["dti"] for r in rows
                                   if r["role"] == "development"])) for p in policies}
        best = max(policies, key=lambda p: means[p])
        conf = [r["scores"][arm][best]["dti"] for r in rows if r["role"] == "confirmation"][0]
        summary[arm] = {"development_mean_by_policy": means, "best_extra_policy": best,
                        "best_extra_development_mean": means[best],
                        "best_extra_confirmation": conf}
    out["summary"] = summary
    out["elapsed_seconds"] = round(time.time() - started, 1)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
