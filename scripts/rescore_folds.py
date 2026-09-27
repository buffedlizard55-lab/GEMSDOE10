#!/usr/bin/env python3
"""Post-hoc rescoring of saved fold probability grids under any GT definition.

    python scripts/rescore_folds.py --work-dir /tmp/gems10cv_base --data-dir /tmp/gems10data

Loads <work-dir>/fold{k}_prob.npy, rebuilds the system folds (seed 7), and
scores GT_ALL / GT_FAR10 (>1 km from train systems) / GT_FAR20 (>2 km) with
fp_ignore = train systems (masked) plus unmasked. Writes
<work-dir>/rescore.json. Lets the campaign keep running while GT definitions
evolve — no retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gems10 import metric, modeling, systems  # noqa: E402

POLICIES = ["topk01_binary", "topk02_binary", "topk03_binary", "topk04_binary",
            "topk02_soft", "topk03_soft", "topk02_envelope", "topk03_envelope",
            "topk03_halo2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--folds-subset", default=None)
    ap.add_argument("--policies", default=",".join(POLICIES))
    args = ap.parse_args()
    policies = args.policies.split(",")
    work_dir = Path(args.work_dir)
    data_dir = Path(args.data_dir)

    with rasterio.open(data_dir / "labels.tif") as s:
        labels = s.read(1)
    with rasterio.open(data_dir / "sample_submission.tif") as s:
        footprint = np.isfinite(s.read(1))
    sf = systems.make_system_folds(labels, n_folds=args.n_folds, buffer_px=3,
                                   seed=7)
    out = {"work_dir": str(work_dir), "policies": policies, "folds": []}
    folds = list(range(args.n_folds))
    if args.folds_subset:
        folds = [int(v) for v in args.folds_subset.split(",")]
    for k in folds:
        p = work_dir / f"fold{k}_prob.npy"
        if not p.exists():
            print(f"fold {k}: missing, skip")
            continue
        prob = np.load(p)
        held = sf.heldout_mask(k) & footprint
        ign = sf.train_system_mask(k)
        d = ndimage.distance_transform_edt(~ign)
        gts = {"GT_ALL": held, "GT_FAR10": held & (d > 10),
               "GT_FAR20": held & (d > 20)}
        row = {"fold": k, "n_gt": {n: int(g.sum()) for n, g in gts.items()},
               "scores": {}}
        for name, gt in gts.items():
            if gt.sum() == 0:
                continue
            for pol in policies:
                emis = np.nan_to_num(modeling.apply_policy(prob, footprint, pol),
                                     nan=0.0)
                cm = metric.components(emis, gt, fp_ignore_mask=ign)
                cu = metric.components(emis, gt)
                row["scores"][f"{name}/{pol}"] = {
                    "masked": cm.dti, "unmasked": cu.dti,
                    "tp_w": cm.tp_w, "fp_w": cm.fp_w, "fn_w": cm.fn_w}
        out["folds"].append(row)
        f = row["scores"].get("GT_FAR10/topk02_binary", {})
        print(f"fold {k}: n={row['n_gt']} far10/topk02b masked={f.get('masked', float('nan')):.4f}",
              flush=True)
    agg: dict[str, dict] = {}
    keys = out["folds"][0]["scores"].keys() if out["folds"] else []
    for key in keys:
        m = np.array([f["scores"][key]["masked"] for f in out["folds"]])
        u = np.array([f["scores"][key]["unmasked"] for f in out["folds"]])
        agg[key] = {"masked_mean": float(m.mean()), "masked_min": float(m.min()),
                    "masked_max": float(m.max()), "unmasked_mean": float(u.mean())}
    out["aggregate"] = agg
    (work_dir / "rescore.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n== aggregate (GT_FAR10/masked is the decision column) ==")
    for key in sorted(agg, key=lambda q: -agg[q]["masked_mean"]):
        a = agg[key]
        print(f"  {key:28s} masked {a['masked_mean']:.4f} "
              f"[{a['masked_min']:.4f},{a['masked_max']:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
