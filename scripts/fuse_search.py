#!/usr/bin/env python3
"""Post-hoc discovery-fusion search from saved BASE fold probabilities.

    python scripts/fuse_search.py --base-dir /tmp/gems10cv_base --data-dir /tmp/gems10data

For each fold: load base prob, rebuild train-system strikes/rays/corridors +
lidar ridge, grid-search noisy-OR weights WITHOUT retraining, and score
GT_ALL/GT_FAR10/GT_FAR20 (masked) at topk01/02/03_binary. Writes
fuse_search.json next to the base dir. The winning weights feed train_final.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gems10 import discovery, external, metric, modeling, systems  # noqa: E402

POLICIES = ["topk01_binary", "topk02_binary", "topk03_binary"]


def cached_scorer_setup(gt: np.ndarray, radius_px: float = 3.0):
    """Precompute the FP kernel weights (depend on GT only, not on pred)."""
    from scipy import ndimage as _ndi

    g = np.asarray(gt, dtype=bool)
    if g.sum() == 0:
        return None
    d = _ndi.distance_transform_edt(~g, sampling=1.0)
    k_nearest = np.maximum(1.0 - d / radius_px, 0.0)
    return {"k": k_nearest, "n_gt": int(g.sum())}


def cached_dti(pred01: np.ndarray, gt: np.ndarray, setup: dict,
               fp_ignore: np.ndarray, alpha: float = 0.2, beta: float = 0.8,
               radius_px: float = 3.0) -> float:
    """DTI with cached FP weights — identical to metric.components()."""
    from gems10 import metric as _metric

    p = np.clip(np.asarray(pred01, dtype=np.float64), 0.0, 1.0)
    pos = (p > 0.0) & ~np.asarray(fp_ignore, dtype=bool)
    fp_w = float((p * (1.0 - setup["k"]))[pos].sum()) if pos.any() else 0.0
    m = _metric.best_weighted_prediction(p, radius_px)
    mg = m[np.asarray(gt, dtype=bool)]
    tp_w = float(mg.sum())
    fn_w = float((1.0 - mg).sum())
    denom = tp_w + alpha * fp_w + beta * fn_w
    return float(tp_w / denom) if denom > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--folds-subset", default=None)
    ap.add_argument("--ray-lens", default="5,10,15")
    ap.add_argument("--weights", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--lidar-w", default="0.0,0.3,0.5")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    t0 = time.time()
    base_dir = Path(args.base_dir)
    data_dir = Path(args.data_dir)
    ray_lens = [float(v) for v in args.ray_lens.split(",")]
    weights = [float(v) for v in args.weights.split(",")]
    lidar_ws = [float(v) for v in args.lidar_w.split(",")]

    with rasterio.open(data_dir / "labels.tif") as s:
        labels = s.read(1)
    with rasterio.open(data_dir / "sample_submission.tif") as s:
        footprint = np.isfinite(s.read(1))
    sf = systems.make_system_folds(labels, n_folds=args.n_folds, buffer_px=3,
                                   seed=7)
    lidar = external.read_u8_stack(
        str(data_dir / "external" / "lidar_scarp_features_u8.tif"),
        expect_shape=labels.shape)
    ex = lidar[:, :, 0]
    valid = np.isfinite(ex) & footprint
    thr = float(np.nanquantile(np.where(valid, ex, np.nan), 0.98))
    ridge = (valid & (ex >= thr)).astype(np.float64)

    combos = [(rl, wr, wc, wl) for rl in ray_lens for wr in weights
              for wc in weights for wl in lidar_ws]
    print(f"{len(combos)} combos x {args.n_folds} folds", flush=True)
    fold_cache: dict[int, dict] = {}
    folds = list(range(args.n_folds))
    if args.folds_subset:
        folds = [int(v) for v in args.folds_subset.split(",")]
    for k in folds:
        p = base_dir / f"fold{k}_prob.npy"
        if not p.exists():
            continue
        train_sys = sf.train_system_mask(k)
        use = np.unique(sf.system_id[train_sys])
        strikes = discovery.system_strikes(sf.system_id, use)
        held = sf.heldout_mask(k) & footprint
        d = ndimage.distance_transform_edt(~train_sys)
        gts = {"ALL": held, "FAR10": held & (d > 10), "FAR20": held & (d > 20)}
        setups = {name: cached_scorer_setup(gt) for name, gt in gts.items()}
        fold_cache[k] = {
            "prob": np.nan_to_num(np.load(p).astype(np.float64), nan=0.0),
            "strikes": strikes,
            "corr": discovery.relay_corridors(footprint.shape, strikes),
            "gts": gts, "setups": setups, "ign": train_sys}
        print(f"fold {k} cached ({len(strikes)} strikes)", flush=True)

    results: dict[str, dict] = {}
    for ci, (rl, wr, wc, wl) in enumerate(combos):
        key = f"rl{rl:g}_wr{wr:g}_wc{wc:g}_wl{wl:g}"
        agg: dict[str, list[float]] = {}
        for k, fc in fold_cache.items():
            rays = discovery.strike_rays(footprint.shape, fc["strikes"],
                                         ray_len_px=rl)
            fused = discovery.noisy_or_fuse(
                fc["prob"], (rays, wr), (fc["corr"], wc), (ridge, wl))
            fused = np.where(footprint, fused, np.nan)
            emis_cache = {pol: np.nan_to_num(
                modeling.apply_policy(fused, footprint, pol), nan=0.0)
                for pol in POLICIES}
            for gt_name, gt in fc["gts"].items():
                setup = fc["setups"][gt_name]
                for pol in POLICIES:
                    m = cached_dti(emis_cache[pol], gt, setup, fc["ign"])
                    agg.setdefault(f"{gt_name}/{pol}", []).append(m)
        results[key] = {kk: float(np.mean(vv)) for kk, vv in agg.items()}
        if ci % 10 == 0:
            print(f"[{time.time()-t0:.0f}s] {ci}/{len(combos)} {key} "
                  f"FAR10/topk02={results[key]['FAR10/topk02_binary']:.4f}",
                  flush=True)
    # rank by FAR10/topk02_binary (decision), show top 15
    ranked = sorted(results, key=lambda q: -results[q]["FAR10/topk02_binary"])
    print("\n== top 15 by FAR10/topk02_binary ==")
    for key in ranked[:15]:
        r = results[key]
        print(f"  {key:24s} FAR10/02b={r['FAR10/topk02_binary']:.4f} "
              f"ALL/02b={r['ALL/topk02_binary']:.4f} FAR20/02b={r['FAR20/topk02_binary']:.4f}")
    report = Path(args.report) if args.report else base_dir / "fuse_search.json"
    report.write_text(json.dumps({"combos": results, "ranked": ranked,
                                  "seconds": round(time.time() - t0, 1)},
                                 indent=1) + "\n")
    print("->", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
