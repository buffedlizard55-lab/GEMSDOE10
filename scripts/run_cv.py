#!/usr/bin/env python3
"""System-holdout CV: the GEMSDOE10 yardstick for hidden-fault detection.

    python scripts/run_cv.py --data-dir /tmp/gems10data --work-dir /tmp/gems10cv

Per fold: hold out whole fault systems (the "unmapped faults" proxy), train an
HGB on the rest, predict the full grid, sweep placement policies, and score
GT = held-out systems with fp_ignore = train systems (masked = official-like
scoring per forum 11516) plus the unmasked literal formula.

Options:
  --folds-subset  e.g. "0,1" for quick iteration (default: all 5)
  --negatives     train negatives per fold (default 200000)
  --iterations/--lr/--depth/--l2  HGB hyperparameters
  --max-channels  use only the first C feature channels (smoke runs)
  --with-discovery  add strike rays + relay corridors (noisy-OR, weights via
                    --discovery-w, e.g. "0.5,0.4")
  --with-selftrain  semi-supervised second pass (top 0.5% + agreement>=2)
  --policies  comma list; default topk{1,2,3,4,5}x{binary,soft} + envelopes@3

Writes: <work-dir>/fold{k}_prob.npy, reports/cv_system_holdout.json (repo),
and prints the policy x fold table (masked DTI is the decision column).
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gems10 import discovery, external, metric, modeling, selftrain, spec, systems  # noqa: E402

DEFAULT_POLICIES = [f"topk{b:02d}_{m}" for b in (1, 2, 3, 4, 5)
                    for m in ("binary", "soft")]
DEFAULT_POLICIES += ["topk03_envelope", "topk03_halo2", "topk02_envelope"]


def train_proximity_channel(train_sys: np.ndarray, cap_px: float = 50.0) -> np.ndarray:
    """Per-fold label-derived channel: 1 on train systems, decaying to 0.

    MUST be recomputed per fold from train systems only (done here) — never
    stored globally, never built from all labels inside CV.
    """
    d = ndimage.distance_transform_edt(~np.asarray(train_sys, dtype=bool))
    return (1.0 - np.minimum(d, cap_px) / cap_px).astype(np.float32)


def run_fold(feat: np.ndarray, channels: list[str], labels: np.ndarray,
             footprint: np.ndarray, sf: systems.SystemFolds, k: int,
             args, work_dir: Path,
             ext_grids: list[np.ndarray] | None = None) -> dict:
    t0 = time.time()
    held = sf.heldout_mask(k)
    train_sys = sf.train_system_mask(k)
    trainable = sf.trainable_mask(k) & footprint
    n_gt = int(held.sum())
    print(f"[fold {k}] gt(hidden)={n_gt} train_sys={int(train_sys.sum())} "
          f"trainable={int(trainable.sum())}", flush=True)

    # The proximity channel is OFF by default: harness finding 2026-09-27 —
    # with it, the model predicts train systems at 0.98 and held-out systems
    # at background level (0.056) — pure catalogue memorisation, zero hidden-
    # fault transfer (masked DTI 0.012). Kept behind a flag for ablation only.
    prox = train_proximity_channel(train_sys) if args.with_proximity else None
    rows, cols, y = systems.sample_training_pixels(
        trainable, train_sys, footprint, n_neg=args.negatives, seed=1000 + k)
    X = feat[rows, cols].astype(np.float32)
    if ext_grids:
        for g in ext_grids:
            X = np.concatenate([X, g[rows, cols].astype(np.float32)], axis=1)
    if prox is not None:
        X = np.concatenate([X, prox[rows, cols].reshape(-1, 1)], axis=1)
    print(f"[fold {k}] train X={X.shape} pos={int(y.sum())}", flush=True)

    extra = None
    if prox is not None or ext_grids:
        parts = []
        if ext_grids:
            parts.append(np.concatenate(ext_grids, axis=2))
        if prox is not None:
            parts.append(prox[:, :, None])
        extra = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=2)
    if args.pu_bags and args.pu_bags > 1:
        # Bagging PU (Mordelet & Vert 2014): negatives are contaminated by
        # unmapped faults, so average K models trained on all positives plus
        # independent negative subsamples. Costs Kx training, decided by FAR.
        probs = []
        for b in range(args.pu_bags):
            rb, cb, yb = systems.sample_training_pixels(
                trainable, train_sys, footprint, n_neg=args.negatives,
                seed=1000 + k + 7919 * (b + 1))
            Xb = feat[rb, cb].astype(np.float32)
            if ext_grids:
                for g in ext_grids:
                    Xb = np.concatenate([Xb, g[rb, cb].astype(np.float32)],
                                        axis=1)
            if prox is not None:
                Xb = np.concatenate([Xb, prox[rb, cb].reshape(-1, 1)], axis=1)
            clfb = modeling.fit_hgb(Xb, yb, iterations=args.iterations,
                                    lr=args.lr, depth=args.depth, l2=args.l2,
                                    seed=7 + b)
            probs.append(modeling.predict_grid(clfb, feat, footprint,
                                               extra=extra))
            print(f"[fold {k}] pu bag {b + 1}/{args.pu_bags} done", flush=True)
        prob = np.nanmean(np.stack(probs), axis=0)
    else:
        clf = modeling.fit_hgb(X, y, iterations=args.iterations, lr=args.lr,
                               depth=args.depth, l2=args.l2, seed=7)
        prob = modeling.predict_grid(clf, feat, footprint, extra=extra)
    if args.with_selftrain:
        elig = trainable & footprint & ~train_sys
        # agreement on a stable full-grid sample (no labels involved)
        agree_full = np.zeros(footprint.shape, dtype=np.int8)
        ys, xs = np.nonzero(footprint)
        Xs = feat[ys, xs].astype(np.float32)
        agree_full[ys, xs] = selftrain.family_agreement(Xs, channels)
        pseudo = selftrain.select_pseudo(prob, elig, agree_full, top_frac=0.005,
                                         min_agree=2)
        pr, pc = np.nonzero(pseudo)
        print(f"[fold {k}] pseudo-positives: {pr.size}", flush=True)
        if pr.size > 100:
            Xp = feat[pr, pc].astype(np.float32)
            if ext_grids:
                for g in ext_grids:
                    Xp = np.concatenate([Xp, g[pr, pc].astype(np.float32)], axis=1)
            if prox is not None:
                Xp = np.concatenate([Xp, prox[pr, pc].reshape(-1, 1)], axis=1)
            X2 = np.concatenate([X, Xp], axis=0)
            y2 = np.concatenate([y, np.ones(pr.size, dtype=np.int8)])
            w2 = np.concatenate([np.ones(y.size), np.full(pr.size, 0.5)])
            clf = modeling.fit_hgb(X2, y2, sample_weight=w2,
                                   iterations=args.iterations, lr=args.lr,
                                   depth=args.depth, l2=args.l2, seed=8)
            prob = modeling.predict_grid(clf, feat, footprint, extra=extra)
    if args.with_discovery:
        use = np.unique(sf.system_id[train_sys])
        strikes = discovery.system_strikes(sf.system_id, use)
        rays = discovery.strike_rays(footprint.shape, strikes, ray_len_px=10.0)
        corr = discovery.relay_corridors(footprint.shape, strikes)
        wr, wc = (float(v) for v in args.discovery_w.split(","))
        prob = discovery.noisy_or_fuse(np.nan_to_num(prob, nan=0.0),
                                       (rays, wr), (corr, wc))
        prob = np.where(footprint, prob, np.nan)
        print(f"[fold {k}] discovery: {len(strikes)} strikes, "
              f"rays={int((rays>0).sum())} corr={int((corr>0).sum())}", flush=True)

    np.save(work_dir / f"fold{k}_prob.npy", prob.astype(np.float32))
    scored_gt = held & footprint
    res = modeling.evaluate_policies(prob, footprint, scored_gt, train_sys,
                                     args.policies)
    # Distance-stratified GT: hidden faults are anti-selected AWAY from the
    # catalogue (median unseen-fault distance 2.2 km, r7 audit), so GT_FAR10
    # (>1 km from train systems) is the decision column; GT_ALL is reference.
    d_to_train = ndimage.distance_transform_edt(~train_sys)
    for gt_name, gt in (("FAR10", scored_gt & (d_to_train > 10)),
                        ("FAR20", scored_gt & (d_to_train > 20))):
        if gt.sum() == 0:
            continue
        for pol, row in modeling.evaluate_policies(
                prob, footprint, gt, train_sys, args.policies).items():
            res[f"{gt_name}/{pol}"] = row
    dt = time.time() - t0
    print(f"[fold {k}] done in {dt:.0f}s; " +
          " ".join(f"{p}={res[p]['dti_masked']:.4f}/{res[p]['dti_unmasked']:.4f}"
                   for p in args.policies), flush=True)
    return {"fold": k, "n_gt": n_gt, "seconds": round(dt, 1), "policies": res}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    ap.add_argument("--features", default=None)
    ap.add_argument("--work-dir", default="/tmp/gems10cv")
    ap.add_argument("--report", default=str(REPO_ROOT / "reports" / "cv_system_holdout.json"))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--folds-subset", default=None)
    ap.add_argument("--negatives", type=int, default=200000)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--max-channels", type=int, default=None)
    ap.add_argument("--externals", default=None,
                    help="comma list of u8 stacks to append (lidar,rad)")
    ap.add_argument("--with-discovery", action="store_true")
    ap.add_argument("--discovery-w", default="0.5,0.4")
    ap.add_argument("--with-selftrain", action="store_true")
    ap.add_argument("--with-proximity", action="store_true",
                    help="ablation only: add dist-to-train channel (memorises)")
    ap.add_argument("--pu-bags", type=int, default=0,
                    help="bagging-PU bags (0/1 = off)")
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    args = ap.parse_args()
    args.policies = args.policies.split(",")

    t0 = time.time()
    data_dir = Path(args.data_dir)
    feat_path = Path(args.features) if args.features else data_dir / "features107.f32.npy"
    meta = json.loads(feat_path.with_suffix(".meta.json").read_text())
    channels = meta["channels"]
    feat = np.lib.format.open_memmap(str(feat_path), mode="r")
    if args.max_channels:
        channels = channels[:args.max_channels]
        feat = feat[:, :, :args.max_channels]
    print(f"features: {feat.shape} ({len(channels)} ch)")

    with rasterio.open(data_dir / "labels.tif") as s:
        labels = s.read(1)
    with rasterio.open(data_dir / "sample_submission.tif") as s:
        footprint = np.isfinite(s.read(1))
    sf = systems.make_system_folds(labels, n_folds=args.n_folds, buffer_px=3, seed=7)
    print(f"systems: {sf.n_systems}, fold px: {sf.fold_pixel_counts()}")

    ext_grids: list[np.ndarray] = []
    if args.externals:
        for name in args.externals.split(","):
            p = data_dir / "external" / ({"lidar": "lidar_scarp_features_u8.tif",
                                          "rad": "geodawn_rad_u8.tif"}[name])
            g = external.read_u8_stack(
                str(p), expect_shape=(spec.HEIGHT, spec.WIDTH),
                grid_transform=(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0),
                grid_crs="EPSG:32611")
            ext_grids.append(g)
            channels += [f"{name}_u8_{i}" for i in range(g.shape[2])]
            print(f"external {name}: {g.shape}")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    folds = list(range(args.n_folds))
    if args.folds_subset:
        folds = [int(v) for v in args.folds_subset.split(",")]
    out = {"config": {k: (v if not isinstance(v, Path) else str(v))
                      for k, v in vars(args).items()},
           "channels": channels, "fold_pixel_counts": sf.fold_pixel_counts(),
           "folds": []}
    for k in folds:
        out["folds"].append(run_fold(feat, channels, labels, footprint, sf, k,
                                     args, work_dir, ext_grids))
    # aggregate
    agg: dict[str, dict[str, float]] = {}
    for p in args.policies:
        m = np.array([f["policies"][p]["dti_masked"] for f in out["folds"]])
        u = np.array([f["policies"][p]["dti_unmasked"] for f in out["folds"]])
        agg[p] = {"masked_mean": float(m.mean()), "masked_min": float(m.min()),
                  "masked_max": float(m.max()), "unmasked_mean": float(u.mean())}
    out["aggregate"] = agg
    out["total_seconds"] = round(time.time() - t0, 1)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(out, indent=1) + "\n")
    print("\n== aggregate (masked_mean is the decision column) ==")
    for p in sorted(agg, key=lambda q: -agg[q]["masked_mean"]):
        a = agg[p]
        print(f"  {p:16s} masked {a['masked_mean']:.4f} "
              f"[{a['masked_min']:.4f},{a['masked_max']:.4f}]  "
              f"unmasked {a['unmasked_mean']:.4f}")
    print(f"total {out['total_seconds']:.0f}s -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
