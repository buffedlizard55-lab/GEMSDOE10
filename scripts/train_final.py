#!/usr/bin/env python3
"""Final full-grid fit: train on ALL catalogue systems, emit probability map.

    python scripts/train_final.py --data-dir /tmp/gems10data --work-dir /tmp/gems10final

Mirrors the winning run_cv.py configuration (same sampling, HGB params,
proximity channel, externals, discovery fusion) but trains on every system and
predicts the full footprint. Output <work-dir>/prob_final.npy (+ .meta.json)
feeds scripts/build_submission.py. Deterministic for fixed seeds (recorded).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gems10 import discovery, external, modeling, selftrain, spec, systems  # noqa: E402
from run_cv import train_proximity_channel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    ap.add_argument("--features", default=None)
    ap.add_argument("--work-dir", default="/tmp/gems10final")
    ap.add_argument("--negatives", type=int, default=200000)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--externals", default=None,
                    help="comma list of u8 stacks to append as features "
                         "(lidar,rad) from <data-dir>/external/")
    ap.add_argument("--with-discovery", action="store_true")
    ap.add_argument("--discovery-w", default="0.5,0.4")
    ap.add_argument("--with-selftrain", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    data_dir = Path(args.data_dir)
    feat_path = Path(args.features) if args.features else data_dir / "features107.f32.npy"
    meta = json.loads(feat_path.with_suffix(".meta.json").read_text())
    channels = list(meta["channels"])
    feat = np.lib.format.open_memmap(str(feat_path), mode="r")
    with rasterio.open(data_dir / "labels.tif") as s:
        labels = s.read(1)
    with rasterio.open(data_dir / "sample_submission.tif") as s:
        footprint = np.isfinite(s.read(1))
    pos = (labels == 1) & footprint
    print(f"positives: {int(pos.sum())}, footprint: {int(footprint.sum())}")

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
    prox = train_proximity_channel(pos)
    channels = channels + ["dist_to_catalogue"]

    rng = np.random.default_rng(args.seed)
    py, px = np.nonzero(pos)
    ny, nx = np.nonzero(footprint & ~pos)
    sel = rng.choice(ny.size, size=min(args.negatives, ny.size), replace=False)
    rows = np.concatenate([py, ny[sel]])
    cols = np.concatenate([px, nx[sel]])
    y = np.concatenate([np.ones(py.size, dtype=np.int8),
                        np.zeros(sel.size, dtype=np.int8)])
    X = feat[rows, cols].astype(np.float32)
    for g in ext_grids:
        X = np.concatenate([X, g[rows, cols].astype(np.float32)], axis=1)
    X = np.concatenate([X, prox[rows, cols].reshape(-1, 1)], axis=1)
    print(f"train X={X.shape}", flush=True)
    clf = modeling.fit_hgb(X, y, iterations=args.iterations, lr=args.lr,
                           depth=args.depth, l2=args.l2, seed=args.seed)

    ext_full = None
    if ext_grids:
        ext_full = np.concatenate(
            [np.concatenate(ext_grids, axis=2), prox[:, :, None]], axis=2)
    else:
        ext_full = prox[:, :, None]
    prob = modeling.predict_grid(clf, np.asarray(feat), footprint, extra=ext_full)

    if args.with_selftrain:
        elig = footprint & ~pos
        agree_full = np.zeros(footprint.shape, dtype=np.int8)
        ys, xs = np.nonzero(footprint)
        Xs = feat[ys, xs].astype(np.float32)
        agree_full[ys, xs] = selftrain.family_agreement(Xs, meta["channels"])
        pseudo = selftrain.select_pseudo(prob, elig, agree_full, top_frac=0.005,
                                         min_agree=2)
        pr, pc = np.nonzero(pseudo)
        print(f"pseudo-positives: {pr.size}", flush=True)
        if pr.size > 100:
            Xp = feat[pr, pc].astype(np.float32)
            for g in ext_grids:
                Xp = np.concatenate([Xp, g[pr, pc].astype(np.float32)], axis=1)
            Xp = np.concatenate([Xp, prox[pr, pc].reshape(-1, 1)], axis=1)
            X2 = np.concatenate([X, Xp])
            y2 = np.concatenate([y, np.ones(pr.size, dtype=np.int8)])
            w2 = np.concatenate([np.ones(y.size), np.full(pr.size, 0.5)])
            clf = modeling.fit_hgb(X2, y2, sample_weight=w2,
                                   iterations=args.iterations, lr=args.lr,
                                   depth=args.depth, l2=args.l2,
                                   seed=args.seed + 1)
            prob = modeling.predict_grid(clf, np.asarray(feat), footprint,
                                         extra=ext_full)
    if args.with_discovery:
        sys_id, n, _ = systems.label_systems(labels)
        strikes = discovery.system_strikes(sys_id, np.arange(1, n + 1))
        rays = discovery.strike_rays(footprint.shape, strikes, ray_len_px=10.0)
        corr = discovery.relay_corridors(footprint.shape, strikes)
        wr, wc = (float(v) for v in args.discovery_w.split(","))
        prob = discovery.noisy_or_fuse(np.nan_to_num(prob, nan=0.0),
                                       (rays, wr), (corr, wc))
        prob = np.where(footprint, prob, np.nan)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    np.save(work_dir / "prob_final.npy", prob.astype(np.float32))
    (work_dir / "prob_final.meta.json").write_text(json.dumps(
        {"config": vars(args), "channels": channels,
         "prob_min": float(np.nanmin(prob)), "prob_max": float(np.nanmax(prob)),
         "prob_mean": float(np.nanmean(prob)),
         "seconds": round(time.time() - t0, 1)}, indent=1, default=str) + "\n")
    print(f"done in {time.time()-t0:.0f}s -> {work_dir}/prob_final.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
