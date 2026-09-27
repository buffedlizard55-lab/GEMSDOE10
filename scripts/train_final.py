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
    ap.add_argument("--extra", default=None,
                    help="static (H,W,C) feature npy to append, e.g. "
                         "data/features_offset.npy (H13)")
    ap.add_argument("--hypothesis", default="baseline107",
                    help="candidate label for the training manifest")
    ap.add_argument("--bind-to", default=None,
                    help="reports/<h>_blocked.json: bind the final prediction "
                         "and training manifest into this report (release gate)")
    ap.add_argument("--with-discovery", action="store_true")
    ap.add_argument("--discovery-w", default="0.5,0.4")
    ap.add_argument("--with-selftrain", action="store_true")
    ap.add_argument("--with-proximity", action="store_true",
                    help="ablation only: dist-to-catalogue channel (memorises)")
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
        names = args.externals.split(",")
        ext_grids = external.open_memmaps(str(data_dir), names)
        for name, g in zip(names, ext_grids):
            channels += [f"{name}_u8_{i}" for i in range(g.shape[2])]
            print(f"external {name}: {g.shape} (disk memmap)")
    if args.extra:
        extra_path = Path(args.extra)
        ext_grids.append(np.lib.format.open_memmap(str(extra_path), mode="r"))
        emeta = json.loads(extra_path.with_suffix(".meta.json").read_text()) \
            if extra_path.with_suffix(".meta.json").exists() else None
        if emeta and emeta.get("channels"):
            channels += list(emeta["channels"])
        else:
            channels += [f"extra_{i}" for i in range(ext_grids[-1].shape[2])]
        print(f"extra features {extra_path.name}: {ext_grids[-1].shape} (disk memmap)")
    # Proximity OFF by default (harness: it memorises, DTI 0.012). Only for
    # ablation via --with-proximity.
    prox = train_proximity_channel(pos) if args.with_proximity else None
    if prox is not None:
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
    if prox is not None:
        X = np.concatenate([X, prox[rows, cols].reshape(-1, 1)], axis=1)
    print(f"train X={X.shape}", flush=True)
    clf = modeling.fit_hgb(X, y, iterations=args.iterations, lr=args.lr,
                           depth=args.depth, l2=args.l2, seed=args.seed)

    extra = prox[:, :, None] if prox is not None else None
    prob = modeling.predict_grid(clf, feat, footprint, extra=extra,
                                 extra_grids=ext_grids or None)

    if args.with_selftrain:
        elig = footprint & ~pos
        agree_full = selftrain.agreement_grid(feat, footprint,
                                              meta["channels"])
        pseudo = selftrain.select_pseudo(prob, elig, agree_full, top_frac=0.005,
                                         min_agree=2)
        pr, pc = np.nonzero(pseudo)
        print(f"pseudo-positives: {pr.size}", flush=True)
        if pr.size > 100:
            Xp = feat[pr, pc].astype(np.float32)
            for g in ext_grids:
                Xp = np.concatenate([Xp, g[pr, pc].astype(np.float32)], axis=1)
            if prox is not None:
                Xp = np.concatenate([Xp, prox[pr, pc].reshape(-1, 1)], axis=1)
            X2 = np.concatenate([X, Xp])
            y2 = np.concatenate([y, np.ones(pr.size, dtype=np.int8)])
            w2 = np.concatenate([np.ones(y.size), np.full(pr.size, 0.5)])
            clf = modeling.fit_hgb(X2, y2, sample_weight=w2,
                                   iterations=args.iterations, lr=args.lr,
                                   depth=args.depth, l2=args.l2,
                                   seed=args.seed + 1)
            prob = modeling.predict_grid(clf, feat, footprint,
                                         extra=extra,
                                         extra_grids=ext_grids or None)
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
        {"config": vars(args), "hypothesis": args.hypothesis,
         "channels": channels,
         "prob_min": float(np.nanmin(prob)), "prob_max": float(np.nanmax(prob)),
         "prob_mean": float(np.nanmean(prob)),
         "seconds": round(time.time() - t0, 1)}, indent=1, default=str) + "\n")
    if args.bind_to:
        import hashlib
        import pickle
        from gems10.raster import sha256_file
        p = Path(args.bind_to)
        report_path = p if p.is_absolute() else REPO_ROOT / p
        if not report_path.exists() and (REPO_ROOT / "reports" / p.name).exists():
            report_path = REPO_ROOT / "reports" / p.name
        report = json.loads(report_path.read_text())
        if not report.get("decision", {}).get("eligible"):
            raise SystemExit("REFUSED: validation decision is not eligible; "
                             "no final-training binding is written")
        model_path = work_dir / "model_final.joblib"
        with open(model_path, "wb") as fh:
            pickle.dump(clf, fh)
        prob_sha = sha256_file(work_dir / "prob_final.npy")
        manifest_name = f"final_manifest_{args.hypothesis.lower()}.json"
        manifest = {
            "hypothesis": args.hypothesis,
            "probability_sha256": prob_sha,
            "validation_protocol": report.get("protocol"),
            "validated_inputs": report.get("inputs"),
            "config": report.get("config"),
            "model_sha256": sha256_file(model_path),
            "final_prob_file": str((work_dir / "prob_final.npy").name),
            "code": report.get("code"),
        }
        manifest_path = REPO_ROOT / "reports" / manifest_name
        manifest_path.write_text(json.dumps(manifest, indent=2,
                                            allow_nan=False) + "\n")
        report["final_prediction"] = {
            "sha256": prob_sha,
            "training_manifest_file": manifest_name,
            "training_manifest_sha256": sha256_file(manifest_path),
        }
        report["promotion_allowed"] = True
        report_path.write_text(json.dumps(report, indent=2,
                                          allow_nan=False) + "\n")
        print(f"bound final prediction into {report_path.name} "
              f"(manifest {manifest_name})")
    print(f"done in {time.time()-t0:.0f}s -> {work_dir}/prob_final.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
