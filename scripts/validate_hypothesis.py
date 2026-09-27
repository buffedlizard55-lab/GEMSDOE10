#!/usr/bin/env python3
"""Paired, spatially blocked H12 evaluation; never uploads a submission.

Run AFTER build_features.py and build_scarp.py. All settings preregistered in
HYPOTHESES.md. Outputs JSON evidence + ignored OOF grids, not a shipping file.
"""
from __future__ import annotations
import gc
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import scipy
import sklearn
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems10 import cv, discovery, metric, modeling, scarp
from gems10.raster import sha256_file


def mask_hash(mask):
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


def paired_decision(folds):
    """Fail closed on absent, duplicate, nonfinite, or non-improving folds."""
    if sorted(r["fold"] for r in folds) != [0, 1, 2, 3]:
        return {"eligible": False, "reason": "incomplete or duplicate spatial folds"}
    rows = sorted(folds, key=lambda x: x["fold"])
    candidate = np.array([r["scores"]["H12"]["dti"] for r in rows])
    competitors = ["baseline107", "baseline107_discovery"]
    if not np.isfinite(candidate).all():
        return {"eligible": False, "reason": "nonfinite candidate score"}
    comparisons = {}
    for name in competitors:
        baseline = np.array([r["scores"][name]["dti"] for r in rows])
        if not np.isfinite(baseline).all():
            return {"eligible": False, "reason": "nonfinite baseline score"}
        delta = candidate - baseline
        comparisons[name] = {"delta_by_fold": delta.tolist(),
                             "development_mean_delta": float(delta[:3].mean()),
                             "confirmation_delta": float(delta[3]),
                             "passes": bool(delta[:3].mean() > 0 and delta[3] > 0)}
    eligible = all(c["passes"] for c in comparisons.values())
    return {"eligible": eligible, "comparisons": comparisons,
            "reason": ("matched baseline and discovery beaten; further release checks required"
                       if eligible else "no slot: development and confirmation must both improve")}


def main():
    started = time.time()
    report_path = ROOT / "reports/h12_blocked.json"
    work = ROOT / "scratch/h12"
    work.mkdir(parents=True, exist_ok=True)
    feat_path = ROOT / "data/features107.f32.npy"
    extra_path = ROOT / "data/features_scarp.npy"
    feat = np.load(feat_path, mmap_mode="r")
    extra = np.load(extra_path, mmap_mode="r")
    with rasterio.open(ROOT / "data/labels.tif") as ds:
        labels = ds.read(1)
    with rasterio.open(ROOT / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    folds = cv.make_folds(n_blocks=4, n_folds=4, buffer_px=40)
    report = {
        "hypothesis": "H12", "status": "running", "promotion_allowed": False,
        "protocol": "spatial-4x4-strided-v1-buffer40-top2",
        "score_population": "held-out catalogue pixels in spatial score blocks (proxy only)",
        "confirmation_fold": 3, "policy": "topk02_binary",
        "config": {"iterations": 200, "lr": .05, "depth": 7, "l2": 1,
                   "seed": 7, "negatives": 200000, "buffer_px": 40},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "inputs": {p.name: sha256_file(p) for p in
                   [feat_path, extra_path, ROOT / "data/labels.tif",
                    ROOT / "data/sample_submission.tif"]},
        "code": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                 [Path(__file__), ROOT / "src/gems10/scarp.py", ROOT / "src/gems10/cv.py",
                  ROOT / "src/gems10/modeling.py", ROOT / "src/gems10/metric.py",
                  ROOT / "scripts/build_features.py", ROOT / "src/gems10/features.py",
                  ROOT / "src/gems10/discovery.py", ROOT / "HYPOTHESES.md"]},
        "new_channels": scarp.CHANNELS, "folds": [],
        "limitations": ["Not private new-fault truth; does not forecast leaderboard DTI",
                        "Four geographic stripe folds, not independent statistical replicates",
                        "Baseline107 Frangi normalization inherits tile-local scale",
                        "External prior experiments not reproducible from checked-in reports alone"]}
    def save():
        report["elapsed_seconds"] = round(time.time() - started, 2)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    for k in range(4):
        t0 = time.time()
        train, score = folds.train_test_masks(fp.shape, k)
        train &= fp
        score &= fp
        gt = (labels == 1) & score
        # Sampling cannot look at held-out labels (only geography chooses masks).
        pos = np.flatnonzero(train & (labels == 1))
        neg = np.flatnonzero(train & (labels == 0))
        rng = np.random.default_rng(1000 + k)
        neg = rng.choice(neg, size=min(200000, neg.size), replace=False)
        sample = np.concatenate([pos, neg])
        rows, cols = np.unravel_index(sample, fp.shape)
        y = np.r_[np.ones(pos.size, dtype=np.int8), np.zeros(neg.size, dtype=np.int8)]
        if not gt.any() or pos.size == 0 or neg.size == 0:
            raise ValueError(f"fold {k}: empty class or score population")
        result = {"fold": k, "role": "confirmation" if k == 3 else "development",
                  "train_mask_sha256": mask_hash(train), "score_mask_sha256": mask_hash(score),
                  "train_pixels": int(train.sum()), "score_pixels": int(score.sum()),
                  "training_positives": int(pos.size), "training_negatives": int(neg.size),
                  "gt_pixels": int(gt.sum()), "scores": {}}
        for name, grids in [("baseline107", None), ("H12", [extra])]:
            X = feat[rows, cols].astype(np.float32)
            if grids:
                X = np.concatenate([X, extra[rows, cols]], axis=1)
            print(f"fold {k} {name} fitting {X.shape}", flush=True)
            model = modeling.fit_hgb(X, y, iterations=200, lr=.05, depth=7, l2=1, seed=7)
            # Predict just held-out cells in small batches; avoid full-grid model inference.
            p = np.full(fp.shape, np.nan, dtype=np.float32)
            sy, sx = np.nonzero(score)
            for i in range(0, len(sy), 50000):
                rr, cc = sy[i:i+50000], sx[i:i+50000]
                batch = feat[rr, cc].astype(np.float32)
                if grids:
                    batch = np.concatenate([batch, extra[rr, cc]], axis=1)
                p[rr, cc] = model.predict_proba(batch)[:, 1]
            emission = np.nan_to_num(modeling.apply_policy(p, score, "topk02_binary"), nan=0)
            result["scores"][name] = metric.components(emission, gt).as_dict()
            if not np.isfinite(p[score]).all() or np.isfinite(p[~score]).any():
                raise ValueError("OOF footprint differs from score geography")
            # Buffered serialization plus readback, rather than relying on tofile.
            output_path = work / f"{name}_fold{k}.npy"
            buffer = io.BytesIO()
            np.save(buffer, p, allow_pickle=False)
            output_path.write_bytes(buffer.getvalue())
            del buffer
            persisted = np.load(output_path, allow_pickle=False)
            if not np.array_equal(p, persisted, equal_nan=True):
                raise ValueError("OOF serialization roundtrip failed")
            del persisted
            result.setdefault("prediction_files", {})[name] = {
                "file": str(output_path.relative_to(ROOT)),
                "sha256": sha256_file(output_path)}
            if name == "baseline107":
                # Recompute label geometry ONLY from buffered training labels, never test labels.
                systems, _ = ndimage.label(train & (labels == 1), np.ones((3, 3)))
                ids = np.unique(systems); ids = ids[ids > 0]
                strikes = discovery.system_strikes(systems, ids)
                rays = discovery.strike_rays(fp.shape, strikes, ray_len_px=10)
                corridors = discovery.relay_corridors(fp.shape, strikes)
                fused = discovery.noisy_or_fuse(np.nan_to_num(p, nan=0), (rays, .5), (corridors, .4))
                e = np.nan_to_num(modeling.apply_policy(fused, score, "topk02_binary"), nan=0)
                result["scores"]["baseline107_discovery"] = metric.components(e, gt).as_dict()
                del systems, rays, corridors, fused, e
            print(f"fold {k} {name}: {result['scores'][name]['dti']:.6f}", flush=True)
            del X, batch, model, p, emission
            gc.collect()
        random_p = np.zeros(fp.shape, dtype=np.float32)
        random_p[score] = np.random.default_rng(7000+k).random(int(score.sum()))
        e = np.nan_to_num(modeling.apply_policy(random_p, score, "topk02_binary"), nan=0)
        result["scores"]["random_budget_control"] = metric.components(e, gt).as_dict()
        result["seconds"] = round(time.time()-t0, 2)
        report["folds"].append(result)
        save()
        del random_p, e, train, score, gt
        gc.collect()
    report["decision"] = paired_decision(report["folds"])
    report["status"] = "completed"
    # A passed local comparison alone does not attest final-model provenance / novelty.
    report["promotion_allowed"] = False
    save()
    print(json.dumps(report["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
