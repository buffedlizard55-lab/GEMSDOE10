#!/usr/bin/env python3
"""Paired spatially-blocked candidate evaluation with preregistered policy sweep.

Session-2 protocol (frozen in HYPOTHESES.md, "Preregistered H16 + H13
validation"): same 4x4 strided blocks, 40 px buffer, HGB frozen settings,
folds 0-2 development / fold 3 confirmation, PLUS a policy axis evaluated on
the saved OOF probabilities (no retraining for the sweep).

    python scripts/validate_candidate.py --hypothesis H13 --extra data/features_offset.npy
    python scripts/validate_candidate.py --hypothesis H16   # rebuilds per fold from train systems

Never uploads. Writes reports/<h>_blocked.json and OOF grids to scratch/.
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
from gems10 import (alignment, cv, discovery, features, metric, modeling,
                    systems)
from gems10.raster import sha256_file

# Preregistered policy set (HYPOTHESES.md session-2 item 4).
POLICIES = ["topk01_binary", "topk02_binary", "topk03_binary", "topk04_binary",
            "topk06_binary", "topk02_soft", "topk02_envelope", "topk02_halo2"]
PRIMARY = "topk02_binary"
CONFIG = {"iterations": 200, "lr": 0.05, "depth": 7, "l2": 1,
          "seed": 7, "negatives": 200000, "buffer_px": 40}


def mask_hash(mask) -> str:
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


def score_policies(prob: np.ndarray, score: np.ndarray, gt: np.ndarray) -> dict:
    out = {}
    for pol in POLICIES:
        e = np.nan_to_num(modeling.apply_policy(prob, score, pol), nan=0)
        out[pol] = metric.components(e, gt).as_dict()
    return out


def select_policy(folds: list, arm: str) -> str:
    """Argmax mean development DTI; deterministic tie-break:
    PRIMARY first, then list order (preregistered)."""
    means = {}
    for p in POLICIES:
        vals = [r["scores"][arm][p]["dti"] for r in folds
                if r["role"] == "development"]
        means[p] = float(np.mean(vals))
    top = max(means.values())
    tied = [p for p in POLICIES if abs(means[p] - top) < 1e-12]
    if PRIMARY in tied:
        return PRIMARY
    return min(tied, key=lambda p: POLICIES.index(p))


def h16_extra_for_fold(train: np.ndarray, labels: np.ndarray,
                       bands_cache: dict, lineaments: dict):
    m = train & (labels == 1)
    sys_id, n, _ = systems.label_systems(m.astype(np.int8))
    strikes = discovery.system_strikes(sys_id, np.arange(1, n + 1))
    grid, names = alignment.ray_continuation_channels(
        bands_cache["tmi"], lineaments, strikes, ncc_band=bands_cache["rtp"])
    return grid, names, len(strikes)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", required=True, choices=["H13", "H16"])
    ap.add_argument("--extra", type=Path, default=None,
                    help="static extra grid (H13). Ignored for H16 (per-fold).")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--work", type=Path, default=None)
    args = ap.parse_args()
    h = args.hypothesis
    report_path = args.report or ROOT / f"reports/{h.lower()}_blocked.json"
    work = args.work or ROOT / f"scratch/{h.lower()}"
    work.mkdir(parents=True, exist_ok=True)

    started = time.time()
    feat_path = ROOT / "data/features107.f32.npy"
    if not feat_path.exists():
        raise SystemExit("run scripts/build_features.py first")
    extra_static = None
    if h == "H13":
        if args.extra is None or not args.extra.exists():
            raise SystemExit("H13 requires --extra data/features_offset.npy (build_offset.py)")
        extra_static = np.load(args.extra, mmap_mode="r")
    feat = np.load(feat_path, mmap_mode="r")
    with rasterio.open(ROOT / "data/labels.tif") as ds:
        labels = ds.read(1)
    with rasterio.open(ROOT / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))

    # Per-fold H16 inputs: bands + lineaments are label-free, compute once.
    # Keep memory bounded: float32 lineaments, bands freed after lineaments.
    bands_cache, lineaments = None, None
    if h == "H16":
        with rasterio.open(ROOT / "data/training_features.tif") as ds:
            raw = {b: ds.read(spec_band(b)).astype(np.float64)
                   for b in ("tmi", "det_elev", "rtp")}
        for b in raw:
            raw[b][raw[b] < -1e38] = np.nan
        tmi_invalid = ~np.isfinite(raw["tmi"])  # mask source kept for H16 builder
        lineaments = {}
        for b in ("tmi", "det_elev"):
            lin = features.structure_tensor(raw[b], sigma=1.5, integration_sigma=4.0)
            lineaments[b] = type(lin)(energy=lin.energy.astype(np.float32),
                                      coherence=lin.coherence.astype(np.float32),
                                      orientation=lin.orientation.astype(np.float32))
            del raw[b]
            gc.collect()
        # 'tmi' entry needs only the invalid pattern (0/1 fill is harmless: the
        # builder derives bad=~isfinite and stamps scores at finite ray points).
        bands_cache = {"tmi": np.where(tmi_invalid, np.nan, 0.0).astype(np.float32),
                       "rtp": raw["rtp"].astype(np.float32)}
        del tmi_invalid, raw
        gc.collect()

    folds = cv.make_folds(n_blocks=4, n_folds=4, buffer_px=40)
    code_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in
                   [Path(__file__), ROOT / "src/gems10/alignment.py",
                    ROOT / "src/gems10/cv.py", ROOT / "src/gems10/modeling.py",
                    ROOT / "src/gems10/metric.py", ROOT / "src/gems10/discovery.py",
                    ROOT / "src/gems10/systems.py", ROOT / "src/gems10/features.py",
                    ROOT / "scripts/build_features.py", ROOT / "HYPOTHESES.md"]}
    inputs = {p.name: sha256_file(p) for p in
              [feat_path, ROOT / "data/labels.tif", ROOT / "data/sample_submission.tif",
               ROOT / "data/training_features.tif"]}
    if extra_static is not None:
        inputs[Path(args.extra).name] = sha256_file(args.extra)
    report = {
        "hypothesis": h, "status": "running", "promotion_allowed": False,
        "protocol": "spatial-4x4-strided-v1-buffer40-policy-sweep-v2",
        "score_population": "held-out catalogue pixels in spatial score blocks (proxy only)",
        "confirmation_fold": 3, "primary_policy": PRIMARY, "policies": POLICIES,
        "config": CONFIG,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "inputs": inputs, "code": code_hashes, "folds": [],
        "limitations": ["Not private new-fault truth; does not forecast leaderboard DTI",
                        "Four geographic stripe folds, not independent statistical replicates",
                        "Confirmation geography re-used from H12 (inspected; not pristine)",
                        "Baseline107 Frangi normalization inherits tile-local scale",
                        "H16 features rebuild from train systems per fold; H13 is label-free"]}
    if h == "H13":
        report["new_channels"] = json.loads(
            args.extra.with_suffix(".meta.json").read_text())["channels"]
    else:
        report["new_channels"] = [f"cont_{b}_r{t}" for b in ("tmi", "det_elev")
                                  for t in alignment.RAY_RADII]
        report["new_channels"] += [f"cont_ncc_r{t}" for t in alignment.NCC_RADII]

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
        pos = np.flatnonzero(train & (labels == 1))
        neg = np.flatnonzero(train & (labels == 0))
        rng = np.random.default_rng(1000 + k)
        neg = rng.choice(neg, size=min(CONFIG["negatives"], neg.size), replace=False)
        sample = np.concatenate([pos, neg])
        rows, cols = np.unravel_index(sample, fp.shape)
        y = np.r_[np.ones(pos.size, dtype=np.int8), np.zeros(neg.size, dtype=np.int8)]
        if not gt.any() or pos.size == 0 or neg.size == 0:
            raise ValueError(f"fold {k}: empty class or score population")
        result = {"fold": k, "role": "confirmation" if k == 3 else "development",
                  "train_mask_sha256": mask_hash(train),
                  "score_mask_sha256": mask_hash(score),
                  "train_pixels": int(train.sum()), "score_pixels": int(score.sum()),
                  "training_positives": int(pos.size),
                  "training_negatives": int(neg.size), "gt_pixels": int(gt.sum()),
                  "scores": {}}
        extra_fold = None
        if h == "H16":
            extra_fold, extra_names, n_strikes = h16_extra_for_fold(
                train, labels, bands_cache, lineaments)
            fin = np.isfinite(extra_fold)
            result["h16_train_systems"] = n_strikes
            result["h16_coverage_train_px"] = {
                nm: int(fin[..., i][train].sum())
                for i, nm in enumerate(extra_names)}
        for name in ("baseline107", h):
            extra = extra_fold if (h == "H16" and name == h) else (
                extra_static if (h == "H13" and name == h) else None)
            X = feat[rows, cols].astype(np.float32)
            if extra is not None:
                X = np.concatenate([X, extra[rows, cols]], axis=1)
            print(f"{h} fold {k} {name} fitting {X.shape}", flush=True)
            model = modeling.fit_hgb(X, y, **{kk: CONFIG[kk] for kk in
                                              ("iterations", "lr", "depth", "l2", "seed")})
            p = np.full(fp.shape, np.nan, dtype=np.float32)
            sy, sx = np.nonzero(score)
            for i in range(0, len(sy), 50000):
                rr, cc = sy[i:i + 50000], sx[i:i + 50000]
                batch = feat[rr, cc].astype(np.float32)
                if extra is not None:
                    batch = np.concatenate([batch, extra[rr, cc]], axis=1)
                p[rr, cc] = model.predict_proba(batch)[:, 1]
            if not np.isfinite(p[score]).all() or np.isfinite(p[~score]).any():
                raise ValueError("OOF footprint differs from score geography")
            result["scores"][name] = score_policies(p, score, gt)
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
                systems_grid, _ = ndimage.label(train & (labels == 1),
                                                np.ones((3, 3)))
                ids = np.unique(systems_grid); ids = ids[ids > 0]
                strikes = discovery.system_strikes(systems_grid, ids)
                rays = discovery.strike_rays(fp.shape, strikes, ray_len_px=10)
                corridors = discovery.relay_corridors(fp.shape, strikes)
                fused = discovery.noisy_or_fuse(np.nan_to_num(p, nan=0),
                                                (rays, .5), (corridors, .4))
                result["scores"]["baseline107_discovery"] = score_policies(
                    np.where(score, fused, np.nan).astype(np.float32), score, gt)
                del systems_grid, rays, corridors, fused
            print(f"{h} fold {k} {name} {PRIMARY}: "
                  f"{result['scores'][name][PRIMARY]['dti']:.6f}", flush=True)
            del X, batch, model, p
            gc.collect()
        random_p = np.zeros(fp.shape, dtype=np.float32)
        random_p[score] = np.random.default_rng(7000 + k).random(int(score.sum()))
        e = np.nan_to_num(modeling.apply_policy(random_p, score, PRIMARY), nan=0)
        result["scores"]["random_budget_control"] = {PRIMARY:
                                                     metric.components(e, gt).as_dict()}
        if extra_fold is not None:
            result["extra_fold_sha256"] = hashlib.sha256(
                np.ascontiguousarray(extra_fold).tobytes()).hexdigest()
            del extra_fold
        result["seconds"] = round(time.time() - t0, 2)
        report["folds"].append(result)
        save()
        del random_p, e, train, score, gt
        gc.collect()

    # Preregistered policy selection on development folds only.
    selection = {arm: select_policy(report["folds"], arm)
                 for arm in ("baseline107", "baseline107_discovery", h)}
    comparisons = {}
    cand = selection[h]
    eligible = True
    for base in ("baseline107", "baseline107_discovery"):
        bpol = selection[base]
        ddev, dconf = [], []
        for r in report["folds"]:
            c = r["scores"][h][cand]["dti"]
            b = r["scores"][base][bpol]["dti"]
            delta = c - b
            (dconf if r["role"] == "confirmation" else ddev).append(delta)
        passes = bool(np.mean(ddev) > 0 and dconf[0] > 0)
        comparisons[base] = {"baseline_policy": bpol,
                             "delta_by_fold": [c - b for c, b in
                                               [(r["scores"][h][cand]["dti"],
                                                 r["scores"][base][bpol]["dti"])
                                                for r in report["folds"]]],
                             "development_mean_delta": float(np.mean(ddev)),
                             "confirmation_delta": float(dconf[0]),
                             "passes": passes}
        eligible = eligible and passes
    report["policy_selection"] = selection
    report["decision"] = {"eligible": eligible, "candidate_policy": cand,
                          "comparisons": comparisons,
                          "reason": ("matched baseline and discovery beaten under "
                                     "selected policies; final-training binding still "
                                     "required" if eligible
                                     else "no slot: development AND confirmation must "
                                          "both improve")}
    report["status"] = "completed"
    report["promotion_allowed"] = False  # never by local comparison alone
    save()
    print(json.dumps(report["decision"], indent=2), flush=True)


def spec_band(name: str) -> int:
    from gems10 import spec
    return spec.BAND_INDEX[name]


if __name__ == "__main__":
    main()
