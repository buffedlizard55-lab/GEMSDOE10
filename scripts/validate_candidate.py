#!/usr/bin/env python3
"""Paired spatially-blocked candidate evaluation with preregistered policy sweep.

Session-2 protocol (frozen in HYPOTHESES.md, "Preregistered H16 + H13
validation"): same 4x4 strided blocks, 40 px buffer, HGB frozen settings,
folds 0-2 development / fold 3 confirmation, PLUS a policy axis evaluated on
the saved OOF probabilities (no retraining for the sweep).

Session-3 extension (HYPOTHESES.md, "Session-3 register"): the policy union
adds the thin-line family (`thinNN_binary`, H19); H20 adds a static external
grid on top of the per-fold H16 channels; every candidate is also compared
with the incumbent H16 arm (same run) and, optionally, with the H16 numbers
recorded in an earlier report (`--incumbent-report`), which doubles as an
exact-reproduction check when the hypothesis is H16 itself.

    python scripts/validate_candidate.py --hypothesis H13 --extra data/features_offset.npy
    python scripts/validate_candidate.py --hypothesis H16   # rebuilds per fold from train systems
    python scripts/validate_candidate.py --hypothesis H16 --policy-set v3 \
        --incumbent-report reports/h16_blocked.json --report reports/h19_blocked.json
    python scripts/validate_candidate.py --hypothesis H20 --policy-set v3 \
        --extra data/external/dem10/dem10_channels.f32.npy \
        --incumbent-report reports/h16_blocked.json

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

# Preregistered policy sets. v2: HYPOTHESES.md session-2 item 4.
# v3: session-3 item 2 — the v2 eight plus the thin-line family (H19).
# v4: session-4 register — v3 plus the probability-ridge family (H24).
POLICY_SETS = {
    "v2": ["topk01_binary", "topk02_binary", "topk03_binary", "topk04_binary",
           "topk06_binary", "topk02_soft", "topk02_envelope", "topk02_halo2"],
}
POLICY_SETS["v3"] = POLICY_SETS["v2"] + [f"thin{b:02d}_binary"
                                         for b in (2, 4, 6, 8, 10, 12, 15, 20)]
POLICY_SETS["v4"] = POLICY_SETS["v3"] + [f"ridge{b:02d}_binary"
                                         for b in (4, 6, 8, 10, 12, 15)]
PROTOCOL = {"v2": "spatial-4x4-strided-v1-buffer40-policy-sweep-v2",
            "v3": "spatial-4x4-strided-v1-buffer40-policy-sweep-v3",
            "v4": "spatial-4x4-strided-v1-buffer40-policy-sweep-v4"}
PRIMARY = "topk02_binary"
CONFIG = {"iterations": 200, "lr": 0.05, "depth": 7, "l2": 1,
          "seed": 7, "negatives": 200000, "buffer_px": 40}
STATIC_EXTRA = {"H13", "H20", "H25", "H27"}  # hypotheses that take --extra
NEEDS_H16 = {"H16", "H20", "H25", "H27"}  # hypotheses whose stack includes H16 channels
# Extra-grid slice per arm: arm name -> how many leading --extra grids it takes
# (None = all). The H20 arm inside an H25 run takes ONLY the dem10 grid so its
# column layout matches the released incumbent exactly.
ARM_EXTRAS = {"H13": None, "H20": 1, "H25": None, "H27": None}
DEV_FOLDS, CONF_FOLD = (0, 1, 2), 3


def mask_hash(mask) -> str:
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


def score_policies(prob: np.ndarray, score: np.ndarray, gt: np.ndarray,
                   policies: list[str]) -> dict:
    out = {}
    for pol in policies:
        e = np.nan_to_num(modeling.apply_policy(prob, score, pol), nan=0)
        out[pol] = metric.components(e, gt).as_dict()
    return out


def select_policy(folds: list, arm: str, policies: list[str]) -> str:
    """Argmax mean development DTI; deterministic tie-break:
    PRIMARY first, then list order (preregistered)."""
    means = {}
    for p in policies:
        vals = [r["scores"][arm][p]["dti"] for r in folds
                if r["role"] == "development"]
        means[p] = float(np.mean(vals))
    top = max(means.values())
    tied = [p for p in policies if abs(means[p] - top) < 1e-12]
    if PRIMARY in tied:
        return PRIMARY
    return min(tied, key=lambda p: policies.index(p))


def h16_extra_for_fold(train: np.ndarray, labels: np.ndarray,
                       bands_cache: dict, lineaments: dict,
                       gate_phase: str = "asbuilt"):
    m = train & (labels == 1)
    sys_id, n, _ = systems.label_systems(m.astype(np.int8))
    strikes = discovery.system_strikes(sys_id, np.arange(1, n + 1))
    grid, names = alignment.ray_continuation_channels(
        bands_cache["tmi"], lineaments, strikes, ncc_band=bands_cache["rtp"],
        gate_phase=gate_phase)
    return grid, names, len(strikes)


def paired_comparison(folds: list, cand_arm: str, cand_pol: str,
                      base_scores) -> dict:
    """base_scores: callable(fold_row) -> baseline DTI for that fold."""
    deltas = []
    for r in sorted(folds, key=lambda x: x["fold"]):
        deltas.append(r["scores"][cand_arm][cand_pol]["dti"] - base_scores(r))
    dev = [d for r, d in zip(sorted(folds, key=lambda x: x["fold"]), deltas)
           if r["role"] == "development"]
    conf = [d for r, d in zip(sorted(folds, key=lambda x: x["fold"]), deltas)
            if r["role"] == "confirmation"]
    return {"delta_by_fold": deltas, "development_mean_delta": float(np.mean(dev)),
            "confirmation_delta": float(conf[0]),
            "passes": bool(np.mean(dev) > 0 and conf[0] > 0)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", required=True,
                    choices=["H13", "H16", "H20", "H25", "H27"])
    ap.add_argument("--extra", type=str, default=None,
                    help="static extra grid(s) .npy, comma-separated in column "
                         "order (H13 offset grid; H20 dem10 grid; H25 "
                         "dem10,dem10_context). Ignored for H16.")
    ap.add_argument("--policy-set", choices=sorted(POLICY_SETS), default="v3")
    ap.add_argument("--incumbent-report", type=Path, default=None,
                    help="earlier completed report whose named arm (under its "
                         "selected policy) the candidate must also beat; for "
                         "--hypothesis H16 it doubles as an exact-reproduction "
                         "check of the shared policies (any run: reproduction "
                         "over arms present in both reports)")
    ap.add_argument("--incumbent-arm", default="H16",
                    help="arm of --incumbent-report to compare against "
                         "(session 4: 'H20' when the incumbent is the H20 report)")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--work", type=Path, default=None)
    args = ap.parse_args()
    h = args.hypothesis
    policies = POLICY_SETS[args.policy_set]
    report_path = (args.report or ROOT / f"reports/{h.lower()}_blocked.json").resolve()
    work = (args.work or ROOT / f"scratch/{h.lower()}").resolve()
    if not work.is_relative_to(ROOT):
        raise SystemExit("--work must live inside the repository")
    work.mkdir(parents=True, exist_ok=True)

    started = time.time()
    feat_path = ROOT / "data/features107.f32.npy"
    if not feat_path.exists():
        raise SystemExit("run scripts/build_features.py first")
    extra_paths: list[Path] = []
    extra_static: list[np.ndarray] = []
    if h in STATIC_EXTRA:
        if not args.extra:
            raise SystemExit(f"{h} requires --extra <static grid .npy>[,...]")
        for item in args.extra.split(","):
            ep = Path(item.strip())
            if not ep.exists():
                raise SystemExit(f"--extra grid missing: {ep}")
            extra_paths.append(ep)
            grid = np.load(ep, mmap_mode="r")
            if grid.ndim != 3 or grid.shape[:2] != (3730, 3292):
                raise SystemExit(f"--extra must be (3730, 3292, C); got {grid.shape}")
            extra_static.append(grid)
    feat = np.load(feat_path, mmap_mode="r")
    with rasterio.open(ROOT / "data/labels.tif") as ds:
        labels = ds.read(1)
    with rasterio.open(ROOT / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))

    incumbent = None
    if args.incumbent_report is not None:
        incumbent = json.loads(args.incumbent_report.read_text())
        if not args.incumbent_report.resolve().is_relative_to(ROOT):
            raise SystemExit("--incumbent-report must live inside the repository")
        if incumbent.get("status") != "completed" or \
                args.incumbent_arm not in incumbent.get("policy_selection", {}):
            raise SystemExit("--incumbent-report must be a completed report with "
                             f"the arm {args.incumbent_arm!r}")

    # Per-fold H16 inputs: bands + lineaments are label-free, compute once.
    # Keep memory bounded: float32 lineaments, bands freed after lineaments.
    bands_cache, lineaments = None, None
    if h in NEEDS_H16:
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
                    ROOT / "src/gems10/placement.py",
                    ROOT / "src/gems10/metric.py", ROOT / "src/gems10/discovery.py",
                    ROOT / "src/gems10/systems.py", ROOT / "src/gems10/features.py",
                    ROOT / "scripts/build_features.py", ROOT / "HYPOTHESES.md"]}
    inputs = {p.name: sha256_file(p) for p in
              [feat_path, ROOT / "data/labels.tif", ROOT / "data/sample_submission.tif",
               ROOT / "data/training_features.tif"]}
    for ep in extra_paths:
        inputs[ep.name] = sha256_file(ep)
    arms = ["baseline107"]
    if h in ("H20", "H25", "H27"):
        arms.append("H16")
    if h in ("H25", "H27"):
        arms.append("H20")          # same-run reproduction anchor vs the incumbent
    arms.append(h)
    report = {
        "hypothesis": h, "status": "running", "promotion_allowed": False,
        "protocol": PROTOCOL[args.policy_set],
        "policy_set": args.policy_set,
        "arms": arms,
        "score_population": "held-out catalogue pixels in spatial score blocks (proxy only)",
        "confirmation_fold": CONF_FOLD, "primary_policy": PRIMARY, "policies": policies,
        "config": CONFIG,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "inputs": inputs, "code": code_hashes, "folds": [],
        "incumbent_report": (str(args.incumbent_report.resolve().relative_to(ROOT))
                             if args.incumbent_report else None),
        "limitations": ["Not private new-fault truth; does not forecast leaderboard DTI",
                        "Four geographic stripe folds, not independent statistical replicates",
                        "Confirmation geography re-used from H12 (inspected; not pristine)",
                        "Baseline107 Frangi normalization inherits tile-local scale",
                        "H16 features rebuild from train systems per fold; H13/H20 extras are label-free"]}
    if h == "H27":
        report["gate_phase"] = ("aligned (H27 corrected-phase continuation "
                                "channels; H16/H20 arms stay as-built)")
    if h in STATIC_EXTRA:
        channels, sources = [], []
        for i, ep in enumerate(extra_paths):
            meta_p = ep.with_suffix(".meta.json")
            if meta_p.exists():
                meta = json.loads(meta_p.read_text())
                channels += meta.get("channels",
                                     [f"extra_{j}" for j in range(extra_static[i].shape[2])])
                if meta.get("source"):
                    sources.append(meta["source"])
            else:
                channels += [f"extra_{j}" for j in range(extra_static[i].shape[2])]
        report["new_channels"] = channels
        if h in ("H20", "H25"):
            report["external_data"] = (sources[0] if len(sources) == 1
                                       else sources) if sources else None
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
        result = {"fold": k, "role": "confirmation" if k == CONF_FOLD else "development",
                  "train_mask_sha256": mask_hash(train),
                  "score_mask_sha256": mask_hash(score),
                  "train_pixels": int(train.sum()), "score_pixels": int(score.sum()),
                  "training_positives": int(pos.size),
                  "training_negatives": int(neg.size), "gt_pixels": int(gt.sum()),
                  "scores": {}}
        extra_fold = None
        extra_fold_aligned = None
        if h in NEEDS_H16:
            extra_fold, extra_names, n_strikes = h16_extra_for_fold(
                train, labels, bands_cache, lineaments)
            if h == "H27":
                extra_fold_aligned, _, _ = h16_extra_for_fold(
                    train, labels, bands_cache, lineaments,
                    gate_phase="aligned")
            fin = np.isfinite(extra_fold)
            result["h16_train_systems"] = n_strikes
            result["h16_coverage_train_px"] = {
                nm: int(fin[..., i][train].sum())
                for i, nm in enumerate(extra_names)}
            del fin
        for name in arms:
            # Column order per arm (must match train_final.py):
            #   baseline107: [feat]
            #   H13:         [feat | statics...]
            #   H16:         [feat | h16]
            #   H20:         [feat | h16 | static0]        (dem10 only — the
            #                released layout, even inside an H25 run)
            #   H25:         [feat | h16 | statics...]     (dem10 + context)
            grids = []
            if name in ("H16", "H20", "H25", "H27"):
                grids.append(extra_fold if name != "H27"
                             else extra_fold_aligned)
            if name == "H13":
                grids.extend(extra_static)
            elif name == "H20":
                grids.extend(extra_static[:1])
            elif name == "H25":
                grids.extend(extra_static)
            X = feat[rows, cols].astype(np.float32)
            for g in grids:
                X = np.concatenate([X, np.asarray(g[rows, cols], dtype=np.float32)], axis=1)
            print(f"{h} fold {k} {name} fitting {X.shape}", flush=True)
            model = modeling.fit_hgb(X, y, **{kk: CONFIG[kk] for kk in
                                              ("iterations", "lr", "depth", "l2", "seed")})
            p = np.full(fp.shape, np.nan, dtype=np.float32)
            sy, sx = np.nonzero(score)
            for i in range(0, len(sy), 50000):
                rr, cc = sy[i:i + 50000], sx[i:i + 50000]
                batch = feat[rr, cc].astype(np.float32)
                for g in grids:
                    batch = np.concatenate([batch, np.asarray(g[rr, cc], dtype=np.float32)],
                                           axis=1)
                p[rr, cc] = model.predict_proba(batch)[:, 1]
            if not np.isfinite(p[score]).all() or np.isfinite(p[~score]).any():
                raise ValueError("OOF footprint differs from score geography")
            result["scores"][name] = score_policies(p, score, gt, policies)
            result["n_features"] = result.get("n_features", {})
            result["n_features"][name] = int(X.shape[1])
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
                    np.where(score, fused, np.nan).astype(np.float32), score, gt, policies)
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
        if extra_fold_aligned is not None:
            result["extra_fold_aligned_sha256"] = hashlib.sha256(
                np.ascontiguousarray(extra_fold_aligned).tobytes()).hexdigest()
            del extra_fold_aligned
        result["seconds"] = round(time.time() - t0, 2)
        report["folds"].append(result)
        save()
        del random_p, e, train, score, gt
        gc.collect()

    # Preregistered policy selection on development folds only.
    scored_arms = arms + ["baseline107_discovery"]
    selection = {arm: select_policy(report["folds"], arm, policies) for arm in scored_arms}
    comparisons = {}
    cand = selection[h]
    rows_sorted = sorted(report["folds"], key=lambda x: x["fold"])
    for base in [a for a in scored_arms if a != h]:
        bpol = selection[base]
        comparisons[base] = {"baseline_policy": bpol,
                             **paired_comparison(rows_sorted, h, cand,
                                                 lambda r, b=base, bp=bpol: r["scores"][b][bp]["dti"])}
    reproduction = None
    if incumbent is not None:
        inc_arm = args.incumbent_arm
        inc_pol = incumbent["policy_selection"][inc_arm]
        inc_rows = {r["fold"]: r for r in incumbent["folds"]}
        comparisons[f"incumbent_report:{inc_arm}"] = {
            "report": report["incumbent_report"], "arm": inc_arm,
            "baseline_policy": inc_pol,
            **paired_comparison(rows_sorted, h, cand,
                                lambda r: inc_rows[r["fold"]]["scores"][inc_arm][inc_pol]["dti"])}
        # Exact-reproduction anchor: every arm of this run that also exists in
        # the incumbent report, over shared policies (session 4; for
        # --hypothesis H16 vs its own report this reduces to the session-2/3
        # three-arm check, unchanged).
        shared = [p for p in incumbent.get("policies", []) if p in policies]
        common_arms = [a for a in scored_arms
                       if all(a in r["scores"] for r in incumbent.get("folds", []))]
        diffs = []
        for r in rows_sorted:
            for arm in common_arms:
                for p in shared:
                    diffs.append(abs(r["scores"][arm][p]["dti"]
                                     - inc_rows[r["fold"]]["scores"][arm][p]["dti"]))
        if diffs:
            reproduction = {"shared_policies": shared, "arms": common_arms,
                            "max_abs_dti_diff": float(max(diffs)),
                            "exact": bool(max(diffs) == 0.0)}
    eligible = all(c["passes"] for c in comparisons.values())
    report["policy_selection"] = selection
    report["reproduction"] = reproduction
    report["decision"] = {"eligible": eligible, "candidate_arm": h, "candidate_policy": cand,
                          "comparisons": comparisons,
                          "reason": (("all matched incumbents beaten under selected policies; "
                                      "final-training binding still required") if eligible
                                     else "no slot: development AND confirmation must both "
                                          "improve over every incumbent")}
    report["status"] = "completed"
    report["promotion_allowed"] = False  # never by local comparison alone
    save()
    print(json.dumps(report["decision"], indent=2), flush=True)
    if reproduction is not None:
        print("reproduction:", json.dumps(reproduction), flush=True)


def spec_band(name: str) -> int:
    from gems10 import spec
    return spec.BAND_INDEX[name]


if __name__ == "__main__":
    main()
