#!/usr/bin/env python3
"""Protocol v5 (session 5): density-matched decision from saved OOF grids.

    MALLOC_ARENA_MAX=2 .venv/bin/python scripts/evaluate_density_matched.py \
        --source reports/h29_oof.json --incumbent-report reports/h25_blocked.json

WHY (HYPOTHESES.md session-5 register). Every earlier decision scored the
held-out CATALOGUE at its full density (1.1-1.3 % of a score block). The
leaderboard scores NEW faults only, and inverting the group's seven scored
artifacts (reports/lb_probe.json) bounds that truth to ~0.13-0.61 % of the
scored area — 2-8x sparser. The DTI trade-off between recall and emitted
pixels depends on that density, so the full-density proxy is blind to exactly
the emission changes that matter most at leaderboard density. v5 scores every
policy both ways and DECIDES on the density-matched score:

  * masked-known simulation (as scripts/budget_density_sweep.py): within each
    fold's score region, 8-connected catalogue systems are split at random
    (seed 100*fold + draw) into kept "new" truth (fraction f) and "known"
    systems that are masked pixel-exactly, for f in {0.5, 0.25} x 3 draws;
  * masking semantics `zero_ignored=True` (known-pixel predictions neither
    count as FP nor credit nearby truth — staff: "excluded from evaluation");
    the other semantics is recorded as `dti_credit` for transparency;
  * per policy "dti" = mean of the six simulated DTIs; "dti_full" = the
    ordinary full-density DTI (the v4 number).

Decision rules (preregistered; all comparisons must pass):
  H28 (emission only, H25 grids): candidate = the H25 arm under its v5-selected
    policy. Beat baseline107 and baseline107_discovery (their v5 selections)
    and the incumbent H25@ridge15_binary on the density-matched development
    mean AND confirmation fold; full-density non-inferiority vs the incumbent
    (development-mean and confirmation deltas >= -0.01); the H25 and
    baseline107 grids must reproduce reports/h25_blocked.json byte-exactly.
  H29 (line-support channels): candidate = the H29 arm under its v5 selection;
    beat baseline107, baseline107_discovery, H25 (its v5 selection) and the
    incumbent H25@ridge15_binary (density-matched dev mean AND confirmation);
    same full-density non-inferiority.
Selection per arm: argmax density-matched development mean over V5_POLICIES,
ties by list order. Nothing here trains or releases; train_final --bind-to and
release.check_release do that.
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
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems10 import cv, metric, modeling  # noqa: E402
from gems10.raster import sha256_file  # noqa: E402

PROTOCOL_V5 = "spatial-4x4-strided-v1-buffer40-policy-sweep-v5-density-matched"
V5_POLICIES = (["topk02_binary", "thin12_binary"]
               + [f"ridge{b:02d}_binary" for b in (4, 6, 10, 15, 20, 30)]
               + [f"ridge{b:02d}_d2" for b in (10, 15, 20, 30)]
               + [f"ridge{b:02d}_d3" for b in (15, 20, 30, 40)])
FRACTIONS = (0.5, 0.25)
DRAWS = 3
INCUMBENT_ARM, INCUMBENT_POLICY = "H25", "ridge15_binary"
NONINFERIORITY_MARGIN = 0.01
DEV_FOLDS, CONF_FOLD = (0, 1, 2), 3
SHARED_CODE = ("src/gems10/modeling.py", "src/gems10/placement.py",
               "src/gems10/metric.py", "src/gems10/cv.py")


def split_truth(gt_all: np.ndarray, fold: int, draw: int, f: float):
    """(truth, known) masks — same seed scheme as budget_density_sweep.py."""
    sys_id, n_sys = ndimage.label(gt_all, np.ones((3, 3)))
    rng = np.random.default_rng(100 * fold + draw)
    keep = rng.random(n_sys + 1) < f
    keep[0] = False
    truth = keep[sys_id]
    return truth, gt_all & ~truth


def score_arm_fold(prob, score, gt_all, splits, policies):
    """{policy: row} with dti (density-matched mean), dti_full, components."""
    out = {}
    for pol in policies:
        e = np.nan_to_num(modeling.apply_policy(prob, score, pol), nan=0.0)
        full = metric.binary_components(e, gt_all)
        sims, credit = [], []
        for truth, known in splits:
            sims.append(metric.binary_components(e, truth, ignore=known).dti)
            credit.append(metric.binary_components(e, truth, ignore=known,
                                                   zero_ignored=False).dti)
        out[pol] = {"dti": float(np.mean(sims)), "dti_full": full.dti,
                    "dti_by_draw": [float(v) for v in sims],
                    "dti_credit": float(np.mean(credit)),
                    "tp_w": full.tp_w, "fp_w": full.fp_w, "fn_w": full.fn_w,
                    "n_pos_pred": full.n_pos_pred, "n_gt": full.n_gt}
    return out


def select(folds, arm, policies, key="dti"):
    means = {p: float(np.mean([r["scores"][arm][p][key] for r in folds
                               if r["role"] == "development"])) for p in policies}
    top = max(means.values())
    return min([p for p in policies if abs(means[p] - top) < 1e-12],
               key=policies.index)


def paired(folds, cand, cpol, base, bpol, key="dti"):
    rows = sorted(folds, key=lambda r: r["fold"])
    d = [r["scores"][cand][cpol][key] - r["scores"][base][bpol][key] for r in rows]
    dev = float(np.mean([x for r, x in zip(rows, d) if r["role"] == "development"]))
    conf = float([x for r, x in zip(rows, d) if r["role"] == "confirmation"][0])
    return {"baseline_policy": bpol, "delta_by_fold": d,
            "development_mean_delta": dev, "confirmation_delta": conf}


def decide(folds, cand_arm, cand_scores_arm, selection, others):
    """Comparisons for one candidate. `cand_scores_arm` is the fold-score key."""
    cpol = selection[cand_arm]
    comps = {}
    for base in others:
        c = paired(folds, cand_scores_arm, cpol, base, selection[base])
        c["passes"] = bool(c["development_mean_delta"] > 0 and c["confirmation_delta"] > 0)
        comps[base] = c
    inc = paired(folds, cand_scores_arm, cpol, INCUMBENT_ARM, INCUMBENT_POLICY)
    inc["passes"] = bool(inc["development_mean_delta"] > 0 and inc["confirmation_delta"] > 0)
    comps[f"incumbent_fixed:{INCUMBENT_ARM}@{INCUMBENT_POLICY}"] = inc
    nin = paired(folds, cand_scores_arm, cpol, INCUMBENT_ARM, INCUMBENT_POLICY,
                 key="dti_full")
    nin["margin"] = -NONINFERIORITY_MARGIN
    nin["passes"] = bool(nin["development_mean_delta"] >= -NONINFERIORITY_MARGIN
                         and nin["confirmation_delta"] >= -NONINFERIORITY_MARGIN)
    comps["full_density_noninferiority"] = nin
    return cpol, comps


def _reason(eligible: bool, comps: dict) -> str:
    if eligible:
        return ("all preregistered v5 comparisons passed; final-training binding still required")
    return "not eligible — failed: " + "; ".join(
        f"{k} (dev {v.get('development_mean_delta') or 0:+.4f}, "
        f"conf {v.get('confirmation_delta') or 0:+.4f})" for k, v in comps.items()
        if not v.get("passes"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=ROOT / "reports/h29_oof.json")
    ap.add_argument("--incumbent-report", type=Path,
                    default=ROOT / "reports/h25_blocked.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--summary", type=Path,
                    default=ROOT / "reports/density_matched_v5.json")
    args = ap.parse_args()
    t0 = time.time()
    src = json.loads(args.source.read_text())
    if src.get("status") != "completed" or src.get("hypothesis") != "H29":
        raise SystemExit("source must be a completed H29 OOF-generation report")
    for rel in SHARED_CODE:  # emission/metric code must be what produced the grids
        if src["code"].get(rel) != sha256_file(ROOT / rel):
            raise SystemExit(f"{rel} changed since the OOF run; refusing")
    inc_rep = json.loads(args.incumbent_report.read_text())
    inc_files = {r["fold"]: r.get("prediction_files", {}) for r in inc_rep["folds"]}
    with rasterio.open(ROOT / "data/labels.tif") as ds:
        labels = ds.read(1) == 1
    with rasterio.open(ROOT / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    folds = cv.make_folds(n_blocks=4, n_folds=4, buffer_px=src["config"]["buffer_px"])
    arms = ["baseline107", "baseline107_discovery", "H25", "H29"]
    rows, repro = [], {}
    for srow in sorted(src["folds"], key=lambda r: r["fold"]):
        k = srow["fold"]
        _, score = folds.train_test_masks(fp.shape, k)
        score &= fp
        gt_all = labels & score
        splits = [split_truth(gt_all, k, d, f) for f in FRACTIONS for d in range(DRAWS)]
        row = {"fold": k, "role": srow["role"], "gt_pixels": int(gt_all.sum()),
               "truth_pixels_by_draw": [int(t.sum()) for t, _ in splits],
               "scores": {}, "prediction_files": {}}
        for arm in arms:
            pf = srow["prediction_files"][arm]
            path = ROOT / pf["file"]
            if sha256_file(path) != pf["sha256"]:
                raise SystemExit(f"fold {k} {arm}: OOF grid hash mismatch")
            row["prediction_files"][arm] = pf
            if arm in ("baseline107", "H25"):
                ref = inc_files.get(k, {}).get(arm, {}).get("sha256")
                repro[f"{arm}_fold{k}"] = bool(ref == pf["sha256"])
            prob = np.load(path, allow_pickle=False)
            row["scores"][arm] = score_arm_fold(prob, score, gt_all, splits, V5_POLICIES)
            # cross-check: fast exact binary metric == reference metric (one policy)
            e = np.nan_to_num(modeling.apply_policy(prob, score, INCUMBENT_POLICY), nan=0)
            ref_dti = metric.components(e, gt_all).dti
            got = row["scores"][arm][INCUMBENT_POLICY]["dti_full"]
            if abs(ref_dti - got) > 1e-12:
                raise SystemExit(f"fast metric mismatch fold {k} {arm}: {got} vs {ref_dti}")
            print(f"[{time.time() - t0:6.0f}s] fold {k} {arm}: "
                  + " ".join(f"{p}={row['scores'][arm][p]['dti']:.4f}"
                             for p in ("ridge15_binary", "ridge15_d2", "ridge15_d3",
                                       "ridge06_binary")), flush=True)
            del prob, e
        rows.append(row)
    # H28 rows: alias the H25 grid scores as the candidate arm "H28".
    for r in rows:
        r["scores"]["H28"] = r["scores"]["H25"]
        r["prediction_files"]["H28"] = r["prediction_files"]["H25"]
    sel_free = {a: select(rows, a, V5_POLICIES) for a in arms}
    code = dict(src["code"])
    code["scripts/evaluate_density_matched.py"] = sha256_file(Path(__file__))
    code["src/gems10/release.py"] = sha256_file(ROOT / "src/gems10/release.py")
    common = {"status": "completed", "promotion_allowed": False,
              "protocol": PROTOCOL_V5, "policies": V5_POLICIES,
              "fractions": list(FRACTIONS), "draws": DRAWS,
              "masking_semantics": "zero_ignored (known-pixel predictions removed "
                                   "before scoring); dti_credit = alternative",
              "incumbent": f"{INCUMBENT_ARM}@{INCUMBENT_POLICY} (same-run grids; "
                           "released session-4 artifact recipe)",
              "noninferiority_margin_full_density": NONINFERIORITY_MARGIN,
              "source_report": str(args.source.resolve().relative_to(ROOT)),
              "source_report_sha256": sha256_file(args.source),
              "config": src["config"], "inputs": src["inputs"], "code": code,
              "versions": src.get("versions"), "confirmation_fold": CONF_FOLD,
              "score_population": "held-out catalogue systems in spatial score blocks; "
                                  "density-matched by masked-known system subsampling "
                                  "(proxy only — not the private new-fault truth)"}
    reproduction = {"grids_byte_identical_to_incumbent_report": repro,
                    "exact": bool(repro and all(repro.values()))}
    outputs = {}
    # ---- H28 -------------------------------------------------------------------
    sel28 = {"baseline107": sel_free["baseline107"],
             "baseline107_discovery": sel_free["baseline107_discovery"],
             "H25": INCUMBENT_POLICY, "H28": sel_free["H25"]}
    cpol, comps = decide(rows, "H28", "H28", sel28,
                         ["baseline107", "baseline107_discovery"])
    comps["reproduction_of_incumbent_grids"] = {"passes": reproduction["exact"],
                                                **reproduction}
    elig28 = all(c["passes"] for c in comps.values())
    outputs["H28"] = {**common, "hypothesis": "H28",
                      "arms": ["baseline107", "baseline107_discovery", "H25", "H28"],
                      "policy_selection": sel28, "folds": rows,
                      "reproduction": reproduction,
                      "decision": {"eligible": elig28, "candidate_arm": "H28",
                                   "candidate_policy": cpol, "comparisons": comps,
                                   "reason": _reason(elig28, comps),
                                   "final_training_recipe": "identical to H25 (same "
                                   "extras, config, seeds): the H28 change is emission only"}}
    # ---- H29 -------------------------------------------------------------------
    sel29 = dict(sel_free)
    cpol29, comps29 = decide(rows, "H29", "H29", sel29,
                             ["baseline107", "baseline107_discovery", "H25"])
    elig29 = all(c["passes"] for c in comps29.values())
    rows29 = [{**r, "scores": {a: r["scores"][a] for a in arms},
               "prediction_files": {a: r["prediction_files"][a] for a in arms}}
              for r in rows]
    outputs["H29"] = {**common, "hypothesis": "H29", "arms": arms,
                      "policy_selection": sel29, "folds": rows29,
                      "reproduction": reproduction,
                      "new_channels": src.get("new_channels"),
                      "decision": {"eligible": elig29, "candidate_arm": "H29",
                                   "candidate_policy": cpol29, "comparisons": comps29,
                                   "reason": _reason(elig29, comps29)}}
    for h, rep in outputs.items():
        rep["elapsed_seconds"] = round(time.time() - t0, 1)
        path = args.out_dir / f"{h.lower()}_blocked.json"
        path.write_text(json.dumps(rep, indent=2, allow_nan=False) + "\n")
    # compact summary for the site / docs
    def means(arm, pol, key):
        return {"development_mean": float(np.mean([r["scores"][arm][pol][key] for r in rows
                                                   if r["role"] == "development"])),
                "confirmation": float([r["scores"][arm][pol][key] for r in rows
                                       if r["role"] == "confirmation"][0])}
    summary = {"protocol": PROTOCOL_V5, "generated_utc": time.strftime(
                   "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "selection": sel_free, "reproduction": reproduction,
               "table": {a: {p: {"density_matched": means(a, p, "dti"),
                                 "full_density": means(a, p, "dti_full")}
                             for p in V5_POLICIES} for a in arms},
               "H28": {"eligible": elig28, "policy": cpol,
                       "comparisons": {k: {kk: v.get(kk) for kk in (
                           "development_mean_delta", "confirmation_delta", "passes")}
                           for k, v in comps.items()}},
               "H29": {"eligible": elig29, "policy": cpol29,
                       "comparisons": {k: {kk: v.get(kk) for kk in (
                           "development_mean_delta", "confirmation_delta", "passes")}
                           for k, v in comps29.items()}},
               "release_rule": "H29 if eligible, else H28 if eligible, else none",
               "elapsed_seconds": round(time.time() - t0, 1)}
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("selection", "H28", "H29", "reproduction")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
