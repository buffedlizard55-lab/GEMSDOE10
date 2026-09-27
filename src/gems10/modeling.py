"""Model fitting, full-grid prediction, placement policies, evaluation.

Shared by scripts/run_cv.py (system-holdout yardstick), scripts/train_final.py
(full-grid fit) and scripts/build_submission.py (gated emission). Policies are
parsed from strings so sweeps are declarative: "<budget>_<mode>", e.g.
"topk03_binary", "topk02_soft", "topk04_envelope", "topk03_halo2".
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:  # pragma: no cover
    HistGradientBoostingClassifier = None

from . import metric, placement


def fit_hgb(X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None,
            iterations: int = 200, lr: float = 0.05, depth: int = 7,
            l2: float = 1.0, seed: int = 7):
    """HistGradientBoosting with native NaN support (no imputation)."""
    if HistGradientBoostingClassifier is None:  # pragma: no cover
        raise SystemExit("scikit-learn required: pip install scikit-learn")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(np.int64).ravel()
    clf = HistGradientBoostingClassifier(
        max_iter=iterations, learning_rate=lr, max_depth=depth,
        l2_regularization=l2, early_stopping=False, random_state=seed,
        validation_fraction=None)
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


def predict_grid(model, feat: np.ndarray, footprint: np.ndarray,
                 extra: np.ndarray | None = None,
                 batch_rows: int = 256) -> np.ndarray:
    """Probability grid (float64, NaN outside footprint), row-batched.

    `feat` is (H, W, C) float32 (memmap ok); `extra` is optional (H, W, E).
    Only footprint rows are scored; NaN rows inside the footprint still get a
    prediction (HGB handles NaN natively).
    """
    h, w = footprint.shape
    out = np.full((h, w), np.nan, dtype=np.float64)
    for r0 in range(0, h, batch_rows):
        r1 = min(r0 + batch_rows, h)
        blk_fp = footprint[r0:r1]
        if not blk_fp.any():
            continue
        X = feat[r0:r1].reshape(-1, feat.shape[2]).astype(np.float32)
        if extra is not None:
            E = extra[r0:r1].reshape(-1, extra.shape[2]).astype(np.float32)
            X = np.concatenate([X, E], axis=1)
        p = model.predict_proba(X)[:, 1].reshape(r1 - r0, w)
        Patch = np.where(blk_fp, p, np.nan)
        out[r0:r1] = Patch
    return out


def parse_policy(policy: str) -> tuple[float, str]:
    """'topk03_binary' -> (0.03, 'binary'). Budgets in whole percent."""
    budget_s, mode = policy.split("_", 1)
    assert budget_s.startswith("topk"), policy
    frac = int(budget_s[4:]) / 100.0
    assert mode in ("binary", "soft", "envelope", "halo2"), policy
    return frac, mode


def apply_policy(prob: np.ndarray, footprint: np.ndarray, policy: str) -> np.ndarray:
    """Emit a submission-shaped grid (NaN outside footprint) for a policy.

    Modes:
      binary   — top-k pixels at 1.0, rest 0.0
      soft     — top-k pixels keep their probability, rest 0.0
      envelope — binary core + 1-px ring at 0.5 (off-by-one insurance)
      halo2    — binary core + ring1 at 2/3 + ring2 at 1/3 (metric-shaped)
    """
    frac, mode = parse_policy(policy)
    fp = np.asarray(footprint, dtype=bool)
    p = np.where(fp, np.nan_to_num(np.asarray(prob, dtype=np.float64),
                                   nan=0.0), 0.0)
    core = placement.topk_mask(p, fp, frac)
    if mode == "binary":
        out = np.where(core, 1.0, 0.0)
    elif mode == "soft":
        out = np.where(core, np.clip(p, 0.0, 1.0), 0.0)
    else:
        out = np.where(core, 1.0, 0.0)
        ring1 = ndimage.binary_dilation(core, iterations=1) & ~core & fp
        if mode == "envelope":
            out = np.where(ring1, 0.5, out)
        else:  # halo2
            ring2 = (ndimage.binary_dilation(core, iterations=2)
                     & ~core & ~ring1 & fp)
            out = np.where(ring1, 2.0 / 3.0, out)
            out = np.where(ring2, 1.0 / 3.0, out)
    return np.where(fp, out.astype(np.float64), np.nan)


def evaluate_policies(prob: np.ndarray, footprint: np.ndarray, gt: np.ndarray,
                      fp_ignore: np.ndarray | None,
                      policies: list[str]) -> dict[str, dict]:
    """Score each policy: DTI masked (official-like) AND unmasked (literal)."""
    res: dict[str, dict] = {}
    for pol in policies:
        emis = apply_policy(prob, footprint, pol)
        scored = np.nan_to_num(emis, nan=0.0)
        c_plain = metric.components(scored, gt)
        row = {"dti_unmasked": c_plain.dti, "tp_w": c_plain.tp_w,
               "fp_w": c_plain.fp_w, "fn_w": c_plain.fn_w,
               "n_pos_pred": c_plain.n_pos_pred, "n_gt": c_plain.n_gt}
        if fp_ignore is not None:
            c_mask = metric.components(scored, gt, fp_ignore_mask=fp_ignore)
            row["dti_masked"] = c_mask.dti
            row["fp_w_masked"] = c_mask.fp_w
        res[pol] = row
    return res
