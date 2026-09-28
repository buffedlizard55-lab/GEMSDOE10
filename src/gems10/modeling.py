"""Model fitting, full-grid prediction, placement policies, evaluation.

Shared by scripts/run_cv.py (system-holdout yardstick), scripts/train_final.py
(full-grid fit) and scripts/build_submission.py (gated emission). Policies are
parsed from strings so sweeps are declarative: "<budget>_<mode>", e.g.
"topk03_binary", "topk02_soft", "topk04_envelope", "topk03_halo2".
"""

from __future__ import annotations

import weakref

import numpy as np
from scipy import ndimage

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:  # pragma: no cover
    HistGradientBoostingClassifier = None

from . import metric, placement

# Single-entry cache for the H24 ridge-NMS normal offsets: the structure
# tensor of a 12.3 Mpx probability field costs seconds, and the policy sweep
# scores six `ridge*` budgets on the same field. Keyed by object identity of
# (prob, footprint) via weakrefs — a dead or replaced array is a miss, never
# a stale hit.
_RIDGE_NRM: dict = {"ref": None, "fp_ref": None, "offsets": None}


def _ridge_offsets_cached(prob: np.ndarray, fp: np.ndarray, p: np.ndarray):
    ent = _RIDGE_NRM
    if (ent["ref"] is not None and ent["ref"]() is prob
            and ent["fp_ref"]() is fp):
        return ent["offsets"]
    offsets = placement.ridge_offsets(np.where(fp, p, np.nan))
    try:
        ent["ref"] = weakref.ref(prob)
        ent["fp_ref"] = weakref.ref(fp)
    except TypeError:  # not weakref-able input: just skip caching
        return offsets
    ent["offsets"] = offsets
    return offsets


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
                 extra_grids: list | None = None,
                 batch_rows: int = 256) -> np.ndarray:
    """Probability grid (float64, NaN outside footprint), row-batched.

    `feat` is (H, W, C) float32 (memmap ok); `extra` is optional (H, W, E);
    `extra_grids` is an optional list of additional (H, W, Ei) grids that are
    concatenated PER BATCH (so disk memmaps are never materialised in RAM —
    concatenating the 800 MB externals eagerly OOM-killed a 3.8 GB host).
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
        # column order must match training: [feat | extra_grids... | extra]
        if extra_grids:
            for g in extra_grids:
                G = g[r0:r1].reshape(-1, g.shape[2]).astype(np.float32)
                X = np.concatenate([X, G], axis=1)
        if extra is not None:
            E = extra[r0:r1].reshape(-1, extra.shape[2]).astype(np.float32)
            X = np.concatenate([X, E], axis=1)
        p = model.predict_proba(X)[:, 1].reshape(r1 - r0, w)
        Patch = np.where(blk_fp, p, np.nan)
        out[r0:r1] = Patch
    return out


def parse_policy(policy: str) -> tuple[float, str]:
    """'topk03_binary' -> (0.03, 'binary'); 'thin06_binary' -> (0.06, 'thin');
    'ridge10_binary' -> (0.10, 'ridge').

    Budgets are whole percents of the footprint. `thinNN_binary` (session 3,
    H19) is the top-NN% mask reduced to its Zhang-Suen skeleton and emitted at
    1.0. `ridgeNN_binary` (session 4, H24) is the top-NN% mask reduced to its
    across-strike probability-ridge NMS (`placement.ridge_nms`) and emitted at
    1.0 — NN is the PRE-NMS budget; in both families the emitted pixel count is
    smaller and is reported by the scorer (`n_pos_pred`). `ridgeNN_d2` /
    `ridgeNN_d3` (session 5, H28) further keep every 2nd/3rd ridge pixel along
    strike (`placement.dot_nms`, probability-ordered): 'ridge15_d3' ->
    (0.15, 'ridgedot3').
    """
    budget_s, mode = policy.split("_", 1)
    if budget_s.startswith("thin"):
        assert mode == "binary", policy
        return int(budget_s[4:]) / 100.0, "thin"
    if budget_s.startswith("ridge"):
        # `ridgeNN_d2` / `ridgeNN_d3` (session 5, H28): the ridgeNN line
        # thinned ALONG strike by probability-ordered radius-2/3 suppression.
        assert mode in ("binary", "d2", "d3"), policy
        return int(budget_s[5:]) / 100.0, ("ridge" if mode == "binary"
                                           else "ridgedot" + mode[1:])
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
      thin     — skeleton of the top-k mask at 1.0 (H19: across-strike width
                 is pure FP under the published metric; the truth is a 1-px line)
      ridge    — across-strike NMS of p inside the top-k mask at 1.0 (H24:
                 follows the probability ridge instead of the mask geometry,
                 so it cannot leave the top-k support)
      ridgedotR — the ridge line dotted ALONG strike: probability-ordered
                 suppression of kept pixels closer than R px (H28, R = 2|3)
    """
    frac, mode = parse_policy(policy)
    fp = np.asarray(footprint, dtype=bool)
    p = np.where(fp, np.nan_to_num(np.asarray(prob, dtype=np.float64),
                                   nan=0.0), 0.0)
    core = placement.topk_mask(p, fp, frac)
    if mode == "thin":
        out = np.where(placement._skeleton(core) & fp, 1.0, 0.0)
    elif mode == "ridge":
        offsets = _ridge_offsets_cached(prob, fp, p)
        out = np.where(placement.ridge_nms(core, p, offsets=offsets) & fp,
                       1.0, 0.0)
    elif mode.startswith("ridgedot"):
        offsets = _ridge_offsets_cached(prob, fp, p)
        ridge = placement.ridge_nms(core, p, offsets=offsets) & fp
        out = np.where(placement.dot_nms(ridge, p, int(mode[8:])), 1.0, 0.0)
    elif mode == "binary":
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
