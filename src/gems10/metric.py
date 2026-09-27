# Provenance: adapted from buffedlizard55-lab/6GEMSDOE src/gems (same owner, re-verified here).
"""Distance-weighted Tversky index (DTI) — the official competition metric.

SOURCE OF TRUTH (quoted, verified 2026-09-25):
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
  section "Performance metric" -> "Mathematical representation"

    TI(a,b) = sum_x p(x)g(x)
              / ( sum_x p(x)g(x) + a*sum_x p(x)(1-g(x)) + b*sum_x (1-p(x))g(x) )

    k(d) = (1 - d/R)_+ = max(1 - d/R, 0),   R = 300 metres

    TP_w = sum_{g in G} max_{x : d(x,g) <= R} p(x) * k(d(x,g))
    FP_w = sum_{x : p(x) > 0} p(x) * [1 - max_{g in G} k(d(x,g))]
    FN_w = sum_{g in G} [1 - max_{x : d(x,g) <= R} p(x) * k(d(x,g))]

    DTI(a,b) = TP_w / (TP_w + a*FP_w + b*FN_w + eps)

    For this competition a = 0.2 (false positives), b = 0.8 (false negatives).

MASKED SCORING (GEMSDOE10 extension — models the staff-clarified scoring):
  DrivenData staff clarified (forum thread 11516, 2026-09-16) that "pixels
  corresponding to known USGS/INGENIOUS faults are masked / excluded from
  evaluation, so they do not count towards penalty terms" — in both rounds.
  `components(..., fp_ignore_mask=M)` implements exactly that: predicted mass
  on M pixels contributes neither to FP_w nor to anything else (M pixels are
  never in the new-fault ground truth). Pass M=None (default) for the literal
  page-967 formula. The harness in systems.py reports both numbers.

Geometry note: the grid is 100 m/pixel (EPSG:32611), so R = 300 m = 3.0 pixels.
The kernel therefore takes the discrete values
    d=0 px (0 m)      -> k = 1
    d=1 px (100 m)    -> k = 2/3
    d=2 px (200 m)    -> k = 1/3
    d=3 px (300 m)    -> k = 0
plus the diagonal distances sqrt(2), sqrt(5), sqrt(8) px -> k = 0.5286, 0.2546, 0.0572.

The two "max" terms are computed exactly (not approximated by a per-pixel match):
  * FP_w needs only Dg(x) = Euclidean distance from x to the nearest ground-truth
    pixel, because k is strictly decreasing, so max_g k(d(x,g)) == k(Dg(x)).
  * TP_w / FN_w need, for every GT pixel, the best weighted prediction inside the
    radius. That is a small-kernel max-dilation over the 29 integer offsets with
    d <= 3 px, which is exact.

eps: the page writes "+ eps" but does not give a value. It can only change the
result when the numerator and all three terms are exactly 0 (i.e. no predictions
and no ground truth). We default to eps=0.0 and define DTI=0.0 in that fully
degenerate case instead of returning NaN, so that no submission can ever receive a
NaN score. Use `eps=` to reproduce a specific platform epsilon if one is published.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# --- official metric constants -------------------------------------------------
ALPHA = 0.2  # false-positive weight  (page 967)
BETA = 0.8  # false-negative weight   (page 967)
RADIUS_M = 300.0  # triangular-kernel support, metres (page 967)
PIXEL_SIZE_M = 100.0  # official grid resolution (page 967)
RADIUS_PX = RADIUS_M / PIXEL_SIZE_M  # = 3.0 pixels exactly


def kernel_offsets(radius_px: float = RADIUS_PX) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer pixel offsets inside the kernel support and their weights.

    Returns (dy, dx, k) ordered by increasing (|dy|, |dx|) so the max-dilation is
    deterministic. Distances are Euclidean in pixel units, matching the metric's
    `d(.,.)` = Euclidean distance in metres on a square 100 m grid.
    """
    r = int(np.ceil(radius_px))
    offs = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = float(np.hypot(dy, dx))
            if d <= radius_px + 1e-12:
                k = max(1.0 - d / radius_px, 0.0)
                offs.append((dy, dx, d, k))
    offs.sort(key=lambda t: (t[2], t[0], t[1]))
    arr = np.array(offs, dtype=np.float64)
    return arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64), arr[:, 3]


@dataclass(frozen=True)
class Components:
    """The three weighted counts and the resulting index."""

    tp_w: float
    fp_w: float
    fn_w: float
    dti: float
    n_pos_pred: int
    n_gt: int

    def as_dict(self) -> dict:
        return {
            "tp_w": self.tp_w,
            "fp_w": self.fp_w,
            "fn_w": self.fn_w,
            "dti": self.dti,
            "n_pos_pred": self.n_pos_pred,
            "n_gt": self.n_gt,
        }


def _shift_zero(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift `arr` by (dy, dx) filling the vacated border with zeros (no wrap)."""
    out = np.zeros_like(arr)
    h, w = arr.shape
    ys_src = slice(max(0, -dy), h - max(0, dy))
    ys_dst = slice(max(0, dy), h - max(0, -dy))
    xs_src = slice(max(0, -dx), w - max(0, dx))
    xs_dst = slice(max(0, dx), w - max(0, -dx))
    out[ys_dst, xs_dst] = arr[ys_src, xs_src]
    return out


def best_weighted_prediction(
    pred: np.ndarray, radius_px: float = RADIUS_PX
) -> np.ndarray:
    """M(x) = max over predictions x' within R of p(x') * k(d(x, x')).

    Returned on the *prediction* grid, so M[g] is the term used by TP_w/FN_w for a
    ground-truth pixel g.
    """
    pred = np.asarray(pred, dtype=np.float64)
    dy, dx, k = kernel_offsets(radius_px)
    best = np.zeros_like(pred)
    for d_y, d_x, kk in zip(dy, dx, k):
        if kk <= 0.0:
            continue
        np.maximum(best, _shift_zero(pred, int(d_y), int(d_x)) * kk, out=best)
    return best


def components(
    pred: np.ndarray,
    target: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
    radius_px: float = RADIUS_PX,
    eps: float = 0.0,
    fp_ignore_mask: np.ndarray | None = None,
) -> Components:
    """Compute TP_w, FP_w, FN_w and DTI exactly as published.

    `pred`   — float array in [0, 1] (values outside are not clipped silently;
               they are clipped for scoring and the caller should have validated).
    `target` — boolean/0-1 array of ground truth (1 = fault).
    `fp_ignore_mask` — optional boolean array: predicted mass on these pixels is
               excluded from FP_w (staff-clarified masking of known-catalogue
               pixels, forum thread 11516). Must be disjoint from `target`; a
               ValueError is raised otherwise, since a ground-truth pixel can
               never be a masked known-fault pixel.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")

    p = np.clip(pred, 0.0, 1.0)
    g = target.astype(bool)
    n_gt = int(g.sum())

    # ---- FP_w -----------------------------------------------------------------
    pos = p > 0.0
    if fp_ignore_mask is not None:
        ign = np.asarray(fp_ignore_mask).astype(bool)
        if ign.shape != g.shape:
            raise ValueError(
                f"shape mismatch: fp_ignore_mask {ign.shape} vs target {g.shape}"
            )
        if bool((ign & g).any()):
            raise ValueError("fp_ignore_mask overlaps the ground truth")
        pos = pos & ~ign
    n_pos_pred = int(pos.sum())
    if n_gt == 0:
        k_nearest = np.zeros_like(p)
    else:
        d_to_gt = ndimage.distance_transform_edt(~g, sampling=1.0)
        k_nearest = np.maximum(1.0 - d_to_gt / radius_px, 0.0)
    fp_w = float((p * (1.0 - k_nearest))[pos].sum()) if n_pos_pred else 0.0

    # ---- TP_w / FN_w ----------------------------------------------------------
    if n_gt == 0:
        tp_w = fn_w = 0.0
    else:
        m = best_weighted_prediction(p, radius_px)
        mg = m[g]
        tp_w = float(mg.sum())
        fn_w = float((1.0 - mg).sum())

    denom = tp_w + alpha * fp_w + beta * fn_w + eps
    dti = float(tp_w / denom) if denom > 0.0 else 0.0
    return Components(tp_w, fp_w, fn_w, dti, n_pos_pred, n_gt)


def distance_weighted_tversky(
    pred: np.ndarray,
    target: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
    radius_px: float = RADIUS_PX,
    eps: float = 0.0,
    fp_ignore_mask: np.ndarray | None = None,
) -> float:
    """Convenience wrapper returning only the index."""
    return components(pred, target, alpha, beta, radius_px, eps, fp_ignore_mask).dti


def analytic_everywhere_score(coverage: float, alpha: float = ALPHA) -> float:
    """DTI of the "predict p=1 on every pixel" submission, analytically.

    With p == 1 everywhere, every GT pixel achieves M=1 (distance 0, k=1), so
    TP_w = n_gt and FN_w = 0. FP_w = sum_x (1 - k(Dg(x))), i.e. every pixel is
    penalised except the fraction of pixels lying exactly on a GT pixel. On a grid
    with GT fraction c (and R=3 px), 1 - k is ~1 almost everywhere except the
    immediate neighbourhood of the GT, and that neighbourhood has measure O(R^2)
    times the number of separate GT pixels — small for thin line networks.

    To first order FP_w ~ (1 - c) * N and n_gt = c*N, giving
        DTI ~ c / (c + alpha*(1 - c))
    which is the closed form below. It is an approximation of the exact value
    (exact() is available in the analysis script); it exists to make the
    "coverage decides your floor" argument auditable without loading the data.
    """
    return coverage / (coverage + alpha * (1.0 - coverage))
