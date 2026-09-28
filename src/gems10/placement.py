# Provenance: adapted from buffedlizard55-lab/6GEMSDOE src/gems (same owner, re-verified here).
"""Metric-aware placement of predictions.

The metric (see metric.py) rewards a prediction pixel p(x) in two ways and
punishes it once:

  * it raises TP_w for every ground-truth pixel within 300 m, weighted p(x)*k(d);
  * it lowers FN_w by the same amount (beta = 0.8 vs alpha = 0.2);
  * it adds alpha * p(x) * (1 - k(d_nearest_gt)) to FP_w.

So the marginal value of putting probability p at a pixel where the model
believes a fault is present with posterior q, and where the nearest ground-truth
pixel is expected about d away, is

    gain(p) ~ q * k(d) * (1 + beta) - (1 - q) * alpha * (1 - k(d))     [per unit p]

which is positive while the local posterior exceeds

    q* = alpha * (1 - k) / (k * (1 + beta) + alpha * (1 - k)).

Two consequences that decide how predictions should be placed:

  1. Along a *true* line, densifying is never harmful. Putting p at every pixel
     of the line gives each ground-truth pixel k = 1 at distance 0, hence maximum
     TP_w and zero FN_w for that line. Thinning a true line to "every 4th-5th
     pixel" (the task brief's hypothesis) can only lower TP_w and raise FN_w,
     because the skipped ground-truth pixels are then served at best by a
     neighbour ~2 px away, i.e. k ~ 1/3 (200 m). The 300 m kernel does not imply
     4-5 px spacing; that rule is a false-positive control, not free geometry.
  2. At d = 1 px (the realistic localisation error) q* = 0.0526, i.e. ~5%. The
     break-even posterior is very low, so aggressively keeping only the top 1% of
     pixels is usually the wrong move for this metric.

`strategies()` exposes several placements precisely so that these claims are
MEASURED (scripts/analysis.py scores every strategy on spatially blocked held-out
folds against the real labels) instead of asserted.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import metric


def breakeven_posterior(k: float) -> float:
    """q* for a given kernel weight k (see module docstring)."""
    return metric.ALPHA * (1.0 - k) / (k * (1.0 + metric.BETA) + metric.ALPHA * (1.0 - k))


def breakeven_table() -> dict[float, float]:
    """q* at each achievable kernel weight on the 100 m grid."""
    # kernel weights for integer offsets with d <= 3 px: 0, 1, sqrt2, 2, sqrt5, sqrt8, 3 px
    ks = [1.0, 2.0 / 3.0, 1.0 - np.sqrt(2) / 3,
          1.0 / 3.0, 1.0 - np.sqrt(5) / 3, 1.0 - np.sqrt(8) / 3, 0.0]
    return {round(k, 4): round(breakeven_posterior(k), 5) for k in sorted(set(ks), reverse=True)}


def _skeleton(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, implemented locally to avoid a scikit-image dep."""
    img = mask.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2 = p[0:-2, 1:-1]; P3 = p[0:-2, 2:]; P4 = p[1:-1, 2:]; P5 = p[2:, 2:]
            P6 = p[2:, 1:-1]; P7 = p[2:, 0:-2]; P8 = p[1:-1, 0:-2]; P9 = p[0:-2, 0:-2]
            Bn = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            An = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
                     for i in range(8))
            if step == 0:
                cond = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            kill = (img == 1) & (Bn >= 2) & (Bn <= 6) & (An == 1) & cond
            if kill.any():
                img[kill] = 0
                changed = True
    return img.astype(bool)


def ridge_nms(core: np.ndarray, p: np.ndarray,
              offsets: list[tuple[np.ndarray, np.ndarray]] | None = None
              ) -> np.ndarray:
    """Probability-ridge non-maximum suppression across the local strike (H24).

    Keep pixels of `core` that are local maxima of `p` along the local NORMAL
    direction (structure-tensor orientation of `p`, measured to be the
    direction of greatest change — see HYPOTHESES.md session-4 register),
    comparing at ±1 and ±2 px offsets along that normal. Ties (common: HGB
    outputs are piecewise constant) are broken lexicographically by
    (p, distance-to-mask-edge), so a flat plateau reduces to its interior
    spine instead of surviving whole.

    `offsets` may supply precomputed (dr, dc) neighbour displacements for the
    two distances (as `modeling` does, cached per probability field); when
    None they are derived from the structure tensor of `p`.

    Properties, by construction:
      * output ⊆ core (never leaves the top-k support, unlike a skeleton that
        can wander off the probability mass);
      * along-strike continuity is preserved — comparisons are across strike
        only, so a line rising along strike is never sampled against its own
        downstream values;
      * deterministic; NaN-free binary output.
    """
    core = np.asarray(core, dtype=bool)
    p = np.asarray(p, dtype=np.float64)
    if not core.any():
        return np.zeros_like(core)
    if offsets is None:
        offsets = ridge_offsets(np.where(np.isfinite(p), p, np.nan))

    dt = ndimage.distance_transform_edt(core)
    H, W = core.shape
    rr = np.arange(H)[:, None]
    cc = np.arange(W)[None, :]
    keep = core.copy()

    def neighbor_ge(r: np.ndarray, c: np.ndarray, dr: np.ndarray,
                    dc_: np.ndarray) -> np.ndarray:
        """True where p at (r, c) beats (lexicographically: p, then interior
        distance) p at the shifted neighbour; out-of-image counts as smaller
        (keeps edge pixels)."""
        nr = r + dr
        nc = c + dc_
        inside = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        nr_c = np.clip(nr, 0, H - 1)
        nc_c = np.clip(nc, 0, W - 1)
        q = p[nr_c, nc_c]
        qdt = dt[nr_c, nc_c]
        me = p[r, c]
        medt = dt[r, c]
        return ~inside | (me > q) | ((me == q) & (medt >= qdt))

    for dr, dc in offsets:
        for sgn in (1, -1):
            keep &= core & neighbor_ge(rr, cc, sgn * dr, sgn * dc)
    return keep


def dot_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer (dr, dc) displacements with 0 < Euclidean distance < `radius`.

    radius 2 -> the 8-neighbourhood (spacing 2 along any straight run);
    radius 3 -> 24 offsets up to sqrt(8) (spacing 3 along a straight run).
    """
    r = int(radius)
    if r < 2:
        raise ValueError("dot radius must be >= 2 (radius 1 suppresses nothing)")
    dr, dc = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.hypot(dr, dc)
    sel = (d > 0) & (d < r)
    return dr[sel].astype(np.int64), dc[sel].astype(np.int64)


def dot_nms(mask: np.ndarray, p: np.ndarray, radius: int) -> np.ndarray:
    """Along-strike dotting of a thin emission (session 5, H28).

    Greedy probability-ordered suppression: visit the pixels of `mask` from the
    highest to the lowest `p` (ties broken by raster index, so the result is
    deterministic), keep a pixel unless an already-kept pixel lies at Euclidean
    distance < `radius`, and then block its < `radius` neighbourhood.

    Applied to a 1-px ridge line (`ridge_nms` output) this keeps every
    `radius`-th pixel along strike. Why that can raise DTI: the 300 m kernel
    credits a ground-truth pixel with k = 2/3 from a prediction 1 px away, so a
    dotted TRUE trace keeps (1 + 2*2/3)/3 ~ 0.78 (radius 3) or (1 + 2/3)/2 ~ 0.83
    (radius 2) of its recall with 1/3 or 1/2 of the pixels, while a dotted FALSE
    trace costs 1/3 or 1/2 of the false-positive mass. When most candidate
    traces are false (sparse truth), the FP saving dominates. Output ⊆ mask.
    """
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return out
    H, W = mask.shape
    pv = np.nan_to_num(np.asarray(p, dtype=np.float64)[ys, xs], nan=-np.inf)
    flat = ys.astype(np.int64) * W + xs
    order = np.lexsort((flat, -pv))  # primary: p descending; then raster index
    dr, dc = dot_offsets(radius)
    blocked = np.zeros_like(mask)
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if blocked[y, x]:
            continue
        out[y, x] = True
        yy = y + dr
        xx = x + dc
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        blocked[yy[ok], xx[ok]] = True
    return out


def ridge_offsets(filled: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """±d-neighbour displacements along the structure-tensor normal of p.

    `filled` is (H, W) float with NaN outside; returns int arrays
    [(dr1, dc1), (dr2, dc2)] rounding d·(sin o, cos o) for d in (1, 2)
    ((row, col) basis; the tensor orientation o is in the (x=col, y=row)
    basis and is the normal to the local line — verified empirically,
    HYPOTHESES.md session-4 register).
    """
    from . import features
    # float32 on purpose: the offsets are rounded to integer pixels, and on
    # this 4 GB host the float64 tensor temporaries (~1 GB peak) caused an
    # OOM-kill during a policy sweep (REVIEW.md session 4). float32 halves
    # the transient; offsets are identical in every synthetic control.
    lin = features.structure_tensor(filled.astype(np.float32, copy=False),
                                    sigma=1.5, integration_sigma=4.0)
    o = lin.orientation
    offsets = []
    for d in (1.0, 2.0):
        dr = np.round(d * np.sin(o))
        dc = np.round(d * np.cos(o))
        dr = np.where(np.isfinite(dr), dr, 0.0)
        dc = np.where(np.isfinite(dc), dc, 0.0)
        bad = (dr == 0) & (dc == 0)  # cannot happen for |n| = 1 at d >= 1
        if bad.any():
            dc = np.where(bad, 1.0, dc)
        offsets.append((dr.astype(np.int16), dc.astype(np.int16)))
    return offsets


def thin_keep(mask: np.ndarray, spacing: int) -> np.ndarray:
    """Skeleton, then keep one pixel per `spacing` pixels of arc length.

    `spacing=1` returns the skeleton unchanged — the geometry the metric prefers
    for a true line. Larger spacing reproduces the brief's "4-5 px" hypothesis so
    it can be scored against the alternative.
    """
    skel = _skeleton(mask)
    if spacing <= 1 or not skel.any():
        return skel
    ys, xs = np.nonzero(skel)
    order = np.lexsort((xs, ys))
    ys, xs = ys[order], xs[order]
    keep = np.zeros_like(skel)
    keep[ys[::spacing], xs[::spacing]] = True
    return keep & skel


def densify_along_lineaments(prob: np.ndarray, valid: np.ndarray,
                             threshold: float, dilation: int = 1) -> np.ndarray:
    """Raise p toward the local maximum next to supra-threshold pixels.

    This adds no information: it compensates for a ridge detector that is one
    pixel off centre, which is exactly the off-by-one error the competition page
    cites as motivation for distance weighting ("rasterization is lossy and can
    induce off-by-one errors near pixel boundaries").
    """
    strong = (prob >= threshold) & valid
    if not strong.any() or dilation < 1:
        return prob.copy()
    dilated = ndimage.binary_dilation(strong, iterations=dilation) & valid
    filled = ndimage.maximum_filter(np.where(valid, prob, -np.inf),
                                    size=2 * dilation + 1)
    out = np.where(dilated, np.maximum(prob, filled), prob)
    out = np.where(valid, out, 0.0)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)


def topk_mask(p: np.ndarray, valid: np.ndarray, frac: float) -> np.ndarray:
    """Exactly the top `frac` share of valid pixels by probability (>= tie-broken).

    Selecting a BUDGET rather than a probability cut-off is the right control for
    this metric: FP_w is an absolute sum over predicted pixels while TP_w and
    FN_w are per-ground-truth-pixel sums, so the score depends on how many
    pixels you commit to relative to how many fault pixels exist — a quantity a
    probability threshold only controls indirectly.
    """
    p = np.clip(np.nan_to_num(p, nan=0.0), 0.0, 1.0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.zeros_like(valid)
    k = max(1, int(round(frac * n_valid)))
    if k >= n_valid:
        return valid.astype(bool)
    inside = p[valid]
    thr = float(np.partition(inside, n_valid - k)[n_valid - k])
    return (p >= thr) & valid


def gated_topk(p: np.ndarray, valid: np.ndarray, gate: np.ndarray,
               frac: float) -> np.ndarray:
    """Exact-budget top-k that spends the budget on gated pixels first.

    `gate` marks the pixels the cross-signal agreement test accepts. The full
    budget `frac * n_valid` is still spent: all gated pixels up to the budget
    are selected by probability, and only if the gate is sparser than the
    budget does the remainder come from the ungated pixels, again by
    probability. A monotone gate therefore changes WHERE the budget goes,
    never how much — so it is directly comparable to `topk_mask` under the
    same blocked folds.
    """
    p = np.clip(np.nan_to_num(p, nan=0.0), 0.0, 1.0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.zeros_like(valid, dtype=bool)
    k = max(1, int(round(frac * n_valid)))
    gated = valid & gate
    ungated = valid & ~gate
    sel = np.zeros_like(valid, dtype=bool)
    n_g = int(gated.sum())
    if n_g >= k:
        vals = p[gated]
        thr = float(np.partition(vals, n_g - k)[n_g - k])
        sel[gated] = vals >= thr
    else:
        sel[gated] = True
        rem = k - n_g
        n_u = int(ungated.sum())
        if rem > 0 and n_u > 0:
            vals = p[ungated]
            if n_u > rem:
                thr = float(np.partition(vals, n_u - rem)[n_u - rem])
                sel[ungated] = vals >= thr
            else:
                sel[ungated] = True
    return sel & valid


def get_strategy(name: str, prob: np.ndarray, valid: np.ndarray, threshold: float,
                 spacing: int = 4, top_fraction: float = 0.01,
                 gate: np.ndarray | None = None) -> np.ndarray:
    """Compute a single named placement.

    `strategies()` builds all of them, which is what the CV comparison wants but is
    wasteful for the final build (the skeleton pass is the expensive one). This
    dispatcher computes exactly the requested placement.
    """
    if name == "raw":
        return strategies(prob, valid, threshold, spacing, top_fraction)["raw"]
    if name == "threshold":
        return strategies(prob, valid, threshold, spacing, top_fraction)["threshold"]
    if name.startswith("hard@"):
        # p -> 1 on the selected pixels. See the module docstring: the metric can
        # be written as DTI = TP_w / (beta*n_gt + alpha*FP_w + (1-beta)*TP_w),
        # which is strictly increasing under p -> lambda*p because the beta*n_gt
        # term does not scale. Fractional probabilities therefore give away score.
        thr = float(name.split("@", 1)[1])
        p = np.clip(np.nan_to_num(prob, nan=0.0), 0.0, 1.0)
        return np.where((p >= thr) & valid, 1.0, 0.0).astype(np.float32)
    if name.startswith("topk_hard@"):
        frac = float(name.split("@", 1)[1])
        p = np.clip(np.nan_to_num(prob, nan=0.0), 0.0, 1.0)
        return np.where(topk_mask(p, valid, frac), 1.0, 0.0).astype(np.float32)
    if name.startswith("topk_soft@"):
        frac = float(name.split("@", 1)[1])
        p = np.clip(np.nan_to_num(prob, nan=0.0), 0.0, 1.0)
        return np.where(topk_mask(p, valid, frac), p, 0.0).astype(np.float32)
    if name.startswith("topk_gate@"):
        # `topk_gate@<frac>`: same exact budget as topk_hard@<frac>, but the
        # budget is spent on `gate` pixels first (cross-signal agreement).
        # A gate must be supplied; without one this strategy is undefined.
        if gate is None:
            raise ValueError(
                f"strategy {name!r} requires a gate surface (agreement channels "
                f"from the feature stack)")
        frac = float(name.split("@", 1)[1])
        p = np.clip(np.nan_to_num(prob, nan=0.0), 0.0, 1.0)
        return np.where(gated_topk(p, valid, gate, frac), 1.0, 0.0).astype(np.float32)
    if name == "densified":
        return densify_along_lineaments(
            np.where(valid, np.clip(np.nan_to_num(prob, nan=0.0), 0, 1), 0.0),
            valid, threshold, dilation=1)
    if name in ("skeleton", "skeleton_spaced"):
        p = np.clip(np.nan_to_num(prob, nan=0.0), 0.0, 1.0)
        p = np.where(valid, p, 0.0)
        strong = (p >= threshold) & valid
        keep = (_skeleton(strong) if name == "skeleton"
                else thin_keep(strong, spacing))
        return np.where(keep & valid, p, 0.0).astype(np.float32)
    if name.startswith("top"):
        p = np.clip(np.nan_to_num(prob, nan=0.0), 0.0, 1.0)
        p = np.where(valid, p, 0.0)
        q = float(np.quantile(p[valid], max(0.0, 1.0 - top_fraction))) if valid.any() else 1.0
        return np.where(p >= q, p, 0.0).astype(np.float32)
    raise KeyError(f"unknown placement strategy: {name}")


def strategies(prob: np.ndarray, valid: np.ndarray, threshold: float,
               spacing: int = 4, top_fraction: float = 0.01) -> dict[str, np.ndarray]:
    """Named placements. Every value is 0 outside `valid`, so all of them pass
    the submission gate's NaN rule by construction."""
    p = np.clip(np.nan_to_num(prob, nan=0.0), 0.0, 1.0)
    p = np.where(valid, p, 0.0)

    out: dict[str, np.ndarray] = {"raw": p}
    out["threshold"] = np.where(p >= threshold, p, 0.0)

    if valid.any():
        q = float(np.quantile(p[valid], max(0.0, 1.0 - top_fraction)))
    else:
        q = 1.0
    out[f"top{int(round(top_fraction * 100))}pct"] = np.where(p >= q, p, 0.0)

    strong = (p >= threshold) & valid
    out["skeleton"] = np.where(_skeleton(strong) & valid, p, 0.0).astype(np.float32)
    out[f"skeleton_spaced{spacing}"] = np.where(
        thin_keep(strong, spacing) & valid, p, 0.0).astype(np.float32)
    out["densified"] = densify_along_lineaments(p, valid, threshold, dilation=1)
    return out
