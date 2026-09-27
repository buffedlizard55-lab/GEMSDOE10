"""Semi-supervised second pass (pseudo-positives from held-out blocks).

NEXT_STEPS item 6 (still open in r6): "build a semi-supervised target: the
model's own high-confidence predictions in held-out blocks, verified against
independent signals (magnetics + gravity + strain + seismicity agreeing),
used as additional positives for a second pass".

Protocol (leak-free inside system-holdout CV):
  1. pass-1 model trained on train systems predicts the full grid;
  2. candidate pseudo-positives = trainable pixels OUTSIDE train systems with
     pass-1 probability above quantile q (default top 0.5%), restricted to
     pixels where >= k of the 6 source families independently agree
     (family evidence = max channel z-score within the family);
  3. pass-2 model trains on train positives + pseudo-positives (weight 0.5)
     + negatives, and is scored on the held-out systems like pass 1.

Kept only if the harness measures a gain on held-out systems.
"""

from __future__ import annotations

import numpy as np

# channel-name -> family, using the official band names + derived prefixes
FAMILY_OF_PREFIX = (
    ("mag", ("mag_anom", "rtp", "tmi", "mag_asa", "mag_tilt", "tmi_hgm",
             "tmi_asa", "rtp_hgm", "mag_anom_hgm", "frangi_tmi", "frangi_rtp",
             "gabor_tmi", "hgm_tmi", "hgm_rtp", "hgm_mag_anom", "vg_tmi",
             "asa_tmi", "tdr_tmi", "std_tmi", "lin_tmi", "lin_rtp", "coh_tmi")),
    ("grav", ("iso_grav_anom", "grav_asa", "grav_tilt", "grav_hgm",
              "frangi_iso_grav", "hgm_iso_grav", "vg_iso_grav", "asa_iso_grav",
              "tdr_iso_grav", "std_iso_grav", "lin_grav", "coh_iso_grav")),
    ("strain", ("geod_2ndinv", "geod_shearrate", "geod_dilaterate",
               "x_geod_shearrate")),
    ("seis", ("ieq_n100a15", "deq_n100a15", "x_ieq_n100a15")),
    ("cond", ("cond_surf", "depth_to_base_surf", "x_cond_surf", "lin_cond")),
    ("topo", ("det_elev", "curv_", "slope", "lin_elev", "frangi_det_elev",
              "gabor_det_elev", "hgm_det_elev", "std_det_elev", "coh_det_elev")),
)


def channel_families(channels: list[str]) -> dict[str, list[int]]:
    """Map family -> channel indices (a channel belongs to the first matching family)."""
    fams: dict[str, list[int]] = {name: [] for name, _ in FAMILY_OF_PREFIX}
    for i, ch in enumerate(channels):
        for name, prefixes in FAMILY_OF_PREFIX:
            if ch == "tc":
                fams["mag"].append(i)
                break
            if ch.startswith(prefixes):
                fams[name].append(i)
                break
    return fams


def fit_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column nanmean/nanstd with degenerate guards (leak-free: no labels)."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return mu, sd


def family_agreement_with_stats(X: np.ndarray, channels: list[str],
                                mu: np.ndarray, sd: np.ndarray,
                                z_thresh: float = 1.0) -> np.ndarray:
    """Count of families (0..6) with max channel |z| above z_thresh."""
    Z = (np.where(np.isfinite(X), X, mu) - mu) / sd
    fams = channel_families(channels)
    agree = np.zeros(X.shape[0], dtype=np.int8)
    for _, idxs in fams.items():
        if not idxs:
            continue
        m = np.abs(Z[:, idxs]).max(axis=1)
        agree += (m > z_thresh).astype(np.int8)
    return agree


def family_agreement(X: np.ndarray, channels: list[str],
                     z_thresh: float = 1.0) -> np.ndarray:
    """Count of families (0..6) with max channel z-score above z_thresh.

    X is (n, C) finite-imputed; z-scores use column nanmean/nanstd computed on
    the passed sample (callers pass the full-grid sample so statistics are
    stable and leak-free — no labels involved).
    """
    mu, sd = fit_stats(X)
    return family_agreement_with_stats(X, channels, mu, sd, z_thresh)


def agreement_grid(feat: np.ndarray, footprint: np.ndarray, channels: list[str],
                   z_thresh: float = 1.0, batch_rows: int = 256,
                   stats_n: int = 500_000, seed: int = 0) -> np.ndarray:
    """Full-grid family-agreement map, row-batched (low peak RAM).

    z-statistics come from a `stats_n` random footprint subset (stable and
    leak-free); the grid is then scored in `batch_rows` row blocks so peak
    memory stays near (batch_rows × W × C) instead of (5.2M × C). The
    unbatched version OOM-killed a 3.8 GB host (2026-09-27) — always use this.
    """
    fp = np.asarray(footprint, dtype=bool)
    h, w = fp.shape
    ys, xs = np.nonzero(fp)
    rng = np.random.default_rng(seed)
    sel = rng.choice(ys.size, size=min(stats_n, ys.size), replace=False)
    mu, sd = fit_stats(feat[ys[sel], xs[sel]].astype(np.float32))
    out = np.zeros((h, w), dtype=np.int8)
    for r0 in range(0, h, batch_rows):
        r1 = min(r0 + batch_rows, h)
        blk = fp[r0:r1]
        if not blk.any():
            continue
        X = feat[r0:r1].reshape(-1, feat.shape[2]).astype(np.float32)
        a = family_agreement_with_stats(X, channels, mu, sd, z_thresh)
        out[r0:r1][blk] = a[blk.ravel()]
    return out


def select_pseudo(prob: np.ndarray, eligible: np.ndarray,
                  agreement: np.ndarray | None, top_frac: float = 0.005,
                  min_agree: int = 2) -> np.ndarray:
    """Boolean pseudo-positive mask over the grid.

    Eligible = trainable & footprint & ~train_systems (i.e. NOT already a
    labelled positive, NOT near held-out GT). Takes the top `top_frac` of
    eligible pixels by probability, then requires family agreement >= min_agree
    when an agreement grid is supplied.
    """
    p = np.asarray(prob, dtype=np.float64)
    elig = np.asarray(eligible, dtype=bool)
    vals = p[elig]
    if vals.size == 0:
        return np.zeros_like(elig)
    thr = np.quantile(vals, 1.0 - top_frac)
    sel = elig & (p >= thr)
    if agreement is not None:
        sel &= np.asarray(agreement) >= min_agree
    return sel
