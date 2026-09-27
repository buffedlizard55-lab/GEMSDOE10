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


def family_agreement(X: np.ndarray, channels: list[str],
                     z_thresh: float = 1.0) -> np.ndarray:
    """Count of families (0..6) with max channel z-score above z_thresh.

    X is (n, C) finite-imputed; z-scores use column nanmean/nanstd computed on
    the passed sample (callers pass the full-grid sample so statistics are
    stable and leak-free — no labels involved).
    """
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Z = (np.where(np.isfinite(X), X, mu) - mu) / sd
    fams = channel_families(channels)
    agree = np.zeros(X.shape[0], dtype=np.int8)
    for _, idxs in fams.items():
        if not idxs:
            continue
        m = np.abs(Z[:, idxs]).max(axis=1)
        agree += (m > z_thresh).astype(np.int8)
    return agree


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
