"""H13/H16 alignment engine: strip NCC (texture displacement) and ray continuation.

Pre-registration and scientific rationale: HYPOTHESES.md (session-2 register).
These are morphological/geophysical operators. They are NOT validated fault or
geothermal-resource classifiers.

Core idea (H13)
---------------
A strike-slip fault laterally displaces geophysical marker patterns (magnetic
lineaments, gravity contacts, topographic lines). For a candidate trace with a
given strike, the marker pattern on one side of the trace should re-appear on
the other side at a NONZERO along-strike lag. We measure the zero-difference
normalized cross-correlation (NCC) between the two marker strips as a function
of along-strike lag. A displacement produces best alignment at |lag| > 0; a
simple boundary/valley does not.

Implementation notes
--------------------
* Strips for 0/90-degree strikes are exact rows/columns. 45/135 degrees use one
  order-1 bilinear rotation of the band (documented approximation; the local
  per-pixel operator is then identical in rotated coordinates).
* NCC is computed with a local window of half-length `window` along strike:
  values are window-centered, and the dot products / energy terms are local
  window means (centered uniform filter). Zero padding near grid edges and
  around NaNs is invalidated with a conservative halo, never silently kept.
* Lag shifts are edge-aware (no wrap); shifted-in regions are NaN.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# Pre-registered constants (HYPOTHESES.md, session 2, item 1, as calibrated
# on a synthetic white-noise NULL field before any contact with holdout data:
# dense lags 1-6 with window 8 gave mean adv ~ +0.35 on pure noise — a
# max-selection bias. Lags {2,4,6} px (200/400/600 m, the geological offset
# range) with window 16 px (3.3 km) reduce the null mean adv to ~0.)
WINDOW = 16         # px, half-length along strike
SEPS = (1, 2, 3)    # px, strip separation across the candidate trace
LAGS = (2, 4, 6)    # px, nonzero lags considered


def _window_center(x: np.ndarray, w: int, axis: int) -> np.ndarray:
    """x minus its local window mean along `axis` (centered window)."""
    m = ndimage.uniform_filter1d(x, size=w, axis=axis, mode="constant", cval=0.0)
    return x - m


def _local_mean(a: np.ndarray, w: int, axis: int) -> np.ndarray:
    return ndimage.uniform_filter1d(a, size=w, axis=axis, mode="constant", cval=0.0)


def _invalid_halo(mask: np.ndarray, halo: int) -> np.ndarray:
    if halo <= 0 or not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    return ndimage.binary_dilation(mask, iterations=halo)


def strip_ncc(a: np.ndarray, b: np.ndarray, window: int = WINDOW,
              axis: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-difference NCC between two same-shape arrays at the preregistered
    lags (0 and LAGS).

    `a` and `b` are the two marker strips (e.g. rows i and i+d of a band).
    Returns per pixel (ncc0, best_nonzero_ncc, mean_nonzero_ncc); NaN where
    the window is invalid. Both inputs must be NaN-free (0-filled by caller).

    Statistic notes (measured on a white-noise null field before data contact):
    ncc_best carries a positive max-selection bias (documented); the unbiased
    displacement contrast is (mean of nonzero-lag NCC) minus ncc0 — a true
    lateral offset dips at zero lag while matching at the offset lags.
    """
    if a.shape != b.shape:
        raise ValueError("strip arrays must share shape")
    w = 2 * window + 1
    ac = _window_center(a, w, axis)
    aa = _local_mean(ac * ac, w, axis)
    eps = 1e-12
    out0 = np.full(a.shape, np.nan, dtype=np.float32)
    best = np.full(a.shape, np.nan, dtype=np.float32)
    sum_nz = np.zeros(a.shape, dtype=np.float64)
    n_nz = 0
    # Edge halo: windows touching the grid border are invalid.
    border = np.zeros(a.shape, dtype=bool)
    border[:window + 1, :] = True; border[-window - 1:, :] = True
    border[:, :window + 1] = True; border[:, -window - 1:] = True
    for lag in (0,) + tuple(LAGS):
        if lag == 0:
            bc = _window_center(b, w, axis)
        else:
            # b shifted by -lag along axis (no wrap); shifted-in zone invalid.
            bs = np.full(b.shape, 0.0, dtype=b.dtype)
            sl = [slice(None)] * b.ndim
            src = [slice(None)] * b.ndim
            if axis == 1:
                sl[1] = slice(0, b.shape[1] - lag); src[1] = slice(lag, b.shape[1])
            else:
                sl[0] = slice(0, b.shape[0] - lag); src[0] = slice(lag, b.shape[0])
            bs[tuple(sl)] = b[tuple(src)]
            shifted_in = np.zeros(b.shape, dtype=bool)
            if axis == 1:
                shifted_in[:, b.shape[1] - lag:] = True
            else:
                shifted_in[b.shape[0] - lag:, :] = True
            bc = _window_center(bs, w, axis)
            bc = np.where(shifted_in, 0.0, bc)
        ab = _local_mean(ac * bc, w, axis)
        bb = _local_mean(bc * bc, w, axis)
        with np.errstate(invalid="ignore", divide="ignore"):
            ncc = ab / np.sqrt(aa * bb + eps)
        ncc = np.where((aa <= eps) | (bb <= eps), np.nan, ncc)
        ncc = np.where(border, np.nan, ncc)
        if lag == 0:
            out0 = ncc
        else:
            n_nz += 1
            finite = np.isfinite(ncc)
            better = (np.isnan(best) | (ncc > best)) & finite
            best[better] = ncc[better]
            sum_nz = np.where(finite, sum_nz + np.where(finite, ncc, 0.0),
                              sum_nz)
            sum_nz = sum_nz  # accumulate only where finite; count below
    mean_nz = np.full(a.shape, np.nan, dtype=np.float64)
    # Per-pixel count of finite nonzero lags (a lag column is either fully
    # finite or fully invalid, except the shifted-in zone, handled above).
    valid_nz = np.isfinite(sum_nz)
    mean_nz[valid_nz] = sum_nz[valid_nz] / max(n_nz, 1)
    return out0, best, mean_nz.astype(np.float32)


def _rotated(band: np.ndarray, angle_deg: float) -> np.ndarray:
    return ndimage.rotate(band, angle_deg, order=1, reshape=False,
                          mode="constant", cval=0.0)


def strip_triples(arr: np.ndarray, d: int, window: int = WINDOW):
    """Strip pair (row i, row i-d) NCC triplets for a strip-space tile.

    `arr` is the strip space (strips = its rows); returns (ncc0, best,
    mean_nz) per pixel.
    """
    b = np.zeros_like(arr)
    b[d:, :] = arr[:-d, :]
    return strip_ncc(arr, b, window=window, axis=1)


def _aggregate_angle(per_d: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-angle (ncc0*, best*, adv_pool) from per-separation triples.

    adv_pool = mean over separations of (mean nonzero-lag NCC - ncc0) — the
    unbiased zero-lag dip contrast; ncc0*/best* are reported from the
    separation with the strongest contrast at each pixel.
    """
    adv_d = [np.clip(m - z, -1.0, 1.0) for z, _b, m in per_d]
    adv_pool = np.nanmean(np.stack(adv_d, axis=-1), axis=-1)
    stacked = np.stack([np.where(np.isfinite(a), a, -2.0) for a in adv_d], axis=-1)
    dstar = np.nanargmax(stacked, axis=-1)
    z_star = np.take_along_axis(np.stack([z for z, _b, _m in per_d], axis=-1),
                                dstar[..., None], axis=-1)[..., 0]
    b_star = np.take_along_axis(np.stack([b for _z, b, _m in per_d], axis=-1),
                                dstar[..., None], axis=-1)[..., 0]
    z_star = np.clip(np.where(np.isfinite(z_star), z_star, np.nan), -1.0, 1.0)
    b_star = np.clip(np.where(np.isfinite(b_star), b_star, np.nan), -1.0, 1.0)
    adv_pool = np.where(np.isfinite(adv_pool), adv_pool, np.nan)
    return z_star, b_star, adv_pool


def strip_offset_features(band: np.ndarray, angles: tuple[float, ...] = (0, 45, 90, 135),
                          seps: tuple[int, ...] = SEPS,
                          window: int = WINDOW) -> dict[tuple[float, int], tuple[np.ndarray, ...]]:
    """Per (angle, sep) strip-NCC triplets for one NaN-free band.

    Caller is responsible for 0-filling NaNs and invalidating the resulting
    halo (see offset_features()).
    """
    res = {}
    for ang in angles:
        if ang in (45, 135):
            arr = _rotated(band, ang)
        else:
            arr = band
        space = arr.T if ang == 90 else arr  # 90: strips are columns
        for d in seps:
            triple = strip_triples(space, d, window=window)
            if ang == 90:
                triple = tuple(t.T for t in triple)
            res[(ang, d)] = triple
    return res


def offset_channels(band: np.ndarray, angles: tuple[float, ...] = (0, 45, 90, 135),
                    seps: tuple[int, ...] = SEPS,
                    window: int = WINDOW) -> np.ndarray:
    """Per-band H13 channel block (full-grid; use offset_channels_tiled for
    the 12.3-px-full raster to bound memory).

    band: (H, W) float with NaN where invalid.
    Returns (H, W, 3*n_angles + 1) float32: per angle (ncc0, ncc_best, adv)
    where adv is the unbiased zero-lag-dip contrast (see strip_ncc), plus one
    per-band aggregate (max adv over angles). HYPOTHESES.md item 1; with the
    preregistered 3 bands this is 39 channels total. A true lateral offset
    shows HIGH adv; noise and symmetric boundaries show adv ~ 0.
    """
    band = np.asarray(band, dtype=np.float32)
    bad = ~np.isfinite(band)
    filled = np.where(bad, 0.0, band)
    res = strip_offset_features(filled, angles=angles, seps=seps, window=window)
    halo = window + max(seps) + max(LAGS) + 3
    invalid = _invalid_halo(bad, halo)
    n_ang = len(angles)
    out = np.full(bad.shape + (n_ang * 3 + 1,), np.nan, dtype=np.float32)
    advs = []
    for k, ang in enumerate(angles):
        z_star, b_star, adv_pool = _aggregate_angle(
            [res[(ang, d)] for d in seps])
        out[..., k * 3 + 0] = z_star
        out[..., k * 3 + 1] = b_star
        out[..., k * 3 + 2] = adv_pool
        advs.append(adv_pool)
    with np.errstate(invalid="ignore", all="ignore"):
        out[..., n_ang * 3] = np.nanmax(np.stack(advs, axis=-1), axis=-1)
    out = np.where(invalid[..., None], np.nan, out)
    return out.astype(np.float32)


def offset_channels_tiled(band: np.ndarray,
                          angles: tuple[float, ...] = (0, 45, 90, 135),
                          seps: tuple[int, ...] = SEPS,
                          window: int = WINDOW,
                          block_rows: int = 384,
                          out: np.ndarray | None = None) -> np.ndarray:
    """Memory-tiled per-band channel block; bit-identical to
    offset_channels (verified by tests).

    The strip window runs along the strip row (strike) axis, so a block of
    strip-space rows with P = window+1 padded rows on each side reproduces
    the full-array computation exactly: cross-strip dependence is only the
    (window+1) border halo and the <= max(seps) strip pairing.

    `out` may be a writable array or memmap slice of shape
    (H, W, 3*n_angles+1); it is written in place and returned.
    """
    band = np.asarray(band, dtype=np.float32)
    bad = ~np.isfinite(band)
    filled = np.where(bad, 0.0, band)
    halo = window + max(seps) + max(LAGS) + 3
    invalid = _invalid_halo(bad, halo)
    n_ang = len(angles)
    agg_ch = n_ang * 3
    if out is None:
        out = np.full(bad.shape + (n_ang * 3 + 1,), np.nan, dtype=np.float32)
    # Aggregate (max adv over angles) accumulates with a NaN-safe running
    # max; -2.0 is the "no angle seen yet" sentinel (adv is clipped to [-1,1]).
    out[..., agg_ch] = np.float32(-2.0)
    P = window + 1
    for k, ang in enumerate(angles):
        if ang in (45, 135):
            space = _rotated(filled, ang)
        else:
            space = filled.T if ang == 90 else filled
        SH = space.shape[0]
        for r0 in range(0, SH, block_rows):
            r1 = min(r0 + block_rows, SH)
            rs, re = max(0, r0 - P), min(SH, r1 + P)
            tile = np.ascontiguousarray(space[rs:re])
            z, b, adv = _aggregate_angle([strip_triples(tile, d, window=window)
                                          for d in seps])
            off = r0 - rs
            z = z[off:off + (r1 - r0)]
            b = b[off:off + (r1 - r0)]
            adv = adv[off:off + (r1 - r0)]
            if ang == 90:  # strip space rows are band columns
                out[:, r0:r1, k * 3 + 0] = z.T
                out[:, r0:r1, k * 3 + 1] = b.T
                out[:, r0:r1, k * 3 + 2] = adv.T
                np.fmax(out[:, r0:r1, agg_ch], adv.T,
                        out=out[:, r0:r1, agg_ch])
            else:
                out[r0:r1, :, k * 3 + 0] = z
                out[r0:r1, :, k * 3 + 1] = b
                out[r0:r1, :, k * 3 + 2] = adv
                np.fmax(out[r0:r1, :, agg_ch], adv,
                        out=out[r0:r1, :, agg_ch])
        if ang in (45, 135):
            del space
            import gc as _gc
            _gc.collect()
    agg = out[..., agg_ch]
    out[..., agg_ch] = np.where(agg == np.float32(-2.0), np.nan, agg)
    out[invalid] = np.nan
    return out  # float32 by construction (band cast + float32 ops)


# ------------------------------------------------------------------ H16
RAY_RADII = (5, 15, 30, 60)      # px (0.5 / 1.5 / 3 / 6 km)
NCC_RADII = (15, 30)             # px, corridor-continuity sampling radii
NCC_HALF = 16                     # samples each side of the ray point
CONTROL_OFFSET = 6                # px, perpendicular control-ray offset
DISK = 3                          # px radius when stamping ray points


def _sample_1d(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bilinear sample; NaN if any corner of the bilinear quad is NaN."""
    from scipy.ndimage import map_coordinates
    coords = np.vstack([rows, cols])
    v = map_coordinates(arr, coords, order=1, mode="constant", cval=np.nan)
    return v


def _ncc1d(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n = min(x.size, y.size)
    if n < 4:
        return np.nan
    x, y = x[:n], y[:n]
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    if not np.isfinite(denom) or denom < 1e-12:
        return np.nan
    return float((xc * yc).sum() / denom)


def ray_continuation_channels(band: np.ndarray,
                              lineaments: dict,
                              systems,
                              radii: tuple[int, ...] = RAY_RADII,
                              ncc_rays: tuple[int, ...] = NCC_RADII,
                              ncc_band: np.ndarray | None = None,
                              min_n_px: int = 8,
                              max_width: float = 20.0) -> tuple[np.ndarray, list[str]]:
    """H16: endpoint-conditioned continuation evidence (pre-registered).

    band/ncc_band: (H,W) float (NaN ok; 0-filled internally for sampling).
    lineaments: {name: Lineament} with .energy/.coherence/.orientation.
    systems: list[Strike] (TRAIN systems only for CV; all systems for final).
    Returns (H, W, C) float32 (NaN where no ray covers) and channel names.
    """
    from .features import Lineament  # noqa: F401  (type hint only)
    band = np.asarray(band, dtype=np.float32)
    H, W = band.shape
    names: list[str] = []
    per: dict[str, list[np.ndarray]] = {}
    for lname, lin in lineaments.items():
        assert isinstance(lin, Lineament)
        for t in radii:
            ch = f"cont_{lname}_r{t}"
            names.append(ch)
            per[ch] = np.zeros((H, W), dtype=np.float32)
    ncb = np.asarray(ncc_band if ncc_band is not None else band, dtype=np.float32)
    for t in ncc_rays:
        ch = f"cont_ncc_r{t}"
        names.append(ch)
        per[ch] = np.zeros((H, W), dtype=np.float32)
    covered = np.zeros((H, W), dtype=bool)

    bad = ~np.isfinite(band)
    filled = np.where(bad, 0.0, band)
    nbad = ~np.isfinite(ncb)
    ncb_filled = np.where(nbad, 0.0, ncb)

    for s in systems:
        if s.n_px < min_n_px or s.width > max_width:
            continue
        vx, vy = float(np.cos(s.angle)), float(np.sin(s.angle))  # (col, row)
        cr, cc = s.centroid
        for (er, ec) in (s.end_a, s.end_b):
            # Outward direction: away from the centroid through the endpoint.
            dr, dc = er - cr, ec - cc
            norm = np.hypot(dr, dc)
            if norm < 1e-9:
                continue
            ur, uc = dr / norm, dc / norm
            # Safety: the ray must leave roughly along the strike. Strike in
            # (row, col) space is (vy, vx) (discovery.py: vx=col, vy=row).
            if abs(ur * vy + uc * vx) < 0.5:
                ur, uc = -ur, -uc
            for t in set(radii) | set(ncc_rays):
                pr, pc = er + t * ur, ec + t * uc
                if not (1 <= pr < H - 1 and 1 <= pc < W - 1):
                    continue
                pts_r = [int(np.clip(round(pr), 0, H - 1))]
                pts_c = [int(np.clip(round(pc), 0, W - 1))]
                for i in range(-DISK, DISK + 1):
                    for j in range(-DISK, DISK + 1):
                        if i * i + j * j <= DISK * DISK:
                            rr, ccj = int(round(pr)) + i, int(round(pc)) + j
                            if 0 <= rr < H and 0 <= ccj < W:
                                pts_r.append(rr); pts_c.append(ccj)
                pts_r = np.array(pts_r); pts_c = np.array(pts_c)
                for lname, lin in lineaments.items():
                    if t not in radii:
                        continue
                    ch = f"cont_{lname}_r{t}"
                    e = lin.energy[pts_r, pts_c]
                    c = lin.coherence[pts_r, pts_c]
                    o = lin.orientation[pts_r, pts_c]
                    match = np.cos(2.0 * (o - s.angle))
                    score = np.where(np.isfinite(e) & np.isfinite(c),
                                     e * np.clip(c, 0, 1) * np.clip(match, 0, 1),
                                     np.nan)
                    score = np.nanmax(score)
                    if np.isfinite(score):
                        np.maximum.at(per[ch], (pts_r, pts_c),
                                      np.full_like(pts_r, score, dtype=np.float32))
                        covered[pts_r, pts_c] = True
                for t2 in ncc_rays:
                    if t != t2:
                        continue
                    ch = f"cont_ncc_r{t2}"
                    rows = pr + (np.arange(-NCC_HALF, NCC_HALF + 1) * ur)
                    cols = pc + (np.arange(-NCC_HALF, NCC_HALF + 1) * uc)
                    sig = _sample_1d(ncb_filled, rows, cols)
                    # perpendicular offset ray (control)
                    rows_c = rows + CONTROL_OFFSET * uc
                    cols_c = cols - CONTROL_OFFSET * ur
                    sig_c = _sample_1d(ncb_filled, rows_c, cols_c)
                    n = 2 * NCC_HALF + 1
                    front, back = sig[:n // 2], sig[n // 2 + 1:]
                    cf, cb = sig_c[:n // 2], sig_c[n // 2 + 1:]
                    c_main = _ncc1d(front, back)
                    c_ctrl = _ncc1d(cf, cb)
                    if not np.isfinite(c_main):
                        continue
                    # Documented convention: a control ray with no measurable
                    # texture (zero variance) contributes nothing to subtract.
                    v = c_main - (c_ctrl if np.isfinite(c_ctrl) else 0.0)
                    if np.isfinite(v):
                        np.maximum.at(per[ch], (pts_r, pts_c),
                                      np.clip(v, -1.0, 1.0).astype(np.float32))
                        covered[pts_r, pts_c] = True
    out = np.stack([arr for arr in per.values()], axis=-1)
    out = np.where(covered[..., None], out, np.nan)
    return out.astype(np.float32), names
