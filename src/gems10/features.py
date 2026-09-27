# Provenance: adapted from buffedlizard55-lab/6GEMSDOE src/gems (same owner, re-verified here).
"""Derived structural features for fault detection.

Rationale, grounded in the official task description and in the literature the
competition's own About page cites:

  * The competition page lists the provided bands and states that the target is a
    fault TRACE — a line — and that faults are the structural marker of geothermal
    systems. Everything here therefore aims at *edges* and *lineaments*, not at
    reproducing the catalogue.
  * The task brief asks specifically for: horizontal gradient magnitude and tilt
    derivative of the potential-field layers; curvature and breaks-in-slope from
    detrended elevation; and cross-checks between strain rate, conductivity and
    earthquake density. Those are implemented below.
  * Reference-comparable prior art cited by the organisers: Mattéo et al. (2021,
    doi:10.1029/2020JB021269) use topography/optical data with deep learning for
    automatic fault mapping; Hermant et al. (2025, Stanford 45th/50th Workshop on
    Geothermal Reservoir Engineering) map Quaternary faults in the western USA
    with deep learning. Classical edge mapping on potential fields uses the tilt
    angle (Miller & Singh 1994) and the analytic-signal amplitude (Roest et al.
    1992; Nabighian 1972) — both are cheap, scale-explicit and physically
    interpretable, which is what we want given no GPU here.

Two important measured facts drive NaN handling in this module (see VERIFICATION.md):

  1. The feature GeoTIFF does not use NaN for holes; it uses the float32
     most-negative sentinel -3.4028234663852886e+38 (the reference solution masks
     it with `X[X < -1e38] = np.nan`). Any arithmetic done on an unmasked sentinel
     produces garbage, so `load_bands()` masks it immediately.
  2. Only 42.07% of pixels are finite in all 19 bands, and that footprint is
     slightly *smaller* than the official submission footprint (5,165,840 vs
     5,167,373 px). Filtering must therefore never move a valid pixel outward.

Derivative operators are central differences with a halo rule: an output pixel is
marked invalid if any input pixel its stencil touched was invalid. This keeps
edge features from smearing invalid data into the scored area.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spec

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

from scipy import ndimage

B = spec.BAND_INDEX  # band-name -> 1-based TIFF band index


# --------------------------------------------------------------------------- IO


def load_bands(path: str, names: list[str], window=None) -> tuple[dict[str, np.ndarray],
                                                                np.ndarray]:
    """Read and mask selected official bands. Returns (bands, invalid)."""
    if rasterio is None:  # pragma: no cover
        raise SystemExit("rasterio required")
    out: dict[str, np.ndarray] = {}
    invalid = None
    with rasterio.open(path) as src:
        for name in names:
            idx = B[name]
            arr = src.read(idx, window=window).astype(np.float32)
            bad = arr < spec.FEATURE_INVALID_BELOW
            arr = np.where(bad, np.nan, arr)
            out[name] = arr
            invalid = bad if invalid is None else (invalid | bad)
    return out, invalid


# ------------------------------------------------------------------- filtering


def _guarded_filter(arr: np.ndarray, func, radius: int) -> np.ndarray:
    """Apply `func` to `arr` (NaN->0) and re-invalidate the affected halo.

    A pixel is marked NaN if any pixel within `radius` of it (the stencil reach)
    was NaN on input. Without this, an edge filter next to the data boundary
    invents structure out of the sentinel / fill value.
    """
    bad = ~np.isfinite(arr)
    filled = np.where(bad, 0.0, arr)
    res = func(filled)
    if radius > 0 and bad.any():
        grown = ndimage.binary_dilation(bad, iterations=radius)
        res = np.where(grown, np.nan, res)
    else:
        res = np.where(bad, np.nan, res)
    return res


def nan_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    r = int(np.ceil(3 * sigma))
    return _guarded_filter(arr, lambda a: ndimage.gaussian_filter(a, sigma), r)


def nan_uniform(arr: np.ndarray, size: int) -> np.ndarray:
    r = size // 2
    return _guarded_filter(arr, lambda a: ndimage.uniform_filter(a, size), r)


def local_std(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Local standard deviation over a Gaussian window (NaN-guarded).

    A textural rather than a geometric channel: fault zones and their damage
    zones show up as locally rougher potential fields / rougher ground than the
    surrounding block, independently of the sign of the anomaly.
    """
    r = int(np.ceil(3 * sigma))
    m1 = nan_gaussian(arr, sigma)
    m2 = nan_gaussian(np.where(np.isfinite(arr), arr * arr, np.nan), sigma)
    with np.errstate(invalid="ignore"):
        var = m2 - m1 * m1
    out = np.sqrt(np.where(var > 0.0, var, 0.0))
    bad = ~np.isfinite(arr)
    if bad.any():
        out = np.where(ndimage.binary_dilation(bad, iterations=r), np.nan, out)
    return out


def curvature_at_scale(arr: np.ndarray, sigma: float,
                       spacing: float = spec.PIXEL_SIZE_M) -> Curvature:
    """`curvature()` of a Gaussian-smoothed surface.

    Smoothing first is what makes curvature a *scale-explicit* measurement: at
    sigma = 1.5 px (150 m) it responds to scarp-scale breaks, at sigma = 3 px
    (300 m) to the broader monocline/fault-block flexure. Raw second derivatives
    of a 100 m DEM-derived layer are dominated by pixel noise.
    """
    return curvature(nan_gaussian(arr, sigma), spacing)


def derivatives(arr: np.ndarray, spacing: float = spec.PIXEL_SIZE_M
                ) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference d/dx and d/dy in metres, NaN-propagating."""
    bad = ~np.isfinite(arr)
    a = np.where(bad, 0.0, arr)
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2.0 * spacing)
    gy[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2.0 * spacing)
    # one-sided at the frame, so the frame is not silently zero
    gx[:, 0] = (a[:, 1] - a[:, 0]) / spacing
    gx[:, -1] = (a[:, -1] - a[:, -2]) / spacing
    gy[0, :] = (a[1, :] - a[0, :]) / spacing
    gy[-1, :] = (a[-1, :] - a[-2, :]) / spacing
    halo = ndimage.binary_dilation(bad, iterations=1) if bad.any() else bad
    return np.where(halo, np.nan, gx), np.where(halo, np.nan, gy)


def second_derivatives(arr: np.ndarray, spacing: float = spec.PIXEL_SIZE_M):
    """r = d2/dx2, t = d2/dy2, s = d2/dxdy (NaN-propagating, radius-1 stencil)."""
    bad = ~np.isfinite(arr)
    a = np.where(bad, 0.0, arr)
    r = np.zeros_like(a)
    t = np.zeros_like(a)
    s = np.zeros_like(a)
    h = spacing * spacing
    r[:, 1:-1] = (a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]) / h
    t[1:-1, :] = (a[2:, :] - 2 * a[1:-1, :] + a[:-2, :]) / h
    s[1:-1, 1:-1] = (a[2:, 2:] - a[2:, :-2] - a[:-2, 2:] + a[:-2, :-2]) / (4.0 * h)
    halo = ndimage.binary_dilation(bad, iterations=1) if bad.any() else bad
    return (np.where(halo, np.nan, r), np.where(halo, np.nan, t),
            np.where(halo, np.nan, s))


# ------------------------------------------------- potential-field edge tools


def analytic_signal(derivs: list[np.ndarray]) -> np.ndarray:
    """sqrt(sum d_i^2) — analytic-signal amplitude (Nabighian 1972; Roest 1992)."""
    acc = None
    for d in derivs:
        term = np.where(np.isfinite(d), d * d, np.nan)
        acc = term if acc is None else acc + term
    return np.sqrt(acc)


def tilt_angle(vertical: np.ndarray, horizontal: np.ndarray) -> np.ndarray:
    """atan2(vertical derivative, horizontal derivative magnitude).

    Miller & Singh (1994). Zero-crossings of the tilt locate the edge; the
    magnitude is dimensionless, which makes it usable across contrasting rock
    units without per-unit calibration.
    """
    return np.arctan2(vertical, horizontal)


def horizontal_gradient_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """HGM = |(dz/dx, dz/dy)| — requested explicitly in the task brief."""
    return np.sqrt(np.where(np.isfinite(gx), gx * gx, np.nan)
                   + np.where(np.isfinite(gy), gy * gy, np.nan))


# ------------------------------------------------------------------ curvature


@dataclass
class Curvature:
    total: np.ndarray      # Laplacian: sum of principal curvatures (x)
    profile: np.ndarray    # curvature in the downslope direction
    plan: np.ndarray       # curvature in the contour (cross-slope) direction
    gaussian: np.ndarray   # r*t - s^2
    slope: np.ndarray      # |grad z|
    slope_of_slope: np.ndarray  # |grad slope| — "break in slope" indicator


def curvature(det_elev: np.ndarray, spacing: float = spec.PIXEL_SIZE_M) -> Curvature:
    """Curvature of the detrended-elevation surface.

    Formulations follow the standard terrain-analysis definitions (Zevenbergen &
    Thorne 1987; Moore, Grayson & Ladson 1991) once terrain is in metres and the
    gradient is dimensionless (dz/dx with both in metres). Signs: positive
    profile curvature = convex (accelerating) downslope — for fault-scarp
    detection the *magnitude* is what matters, and both signs are informative
    because a scarp is convex on one side and concave on the other.
    """
    p, q = derivatives(det_elev, spacing)
    r, t, s = second_derivatives(det_elev, spacing)
    with np.errstate(invalid="ignore", divide="ignore"):
        grad2 = p * p + q * q
        denom_prof = grad2 * np.power(1.0 + grad2, 1.5)
        profile = -(r * p * p + 2.0 * s * p * q + t * q * q) / denom_prof
        plan = (t * p * p - 2.0 * s * p * q + r * q * q) / np.power(grad2, 1.5)
        total = r + t
        gaussian = r * t - s * s
        slope = np.sqrt(grad2)
    slope = np.where(np.isfinite(slope), slope, np.nan)
    gsx, gsy = derivatives(slope, spacing)
    return Curvature(total=total, profile=profile, plan=plan, gaussian=gaussian,
                     slope=slope,
                     slope_of_slope=horizontal_gradient_magnitude(gsx, gsy))


# ------------------------------------------------------- structure tensor / coherence


@dataclass
class Lineament:
    energy: np.ndarray       # larger eigenvalue of the smoothed structure tensor
    coherence: np.ndarray    # (l1-l2)/(l1+l2) — how linear the local pattern is
    orientation: np.ndarray  # radians, direction of least change (the line)


def structure_tensor(arr: np.ndarray, sigma: float = 1.5,
                     integration_sigma: float = 4.0) -> Lineament:
    """Lineament strength from the structure tensor (Weickert 1998; Bigun &
    Granlund 1987), the standard local-orientation estimator used before Hough /
    deep-learning line detectors.

    Smoothing the derivatives (not the image) is what makes the estimate
    rotation-invariant; `integration_sigma` sets the neighbourhood over which a
    line is assumed straight, i.e. its length scale.
    """
    smooth = nan_gaussian(arr, sigma)
    gx, gy = derivatives(smooth)
    gxx = nan_gaussian(np.where(np.isfinite(gx), gx * gx, np.nan), integration_sigma)
    gyy = nan_gaussian(np.where(np.isfinite(gy), gy * gy, np.nan), integration_sigma)
    gxy = nan_gaussian(np.where(np.isfinite(gx) & np.isfinite(gy), gx * gy, np.nan),
                       integration_sigma)
    half = 0.5 * (gxx + gyy)
    diff = 0.5 * (gxx - gyy)
    root = np.sqrt(diff * diff + gxy * gxy)
    l1 = half + root
    l2 = half - root
    with np.errstate(invalid="ignore", divide="ignore"):
        coherence = (l1 - l2) / (l1 + l2)
        orientation = 0.5 * np.arctan2(2.0 * gxy, gxx - gyy)
    return Lineament(energy=l1, coherence=coherence, orientation=orientation)


def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean downsample that stays NaN if any contributing pixel is NaN."""
    if factor == 1:
        return arr
    h, w = arr.shape
    h2, w2 = h // factor, w // factor
    v = arr[: h2 * factor, : w2 * factor]
    finite = np.isfinite(v)
    summed = np.where(finite, v, 0.0).reshape(h2, factor, w2, factor).sum(axis=(1, 3))
    count = finite.reshape(h2, factor, w2, factor).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = summed / count
    return np.where(count > 0, out, np.nan)


# ---------------------------------------------------------------- the stack


def build_feature_stack(path: str, scales: tuple[int, ...] = (1, 2, 4)
                        ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Assemble the derived-feature stack on the official grid.

    Returns (features, invalid_mask). Only features that are computed from the
    official bands are added here; all 19 official bands are passed through
    (sentinel-masked) as well, so the model can use both.
    """
    names = [n for n, _ in spec.FEATURE_BANDS]
    bands, invalid = load_bands(path, names)
    feats: dict[str, np.ndarray] = dict(bands)  # passthrough, masked

    tmi = bands["tmi"]
    rtp = bands["rtp"]
    mag = bands["mag_anom"]
    hg_mag = bands["tmi_hg"]  # horizontal derivative of TMI (per provider metadata)
    vg_mag = bands["tmi_vg"]

    # --- potential-field edges (the task brief's first priority) -------------
    # hg_mag is already the provider's horizontal-gradient magnitude (|∇TMI|);
    # so |hg_mag| (magnitude of a magnitude) is just its absolute value — kept
    # as a passthrough for model convenience, not as new information.
    feats["mag_hgm_from_parts"] = np.abs(hg_mag)
    feats["mag_asa"] = analytic_signal([hg_mag, vg_mag])       # sqrt(hg^2+vg^2)
    feats["mag_tilt"] = tilt_angle(vg_mag, np.abs(hg_mag))
    tmi_gx, tmi_gy = derivatives(tmi)
    feats["tmi_hgm_computed"] = horizontal_gradient_magnitude(tmi_gx, tmi_gy)
    feats["tmi_asa_computed"] = analytic_signal([tmi_gx, tmi_gy])
    rtp_gx, rtp_gy = derivatives(rtp)
    feats["rtp_hgm_computed"] = horizontal_gradient_magnitude(rtp_gx, rtp_gy)
    mag_gx, mag_gy = derivatives(mag)
    feats["mag_anom_hgm_computed"] = horizontal_gradient_magnitude(mag_gx, mag_gy)

    grav = bands["iso_grav_anom"]
    ghg = bands["iso_grav_anom_hg"]
    gvg = bands["iso_grav_anom_vg"]
    feats["grav_asa"] = analytic_signal([ghg, gvg])
    feats["grav_tilt"] = tilt_angle(gvg, np.abs(ghg))
    ggx, ggy = derivatives(grav)
    feats["grav_hgm_computed"] = horizontal_gradient_magnitude(ggx, ggy)

    # --- topography: curvature + breaks in slope (second priority) -----------
    curv = curvature(bands["det_elev"])
    feats["curv_total"] = curv.total
    feats["curv_profile"] = curv.profile
    feats["curv_plan"] = curv.plan
    feats["curv_gaussian"] = curv.gaussian
    feats["slope_computed"] = curv.slope
    feats["slope_of_slope"] = curv.slope_of_slope
    elev_gx, elev_gy = derivatives(bands["det_elev"])
    feats["det_elev_hgm_computed"] = horizontal_gradient_magnitude(elev_gx, elev_gy)

    # --- lineaments: orientation-coherence energy over the key surfaces ------
    for src_name, src in (("elev", bands["det_elev"]), ("tmi", tmi), ("grav", grav),
                          ("cond", bands["cond_surf"])):
        for factor in scales:
            ds = downsample(src, factor)
            if factor > 1:
                ds = np.repeat(np.repeat(ds, factor, axis=0), factor, axis=1)
                ds = ds[: src.shape[0], : src.shape[1]]
                pad_r = src.shape[0] - ds.shape[0]
                pad_c = src.shape[1] - ds.shape[1]
                if pad_r or pad_c:
                    ds = np.pad(ds, ((0, pad_r), (0, pad_c)), constant_values=np.nan)
            tensor = structure_tensor(ds, sigma=1.0,
                                      integration_sigma=2.0 * factor)
            suffix = f"_s{factor}"
            feats[f"lin_{src_name}_energy{suffix}"] = tensor.energy
            feats[f"lin_{src_name}_coherence{suffix}"] = tensor.coherence
            feats[f"lin_{src_name}_cos2theta{suffix}"] = np.cos(2.0 * tensor.orientation)
            feats[f"lin_{src_name}_sin2theta{suffix}"] = np.sin(2.0 * tensor.orientation)

    # --- cross-checks between independent signals (third priority) -----------
    # Agreement between strain rate, conductivity and seismicity is only
    # informative where each is informative, so keep them separate and let the
    # model combine them; also provide explicit products as cheap interactions.
    for a, b in (("geod_shearrate", "geod_dilaterate"),
                 ("cond_surf", "depth_to_base_surf"),
                 ("ieq_n100a15", "deq_n100a15")):
        feats[f"x_{a}__{b}"] = _safe_prod(bands[a], bands[b])

    return feats, invalid


def _safe_prod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return a * b


def standardize(train: np.ndarray, *others: np.ndarray
                ) -> tuple[np.ndarray, ...]:
    """Z-score using TRAIN statistics only (no leakage across CV folds)."""
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return tuple((o - mu) / sd for o in (train,) + others)


# ------------------------------------------- Hessian ridge (Frangi) line detector

# GEMSDOE10 addition. Faults appear in potential-field and elevation grids as
# LINEAMENTS — curvilinear ridges/valleys a few pixels wide. The Frangi
# vesselness filter (Frangi et al. 1998, MICCAI — "Multiscale vessel enhancement
# filtering") is the classical scale-explicit detector for exactly such
# structures: it scores each pixel from the eigenvalues of the local Hessian at
# a matched scale, high only where the field curves strongly across one axis
# and weakly along the perpendicular axis. Symmetric in sign (ridges AND
# valleys) because a fault can be either depending on the sense of offset and
# the field. Orientation (the ridge direction) comes free from the eigenvectors.


@dataclass
class Frangi:
    vesselness: np.ndarray   # [0, 1]-ish line score at this scale
    orientation: np.ndarray  # radians, direction ALONG the ridge
    strength: np.ndarray     # raw second-order structure norm (unnormalised)


def hessian_eigenvalues(arr: np.ndarray, sigma: float
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale-normalised Hessian eigendecomposition (NaN-guarded).

    Returns (l1, l2, vx, vy) with |l1| <= |l2| (l2 = across-ridge curvature)
    and (vx, vy) the unit eigenvector belonging to l1 (the ALONG-ridge
    direction). Gaussian second derivatives are scale-normalised by sigma**2
    (Lindeberg 1998) so responses are comparable across scales.
    """
    bad = ~np.isfinite(arr)
    filled = np.where(bad, 0.0, arr)
    # gaussian_filter(order=...) gives exact Gaussian derivatives, separable.
    ixx = ndimage.gaussian_filter(filled, sigma, order=(0, 2)) * sigma * sigma
    iyy = ndimage.gaussian_filter(filled, sigma, order=(2, 0)) * sigma * sigma
    ixy = ndimage.gaussian_filter(filled, sigma, order=(1, 1)) * sigma * sigma
    r = int(np.ceil(3 * sigma)) + 1
    if bad.any():
        grown = ndimage.binary_dilation(bad, iterations=r)
        ixx = np.where(grown, np.nan, ixx)
        iyy = np.where(grown, np.nan, iyy)
        ixy = np.where(grown, np.nan, ixy)
    # analytic 2x2 eigendecomposition, ordered |l1| <= |l2|
    half_trace = 0.5 * (ixx + iyy)
    half_diff = 0.5 * (ixx - iyy)
    root = np.sqrt(half_diff * half_diff + ixy * ixy)
    la = half_trace + root
    lb = half_trace - root
    swap = np.abs(la) > np.abs(lb)
    l1 = np.where(swap, lb, la)
    l2 = np.where(swap, la, lb)
    # eigenvector of l1: perpendicular to the across-ridge direction.
    # For H = [[a,b],[b,c]] and eigenvalue l: direction (b, l - a) (or (l - c, b)).
    with np.errstate(invalid="ignore", divide="ignore"):
        vx = ixy
        vy = l1 - ixx
        n = np.sqrt(vx * vx + vy * vy)
        vx = np.where(n > 0, vx / n, 1.0)
        vy = np.where(n > 0, vy / n, 0.0)
    return l1, l2, vx, vy


def frangi_vesselness(arr: np.ndarray, sigma: float,
                      beta: float = 0.5) -> Frangi:
    """Frangi line score at one scale, sign-symmetric (ridges and valleys).

    V = exp(-Rb^2 / 2 beta^2) * (1 - exp(-S^2 / 2 c^2)) with Rb = |l1/l2|
    (blobness — 0 for an ideal line) and S = sqrt(l1^2 + l2^2) (structure
    strength). c is set per-tile to half the max S (Frangi's heuristic),
    which keeps the score in [0, 1] and comparable within a tile. NaN
    propagates from invalid input through the Hessian halo rule.
    """
    l1, l2, vx, vy = hessian_eigenvalues(arr, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = np.abs(l1) / np.abs(l2)
        s = np.sqrt(l1 * l1 + l2 * l2)
        cmax = np.nanmax(s)
        c = 0.5 * cmax if np.isfinite(cmax) and cmax > 0 else 1.0
        v = np.exp(-(rb * rb) / (2.0 * beta * beta)) * (1.0 - np.exp(-(s * s) / (2.0 * c * c)))
        v = np.where(np.abs(l2) > 0, v, 0.0)
        orientation = np.arctan2(vy, vx)
    # re-apply NaN where the Hessian itself is undefined
    v = np.where(np.isfinite(l2), v, np.nan)
    orientation = np.where(np.isfinite(l2), orientation, np.nan)
    return Frangi(vesselness=v.astype(np.float64), orientation=orientation,
                  strength=s)


# ------------------------------------------------- Gabor orientation-max detector

# GEMSDOE10 addition. A bank of even-symmetric Gabor kernels at K orientations;
# the per-pixel max over orientations responds to line segments at any strike
# while the argmax gives strike. Complements Frangi (second-order, isotropic
# scale selection) with first-order oriented energy at fixed kernel sizes.


def _gabor_kernel(size: int, theta: float, wavelength: float,
                  gamma: float = 0.5, bandwidth: float = 1.0) -> np.ndarray:
    """Even Gabor kernel (zero-mean) of odd `size`, orientation `theta`."""
    assert size % 2 == 1
    h = size // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float64)
    xp = xx * np.cos(theta) + yy * np.sin(theta)
    yp = -xx * np.sin(theta) + yy * np.cos(theta)
    sigma = wavelength * bandwidth / np.pi * np.sqrt(np.log(2) / 2.0) * (
        2.0 ** bandwidth + 1.0) / (2.0 ** bandwidth - 1.0)
    g = np.exp(-(xp * xp + gamma * gamma * yp * yp) / (2.0 * sigma * sigma))
    g *= np.cos(2.0 * np.pi * xp / wavelength)
    return g - g.mean()


def gabor_max_response(arr: np.ndarray, size: int = 11,
                       n_orientations: int = 8,
                       wavelength: float | None = None,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Max-over-orientations Gabor line energy + dominant orientation.

    Returns (energy, orientation). Energy is |response| standardised by the
    kernel L1 norm so kernel sizes are comparable; orientation is the angle of
    the maximally-responding kernel in [0, pi). NaN-guarded with a halo of
    size//2 + 1 (an output pixel is NaN if any input pixel under its kernel
    was NaN).
    """
    if wavelength is None:
        wavelength = size / 2.5
    bad = ~np.isfinite(arr)
    filled = np.where(bad, 0.0, arr).astype(np.float64)
    energy = np.full(arr.shape, -np.inf)
    orient = np.zeros(arr.shape)
    for k in range(n_orientations):
        theta = np.pi * k / n_orientations
        ker = _gabor_kernel(size, theta, wavelength)
        resp = ndimage.convolve(filled, ker, mode="constant", cval=0.0)
        resp = np.abs(resp) / (np.abs(ker).sum() + 1e-12)
        better = resp > energy
        energy = np.where(better, resp, energy)
        orient = np.where(better, (theta + np.pi / 2.0) % np.pi, orient)
    energy = np.where(np.isfinite(energy), energy, np.nan)
    r = size // 2 + 1
    if bad.any():
        grown = ndimage.binary_dilation(bad, iterations=r)
        energy = np.where(grown, np.nan, energy)
        orient = np.where(grown, np.nan, orient)
    return energy, orient
