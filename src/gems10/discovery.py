"""Discovery operators: geologically-motivated hidden-fault candidates.

All functions take an EXPLICIT source-system mask so CV folds can pass
train-systems-only (no leakage of held-out geometry), and the final fit can
pass all systems. Nothing here reads the label file.

Operators
---------
1. system_strikes: PCA strike per system (centroid, angle, half-length,
   endpoints). Basin-and-Range normal faults are ~linear at 100 m pixels;
   the first principal axis is the strike.
2. strike_rays: continuation rays from each system's endpoints along ±strike.
   Rationale: mapped traces end where exposure ends (alluvium cover), not
   where the fault ends; tip damage zones continue 0.5–1.5 km into the basin
   (i.e. 5–15 px). Default ray 10 px.
3. relay_corridors: breached relay ramps / step-over fracture meshes between
   facing endpoints of distinct subparallel systems 3–15 px apart.
4. noisy_or_fuse: calibrated blend of model probability with discovery grids.

Every operator is evaluated inside the system-holdout harness before it is
allowed into the submission (see scripts/run_cv.py --with-discovery).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Strike:
    system: int
    centroid: tuple[float, float]  # (row, col)
    angle: float  # radians, direction of the first principal axis [0, pi)
    half_len: float  # px, 2-sigma extent along strike
    width: float  # px, 2-sigma extent across strike
    n_px: int
    end_a: tuple[float, float]
    end_b: tuple[float, float]


def system_strikes(system_id: np.ndarray, use_systems: np.ndarray) -> list[Strike]:
    """PCA strike for each system id in `use_systems` (1-based ids)."""
    out: list[Strike] = []
    for s in sorted(int(v) for v in np.unique(use_systems) if int(v) > 0):
        rr, cc = np.nonzero(system_id == s)
        if rr.size < 3:
            continue
        r0, c0 = float(rr.mean()), float(cc.mean())
        dr = rr.astype(np.float64) - r0
        dc = cc.astype(np.float64) - c0
        cov = np.cov(dc, dr)  # note: x=col, y=row order
        try:
            vals, vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        vx, vy = float(vecs[0, 0]), float(vecs[1, 0])
        ang = float(np.arctan2(vy, vx)) % np.pi
        hl = float(2.0 * np.sqrt(max(vals[0], 0.0)))
        wd = float(2.0 * np.sqrt(max(vals[1], 0.0)))
        # endpoints: extreme projections on the strike axis
        t = dc * vx + dr * vy
        tmin, tmax = float(t.min()), float(t.max())
        out.append(Strike(system=s, centroid=(r0, c0), angle=ang, half_len=hl,
                          width=wd, n_px=int(rr.size),
                          end_a=(r0 + vy * tmin, c0 + vx * tmin),
                          end_b=(r0 + vy * tmax, c0 + vx * tmax)))
    return out


def _draw_segment(grid: np.ndarray, a: tuple[float, float],
                  b: tuple[float, float], value: float, width_px: int = 1) -> None:
    """Additive rasterisation of a segment (row/col coords), clipped to grid."""
    h, w = grid.shape
    ar, ac = a
    br, bc = b
    length = float(np.hypot(br - ar, bc - ac))
    n = max(int(np.ceil(length * 2.0)), 1)
    for t in np.linspace(0.0, 1.0, n + 1):
        r = int(round(ar + (br - ar) * t))
        c = int(round(ac + (bc - ac) * t))
        if 0 <= r < h and 0 <= c < w:
            if width_px <= 1:
                grid[r, c] += value
            else:
                r0, r1 = max(0, r - width_px // 2), min(h, r + width_px // 2 + 1)
                c0, c1 = max(0, c - width_px // 2), min(w, c + width_px // 2 + 1)
                grid[r0:r1, c0:c1] += value


def strike_rays(shape: tuple[int, int], strikes: list[Strike],
                ray_len_px: float = 10.0, value: float = 1.0,
                min_half_len: float = 2.0) -> np.ndarray:
    """Continuation rays from system endpoints along ±strike.

    Only systems with half_len >= min_half_len cast rays (a strike fit on a
    3-px blob is noise). Rays START at the endpoint and extend outward.
    """
    grid = np.zeros(shape, dtype=np.float32)
    for s in strikes:
        if s.half_len < min_half_len:
            continue
        dx = float(np.cos(s.angle))
        dy = float(np.sin(s.angle))
        # orient the axis from end_a to end_b
        ar, ac = s.end_a
        br, bc = s.end_b
        along_r, along_c = br - ar, bc - ac
        if along_r * dy + along_c * dx < 0:
            dx, dy = -dx, -dy
        _draw_segment(grid, (ar, ac), (ar - dy * ray_len_px, ac - dx * ray_len_px),
                      value)
        _draw_segment(grid, (br, bc), (br + dy * ray_len_px, bc + dx * ray_len_px),
                      value)
    return grid


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def relay_corridors(shape: tuple[int, int], strikes: list[Strike],
                    min_gap_px: float = 3.0, max_gap_px: float = 15.0,
                    max_strike_diff_deg: float = 30.0,
                    value: float = 1.0, width_px: int = 3) -> np.ndarray:
    """Step-over corridors between facing endpoints of distinct systems.

    For each endpoint, the nearest endpoint of a DIFFERENT system within
    [min_gap, max_gap] whose strike differs by < max_strike_diff is linked
    with a width_px-wide corridor (extensional fracture mesh of a relay ramp).
    O(E^2) with early bounding-box rejection; E ~ 2 systems ≈ 6.4k → ~40M
    pairs worst case, but the gap window prunes nearly all of them.
    """
    grid = np.zeros(shape, dtype=np.float32)
    ends: list[tuple[float, float, float, int]] = []  # r, c, angle, system
    for s in strikes:
        ends.append((s.end_a[0], s.end_a[1], s.angle, s.system))
        ends.append((s.end_b[0], s.end_b[1], s.angle, s.system))
    if not ends:
        return grid
    E = np.array(ends, dtype=np.float64)
    maxd = float(np.radians(max_strike_diff_deg))
    for i in range(len(ends)):
        r0, c0, a0, s0 = E[i]
        # candidate box prune
        dr = np.abs(E[:, 0] - r0)
        dc = np.abs(E[:, 1] - c0)
        cand = np.nonzero((E[:, 3] != s0) & (dr <= max_gap_px) & (dc <= max_gap_px))[0]
        best_j, best_d = -1, max_gap_px + 1.0
        for j in cand:
            if j == i:
                continue
            d = float(np.hypot(E[j, 0] - r0, E[j, 1] - c0))
            if d < min_gap_px or d > max_gap_px or d >= best_d:
                continue
            if _ang_diff(a0, float(E[j, 2])) > maxd:
                continue
            best_j, best_d = int(j), d
        if best_j >= 0:
            _draw_segment(grid, (r0, c0),
                          (float(E[best_j, 0]), float(E[best_j, 1])),
                          value, width_px=width_px)
    return grid


def noisy_or_fuse(prob: np.ndarray, *layers: tuple[np.ndarray, float]) -> np.ndarray:
    """Noisy-OR blend: 1 - (1-p) * prod(1 - w*clip(layer,0,1)).

    Each layer is a non-negative grid (counts ok — clipped to [0,1] after an
    internal max-normalisation ONLY if its max exceeds 1... no: callers pass
    weights; layers are clipped to [0, 1] raw so counts saturate, which is the
    intended semantics for overlapping rays).
    """
    out = 1.0 - np.clip(np.asarray(prob, dtype=np.float64), 0.0, 1.0)
    for layer, w in layers:
        if w <= 0:
            continue
        l = np.clip(np.asarray(layer, dtype=np.float64), 0.0, 1.0)
        out = out * (1.0 - float(w) * l)
    return (1.0 - out).astype(np.float64)


def normalise01(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.float64)
    mx = np.nanmax(g) if np.isfinite(g).any() else 0.0
    if not np.isfinite(mx) or mx <= 0:
        return np.zeros_like(g)
    return np.clip(g / mx, 0.0, 1.0)
