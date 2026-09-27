"""External priors: lidar scarp morphology + radiometrics on the official grid.

Products (built by sibling repo 7GEMSDOE from public sources, reused here with
provenance; paths below are the 7GEMSDOE repo locations):

  lidar_scarp_features_u8.tif  12 bands, uint8, nodata 0
      source: USGS 3DEP lidar (2 m DEMs aggregated to the 100 m official grid)
      bands: ex_max, ex_mean, step_max, lapneg_max, lappos_max, downface_max,
             upface_max, cross_max, relief, coh100, strike, valid
      quantisation: q = 1 + round(254 * t(clip(x/xmax,0,1))), t in {sqrt,linear}
  geodawn_rad_u8.tif  4 bands, uint8, nodata 0
      source: GeoDAWN radiometrics (USGS ScienceBase doi:10.5066/P93LGLVQ)
  sgmc_gap (GapFinder v2, GEMSDOE3): 61,664 px of USGS-SGMC faults >300 m from
      the training labels — REJECTED for the submission (see below).

Measured in GEMSDOE10 (2026-09-27, system-holdout, GT = held-out catalogue
systems, fp_ignore = train systems):
  * lidar ex_max top-2% ridge: masked DTI 0.0254 (unsupervised, no training)
  * SGMC-gap traces: masked DTI 0.0000 — by construction (>300 m from every
    catalogue system, hence from every held-out GT pixel). The private test
    set excludes the public USGS database and SGMC is USGS-public, so these
    traces are near-certain Phase-1 false positives AND ineligible as Phase-2
    "previously-unknown" discoveries. Excluded; kept as a documented negative.
  * QFaults-not-catalogue gap: 100 px in-footprint, DTI 0.003 — negligible.

Licenses: USGS-authored (public domain, verify attribution before reuse);
GeoDAWN via USGS ScienceBase (public). Both satisfy the competition's external-
data rule ("allowed ... provided you possess the necessary licenses").
"""

from __future__ import annotations

import numpy as np

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

LIDAR_BANDS = ["ex_max", "ex_mean", "step_max", "lapneg_max", "lappos_max",
               "downface_max", "upface_max", "cross_max", "relief", "coh100",
               "strike", "valid"]


def read_u8_stack(path: str, expect_shape: tuple[int, int] | None = None,
                  grid_transform: tuple | None = None,
                  grid_crs: str | None = None) -> np.ndarray:
    """Read a uint8 external stack; 0 (nodata) -> NaN; verify grid on request."""
    if rasterio is None:  # pragma: no cover
        raise SystemExit("rasterio required")
    with rasterio.open(path) as src:
        if expect_shape is not None and (src.height, src.width) != expect_shape:
            raise ValueError(f"{path}: shape {(src.height, src.width)} != {expect_shape}")
        if grid_transform is not None and tuple(src.transform)[:6] != grid_transform:
            raise ValueError(f"{path}: transform {tuple(src.transform)[:6]} != official")
        if grid_crs is not None and src.crs is not None and src.crs.to_epsg() != 32611:
            raise ValueError(f"{path}: crs {src.crs} != EPSG:32611")
        n = src.count
        arr = src.read().astype(np.float32)  # (C, H, W)
    arr = np.where(arr == 0, np.nan, arr)
    return np.moveaxis(arr, 0, -1)  # (H, W, C)


def lidar_ridge_from_exmax(lidar: np.ndarray, footprint: np.ndarray,
                           top_frac: float = 0.02) -> np.ndarray:
    """Unsupervised scarp-ridge mask: top `top_frac` of ex_max within lidar-valid."""
    ex = lidar[:, :, 0]
    valid = np.isfinite(ex) & np.asarray(footprint, dtype=bool)
    thr = float(np.nanquantile(np.where(valid, ex, np.nan), 1.0 - top_frac))
    return valid & (ex >= thr)
