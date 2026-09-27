# Provenance: adapted from buffedlizard55-lab/6GEMSDOE src/gems (same owner, re-verified here).
"""Pinned specification of the official competition grid — measured, not assumed.

Every value below was measured on 2026-09-25 from the official bytes whose sha256
is pinned here, after independently re-verifying all eight bridge parts and the
three whole-file hashes (see VERIFICATION.md).

Official statements these values were checked against
(https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/,
section "Submission format"):
  * same projected CRS as the training data — UTM zone 11N, EPSG 32611
  * same resolution as the training data — 100 m
  * same bounds as the training data; data outside the bounds is null or nan
  * a single layer, datatype 32-bit float, values between 0 and 1

Measured facts (all reproduced by scripts/verify_spec.py):

  grid            3730 rows x 3292 columns = 12,279,160 pixels
  crs             EPSG:32611   (UTM 11N, WGS 84)
  pixel size      100.0 m x 100.0 m
  transform       (100, 0, 243350.0, 0, -100, 4508550.0)
  bounds          left 243350.0  bottom 4135550.0  right 572550.0  top 4508550.0
  footprint       5,167,373 finite pixels (the rest, 7,111,787, are NaN)
  labels          int8 with nodata = -1; 60,988 positive pixels (all inside the
                  footprint); nodata mask is byte-identical to the sample
                  submission's NaN mask
  sample subm.    float32, nodata = NaN, and — note — NOT all zeros: it carries
                  exactly 60,988 ones at the catalogue fault pixels. The
                  competition page describes it as "a sample submission that
                  predicts total fault absence"; the file as published is a
                  copy of the existing-fault labels. Verified 2026-09-25.
  features        float32, 19 bands, nodata sentinel -3.4028234663852886e+38
                  (= the most negative float32), 5,165,840 pixels finite in all
                  19 bands

Consequences used elsewhere in this repo:
  * 300 m triangular kernel = exactly 3.0 pixels on this grid (metric.py).
  * The authoritative "inside the footprint" mask for a submission is the sample
    submission's non-NaN mask; a NaN anywhere inside it violates the finite
    confidence contract. The original backend rejection cause is not known.
    That is a hard gate in raster.py.
  * Faults occupy 60,988 / 5,167,373 = 1.1803% of the scored footprint and
    0.4967% of the full grid. The task brief's "roughly 1% of the area" holds for
    the scored footprint.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# --- geometry ----------------------------------------------------------------
EPSG = 32611
PIXEL_SIZE_M = 100.0
WIDTH = 3292
HEIGHT = 3730
ORIGIN_X = 243350.0
ORIGIN_Y = 4508550.0
TOTAL_PIXELS = WIDTH * HEIGHT  # 12,279,160

# --- measured counts ---------------------------------------------------------
FOOTPRINT_PIXELS = 5_167_373  # finite pixels in the official sample submission
NODATA_PIXELS = 7_111_787  # NaN pixels in the official sample submission
LABEL_POSITIVE_PIXELS = 60_988  # int8 == 1 in the official labels
FEATURES_ALL_BAND_VALID_PIXELS = 5_165_840  # finite in all 19 feature bands

COVERAGE_OF_FOOTPRINT = LABEL_POSITIVE_PIXELS / FOOTPRINT_PIXELS  # 0.011803
COVERAGE_OF_GRID = LABEL_POSITIVE_PIXELS / TOTAL_PIXELS  # 0.004967

# --- sentinels ---------------------------------------------------------------
FEATURE_SENTINEL = -3.4028234663852886e38  # float32 most-negative value
FEATURE_INVALID_BELOW = -1e38  # same rule the reference solution uses (cell 5)
LABEL_NODATA = -1
SUBMISSION_NODATA = float("nan")

# --- sha256 pins (independent re-verification, 2026-09-25) -------------------
PINS: dict[str, dict] = {
    "training_features.tif": {
        "bytes": 418_912_844,
        "sha256": "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5",
        "official_data_tab_name": "gems-geodawn-numerical-features.tif",
    },
    "labels.tif": {
        "bytes": 425_830,
        "sha256": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
        "official_data_tab_name": "existing_faults.tif",
    },
    "sample_submission.tif": {
        "bytes": 1_599_597,
        "sha256": "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
        "official_data_tab_name": "example_submission.tif",
    },
}

BRIDGE_PARTS = [
    ("part-000", 94_371_840,
     "0a330f8951af6c921029e25c84a579319d2db554d62d30d894d6ddc97f98cff7"),
    ("part-001", 94_371_840,
     "3c98037b2c997e3bbfcdfc2d9a982e8b820a77410dd7404b05cdb80594922c50"),
    ("part-002", 94_371_840,
     "c375c4dbc40c59bbaece572b5e348700b75935f9a30c879c82b6417e0836f31c"),
    ("part-003", 94_371_840,
     "b164159e6d0cb2595bc9f63a948af2646b124114a9c5921880c7516092137320"),
    ("part-004", 41_425_484,
     "fa0a6f9c936fac1d6f20ca37f5929b2d60bf7a80f3d477dcab886f941aee2696"),
]

# --- the 19 official feature bands (order = band index 1..19) ----------------
# Descriptions read from the official raster's per-band TIFF tags, verbatim.
FEATURE_BANDS: list[tuple[str, str]] = [
    ("mag_anom", "Magnetic anomaly - deviation from expected Earth's magnetic field"),
    ("rtp", "Reduced to pole magnetic data - magnetic anomaly corrected for latitude effects"),
    ("tmi_hg", "Total magnetic intensity horizontal gradient - rate of change in horizontal direction"),
    ("geod_2ndinv", "Geodetic second invariant - measure of strain rate tensor magnitude"),
    ("iso_grav_anom_slope", "Isostatic gravity anomaly slope - gradient of gravity after isostatic correction"),
    ("tc", "Tilt angle or total curvature - magnetic field derivative for edge detection"),
    ("geod_shearrate", "Geodetic shear rate - rate of angular deformation from GPS/InSAR"),
    ("geod_dilaterate", "Geodetic dilatation rate - rate of volumetric strain (expansion/contraction)"),
    ("tmi_vg", "Total magnetic intensity vertical gradient - rate of change in vertical direction"),
    ("deq_n100a15", "Distance to earthquake (n=100km radius, a=15\u00b0 azimuth parameters)"),
    ("iso_grav_anom_vg", "Isostatic gravity anomaly vertical gradient - vertical rate of change"),
    ("det_elev", "Detrended elevation - topography with regional trends removed"),
    ("iso_grav_anom", "Isostatic gravity anomaly - gravity after compensating for topographic mass"),
    ("tmi", "Total magnetic intensity - total strength of magnetic field"),
    ("depth_to_base_surf", "Depth to basement surface - thickness of sedimentary cover"),
    ("ieq_n100a15", "Earthquake intensity or density (n=100km radius, a=15\u00b0 parameters)"),
    ("cond_surf", "Conductivity surface - electrical conductivity of subsurface"),
    ("iso_grav_anom_hg", "Isostatic gravity anomaly horizontal gradient - horizontal rate of change"),
    ("det_elev_slope", "Detrended elevation slope - gradient of elevation after detrending"),
]
BAND_INDEX = {name: i + 1 for i, (name, _) in enumerate(FEATURE_BANDS)}

# --- source URLs (all re-checked 2026-09-25) ---------------------------------
URLS = {
    "home": "https://www.drivendata.org/competitions/306/competition-doe-gems/",
    "problem": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
    "about": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/",
    "data": "https://www.drivendata.org/competitions/306/competition-doe-gems/data/",
    "rules_landing": "https://www.drivendata.org/competitions/306/competition-doe-gems/rules/",
    "rules_pdf": "https://www.nlr.gov/docs/fy26osti/96647.pdf",
    "rules_mirror_hero": "https://www.herox.com/GEMSPrize/resource/2274",
    "reference_solution": "https://github.com/drivendataorg/gems-prize-reference-solution",
    "geodawn_dataset": "https://doi.org/10.5066/P93LGLVQ",
    "ingenious_compilation": "https://doi.org/10.15121/1881483",
}


@dataclass(frozen=True)
class GridSpec:
    """Machine-readable form of the constants above (for the site + tests)."""

    epsg: int = EPSG
    pixel_size_m: float = PIXEL_SIZE_M
    width: int = WIDTH
    height: int = HEIGHT
    origin_x: float = ORIGIN_X
    origin_y: float = ORIGIN_Y
    footprint_pixels: int = FOOTPRINT_PIXELS
    nodata_pixels: int = NODATA_PIXELS
    label_positive_pixels: int = LABEL_POSITIVE_PIXELS

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.origin_x,
            self.origin_y - self.height * self.pixel_size_m,
            self.origin_x + self.width * self.pixel_size_m,
            self.origin_y,
        )

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bounds"] = list(self.bounds)
        d["coverage_of_footprint_pct"] = round(100.0 * COVERAGE_OF_FOOTPRINT, 4)
        d["coverage_of_grid_pct"] = round(100.0 * COVERAGE_OF_GRID, 4)
        d["pins"] = PINS
        d["feature_bands"] = [{"band": i + 1, "name": n, "description": s}
                              for i, (n, s) in enumerate(FEATURE_BANDS)]
        d["urls"] = URLS
        return d


def write_spec(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(GridSpec().as_dict(), indent=2) + "\n")
    return p


if __name__ == "__main__":  # pragma: no cover
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "configs/submission_spec.json"
    print(write_spec(out))
