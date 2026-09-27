# Data — placement, provenance, verification

## Official files (competition data tab, login-gated)

https://www.drivendata.org/competitions/306/competition-doe-gems/data/

| File here | Official name | Bytes | sha256 |
|---|---|---|---|
| `training_features.tif` (**not committed**, gitignored) | `gems-geodawn-numerical-features.tif` | 418,912,844 | `4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5` |
| `labels.tif` (committed, 425 KB) | `existing_faults.tif` | 425,830 | `7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093` |
| `sample_submission.tif` (committed, 1.6 MB) | `example_submission.tif` | 1,599,597 | `2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc` |

Pins live in `src/gems10/spec.py` (`PINS`, `BRIDGE_PARTS`).

## Placement

1. Download the three files from the data tab into `data/` (names above), **or**
2. Copy the five bridge parts (`gems-geodawn-numerical-features.tif.part-000..004`, same layout as sibling repo 6GEMSDOE `data/bridge/`) anywhere and assemble:
   `python scripts/fetch_data.py --data-dir data --bridge-dir <parts-dir>`
3. Verify: `python scripts/fetch_data.py --data-dir data` → `ALL OK` (3/3).

## External priors (`data/external/`, gitignored, fetched at build time)

| File | Source | Provenance |
|---|---|---|
| `lidar_scarp_features_u8.tif` (12 bands) + `.json` | USGS 3DEP lidar → 100 m grid | sibling 7GEMSDOE `external/dem/` |
| `geodawn_rad_u8.tif` (4 bands) + `.json` | GeoDAWN radiometrics, doi:10.5066/P93LGLVQ | sibling 7GEMSDOE `external/geodawn_rad/` |
| `qfaults_prior_u8.tif` + `.json` | USGS QFaults `Qfaults_GIS.zip` sha256 `447eadc5…` | sibling 7GEMSDOE `external/qfaults/` |
| `sgmc_gap.tif` | GapFinder v2 (GEMSDOE3) | rejected for submission — see `src/gems10/external.py` |

All are verified to the official grid (3730×3292, EPSG:32611, 100 m) on load by `src/gems10/external.py::read_u8_stack`.

## Measured facts (re-verified 2026-09-27)

Grid 3730×3292 = 12,279,160 px; footprint 5,167,373 finite px; labels 60,988 positive (1.18% of footprint); sample submission is **not** all-zero — it carries 1.0 at exactly the 60,988 catalogue pixels; feature nodata sentinel is float32-min `-3.4028234663852886e+38` (not NaN); 3,199 fault systems (8-conn), median 12 px, max 360 px.
