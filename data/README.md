# Data — placement, provenance, verification

## Official files (competition data tab, login-gated)

https://www.drivendata.org/competitions/306/competition-doe-gems/data/

| File here | Official name | Bytes | sha256 |
|---|---|---|---|
| `training_features.tif` (**not committed**, gitignored) | `gems-geodawn-numerical-features.tif` | 418,912,844 | `4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5` |
| `labels.tif` (committed, 425 KB) | `existing_faults.tif` | 425,830 | `7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093` |
| `sample_submission.tif` (committed, 1.6 MB) | `example_submission.tif` | 1,599,597 | `2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc` |

Pins live in `src/gems10/spec.py` (`PINS`, `BRIDGE_PARTS`).

## Automatic placement (working in this session)

`python scripts/fetch_data.py --download` restores the feature file from public team
mirror `buffedlizard55-lab/6GEMSDOE` at commit
`e2fe3f41c6f5dd2dcb2fc91958ee67698f114ada` through GitHub's API via `gh`.
Each part and the assembled file are verified before atomic placement. Temporary
parts are removed; the large raster is ignored. This verifies continuity with
inherited team pins, not independent official authenticity.

The user-provided Dropbox feature link failed TLS from this runtime. The official
data tab redirects to login. The mirror is a transport fallback, not an official
publisher. Existing `--bridge-dir` assembly and `--no-features` small-file checks
remain available. No GPU is needed for HGB training.

## External priors: historical, not present here

Earlier sessions reference sibling 7GEMSDOE lidar scarp and radiometric aggregates,
QFault priors, and an SGMC-gap raster. Those files are NOT present in this checkout
and have not been independently regenerated in this session. Do not claim a model
used them based on filenames in an old report. New experiments use provided data
only. H15's exact official archives, CC BY 4.0 attribution and failed accessibility
checks are in HYPOTHESES.md. Large external data belong in ignored `data/external/`.

## Measured facts (re-verified 2026-09-27)

Grid 3730×3292 = 12,279,160 px; footprint 5,167,373 finite px; labels 60,988 positive (1.18% of footprint); sample submission is **not** all-zero — it carries 1.0 at exactly the 60,988 catalogue pixels; feature nodata sentinel is float32-min `-3.4028234663852886e+38` (not NaN); 3,199 raster components (8-conn, not necessarily geological systems), median 12 px, max 360 px.
