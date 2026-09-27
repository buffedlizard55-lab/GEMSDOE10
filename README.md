# GEMSDOE10 — Hidden-Fault Discovery for the DOE GEMS Prize

**Competition:** [The Geologic Enhanced Mapping System (GEMS) Prize Challenge](https://www.drivendata.org/competitions/306/competition-doe-gems/) — predict geological faults from geophysical data to find hidden geothermal systems. **$300,000** prize pool, submissions close **Dec 3, 2026, 23:59 UTC**.

**Thesis (why this repo exists):** both prize rounds score faults **missing** from the catalogue, with known-catalogue pixels **masked** from evaluation ([staff clarification, forum 11516](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516)). Optimising catalogue reproduction is the wrong objective — our sibling 6GEMSDOE reached 0.17 blocked-CV catalogue DTI and scored **0.0286** on the leaderboard. GEMSDOE10 optimises **hidden-fault detection** instead, measured by a system-holdout harness that simulates the official scoring.

| # | Verified result (2026-09-27) | Evidence |
|---|---|---|
| 1 | Official data verified byte-for-byte (3/3 sha256) | `scripts/fetch_data.py`, [VERIFICATION.md](VERIFICATION.md) |
| 2 | Metric = official distance-weighted Tversky (α=0.2, β=0.8, R=300 m) + masked variant; worked example reproduces 0.60 | `tests/test_metric.py` |
| 3 | Format gate catches the reported *"Predicted values must be in range [0, 1]"* rejection (NaN inside footprint) | `tests/test_gate.py` |
| 4 | Proximity-to-catalogue features **memorise**: train systems 0.98, held-out = background → masked DTI 0.012. Removed. | [HARNESS finding](#key-harness-findings) |
| 5 | Geophysics-only HGB (107 ch), 5-fold: GT_ALL **0.0929** @topk02_binary; GT_FAR10 **0.0508** @topk01_binary; GT_FAR20 0.0293 @topk01_binary | `reports/cv_baseline.json`, `reports/rescore_baseline.json` |
| 6 | Unsupervised lidar scarp ridge (no training): masked DTI **0.0254** | `src/gems10/external.py` |
| 7 | SGMC-gap traces score **0.0000** by construction → excluded (documented negative) | `src/gems10/external.py` |
| 8 | 71/71 tests pass; submission writer refuses byte-duplicates of known artifacts | `tests/`, `scripts/build_submission.py` |

## Key harness findings

- **System-holdout CV is the yardstick.** 3,199 fault systems (8-connected components, median 12 px) dealt into 5 folds balanced to ~12.2k hidden px each. GT = held-out systems, `fp_ignore` = train systems — mirroring the official masking. Full footprint scored, like the platform.
- **Catalogue proximity is poison for hidden faults.** See row 4 above. The final model sees geophysics only and therefore covers catalogue faults (masked/free) *and* hidden ones with one probability surface.
- **Binary top-k beats soft** at fixed budget on fold 0 (0.0895 vs 0.0777); 2% budget beats 3% (0.0895 vs 0.0829). Full sweep in `reports/cv_*.json`.
- Public leaderboard (2026-09-26): #1 DARD **0.3049**, our group best 0.1563 (a duplicated file — see [Why 0.1563 repeated](PROJECT_BRIEF.md#why-01563-keeps-repeating)). Gap to close: +0.1486.

## Quickstart

```bash
pip install -r requirements.txt
python scripts/fetch_data.py --data-dir data --bridge-dir <parts>  # verifies sha256
python scripts/build_features.py --data-dir data                    # 107 ch, ~3 min
python scripts/run_cv.py --data-dir data --work-dir cv_out         # 5-fold yardstick
python scripts/train_final.py --data-dir data --work-dir final_out # full-grid fit
python scripts/build_submission.py --prob final_out/prob_final.npy --policy topk02_binary --name gems10-v1
python scripts/validate_submission.py docs/downloads/gems10-v1-*.tif
```

`training_features.tif` (419 MB) is never committed — see [data/README.md](data/README.md). Labels + sample submission (2 MB) are committed for the footprint.

## Layout

- `src/gems10/` — `metric` (DTI + masked), `raster` (hard gate), `spec` (pins), `features` (+Frangi/Gabor), `systems` (holdout folds), `modeling` (HGB + policies), `discovery` (rays/corridors), `selftrain`, `external` (lidar/rad)
- `scripts/` — `fetch_data`, `build_features`, `run_cv`, `train_final`, `build_submission`, `validate_submission`, `build_site`
- `tests/` (71), `reports/` (JSON evidence), `docs/` (GitHub Pages site + downloads)
- [PROJECT_BRIEF.md](PROJECT_BRIEF.md) · [REQUIREMENTS.md](REQUIREMENTS.md) · [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) · [VERIFICATION.md](VERIFICATION.md) · [LIMITATIONS.md](LIMITATIONS.md) · [AI_DISCLOSURE.md](AI_DISCLOSURE.md) · [NEXT_STEPS.md](NEXT_STEPS.md)

## Rules & compliance (account holder must confirm)

- Eligibility (§1.3): US citizen/permanent resident (or US entity with such a captain); no federal employees. **Nothing in this repo can check this.**
- 3 scored submissions per rolling 7 days; one submission chosen blind for both rounds (§3.4/§3.6).
- Generative-AI disclosure is mandatory (§3.2) — draft in [AI_DISCLOSURE.md](AI_DISCLOSURE.md).
- External data allowed with licenses (USGS public domain; GeoDAWN via ScienceBase).
