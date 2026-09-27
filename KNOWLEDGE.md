# KNOWLEDGE — durable, verified project knowledge

**Purpose:** reuse across sessions without re-fetching. Every fact below is tied to an
official/verified source or to a measurement in this repository. Where a source could
not be fetched from this sandbox, that is stated explicitly. Session logs live in the
reports; this file is the distilled layer.

Last updated: 2026-09-27 (session 2, engine calibration + spatial validation launch).

---

## 1. Competition mechanics (verified 2026-09-27)

| Fact | Value | Source |
|---|---|---|
| Prize | $300K total; two rounds; ONE final submission chosen blind to private score | [rules PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf), [competition page](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) |
| Round prizes | $50K split top-5 (Round 1); $250K at 100/70/40/25/15 (Round 2) | competition page |
| DTI | `k(d)=max(1−d/R,0)`, R = 300 m = 3 px (100 m grid); p(x) linear; TPw=Σ_g max p(x)k(d) over GT components; FPw=Σ p(x)[1−max k(d)]; FNw=Σ_g[1−max p(x)k(d)]; DTI = TPw/(TPw+0.2·FPw+0.8·FNw+ε) | competition page (exact formula) |
| Masking | Known USGS/INGENIOUS pixels are **pixel-exactly** excluded (no buffer), both rounds; a prediction near a known trace but far from new GT is fully penalized | staff posts [11516/2](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516/2), [11516/4](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516/4) (2026-09-21) |
| "New fault" definition | any fault pixel not already captured by USGS/INGENIOUS, **including newly mapped geometry of existing systems** (splays, continuations) | staff post [11536/2](https://community.drivendata.org/t/where-do-you-draw-the-line/11536/2) (2026-09-23) |
| New GT near known traces | allowed — "corrections or modifications to existing fault traces", within 300 m, in both new-fault and final-round sets | staff post 11516/4 |
| Hidden test faults | sources/types not disclosed; **Phase 2 test set is updated by expert review of ALL Phase 1 submissions** → novel, defensible predictions matter even if not top Phase 1 | staff post [11527/7](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-an/11527/7) (2026-09-23) |
| Cadence | 3 submissions per week, rolling 7-day reset | staff post [11524/2](https://community.drivendata.org/t/weekly-submissions/11524/2) (2026-09-17) |
| Training labels origin | "obtained from the INGENIOUS project's Great Basin Regional Dataset Compilation" (fn 4 = DOI 10.15121/1881483) | rules PDF |
| Eligibility | U.S. citizen/permanent resident captain; FFRDCs & federal employees barred | rules PDF |
| Finalist requirement | complete reproducible code + documentation | rules PDF |

**Submission format (measured, `SUBMISSION_GUIDE.md`):** single-band float32 GeoTIFF;
EPSG:32611 (UTM 11N); 3,730 × 3,292 px @ 100 m; values finite in [0,1] inside the
footprint (5,167,373 scored cells), **NaN outside**; unique filename + short note.
The historical `[0,1]` upload errors were caused by NaN **inside** the footprint
(9-rule raster gate) and intermittent first-row zeros (fixed by buffered
serialization + readback + SHA).

## 2. Leaderboard & duplicate diagnosis (verified 2026-09-27)

- Public leaderboard: #1 DARD **0.3049**; top-25 floor 0.1587 (extradr19); group best
  **0.1563** (GEMSDOE family) sits below top 25. Gap to #1: 0.1486.
- `reports/submission_audit.json`: GEMSDOE1 / 5GEMSDOE / GEMSDOE2 recall artifacts are
  **byte-identical** (same sha); 8GEMSDOE hedge differs only on 54,533 masked
  catalogue pixels → zero additional non-catalogue predictions. Conclusion: the
  recurring 0.1563 is the same artifact re-uploaded; a new filename is not a new
  discovery. The site must make a genuinely different, evidence-backed TIF available.
- Official reference solution (what "good" looks like to the organizers): Prof. John
  Lipor (Portland State) U-Net + Monte-Carlo dropout CV, PyTorch —
  [drivendataorg/gems-prize-reference-solution](https://github.com/drivendataorg/gems-prize-reference-solution).
  The official benchmark is a CNN; our stack is HGB on hand-built channels (CPU, 2 cores).

## 3. Provided data (hashes pinned in `src/gems10/spec.py`)

- `training_features.tif` (418,912,844 B, sha `4371c82e…`): 19 bands, 100 m, EPSG:32611 —
  conductivity, detrended elevation, geodetic strain rates (shear/dilation), iso-gravity,
  RTP/TMI/depth products, quake density, etc. (band names: `spec.FEATURE_BANDS`).
- `labels.tif`: USGS quaternary + INGENIOUS fault raster (pixel-exact scoring mask source).
- `sample_submission.tif`: footprint template (NaN outside = excluded from scoring).
- External allowed if freely licensed and shareable with the sponsor (competition page).

## 4. INGENIOUS GDR (official external source, verified 2026-09-27 via fetch_page)

[GDR submission 1391](https://gdr.openei.org/submissions/1391) — INGENIOUS Great Basin
Regional Dataset Compilation, Ayling et al. 2022, DOI 10.15121/1881483, CC BY 4.0.

- Downloadable binaries: 2m temperature probes (1.03 MB), seismicity (22.98 MB),
  geodetic shear/dilation (51.99 MB), paleo geothermal (82.04 kB), **Quaternary Faults
  v2 `qfaults_ingenious_nad83conus117_2023-06-27.zip` (5.85 MB, supersedes v1)**,
  quaternary volcanics (9.44 MB), study-area boundary (6.68 kB), well/spring gdb
  (19.85 MB).
- Linked via DOIs: MT conductance, detrended elevation, gravity/magnetics, heat flow,
  slip/dilation tendency, thermal conductivity (sub/1390).
- **Sandbox constraint:** only `api.github.com` + `github.com` are TLS-reachable from
  bash here; GDR/USGS/PNNL binaries fail TLS. Fetch via `gh`-transport, CI, or the user.
  → Hypotheses needing those binaries (e.g. H15 paleo-discharge) are BLOCKED in-sandbox
  until a fetch path exists (`reports/h15` status; H12 evidence in
  `reports/h12_blocked.json`).

## 5. Repository implementation map (session 2)

```
src/gems10/
  spec.py        — band table, pins (sha), footprint, invalid sentinel
  cv.py          — make_folds: 4×4 strided blocks, buffer_px, train/score masks
  features.py    — structure_tensor (σ1.5, integ 4) → Lineament(energy,coherence,orientation)
  discovery.py   — Strike (angle in (col,row) basis: vx=col, vy=row), system_strikes,
                   strike_rays, relay_corridors, noisy_or_fuse
  systems.py     — label_systems → (ids int32, n, sizes); CONNECTIVITY_8
  alignment.py   — NEW session 2: strip NCC engine (H13) + ray continuation (H16)
  modeling.py    — fit_hgb (NaN-native), apply_policy (binary/soft/envelope/halo2),
                   parse_policy ('topkNN_mode')
  metric.py      — components → DTI (official formula, R=3px)
  release.py     — fail-closed release gates (protocols v1 + policy-sweep-v2)
  novelty.py     — raster_identity, refuse_duplicate (sibling audit)
  raster.py      — write_submission, assert_submittable (9-rule gate), sha256_file
  placement.py   — topk_mask
  selftrain.py, external.py — earlier-session modules
scripts/
  build_features.py      — 107-channel plan, HALO=40, --tile-rows (mem-bound safe)
  build_offset.py        — H13: 39 channels → data/features_offset.npy (tiled, low-mem)
  build_continuation.py  — H16 all-systems builder (per-fold use is in validate_candidate)
  validate_candidate.py  — generic H13/H16 paired spatial validation + policy sweep v2
  train_final.py         — full-data baseline107 fit → prob_final.npy + manifest
  build_submission.py    — release-staged TIF/ZIP/JSON publishing (no upload)
  build_site.py          — static evidence hub (docs/)
tests/ — 15 files, 107 tests green (2026-09-27)
```

### H13 strip-NCC engine — calibrated spec (`alignment.py`)

- Bands: rtp, tmi, iso_grav_anom; angles 0/45/90/135 (45/135 = one order-1 rotation,
  documented approximation); strips = (row i, row i−d) in strip space, d ∈ {1,2,3} px;
  window 16 px (3.3 km); lags {0, 2, 4, 6} px (200/400/600 m); 28-px invalid halo.
- Channels per (band, angle): `ncc0`, `ncc_best` (sensitivity indicator, +0.17 null
  bias documented), `adv` = mean_d(mean_ℓ NCC_ℓ − NCC_0) — the **unbiased zero-lag dip**;
  plus per-band `advmax`. 39 channels total.
- **Calibration history (pre-holdout-contact, synthetic nulls):** draft-1
  (adv = max(NCC_ℓ−NCC_0), window 8, lags 1–6) measured mean adv **+0.35** on white
  noise — max-selection bias. Fixed as above; post-fix null mean adv **+0.009**.
  Verified by `tests/test_alignment.py` (displaced lineation, no-shift control, white
  noise, NaN halo, 90° convention, tiled==full bit-identical, ray geometry, 1-D NCC).
- 90° bug fixed in session 2: the draft compared a column with itself row-shifted;
  correct convention compares adjacent columns (consistent with 0°).

### H16 ray-continuation spec

- Systems from labels via `systems.label_systems` → `discovery.system_strikes`;
  per fold **train-side systems only** (n_px ≥ 8, width ≤ 20 px filter in engine).
- 10 channels: `cont_{tmi,det_elev}_r{5,15,30,60}` = energy·coherence·
  max(0, cos 2(φ−θ_ray)) at outward ray points (3-px disk, max over systems);
  `cont_ncc_r{15,30}` (rtp) = 1-D front/back NCC (33-px ray, halves 16) minus
  6-px-perpendicular control ray (zero-variance control contributes nothing —
  documented convention). All NaN where no train ray covers the pixel.

### Validation protocol (frozen, `HYPOTHESES.md` session-2 section)

- 4×4 strided spatial folds, buffer 40 px; folds 0–2 development, fold 3 confirmation
  (re-used geography — explicitly a limitation).
- HGB 200 iters, lr .05, depth 7, L2 1, seed 7; ≤200k negatives; all feature building
  blind to held-out labels (H16 rebuilt from train systems per fold).
- 8 pre-registered policies (topk01/02/03/04/06_binary, topk02_soft/envelope/halo2);
  per-arm policy = argmax dev-mean DTI, ties → topk02_binary; no confirm-fold tuning.
- **Decision rule:** candidate promotes only if it beats BOTH baseline107 and
  baseline107_discovery (each under its own selected policy) on dev mean AND on the
  confirmation fold. `promotion_allowed` is always False in the report — publication
  additionally requires a final-training binding (manifest with probability hash,
  input/config/code hashes) verified by `release.verify_training_binding`.
- Random-budget control reported per fold.

## 6. Known dead ends / errors (do not repeat)

- NCC max-selection bias (see §5) — use the pooled zero-lag dip, never raw max.
- Full-grid float64 offset build OOMs on this 4 GB box → use `offset_channels_tiled`
  (bit-identical, ~500 MB peak).
- `strip_ncc` on arrays smaller than the border halo returns all-NaN (debug artifact).
- 45/135° rotated strips: use (row,col) dot = ur·vy + uc·vx for the outward check
  (vx=col, vy=row convention) — the (ur·vx+uc·vy) mixup silently flips rays.
- H12 REJECTED on identical geography (dev Δ +0.001218, confirm Δ −0.002171,
  `reports/h12_blocked.json`): a shape-polarity feature does not beat the 107-channel
  baseline; don't retune on the confirm fold.
- Fold-0 random control (0.1492) exceeded baseline107 fold-0 (0.1171) once — with only
  4 correlated stripe folds, sub-0.01 deltas are noise; never read them as signal.
- Historical `reports/cv_*.json` (component-holdout: ALL top2 0.0929/0.0508 FAR10) are a
  different protocol — not comparable to spatial blocked scores.
- System python is bare; always use `.venv/bin/python` (lock: numpy 2.4.6, scipy
  1.17.1, sklearn 1.9.1, rasterio 1.4.4, pytest 9.1.1).
- bash/curl reach only api.github.com here; fetch_page tool reaches everything.
- `[0,1]` upload rejections = NaN inside footprint (gate) or first-row zeros
  (serialization) — both fixed; `assert_submittable` re-checks every publish.

## 7. Standing project commitments (user-directed)

1. README carries the full project prompt; read it every session.
2. Core values as decision framework: **Maximize P(Win)**, **Own the Outcome**.
3. No slot is spent on anything that has not beaten the current holdout best.
4. Hypotheses must be genuinely new vs. implemented code (register: `HYPOTHESES.md`).
5. Site: one-click TIF download, obvious at top/executive summary; values strictly in
   [0,1]; unique filename + short DrivenData note.
6. Free/official external data only; unobtainable ⇒ hypothesis not viable here.
7. 3-pass discipline (implement/verify → review & fix → re-check vs original request);
   end with merged PR to main + remaining-work list.
8. Leaderboard goal: top placement (beat 0.3049) — contrarian but scientifically grounded.

## 8. Reproduce

```
.venv/bin/python scripts/fetch_data.py --download          # data (pinned hashes)
OMP_NUM_THREADS=2 .venv/bin/python scripts/build_features.py --tile-rows 128
.venv/bin/python scripts/build_offset.py                   # H13 39-ch (tiled)
.venv/bin/python scripts/validate_candidate.py --hypothesis H16
.venv/bin/python scripts/validate_candidate.py --hypothesis H13 --extra data/features_offset.npy
.venv/bin/python -m pytest tests/ -q                       # 107 tests
```
