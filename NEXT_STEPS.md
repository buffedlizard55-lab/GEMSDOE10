# Next steps — after session 5 (2026-09-28)

Read README.md (including the preserved project prompt), HYPOTHESES.md
(session-5 register + results), KNOWLEDGE.md §5d, REVIEW.md (S5-1…S5-8) and
LIMITATIONS.md first. Ranked by expected effect on P(win) per unit of work;
nothing here promises a leaderboard score.

## Current state (measured)

- **Recommended artifact — H28:** `gems10-h28-dotted-ridge-20260928T020256236880Z-6452ae1d00.tif` (sha256 `7637b72d…`,
  suggested note `gems10-h28-dotted-ridge | ridge20_d3 | sha 7637b72d4c`). The
  H25 model (final probability grid and model pickle byte-identical to the
  session-4 release) with the along-strike dotted ridge emission `ridge20_d3`:
  69,281 cells (1.34 % of the footprint, ~40 % of H25's), all isolated pixels;
  22.2 % of mass within 300 m of catalogue faults (H25: 27.8 %). Protocol v5:
  density-matched dev mean **0.16909** / confirmation **0.18697** vs **0.13718 /
  0.14986** for H25/ridge15 (+23 % / +25 %, 24 of 24 simulated truths); full
  density 0.23687 / 0.26330 (not worse). Hash-bound: `reports/h28_blocked.json` +
  `reports/final_manifest_h28.json`.
- **H29 (km-scale line support) not eligible:** confirmation −0.0060 vs H25
  under v5; neutral at full density. Negative recorded.
- **Suggested use of the next weekly slot (account holder's call):** H28 is the
  only candidate that wins the density-matched test; if it is uploaded, record
  the returned score with its file hash — it is also the first GEMSDOE10
  artifact whose leaderboard score would calibrate the proxy.

- **Leaderboard probe (new evidence):** all seven scored group artifacts are
  binary; inverting their scores bounds the hidden new-fault density to
  ~0.13–0.6 % of the scored area (2–8× sparser than the catalogue proxy). The
  worst artifact put 68 % of its pixels within 300 m of known faults. Our local
  folds cannot see either effect — hence protocol v5 (`reports/lb_probe.json`).
- **1 m lidar coverage measured:** USGS 3DEP 1 m tiles cover **98.3 %** of the
  footprint (686 tiles, 157 GB); GeoDAWN West Central alone 86.8 %
  (`reports/lidar1m_inventory.json`, runner tag `ext/lidar1m-inventory-36365777149`).
- Exact reproduction extended to grids: baseline107 and H25 OOF grids are
  byte-identical to session 4's.
- No slot has been spent by this repository. The account holder decides uploads.

## P0: H31 — metre-scale scarp channels from the GeoDAWN 1 m lidar

Why first: the lidar covers essentially the whole region and is the most
plausible data behind the experts' "new" faults (1–5 m piedmont scarps are
invisible at 10–100 m). H20's 10 m aggregates already gave the largest
single feature gain in this repo.

Design (runner job **`scarp1m`**, sharded; the inventory job `lidar1m` is done). Name it so it does not CONTAIN another job's name: the workflow selects jobs with `contains(JOBS, '<name>')`, a substring test, so `lidar1m_scarp` would also re-run the inventory job:
1. Matrix of ~24 shards over `inventory.json` tiles (deterministic split by key;
   ~6.5 GB per shard). Each shard streams its tiles (never keeps more than one
   10 k × 10 k tile), computes metre-scale statistics in the tile's native
   NAD83 UTM, then accumulates them into 100 m competition cells (EPSG:32611,
   template transform) with sum/max/count accumulators. Zone-10 tiles: map the
   10 m block centres to 32611 with `rasterio.warp.transform` (coarse grid +
   bilinear interpolation of coordinates), not per metre pixel.
2. Channels (label-free, mirror `scripts/ext/dem10_scarp.py` at 1 m scales):
   slope max/p95 per cell; residual relief at 2/5/20 m (z − G_σ z);
   one-sidedness of the residual gradient (scarp vs channel) at 5 m and 20 m;
   steep-run length (longest connected run of slope > 15° along the dominant
   orientation); scarp-template (Hilley et al. 2010-style) amplitude at 5–20 m
   widths; valid fraction.
3. Publish each shard as `ext/lidar1m-<shard>-<run>` (files < 95 MB, float32
   vectors over footprint indices), plus a merge script in the sandbox
   (`scripts/build_lidar1m_grid.py`) that sha-checks and assembles the grid.
4. Validate as H31 = H25 stack + lidar channels under protocol v5 (the
   density-matched decision, with full-density non-inferiority); same release gate.
Budget: ~1 h wall per shard on standard runners (download ~2 min/GB + compute
~30–60 s per tile). Test the per-tile function locally on a synthetic 1 m DEM
first; the sandbox cannot read the real tiles.

## P1: close the known-fault "shadow" blind spot (H30)

Our released fields put 28–46 % of their mass within 300 m of catalogue faults;
the final model is fitted in-sample on those faults, so smooth features make
"shadow" ridges beside them. Build an **in-sample masked-known protocol**: per
fold, add a random subset of the held-out systems to TRAINING as positives
(exactly like the final fit sees known faults), score the remaining held-out
systems with the added ones masked, and compare emission with/without excluding
R ∈ {1, 2} px around known pixels. Only this protocol can measure the effect.

## P2: carried hypotheses

- **H22** ComCat seismicity lineaments (needs a runner job; FDSN event service).
- **H18** cross-field edge coincidence (provided bands only).
- **H27 variant**: H25 layout + corrected-phase (aligned) continuation gate.
- **H15** paleo-discharge points (archives in tag `ext/catalogue-36326816737`).

## P3: hygiene

- Reuse `scripts/evaluate_density_matched.py` for every future candidate (it
  only needs saved OOF grids); keep `--policy-set anchor` for OOF generation.
- Keep `reports/official_feed.json` dated; the leaderboard moves intra-day.
- Never pipe long runs through `tee`; use `MALLOC_ARENA_MAX=2` on 4 GB hosts.

## Limitations in the way

- Private truth is hidden: every local number is a proxy. The density bound is
  derived from user-reported scores whose account/file association is not
  authenticated.
- The sandbox reaches only GitHub/PyPI; USGS/AWS/DrivenData data flow through
  the GitHub runner and immutable `ext/*` tags (or the research fetch tool for
  small pages).
- 2 CPU / 3.8 GB RAM: one validation run at a time (~15 min for 3 arms).
- Uploads, weekly-slot accounting and final-submission choice require the
  authorized DrivenData account holder.
