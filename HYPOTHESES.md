# Hypothesis register — frozen before implementation, 2026-09-27

Read README.md and PROJECT_BRIEF.md first. **No weekly slot without a measured improvement over the current best on the same spatial holdout.** Expected improvement rankings below are qualitative scientific judgments, not predicted DTI gains. None can promise the public leader's score (0.3168, re-verified 2026-09-27 session 4; was 0.3049 earlier the same day). The target is fault pixels, not geothermal vents, heat production, or resource reserves.

## What has already been tried

Code audit at base commit `4805254`: 19 input bands; horizontal/vertical gradients, analytic signal, tilt, multiscale curvature and slope-of-slope; structure tensors; Frangi/Gabor ridges; strain/conductivity/seismicity products; HGB; whole-component holdouts; top-k/soft/halo policies; catalogue strike rays and relay corridors; self-training; lidar ridge and radiometric external stacks. See `scripts/build_features.py`, `src/gems10/{features,discovery,selftrain,external,placement}.py`. The persisted external data and trained models are absent in this checkout. Claims of their prior performance are historical reports, not fresh reproductions.

Sibling websites show neural-network ensembles (1/2/5), gap-catalogue supervision and ridge/node placement (3/4), HGB structural transforms (6), and lidar/radiometric coherence (8). Novelty here means the explicit test below is not implemented in this checkout or described in the reviewed sibling landing pages; it is not a claim to worldwide novelty or an exhaustive audit of every sibling experiment.

## Ranked candidates (four distinct physical mechanisms)

| Expected DTI opportunity rank / cost rank | Hypothesis and specific layers | Physical signature / proposed transform | Why it could find a missing fault, not just redraw the catalogue | Difference from implemented work / falsifier | Data readiness |
|---|---|---|---|---|---|
| **1 / 1 (low)** H12: displaced-surface step versus erosional valley | competition `det_elev` (band 12); existing 107 channels as matched baseline | Fit an odd step kernel orthogonal to eight candidate strikes, explicitly projecting out a local plane. Contrast with even valley/ridge curvature, and require same-polarity response along strike. Radii 2 and 4 pixels (200/400 m); no catalogue distance or endpoints. | Small-offset secondary scarps and strands can interrupt an otherwise continuous surface even where existing trace mapping missed them. One-sided relief and continuity could reject channels that generic line detectors call faults. 100 m aggregation may erase the subtle signal: this is a coarse-resolution pilot, not a substitute for lidar. | Existing Frangi/Gabor are sign-symmetric ridge/curvature detectors; no planar-detrended odd step template with an even-profile discriminator and along-strike polarity persistence. Fails if blocked DTI does not improve or roads/terraces dominate. | In provided feature raster; no new external data. Restore pinned bytes before testing. |
| **2 / 3 (high)** H13: lateral displacement of geophysical texture | `rtp`, `tmi`, `mag_anom`; `det_elev` for surface context | Along-strike lagged normalized cross-correlation between opposing strips of a putative trace; improvement of shifted alignment over zero-lag alignment, supported across independent strips/scales. | A strike-slip discontinuity can offset a magnetic texture without producing a strong ridge at the actual slip plane; could expose missing Walker Lane splays. Not every magnetic contact is tectonic. | Current code detects intensity/curvature, not restored matching of offset marker textures; not another weighted union or catalogue ray. Falsify with shuffled-strip and lithologic-contact controls and inconsistent displacement signs. | Provided stack only; magnetic bands are correlated, not independent corroboration. No expected numeric gain asserted. |
| **3 / 2 (medium)** H14: concealed basin-margin source boundary | `iso_grav_anom` and `rtp`; surface relief as negative/stratification control | Potential-field upward continuation in Fourier space using exp(-height × radial wavenumber), then track edge position and orientation with continuation height; seek stable gravity boundaries with weak surface expression. | Basin-fill cover may hide a surface trace while retaining a basement density boundary; hidden intra-basin faults need not be near catalogue endpoints. A lithologic edge is an alternative explanation, not proof of faulting. | Existing Gaussian multiscale derivatives do not model potential-field continuation or edge-position persistence with height. This is related to, but not identical to, existing edge detection; lower priority than H12/H13 because novelty and 100 m localization are uncertain. | Provided stack only. Edge padding, regional trends and source-depth ambiguity must be tested. |
| **4 / 4 (high; BLOCKED)** H15: diffuse past discharge marking a permeable fault corridor | INGENIOUS paleo-geothermal deposits, 2 m temperature probes, spring/well temperatures and chemistry; `det_elev`, `cond_surf` | Residual temperature after elevation/background adjustment, aligned chains of sinter/tufa occurrences and coincident conductivity contrasts; never distance-to-known-fault as a feature. | Fossil/dispersed discharge can identify permeability not captured by mapped scarps. But hydrologic discharge may occur away from a fault and observation density is biased toward known resources. | No thermal residual or paleo-discharge observation model in this repo; distinct from magnetic/DEM lineaments. Requires geographic coverage census and surveyed-background controls; not usable as negative labels. | Specific official source: [GDR 1391](https://gdr.openei.org/submissions/1391), CC BY 4.0. Landing page and exact archive URLs verified, but binary downloads fail TLS in this runtime. **Not currently viable / not eligible for implementation or submission** until archives are obtained, licensed attribution retained, and in-footprint coverage measured. |

## Scientific support and boundaries of inference

- H12: USGS documents geomorphic mapping of recent fault strands from lidar, including scarps crossing alluvial fans: [USGS Earthquake Hazards maps](https://www.usgs.gov/programs/earthquake-hazards/maps). This supports examining topographic offsets, **not** the effectiveness of our proposed kernel or its 100 m resolution. [USGS-hosted Dog Valley award report](https://earthquake.usgs.gov/cfusion/external_grants/reports/G20AP00055.pdf) describes scarps, benches, linear valleys and alternating facing directions. It is a hosted grantee report, not a universal fault classifier. A fixed-polarity requirement will miss some strike-slip features.
- H13/H14: [USGS Granite Springs Valley publication, 2022](https://pubs.usgs.gov/publication/70259621) describes gravity/magnetic derivatives to delineate buried faults **and contacts**, and a geothermal system without definitive surface manifestations. It motivates testing geophysical structure but does not validate our cross-correlation or continuation algorithm. [USGS OFR 2000-420](https://pubs.usgs.gov/publication/ofr00420) describes gravity profiles locating possible faults concealed by basin fill.
- H15: [official GDR resource list](https://gdr.openei.org/submissions/1391) describes shallow temperature, paleo deposits, and well/spring observations. Exact archives: [paleo](https://gdr.openei.org/files/1391/paleo_geothermal_regional.zip), [2 m probes](https://gdr.openei.org/files/1391/2m_temperature_probe_INGENIOUS_regional_data.zip), [well/spring](https://gdr.openei.org/files/1391/wellspringdata.gdb.zip). Page access is not equivalent to successful binary acquisition.
- [DrivenData problem](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) defines the prediction target and allowed external data. [Staff 11536/2](https://community.drivendata.org/t/where-do-you-draw-the-line/11536/2) defines new as any fault pixel not already captured by USGS/INGENIOUS, including new geometry of existing systems. Thus distance >1 km from known traces is a sensitivity check, **not** a verified property of hidden labels or an exclusive selection target.

## Session-4 register (2026-09-27, session 4) — frozen BEFORE any new holdout contact

**Verified context this session (before any scoring):** feature stack rebuilt from the
mirror-restored `training_features.tif` (sha256 `4371c82e…` matches the spec pin);
`data/features107.f32.npy` rebuilt and measured equal to the session-3 report pin
(`fb0cfb4060474b374bbfabd431cd8ab9c4ef1c55c322244295e326c259a4fe80`); tag
`ext/dem10-36326816737` restored through `scripts/fetch_external.py` (13 channels +
manifest, all sha256-checked) and reassembled with `scripts/build_dem10_grid.py`
(all 13 channels finite on the footprint); 110 tests green. Current **holdout best =
H20 arm, `thin10_binary`** (`reports/h20_blocked.json`: development mean **0.18984**,
confirmation **0.18680**). Every candidate below must beat those two numbers (not the
older H16 incumbent) under the frozen spatial protocol to be eligible.

**New first-hand engineering finding this session (motivates H27):** the
`Lineament.orientation` returned by `features.structure_tensor` is the **normal to the
line** (direction of greatest change), measured empirically this session: a horizontal
stripe (line along columns) yields orientation 90°, a vertical stripe 0°, a 45° stripe
−45° — the module docstring comment ("direction of least change (the line)") is
incorrect. `alignment.ray_continuation_channels` gates lineament energy with
`clip(cos(2·(o − strike)), 0, 1)`, where `strike` is the PCA line direction
(`discovery.Strike.angle`): for a lineament *parallel* to the system strike
(o = strike ± 90°) this equals −1 and is clipped to **0**, so the implemented `cont_*`
channels fire on lineaments *perpendicular* to the continuation direction, the
opposite of the preregistered "orientation-matched" intent (H16 item 2). The released
H16/H20 artifacts are bound to the as-implemented behaviour and are not thereby
invalidated — the measured numbers stand — but the corrected phase has never been
scored. Recorded in REVIEW.md this session.

**Public leaderboard re-read first-hand this session (2026-09-27, session 4,
pages 1–2 via fetch_page):** #1 DARD **0.3168** — the 0.3049 figure in the
preserved prompt and the session-3 register is now stale (DARD improved
intra-day); #2–#5 unchanged (0.2993 / 0.2854 / 0.2843 / 0.2806); the triple
**0.1563** sits at ranks #26–28 (`extradr19`, `SDCF9`, `smashi34`); top-50
floor 0.0982. Recorded in `reports/official_feed.json`.

**External-source checks for candidates that need data (2026-09-27):**
- H22 needs the USGS ComCat event catalogue: the official FDSN service descriptor
  `https://earthquake.usgs.gov/fdsnws/event/1/application.json` was fetched
  successfully this session (fetch_page) → source obtainable; bulk event queries are
  still runner-transport only (sandbox TLS restriction, KNOWLEDGE §5b).
- H24/H25/H27 need **no new external data**: H24 operates on saved OOF probability
  fields; H25 derives from the already-restored `dem10` tag; H27 rebuilds provided-band
  lineaments. No candidate is proposed as viable without its source in hand.
- H23 needs the *hidden truth density*, which no free official source can provide
  (staff explicitly withhold test coverage — [11527/7](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527/7));
  therefore H23 is **analysis-only** this session and cannot consume a slot.
- The 1 m lidar follow-up (H20-a) needs the competition's `1m_DEM_links.csv`
  (login-gated) or its Dropbox PDF mirror (TLS-failed from this runtime; not
  re-verified today) → **not proposed as viable this session**; blocked with a named
  path in NEXT_STEPS.md.

**Ranked candidates (rank = qualitative expected DTI opportunity per unit cost vs the
H20/thin10 holdout best; no numeric forecast):**

| Rank | ID · layers | Physical signature / transform | Why it can catch a fault missing from USGS/INGENIOUS rather than redraw the catalogue | Difference from everything implemented here | Data readiness / cost |
|---|---|---|---|---|---|
| **1 / low–medium** | **H24 — probability-ridge NMS emission.** No new layer: the saved OOF probability fields of the H20 stack (107 baseline + H16 continuation 10 + 3DEP-10 m 13). | New policy family `ridgeNN_binary` (NN ∈ {4,6,8,10,12,15}% pre-NMS budget): take the top-NN% mask, then keep only pixels that are local maxima **across the local strike** of p — normal direction from `structure_tensor(p)` (verified above to be the gradient/normal), comparisons at ±1 and ±2 px along the normal, lexicographic tie-break (p, then distance-to-mask-edge) so flat HGB plateaus reduce to their interior spine — emit 1.0. | It detects nothing new by itself; like H19 it converts across-strike FP width into budget for *additional* candidate structures, but unlike H19's Zhang–Suen skeleton (which follows the **binary mask geometry** and drifted >300 m off the sparse confirmation fold, `reports/h19_blocked.json`) it follows the **probability ridge** — the model's own best line — and by construction cannot leave the top-k support. Whether ridge emission beats the already-localised `thin10` selection on the H20 field is exactly what the holdout will measure. | No ridge/NMS policy exists in `modeling`/`placement` (grepped): the policy union is top-k (binary/soft/envelope/halo2) and mask-skeleton thinning only. Not a union of priors, not a new detector. Falsifier: selected ridge policy fails to beat H20/thin10 on development mean **and** confirmation. | Provided data + OOF fields regenerated by the frozen H20 protocol (the rerun doubles as an exact-reproduction check of `reports/h20_blocked.json`). Low–medium cost: one policy engine + one 3-arm run (~40 min). |
| **2 / medium** | **H25 — spatial context of the external 10 m channels.** Same stack as H24 plus 36 new channels built from the restored `dem10` grid: for the six mechanistic channels {`slope_max`, `steep_ratio_max`, `onesided`, `onesided3`, `hgm200_mean`, `resid_range`}, NaN-aware box **mean / max / std over 7×7 and 15×15** (0.7 / 1.5 km) → `ctx_*` = 6×6 = 36; total 166 features. | Regional context of a local signature: a scarp candidate is more credible where the surrounding kilometre is also steep/one-sided, and less credible where the same local relief is an isolated noise patch. HGB sees only per-pixel values, so it cannot infer "is this pixel part of a region…" without these aggregates. | The 13 external channels enter the model raw (H20); no neighbourhood statistic exists over the external stack anywhere in this checkout (grepped `uniform_filter`/`window`/`context` — only provided-band multiscale builds and 1-D NCC windows). It is not another local edge detector and not a catalogue prior. Falsifier: H25 arm fails to beat the same-run H20 arm and the external H20/thin10 holdout best on development mean **and** confirmation. | Same tag `ext/dem10-36326816737` already in `data/external/dem10/`; builder derives the context grid locally (no network). Medium cost: one builder + arm plumbing + one 4-arm run (~55 min). |
| **3 / high / unresolvable locally** | **H23 — density-robust emission-budget rule.** No new layer; a decision rule mapping (possibly sparsified) truth density to the emission budget of the *released field*. | Decision rule, not physics: choose the budget so the marginal hit-rate estimate satisfies the metric's break-even identity (q\* ≈ DTI/5 family) under simulated sparsity, instead of argmax development DTI at catalogue density. | The hidden truth is the *new*-fault set only — sparser than any local fold (session-3 sweep: best budget moves thin12 → thin04 as simulated density falls, `reports/budget_density_sweep.json`); a budget matched to that regime would spend fewer FP-weight pixels per hit everywhere at once. | Nothing in the repo selects a budget by a density-robust rule; all selection is argmax at catalogue density. **Why it cannot be validated this session:** the rule must be chosen now, but the session-3 sweep has already revealed the density→optimum mapping (any rule written today is informed by it — not blind), the hidden density is not observable from any free official source, and a release of a non-selected policy is refused by the gate by design. → **Analysis-only**; the honest validation paths are (i) a user-approved *measurement* pair of uploads of the same field at two budgets (user decision, not ours) or (ii) future evidence that identifies the hidden density. No slot, no autonomous promotion. | Zero compute; works from existing reports. Cost: documentation only. |
| **4 / medium** | **H27 — corrected-phase lineament gate (aligned-lineament continuation).** Provided bands `tmi`, `det_elev`, `rtp`; same rays/radii/NCC controls as H16; gate becomes `clip(−cos(2·(o − strike)), 0, 1)` so lineaments **parallel** to the strike (o ⊥ … as measured) pass, matching the preregistration's "orientation-matched" wording. Applied to a new arm `[107 + 10 fixed + 13 dem10]` (130 features) so the comparison against the H20 holdout best is like-for-like. | The same continuation evidence as H16 but with the physically intended phase: energy of lineaments that *continue the system's own orientation* beyond the endpoint, minus the same NCC controls. | Staff say new geometry concentrates around existing systems (11516/4, 11536/2); if genuinely strike-parallel continuation structure is what marks unmapped strands, the as-built gate suppressed exactly that signal and the corrected channels can carry it. | The implemented `cont_*` channels are 90° out of phase with their preregistered intent (measured this session, above); the corrected phase has never been trained or scored. Not a re-threshold, not a union. Falsifier: fixed arm fails to beat the same-run as-built H20 arm and external H20/thin10 on development mean **and** confirmation. | Provided data only. Builder gains an explicit `gate_phase` parameter (default = as-built, bit-identical) so all incumbents reproduce unchanged. Medium cost: parameterised builder + one 4-arm run (~55 min); rank-4 only because it must both fix the phase *and* out-score the dem10-extended incumbent. |
| **5 / high; data-checkable but not runnable here** | **H22 — seismicity lineament coherence** (carried from the session-3 register, source re-checked today). USGS ComCat FDSN `https://earthquake.usgs.gov/fdsnws/event/1/` (public domain, application descriptor fetched 2026-09-27). | Strike-aligned anisotropic kernel density of event hypocentres at 2–10 km scales + structure-tensor coherence, vs the provided 100 km-radius density bands. | Micro-seismicity alignments mark active structure without fresh scarps and independent of magnetics/gravity/topography. | No event-level lineament transform exists here (provided bands are regional densities). | Bulk query must run on the GitHub runner (transport pattern of `scripts/ext/`); ranked below H24/H25/H27 for this session because the transport job does not exist yet and most Quaternary faults are seismically quiet on catalogue timescales. Not run this session. |

**Preregistered session-4 protocol (identical folds, buffer, HGB settings, sampling and
policy-selection rule as sessions 2–3; union extended to v4 = v3 ∪ {ridge04, ridge06,
ridge08, ridge10, ridge12, ridge15}_binary):**

1. **Incumbent.** `reports/h20_blocked.json`, H20 arm under `thin10_binary`
   (development mean 0.18984, confirmation 0.18680). The validation runner gains an
   explicit `--incumbent-arm` so the comparison reads the named arm's selected policy
   from the incumbent report; the comparison key is recorded as
   `incumbent_report:H20` and must pass like any other.
2. **H24.** Rerun the frozen H20 protocol with the v4 union
   (`--hypothesis H20 --policy-set v4 --report reports/h24_blocked.json
   --incumbent-report reports/h20_blocked.json --incumbent-arm H20`). First, the
   shared-policy scores of arms baseline107 / baseline107_discovery / H16 / H20 must
   reproduce `reports/h20_blocked.json` exactly (max |ΔDTI| = 0); otherwise stop and
   diagnose. Then per-arm selection = argmax mean development DTI over the v4 union.
   H24 is eligible only if the H20 arm's selected policy beats the external incumbent
   (H20/thin10) on the development mean **and** the confirmation fold and also beats
   every same-run arm under its own selected policy. If selection returns
   `thin10_binary` itself (a tie), deltas are 0 and eligibility correctly fails.
3. **H25.** One 4-arm run (`--hypothesis H25 --policy-set v4 --extra
   dem10_channels.f32.npy,dem10_context.f32.npy --report reports/h25_blocked.json
   --incumbent-report reports/h20_blocked.json --incumbent-arm H20`); arms =
   baseline107, H16, H20 (same-run reproduction anchor on shared v3 policies), H25.
   H25 is eligible only if its selected policy beats the external incumbent
   (H20/thin10) and every same-run arm on development mean **and** confirmation fold.
   The context grid builder, its channel list and window sizes are frozen above; no
   channel subset selection after results.
4. **H27.** Deferred to a later run this session or the next session; if run, arms =
   baseline107, H16, H20 (as-built), H27 (fixed gate); eligibility identical to item 3
   against the external H20/thin10 incumbent and all same-run arms. The default
   `gate_phase` remains as-built so items 2–3 reproduce exactly.
5. **H23.** No run. Documented above as analysis-only; it cannot consume a slot and
   cannot change the site's approved download.
6. **Release.** Only a candidate actually scored under items 2–4 may be released:
   (field, policy) pairs are never mixed post hoc; a passing candidate still needs the
   full-data fit with a hash-bound manifest (`train_final.py --bind-to`) and every
   `build_submission.py` gate. A failure at any step = no slot, negative recorded.
7. **What this cannot show.** Catalogue-generalisation proxy on four correlated stripe
   folds with a re-used confirmation geography and development-selected policy; nothing
   here forecasts the private score; no slot is spent by this repository.

## Session-4 results (2026-09-27, session 4) — measured, no slot spent

**H24 (`reports/h24_blocked.json`, protocol `spatial-4x4-strided-v1-buffer40-policy-sweep-v4`,
22 policies, elapsed 1905 s) — ELIGIBLE under protocol item 2.**

- **Reproduction gate passed exactly first:** the 16 shared v3 policies on arms
  baseline107 / baseline107_discovery / H16 / H20 reproduce `reports/h20_blocked.json`
  with `max_abs_dti_diff: 0.0, exact: true` (all four arms), before any new-policy
  score was used. Score-mask hashes and prediction-file hashes in the report were
  re-checked independently against the saved OOF grids.
- **Selection (preregistered argmax over development folds 0–2):** all four arms
  selected `ridge15_binary`. The candidate is therefore **H20 arm + ridge15_binary**.
- **Per-fold DTIs at the selected policies:**

  | fold | role | baseline107 | H16 | H20 (candidate) |
  |---|---|---|---|---|
  | 0 | development | 0.19443 | 0.19967 | **0.20406** |
  | 1 | development | 0.21744 | 0.22100 | **0.22408** |
  | 2 | development | 0.26341 | 0.26986 | **0.27846** |
  | 3 | confirmation | 0.21046 | 0.20728 | **0.22525** |

- **Eligibility (all `passes: true` in `decision.comparisons`):**
  - vs external incumbent **H20/thin10** (`reports/h20_blocked.json`): development-mean
    delta **+0.04569** (0.23553 vs 0.18984), confirmation delta **+0.03845**
    (0.22525 vs 0.18680);
  - vs same-run baseline107 (ridge15): dev **+0.01044**, conf **+0.01479**;
  - vs same-run H16 (ridge15): dev **+0.00536**, conf **+0.01797**;
  - vs same-run baseline107_discovery (ridge15): dev **+0.01044**, conf **+0.01479**
    — identical to the baseline107 deltas because the discovery fusion (10 px strike
    rays, 0.4/0.5 weights) is provably inside the 40 px score buffer, so the fused and
    raw fields coincide exactly on every score region (observed DTIs equal to all
    printed digits on all four folds).
- **H20-arm policy ladder (fold 0 / 1 / 2 / 3):** `topk02_binary` 0.11475 / 0.14225 /
  0.18679 / 0.19853; `topk06_binary` 0.15370 / 0.16992 / 0.21324 / 0.18021;
  `thin10_binary` 0.16483 / 0.18130 / 0.22339 / 0.18680; `ridge06_binary`
  0.16791 / 0.19147 / 0.25317 / 0.22707; `ridge10_binary` 0.19130 / 0.21569 /
  0.27183 / 0.22928; `ridge15_binary` 0.20406 / 0.22408 / 0.27846 / 0.22525.
  `ridge06` already beats `thin10` on every fold; the ladder is monotone on the
  development folds (ridge15 > ridge10 > ridge06 > thin10) and only ridge10 edges
  ridge15 on the confirmation fold — selection correctly used development folds only.
- **Independent verification performed before recording this result** (same session,
  separate script): (i) `dti = tp/(tp + 0.2·fp + 0.8·fn)` recomputed from stored
  components matches every printed DTI to 6 dp; (ii) ridge15 and thin10 emissions
  re-derived from the saved OOF grid `scratch/h24/H20_fold0.npy` match the stored
  `dti` and `n_pos_pred` exactly (0.204056/39099 and 0.164834/32517); (iii) the
  emission is a strict subset of its top-15% core (39,099 of 139,376 px kept);
  (iv) the fold-0 score-mask hash in the report matches a fresh computation.
  On fold 0 the candidate has TP_w 3,628 / FP_w 35,570 / FN_w 8,798 versus thin10's
  2,700 / 29,491 / 9,727 — TP +34 % against FP +21 %, the mechanism the register
  predicted (the 15 % pre-NMS budget reaches further down the probability ranking
  than thin10's 10 % support, and the strike-normal NMS keeps only the ridge of each
  reached structure).
- **What this cannot show (unchanged):** catalogue-generalisation proxy on four
  correlated stripe folds with a development-selected policy; the confirmation fold's
  geography has been used before in sessions 2–3; nothing here forecasts the private
  score; **no slot was spent**. The run's first attempt was OOM-killed mid-fold-1
  (float64 structure-tensor transients, `| tee` masked the kill); the retry used the
  float32 tensor fix + `MALLOC_ARENA_MAX=2` and reproduced everything exactly —
  recorded in `REVIEW.md`, `KNOWLEDGE.md` §5c.
- **Status:** new holdout best at the time of this writing = **H20/ridge15_binary**
  (development mean 0.23553, confirmation 0.22525) — superseded the same session by
  the H25 result below. Releasing a candidate as a downloadable artifact still
  requires the full-data fit with a hash-bound manifest (protocol item 6); nothing
  is submitted autonomously.

**H25 (`reports/h25_blocked.json`, protocol `spatial-4x4-strided-v1-buffer40-policy-sweep-v4`,
22 policies, elapsed 2507 s) — ELIGIBLE under protocol item 3.**

- **Reproduction gate passed exactly:** the shared v3 policies on arms baseline107 /
  baseline107_discovery / H16 / H20 again reproduce `reports/h20_blocked.json`
  (`max_abs_dti_diff: 0.0, exact: true`) — the H20 anchor arm inside this run is
  bit-for-bit the released incumbent field despite the H25 arm co-running.
- **Selection:** every arm selected `ridge15_binary`. Candidate = **H25 arm
  (166 features = 107 + H16 10 + dem10 13 + context 36) + ridge15_binary**.
- **Per-fold DTIs at the selected policies:**

  | fold | role | baseline107 | H16 | H20 | H25 (candidate) |
  |---|---|---|---|---|---|
  | 0 | development | 0.19443 | 0.19967 | 0.20406 | 0.20097 |
  | 1 | development | 0.21744 | 0.22100 | 0.22408 | **0.22476** |
  | 2 | development | 0.26341 | 0.26986 | 0.27846 | **0.28415** |
  | 3 | confirmation | 0.21046 | 0.20728 | 0.22525 | **0.25179** |

- **Eligibility (all `passes: true`):**
  - vs external incumbent **H20/thin10**: dev **+0.04679**, confirmation **+0.06498**;
  - vs same-run **H20** (each under its selected ridge15): dev **+0.00109**,
    confirmation **+0.02654** — the preregistered falsifier passes, but note the
    shape honestly: fold 0 is *negative* (0.20097 vs 0.20406) and the development
    margin is within fold noise; the confirmation margin is decisive;
  - vs baseline107 / baseline107_discovery: dev +0.01153, conf +0.04133; vs H16:
    dev +0.00645, conf +0.04450.
- **Where the context channels actually help (policy ladder):** under the *incumbent*
  policy `thin10_binary` the H25 arm's development mean is 0.18995 — i.e. **zero**
  versus the incumbent's 0.18984 — while its confirmation jumps to 0.21751
  (+0.0307 over 0.18680). The context gain is concentrated on the sparse
  confirmation geography; on the development folds it only interacts with the ridge
  policy (ridge15: H25 0.23663 vs H20 0.23553 dev mean, +0.00109).
- **Status:** new holdout best = **H25/ridge15_binary** (development mean 0.23663,
  confirmation 0.25179), ahead of H20/ridge15 (0.23553 / 0.22525) and of the released
  H20/thin10 (0.18984 / 0.18680). Same limits as H24: proxy folds, re-used
  confirmation geography, no forecast of the private score, no slot spent.

## Session-3 register (2026-09-27, later session) — frozen BEFORE any new holdout result

**Verified context this session (first-hand reads, links in `reports/official_feed.json`):**
[public leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) #1 DARD 0.3049 (session-3 read; superseded by the session-4 re-read above: 0.3168), then 0.2993 / 0.2854 / 0.2843 / 0.2806; an account named `doegemsDrivendata` sits at 0.1847 (its role is not stated on the page — do not call it the official benchmark without confirmation); **0.1563 appears three times** (`extradr19`, `SDCF9`, `smashi34`, ranks 25–27). The [problem page](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) states the hidden labels were "manually identified by fault experts", that the competition itself distributes `1m_DEM_links.csv` (1 m DEM download links) alongside the 100 m features, and that any external data is allowed if the licence permits use and sharing with the sponsor. Staff [11527/7](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527/7) (re-read; no newer staff post in that thread) still withhold data sources, fault types and coverage.

**What the session-2 numbers say about where the score is lost (measured, `reports/h16_blocked.json`):** at the selected policy `topk06_binary`, fold 0 has TP_w 3,084 / FP_w 49,365 / FN_w 9,342 — i.e. α·FP (9,873) is *larger* than β·FN (7,473). The FP mass comes from predicted pixels ≥300 m from any held-out trace: false structures **and** across-strike thickness of blobs around true structures. The truth is a rasterised 1-px line; TP is a per-truth-pixel *max* inside 300 m, FP is a per-predicted-pixel *sum*. Thickness therefore buys nothing and costs 0.2 per pixel. This is a property of the published metric, not a guess about the hidden labels.

**Ranked candidates (rank = qualitative expected DTI opportunity / implementation cost; no numeric forecast):**

| Rank | ID · layers | Physical signature / transform | Why it can catch a fault missing from USGS/INGENIOUS rather than redraw the catalogue | Difference from everything implemented here | Data readiness / cost |
|---|---|---|---|---|---|
| **1 / low** | **H19 — thin-line emission geometry.** No new layer; operates on the incumbent (H16, 117-ch) probability field. | Zhang–Suen skeleton of the top-k mask, emitted at 1.0 (`thinNN_binary`, NN = pre-thinning budget ∈ {2,4,6,8,10,12,15,20}%); the freed FP budget is spent on more distinct lineaments instead of on width. | It does not detect anything new by itself: it converts wasted across-strike FP mass into coverage of *additional* candidate structures the model already ranks highly — exactly the recall the β = 0.8 metric pays for. Traces in the catalogue are 1-px lines; predictions should be too. | The inherited `placement.skeleton` strategy was threshold-based, soft-valued and never scored under the spatial protocol; the session-2 sweep contained only binary/soft/envelope/halo2 (all at least as thick as top-k). No budget-preserving thinning exists in any report. | Provided data only. Rescoring of saved OOF grids (regenerated by re-running the frozen H16 protocol — which is itself a reproducibility check). |
| **2 / medium** | **H20 — 3DEP 10 m DEM scarp channels** (external, official, public domain: USGS 3DEP 1/3 arc-second seamless DEM, [catalogue item](https://www.sciencebase.gov/catalog/item/4f70aa9fe4b058caae3f8de5), tiles `prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/<tile>/USGS_13_<tile>.tif`, 12 tiles touch the footprint). | 13 label-free channels aggregated 10×10 to the official grid: 10 m slope max/mean/std; |∇G₂₀ₘ| and |∇G₅₀ₘ| maxima; regional |∇G₂₀₀ₘ|; steepening ratio; residual-relief std/range; |Laplacian G₃₀ₘ| max; **one-sidedness of the residual gradient** at 100 m and 300 m (scarp = one-sided step; channel = two facing banks). | A 2–10 m Quaternary scarp is a 20–50 m wide step: invisible at 100 m (`det_elev` cannot carry it) but a 10–25° steepening against a 1–3° fan at 10 m. Expert Quaternary-fault mapping in the Basin and Range is done on DEM/lidar relief; the competition ships 1 m DEM links, so DEM-visible scarps are an anticipated target family. New scarps far from mapped systems are precisely the traces a 100 m geophysical model cannot see. | All topographic channels in the 107 baseline derive from the 100 m `det_elev`; H12 tested an odd-step kernel at 100 m and was rejected — same physics at a resolution where the signal does not exist. Earlier sibling "lidar" stacks are not present, not reproducible and were never scored on the board (the 8GEMSDOE upload was the duplicate artifact). | Sandbox cannot reach the tiles (TLS, measured). Built on a GitHub-hosted runner by `.github/workflows/external-data.yml` → `scripts/ext/dem10_scarp.py`, published under an immutable tag, restored with `scripts/fetch_external.py`. 1 m tiles are the follow-up, not this run. |
| **3 / low–medium** | **H21 — catalogue-version differencing.** Current [USGS QFFD GIS](https://earthquake.usgs.gov/static/lfs/nshm/qfaults/Qfaults_GIS.zip) (public domain, DOI 10.5066/P9BCVRCK) and INGENIOUS Quaternary Faults v1/v2 ([GDR 1391](https://gdr.openei.org/submissions/1391), CC BY 4.0) vs the provided label raster. | Provenance, not physics: rasterise each catalogue on the template (centre and all-touched conventions), measure traces >300 m from any provided label pixel. | If the provided labels are an older snapshot, current-catalogue traces absent from them are unmasked and are the closest public analogue of "faults added by recent mapping" — usable as a near-target validation population and, if non-empty, as direct candidates. If the count is ~0 the result is a cheap, recorded negative. | No catalogue other than the provided raster has ever been compared here; NEXT_STEPS named the v2 audit but nothing was implemented. | Same workflow (`scripts/ext/catalogue_diff.py`); also fetches and hashes the H15 point archives (paleo, 2 m probes, wells/springs) so H15 stops being access-blocked. |
| **4 / medium** | **H22 — seismicity lineament coherence** from the raw event catalogue (USGS ComCat FDSN service, https://earthquake.usgs.gov/fdsnws/event/1/, public domain), `ieq_n100a15`/`deq_n100a15` only as context. | Strike-aligned anisotropic kernel density at 2–10 km scales and its structure-tensor coherence, instead of the provided 100 km-radius density. | Micro-seismicity alignments mark active structures without fresh scarps and independent of magnetics/gravity/topography. | The provided seismicity bands are 100 km-radius products; no event-level lineament transform exists here. | Obtainable only through the runner transport; not built this session (ranked below H20/H21 because most Quaternary faults are seismically quiet on catalogue timescales). |
| **5 / high** | **H15 — paleo-discharge / thermal residual** (carried; see session-2 row). | Residual 2 m temperature and aligned sinter/tufa chains. | Fossil discharge marks permeable structure without a scarp. | No observation model here. | Archives fetched + hashed by the H21 job this session; modelling deferred (sparse, resource-biased sampling). |

**Preregistered session-3 protocol (identical folds, buffer, HGB settings and sampling as session 2):**

1. **Incumbent.** The H16 arm under `topk06_binary` (dev 0.15197 / 0.17242 / 0.20094, mean 0.17511; confirmation 0.17274 — from `reports/h16_blocked.json`). Every session-3 candidate must beat this incumbent, not only baseline107/baseline107_discovery.
2. **H19.** Re-run the frozen H16 protocol (same seeds, same inputs — the feature stack rebuilt this session has sha256 `fb0cfb40…` identical to the report). The per-fold scores must reproduce the report exactly; otherwise stop and diagnose. Then score the policy union {session-2 eight} ∪ {thin02, thin04, thin06, thin08, thin10, thin12, thin15, thin20 (`_binary`)} on the saved OOF grids of every arm. Selected policy per arm = argmax mean development DTI (ties → session-2 rule). H19 is eligible only if the H16 arm's selected policy beats the incumbent number on the **development mean and on the confirmation fold**.
3. **H20.** Arms: baseline107 (107), H16 (117, incumbent), H16+H20 (130). Policy union as in item 2. Eligible only if H16+H20, under its selected policy, beats the H16 arm under *its* selected policy on the development mean **and** the confirmation fold (and, necessarily, both session-2 baselines). NaN handling: `dem10_*` are NaN outside the footprint; HGB handles NaN natively.
4. **H21.** Not a model. Report counts; if catalogue-new traces exist, evaluate them only as a *held-out* population (never as training labels this session) and record the incumbent's recall on them. Any use as prediction requires its own preregistration.
5. **Release.** Only a combination actually scored under items 2–3 may be released: (field, policy) pairs are never mixed post hoc. The passing candidate then needs the full-data fit with a hash-bound manifest (`train_final.py --bind-to`) and all `build_submission.py` gates. A failure at any step = no slot, negative result recorded.
6. **What this cannot show.** Catalogue-generalisation proxy on four correlated stripe folds; the confirmation geography is re-used; nothing here forecasts the private score.

## Session-3 results (2026-09-27, later session) — measured, no slot spent

- **Reproduction check passed.** Re-running the frozen H16 protocol on the rebuilt
  feature stack (`data/features107.f32.npy`, sha256 `fb0cfb40…4fe80`, identical to
  the session-2 report) reproduced **every** session-2 number exactly: 8 shared
  policies × 3 arms × 4 folds, max |ΔDTI| = 0.0 (`reports/h19_blocked.json →
  reproduction.exact = true`). The protocol is deterministic end to end.
- **H19 (thin-line emission): NOT ELIGIBLE** under the preregistered rule
  (`reports/h19_blocked.json`, protocol `…-policy-sweep-v3`, incumbent
  `reports/h16_blocked.json` H16/topk06_binary). Selected policy for the H16 arm =
  `thin15_binary`: development folds 0.16624 / 0.17822 / 0.20321 (mean **0.18256**,
  Δ **+0.00744** vs the incumbent 0.17511) but confirmation fold **0.15163**
  (Δ **−0.02111** vs 0.17274). Anatomy: on folds 0–2 thinning trades width for
  more distinct lineaments (fold 0: thin20 emits 60,279 px for TP_w 3,660 vs
  topk06's 55,750 px for TP_w 3,106); on fold 3 — the sparsest truth (0.84% of the
  score region vs 1.1–1.3%) — the skeleton of wide blobs drifts >300 m from the
  trace and TP_w falls (thin20 35,273 px → TP_w 1,571 vs topk06 38,102 px → 1,957).
  Development-selected policy also lost to the *baseline107* arm's own selected
  policy on the confirmation fold (Δ −0.00925). Decision: no release, negative
  recorded; the thin family is retained in the policy union (v3) for future arms.
  Side observation (not a selection): fold 3's optimum for the H16 field is
  `topk03_binary` (0.18792) — the budget optimum moves with truth density, which
  is analysed separately in `reports/budget_density_sweep.json`.
- **H21 (catalogue-version differencing): DECISIVE NEGATIVE.** Built on the GitHub
  runner (`ext/catalogue-36326816737`, `data/external/catalogue/catalogue_diff.json`,
  archives sha256-recorded, GDAL 3.8.4 reprojection to EPSG:32611, centre and
  all-touched rasterisation on the official template):
  | catalogue (current, official) | traces in bbox | centre-rasterised px in footprint | on label px | within 300 m, off-label | **>300 m from any label** | IoU with labels |
  |---|---|---|---|---|---|---|
  | USGS QFFD 2020 (`Qfaults_GIS.zip`, 32.4 MB) | 14,419 | 60,939 | 60,839 | 99 | **1** | 0.996 |
  | INGENIOUS Quaternary faults v1 | 1,148 | 60,982 | 60,958 | 23 | **1** | 0.999 |
  | INGENIOUS Quaternary faults v2 (2023-06-27) | 1,148 | 60,982 | 60,958 | 23 | **1** | 0.999 |
  The provided label raster (60,988 px) *is* the current public catalogue: only
  2 label pixels lie >300 m from the QFFD lines and 0 from INGENIOUS; the public
  catalogues contain **no** trace absent from the labels. Consequences: (i) there
  is no "catalogue-lag" population to validate on or to predict; (ii) the hidden
  new-fault truth is genuinely outside every public catalogue, exactly as the
  problem page states; (iii) H21 is closed. The H15 point archives were fetched
  and hashed in the same run (paleo-geothermal 709 points, 281 in footprint; 2 m
  probes 3,800 points, 2,782 in footprint; CSVs in `data/external/catalogue/`),
  so H15 is no longer access-blocked — it remains unmodelled.
- **H20 (3DEP 10 m scarp channels): ELIGIBLE under the preregistered rule, and
  RELEASED** (`reports/h20_blocked.json`, protocol `…-policy-sweep-v3`, arms
  baseline107 / H16 / H16+H20 = 130 features; incumbent `reports/h16_blocked.json`
  H16/topk06_binary). Selected policy for the H16+H20 arm = `thin10_binary`.
  | comparison (candidate H20/thin10 vs …) | fold 0 | fold 1 | fold 2 | dev mean Δ | confirmation Δ |
  |---|---|---|---|---|---|
  | incumbent report H16/topk06 (0.15197 / 0.17242 / 0.20094 · 0.17274) | +0.01286 | +0.00888 | +0.02245 | **+0.01473** | **+0.01406** |
  | same-run H16 arm under *its* selected policy (thin15) | −0.00141 | +0.00308 | +0.02018 | +0.00729 | +0.03517 |
  | baseline107 under its selected policy (thin12) | +0.00398 | +0.00888 | +0.01993 | +0.01093 | +0.02593 |
  Absolute H20 scores: 0.16483 / 0.18130 / 0.22339 (dev mean 0.18984), confirmation
  **0.18680**. Robustness reading (not a selection): H16+H20 also beats H16 under the
  incumbent's own `topk06_binary` (dev 0.17895 vs 0.17511; confirmation 0.18021 vs
  0.17274) and under `thin10_binary` on all four folds; the DEM channels are what
  keeps the thin policy from collapsing on the sparse confirmation fold (H20 thin10
  0.18680 vs H16 thin10 0.16803) — the 10 m relief localises the emitted line.
  Fold 1 is the weakest geography (−0.0025 at topk06, +0.0005 at thin10). Random
  budget control 0.1206–0.1492, below every arm. Data readiness: tag
  `ext/dem10-36326816737` (12 tiles, per-tile sha256/ETag/size recorded, 49
  footprint blocks, 440 s on the runner), all 13 channels finite on all 5,167,373
  footprint pixels; georeferencing check corr(`dem10_slope_mean`, slope of the
  provided `det_elev`) = 0.936, falling to 0.810 under a 3-px shift control;
  univariate label-vs-footprint AUCs only 0.49–0.58 (provided 100 m slope 0.564) —
  the gain is interaction-driven, as the register anticipated.
  **Release chain completed:** `build_continuation.py` (all systems) →
  `train_final.py --extra features_continuation.npy,dem10_channels.f32.npy
  --hypothesis H20 --bind-to reports/h20_blocked.json` (260,988 training rows ×
  130 features, 240 s; `reports/final_manifest_h20.json`) → `build_submission.py
  --policy thin10_binary` → **`gems10-h20-dem10-scarp-thin-20260927T155223039488Z-ffc91a1686.tif`**
  (sha256 `a26e5e46…2247`; 153,957 positive cells = 2.98% of the footprint, 143,657
  off-catalogue; format gate + novelty gate passed). It supersedes the H16 artifact
  on the hub; the H16 file stays downloadable for provenance.
  **Caveats that travel with it:** the policy was selected from 16 options on three
  development folds; the confirmation geography is re-used across sessions; the
  budget question (below) is open; nothing here forecasts the private score.
- **Budget-vs-density sensitivity** (`reports/budget_density_sweep.json`, H20 OOF
  grids, catalogue systems randomly thinned to f ∈ {1, ½, ¼, ⅒} with the remaining
  systems masked pixel-exactly as "known"): development-mean best policy thin12 →
  thin08 → thin04 → thin04 as f goes 1 → ½ → ¼ → ⅒ (DTI 0.189 → 0.137 → 0.095 →
  0.058; confirmation fold prefers topk02/topk02/topk01/topk01). Direction only —
  the model was trained on the full catalogue, so wide-budget penalties are
  overstated — but it says a sparser hidden truth wants *thinner* emission than
  the locally selected budget. Recorded so the next session can reason about the
  hidden density instead of assuming the catalogue's.

## Session-2 register (2026-09-27) — ranked untried candidates

**New verified staff fact (re-verified 2026-09-27, [11516/4](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516/4))**: the known-fault mask is **pixel-exact** (identical to the provided training labels); a predicted pixel near a known trace but far from new-fault ground truth is **fully penalized**; and new-fault ground truth **can lie within 300 m of a known trace** as "corrections or modifications to existing fault traces". Predicting on the known line itself is free (masked). This directly ranks strategies that target continuations, splays and corrections of existing systems above generic region-wide line detectors.

Candidates not yet implemented or tested in this checkout (H12 tested and rejected; H15 blocked by data access). Expected-impact ranks are qualitative scientific judgments conditioned on the verified facts above, not predicted DTI gains.

| Rank (impact / cost) | ID · layers | Physical signature / transform | Why it could catch a missing fault, not just redraw the catalogue | Difference from everything already implemented here | Data readiness |
|---|---|---|---|---|---|
| **1 / 2 (medium)** **H16: endpoint continuation by marker alignment.** `tmi`, `det_elev`, `rtp` bands; catalogue system strikes/endpoints (train systems only in CV) | Along each system endpoint ray (outward, 5/15/30/60 px = 0.5–6 km), score orientation-matched structure-tensor lineament energy (cos 2(φ−θ_ray) gate) plus corridor continuity: 1-D zero-lag NCC of the band sampled along the ray, front half vs back half, minus the same on a 6 px-offset control ray | Staff state new labels include corrections/modifications of existing traces within 300 m of known lines; mapped traces end where exposure ends, so unmapped continuations concentrate just beyond endpoints. Evidence that the same marker family keeps running past the endpoint, oriented to the local strike, is a continuation signature. | `discovery.strike_rays` is a fixed 10 px, fixed-weight, *prior* fused after the model; it is not a scored feature, does not use orientation-matched lineament energy, and does not measure marker continuity. This is endpoint-conditioned, multi-length, evidence-scored continuation as model features. | Provided raster + labels (systems). Per-fold rebuild from train systems only. No new external data. |
| **2 / 2 (medium)** **H13: lateral displacement of geophysical texture.** `rtp`, `tmi`, `iso_grav_anom` bands | Zero-difference normalized cross-correlation (NCC) of marker strips on opposite sides of a candidate trace (strikes 0/45/90/135°, separation 1–3 px), zero-lag vs best nonzero along-strike lag (1–6 px): advantage, zero-lag value, dominant lag | A strike-slip discontinuity offsets marker texture without a strong ridge on the slip plane; offset of magnetic/gravity lineaments can expose unmapped splays and en-echelon strands adjacent to existing systems (the region where staff say new geometry lives). Not every offset contact is tectonic — controls required. | Existing code detects intensity/curvature/ridge shape and planar-detrended steps (H12); nothing measures restored matching of offset marker strips. Not a weighted union, not a catalogue ray. Falsified by shuffled-strip and no-shift controls. | Provided stack only; magnetic bands correlated, gravity quasi-independent. No new external data. |
| **3 / 3 (high)** **H18: cross-field edge coincidence.** `tmi_hg`, `tmi_vg`, `iso_grav_anom_hg`, `iso_grav_anom_vg`, `iso_grav_anom_slope` | Linear edge energy in ≥2 independent potential-field bands at matched strike (orientation coherence between bands), with along-strike persistence | A buried or concealed fault boundary should turn up as a coincident, same-strike edge in gravity AND magnetics; a single-band edge is more likely a lithologic contact. Coincidence across fields is cheap discriminative evidence the HGB has to relearn implicitly today. | Existing edge channels are per-band; no cross-field coincidence/orientation-matching transform exists in this checkout. | Provided stack only. |
| **4 / 4 (high)** **H17: geodetic strain-rate linear coherence.** `geod_2ndinv`, `geod_shearrate`, `geod_dilaterate` | Persistent linear coherence of the strain-rate trio (multi-scale orientation consensus), independent of surface topography | Slow aseismic slip on unmapped faults produces strain-rate linearity without fresh surface scarps; the trio is quasi-independent of the magnetic/gravity families used by H13/H16. | No geodetic-band coherence transform exists here; the trio currently enters only as raw bands and one cross product. | Provided stack only. |
| **5 / 4 (high; BLOCKED)** **H14: concealed basin-margin source boundary** (carried from session 1). `iso_grav_anom`, `rtp` | Fourier upward continuation exp(−h·k); edge-position persistence with continuation height | Basin fill hides surface traces while retaining basement density boundaries. | Existing multiscale Gaussian derivatives do not model potential-field continuation. | Provided stack only; padding/source-depth tests required. Lower rank: 100 m localization and novelty uncertain. |

**Not viable without new external data (named sources checked 2026-09-27):** H15 (paleo-discharge) requires the [GDR 1391 archives](https://gdr.openei.org/submissions/1391) — landing page, DOI [10.15121/1881483](https://doi.org/10.15121/1881483), CC BY 4.0 and exact archive URLs re-verified this session, but binary downloads still fail TLS from this runtime (measured). A GitHub Actions runner (unrestricted network) is the obtainable path; until bytes are in hand with hash + license attribution, H15 stays BLOCKED and ineligible. INGENIOUS `Quaternary Faults v2` (2023-06-27, 5.85 MB, same GDR) is an additional candidate: if it contains traces absent from the provided label raster, they are *known-but-unmasked* candidates — download, hash and audited comparison against the label raster required first.

## Session-2 results (2026-09-27)

- **H16: ELIGIBLE** under the preregistered decision rule
  (`reports/h16_blocked.json`, protocol `...-policy-sweep-v2`). Selected policy
  topk06_binary for all three arms. vs baseline107: dev mean Δ **+0.003734**
  (folds +0.001018 / +0.006011 / +0.004173), confirmation Δ **+0.000704**.
  vs baseline107_discovery: identical deltas (the discovery fusion adds nothing
  at a 6% budget on this geography). Random topk02 control: 0.1206–0.1492,
  below every baseline fold. `promotion_allowed` remains **false**: a
  full-data final fit with a hash-bound training manifest is still required
  (`train_final.py --bind-to`). Note: the recorded run predates a reporting-only
  edit to `validate_candidate.py`; a clean re-run on frozen code will supersede
  the report for the release binding.
- **H13: running** (same protocol; label-free 39-channel offset grid
  `data/features_offset.npy`).
- Status register update: H16 eligible (final-binding stage); H12 rejected;
  H15 blocked; H18/H17/H14 remain untested backups.

## Preregistered H16 + H13 validation (session 2, frozen before implementation, 2026-09-27)

1. **Engine (H13).** For band B (rtp, tmi, iso_grav_anom; sentinels → NaN, 0-filled, halo-invalidated), strike θ ∈ {0°, 90°} exact axis strips; θ ∈ {45°, 135°} after one `ndimage.rotate(θ, order=1, reshape=False)` of the band (documented approximation). Strips: rows/cols i and i−d, d ∈ {1,2,3} px (100–300 m). Window 16 px (3.3 km) along strike. Per position, zero-difference NCC at lag 0 and lags ℓ ∈ {2,4,6} px (200/400/600 m; no-wrap). Channels per (band, θ): `ncc0` (zero-lag NCC at the best-contrast separation), `ncc_best` (max nonzero-lag NCC, a sensitivity indicator with documented +0.17 null bias), `adv` = mean_d ( mean_ℓ NCC_ℓ(d) − NCC_0(d) ) — the unbiased zero-lag-dip contrast; plus `advmax` per band (max over θ). 39 channels. Invalid halo (window+seps+lags+3 = 28 px) ⇒ NaN (HGB native).
   - **Calibration addendum (2026-09-27, synthetic null fields, BEFORE any holdout data contact).** The first draft (`adv = max_{d,ℓ≥1}(NCC_ℓ − NCC_0)`, window 8 px, lags 1–6 px) measured mean adv **+0.35** (max 1.0) on a white-noise null — pure max-selection bias, not signal. Recalibrated to window 16 px, lags {2,4,6}, and the pooled contrast above; post-fix null: mean adv **+0.009**, median +0.009 (target ≈ 0). Verified by `tests/test_alignment.py`: displaced lineation (shift 3 px on square-wave texture) gives ncc_best > 0.4, ncc0 < 0.6, adv > 0.15 at the straddling row; no-shift control and white-noise null give adv < 0.15 and |mean adv| < 0.05; NaN halos invalidate; ray geometry and 1-D NCC controls pass.
2. **Engine (H16).** Systems from labels via `discovery.system_strikes` (train systems only per fold). Keep systems with n_px ≥ 8 and width ≤ 20 px. Structure tensor (σ=1.5, integration 4) on `tmi` and `det_elev`. For each endpoint ray (outward from centroid through the endpoint), radii t ∈ {5,15,30,60} px: `cont_tmi_r{t}` / `cont_elev_r{t}` = energy·coherence·max(0, cos 2(φ−θ_ray)) at the ray point, 3 px disk, max over systems; `cont_ncc_r15`, `cont_ncc_r30` (rtp) = 1-D NCC of the ray-sampled band, halves of 16 samples about the point, minus the same computed on a 6 px perpendicular-offset control ray, clipped to [−1,1]. 10 channels, all NaN where no train ray covers the pixel.
3. **Protocol.** Identical to the H12 blocked protocol: `cv.make_folds(n_blocks=4, n_folds=4, buffer_px=40)`; folds 0–2 development, fold 3 confirmation (re-used confirmation geography, explicitly not pristine — recorded as a limitation); HGB 200 iterations, lr .05, depth 7, L2 1, seed 7; ≤200,000 sampled negatives per fold; sampling and all feature construction blind to held-out labels (H16 rebuilds from train systems only).
4. **Policy axis (new, free from saved OOF probabilities).** Pre-registered set: topk01_binary, topk02_binary, topk03_binary, topk04_binary, topk06_binary, topk02_soft, topk02_envelope, topk02_halo2. Per arm, selected policy = argmax mean development DTI (ties → topk02_binary). Primary reported policy remains topk02_binary. No policy tuning on the confirmation fold.
5. **Decision rule.** A candidate (H16 or H13) is promotable only if, under each arm's own selected policy, it beats **both** baseline107 and baseline107_discovery on the mean of development folds **and** on the confirmation fold. Otherwise: no slot, no release, negative result recorded. A passing candidate then needs full-data final training with a manifest binding probability hash, inputs, config and code hashes before any publication (generalized release gate).
6. **Controls.** H13: white-noise null (adv ≈ 0) and no-shift control reported; H16: systems-width and n_px filters fixed; no radius/strike sweep on the lockbox. Unit tests (`tests/test_alignment.py`) include a synthetic displaced-lineation construct that must show the zero-lag dip, and null/control constructs that must not.
7. **No promise.** These are catalogue-generalization proxies. Nothing here predicts the private new-fault score; passing locally is necessary, not sufficient, and publication still requires the full evidence chain.

## Preregistered H12 validation (before code/results)

1. Existing `reports/cv_*.json` use interleaved fault components, **not** spatial blocks. The 0.0929 ALL / 0.0508 FAR10 numbers are not directly comparable with a blocked score. Rerun baseline107 and baseline107+H12 on identical existing `cv.make_folds(n_blocks=4,n_folds=4)` geography.
2. Exclude 40 pixels (4 km, matching baseline feature halo) around held-out blocks from training. Sample positives/negatives only from the training mask; no test label informs sampling, fitting, thresholds, or feature design. This is stricter than the inherited 3-pixel buffer.
3. HGB hyperparameters fixed to inherited baseline: 200 iterations, learning rate .05, depth 7, L2 1, seed 7; up to 200,000 sampled training negatives. Binary top 2% fixed **before** scoring, inherited ALL-best policy. Apply budget only within each score region. Also report uniform random control at equal budget. No policy sweep on the lockbox.
4. Folds 0–2 are development; fold 3 is an untouched confirmation geography for this experiment (historical group activity may have examined the same region; not a newly collected blind test). Require positive paired mean development gain AND positive confirmation gain. Report each fold and worst-fold regression; no significance claim from four correlated folds.
5. Score with the published distance-weighted Tversky formula, predictions and ground truth restricted to the score mask. This is catalogue-generalization, not private-new-fault validation. Known training pixels lie outside score blocks. Component-holdout comparisons remain supplemental and cannot be silently mixed with these values.
6. No training-final/publishing if H12 fails. If it passes this matched baseline, first establish whether the historical campaign winner (including self-training) can be reproduced on the same geography before claiming it beat the current best. Format-valid diagnostic TIFs must be labelled **EXPERIMENT — DO NOT SUBMIT**. Public download promotion requires evidence and content-duplicate checks.
7. Log actual input/code hashes, fold masks, feature definition, config, scores, runtime, decision and limitations. Never infer future leaderboard score from proxy DTI. No competition upload is authorized by a local test alone.
