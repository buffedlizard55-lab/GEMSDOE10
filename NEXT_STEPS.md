# Next steps — after session 3 (2026-09-27)

Read README.md (including the preserved project prompt), HYPOTHESES.md
(session-3 register + results), KNOWLEDGE.md §5b (transport + facts), REVIEW.md
and LIMITATIONS.md first. Everything below is ranked by expected effect on
P(win) per unit of work; nothing here promises a leaderboard score.

## Current state (measured)

- **New recommended artifact — H20:** `gems10-h20-dem10-scarp-thin-20260927T155223039488Z-ffc91a1686.tif`
  (130-feature HGB = 107 baseline + H16 continuation 10 + 3DEP-10 m scarp 13,
  policy `thin10_binary`, 153,957 cells = 2.98% of the footprint; hash-bound to
  `reports/h20_blocked.json` + `reports/final_manifest_h20.json`). Beat the H16
  incumbent under the frozen protocol: dev mean **+0.01473**, confirmation
  **+0.01406** (absolute 0.16483 / 0.18130 / 0.22339 · 0.18680). The H16
  artifact (`…-3431b83c7c.tif`, 6%) stays on the hub as superseded/provenance.
  No slot has been spent on either; both are unmeasured on the leaderboard.
- **H19 thin-line emission — NOT ELIGIBLE** (`reports/h19_blocked.json`): dev
  mean +0.00744 vs the incumbent, confirmation −0.02111. Protocol reproduction
  of the session-2 report was exact (max |ΔDTI| = 0).
- **H20 data pipeline** is reproducible: tag `ext/dem10-36326816737` →
  `scripts/fetch_external.py` → `scripts/build_dem10_grid.py` (49 blocks, 440 s
  on the runner; 12 tiles with sha256). The channels alone are weak (univariate
  AUC ≤ 0.58); the gain is interaction-driven and concentrated in folds 2–3.
- **H21 catalogue differencing — CLOSED (decisive negative):** the provided
  labels are the current public catalogue (1 pixel of USGS QFFD / INGENIOUS v1 /
  v2 lies >300 m from the labels). There is no catalogue-lag population.
- **H15 unblocked, unmodelled:** point archives (paleo-geothermal 281 in-footprint
  points, 2 m probes 2,782) restored from `ext/catalogue-36326816737`.
- **Budget sensitivity measured:** `reports/budget_density_sweep.json` (H20
  OOF grids) — the optimal emission budget shrinks as the truth gets sparser
  (see P0). H20's `thin10_binary` already emits ~3% (vs H16's 6%), which sits
  closer to the sparse-fold optimum, but it was *selected*, not *derived*.

## P0: the emission-budget question (largest known lever, unresolved)

The metric penalty scales with emitted pixels ÷ truth pixels. Locally the H16
field prefers 6% where held-out truth is 1.1–1.3% of the region and 3% where it
is 0.84% (fold 3: topk03 0.18792 vs topk06 0.17274). The hidden truth (new faults
only) is almost certainly sparser than the catalogue, so the budget that wins
locally is probably above the hidden optimum — but the hidden density is
unknown and cannot be estimated from local data. `reports/budget_density_sweep.json`
(H20 OOF grids; catalogue systems randomly kept with probability f, the removed
systems' pixels masked "known") shows the development-mean best policy
moving from `thin12` at f = 1 (0.189) to `thin08` at f = ½ (0.137), `thin04` at
f = ¼ (0.095) and `thin04` at f = ⅒ (0.058); the sparse confirmation fold prefers
`topk02` → `topk02` → `topk01` → `topk01`. Thinned policies beat top-k on the
development folds at every density. Caveat: the
simulation removes truth but the model was trained on the full catalogue, so it
overstates the penalty of wide budgets somewhat; it is a direction, not a
calibration. Options, in order of evidential value:

1. Treat the sweep as the prior: if the hidden truth is ≤ ½ as dense as the
   catalogue, a `thin06`–`thin08` emission of the *same H20 field* is expected to
   beat the released `thin10` — but this is untested on any holdout that has
   that density, and the gate will not publish a non-selected policy.
2. If the account holder is willing to spend slots on *measurement*, two uploads
   of the **same H20 field** at two budgets (e.g. thin06 and thin10) identify the
   density regime from the public score pair (the field is fixed, only n changes).
   This is the user's call — it is not a holdout-validated improvement and the
   repository's release gate will not publish a non-selected policy by itself.
3. Preregister a density-robust policy rule (e.g. budget chosen so that the
   marginal hit-rate estimate q(n) ≈ DTI/5 on the development folds under a
   *sparsified* truth) and evaluate it under the frozen protocol as a new
   hypothesis (H23) — this is the only route that can change the site's approved
   download without user-side probing.

## P1: candidate work, ranked

1. **H19b — ridge (non-maximum-suppression) thinning instead of skeleton thinning.**
   H19 failed where blobs are wide because a skeleton follows the blob's centre,
   not the probability ridge. Keep pixels that are local maxima of the OOF
   probability across the local strike (structure-tensor orientation of p),
   inside the top-k mask; score with `scripts/rescore_blocked.py` on the saved
   grids (second look — label it so), then, if promising, preregister and re-run.
2. **H20 follow-ups (now justified by a measured gain):** (a) 1 m lidar
   versions of the one-sided-step and residual channels from the official
   `1m_DEM_links.csv` tiles (needs the runner; ~100× the 10 m data — stage by
   footprint block, publish per-block tags); (b) multi-scale 10 m channels
   (hgm at 100/400 m, aspect-conditioned steps) — cheap, same pipeline; (c) an
   H20 model with fold-1 diagnostics (the only geography where H20 does not
   beat H16 at top-k budgets). Per-channel ablation needs re-training under the
   frozen protocol (≈ 35 min per arm on this machine).
3. **H22 seismicity lineament coherence** (ComCat event catalogue via the runner;
   `scripts/ext/` pattern) — untested; ranked below H19b/H20 in the register.
4. **H15 paleo-discharge residual** — data now in hand; needs an observation
   model that respects resource-biased sampling before any protocol run.
5. **H18 / H17 / H14** remain untested backups (session-2 register).

## P0-bis: unpushed work (GitHub token expired at the end of session 3)

Commits `0b2cf2b` and later on `arena/01a0e336-gemsdoe10` (H19/H20/H21 results,
v3 gate, H20 artifact, docs) could **not** be pushed: `git push` and `gh api`
returned "Bad credentials" after the earlier pushes in the same session had
succeeded (the last pushed commit is `8d9b513`). First actions next session:
`git push origin arena/01a0e336-gemsdoe10`, open the PR to `main`, merge it, and
confirm the Pages hub shows `gems10-h20-dem10-scarp-thin` as RECOMMENDED. The
`ext/*` tags were published by the runner before the token expired and are safe.

## P2: hygiene

- `data/external/` products are ignored; re-restore with the tag commands in
  KNOWLEDGE.md §8. Tags are immutable evidence — do not delete them.
- The external-data workflow is push-triggered via `scripts/ext/JOBS`
  (dispatch is 403 for this token). Set JOBS to the job you need *before*
  editing anything under `scripts/ext/`, or both jobs run.
- `wellspringdata.gdb` layers were not enumerated by `ogrinfo` in run
  36326816737; fix the layer listing (`ogrinfo -so -q <gdb>` output parsing) if
  H15 needs wells/springs.
- Keep the three-pass discipline and the PR-then-merge rule every session.

## What must not be claimed

- No leaderboard forecast from any local delta (four correlated stripe folds,
  re-used confirmation geography, development-selected policy).
- No geothermal-resource claim; the target is fault pixels.
- No slot spent, no upload made, no credentials touched by this repository.
