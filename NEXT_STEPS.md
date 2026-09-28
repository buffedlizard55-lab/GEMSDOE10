# Next steps — after session 4 (2026-09-27)

Read README.md (including the preserved project prompt), HYPOTHESES.md
(session-4 register + results), KNOWLEDGE.md §5b–§5c (transport, traps, facts),
REVIEW.md and LIMITATIONS.md first. Everything below is ranked by expected
effect on P(win) per unit of work; nothing here promises a leaderboard score.

## Current state (measured)

- **New recommended artifact — H25:** `gems10-h25-ctx-ridge-20260927T232947704150Z-6452ae1d00.tif`
  (166-feature HGB = 107 baseline + H16 continuation 10 + 3DEP-10 m scarp 13 +
  36 DEM-context channels, policy `ridge15_binary`, 174,232 cells = 3.37 % of the
  footprint; hash-bound to `reports/h25_blocked.json` + `reports/final_manifest_h25.json`).
  Development folds 0.20097 / 0.22476 / 0.28415 (mean **0.23663**), confirmation
  **0.25179** — beats the same-run H20+ridge15 (+0.0011 dev, +0.0265 conf) and the
  released H20/thin10 incumbent (+0.0468 dev, +0.0650 conf). Honest caveats: the
  development margin over same-run H20 is within fold noise and fold 0 is *negative*
  (0.20097 vs 0.20406); the decision rests on the confirmation fold. Under thin10
  the context channels alone add nothing on development (+0.0001) — their value is
  the ridge-policy interaction on sparse geography.
- **Close second — H24's H20+ridge15** (`reports/h24_blocked.json`, dev mean
  0.23553 / conf 0.22525): same H20 field as the old artifact with the
  preregistered ridge emission; kept as evidence, superseded by H25.
- **Both session-4 runs reproduced `reports/h20_blocked.json` exactly**
  (max |ΔDTI| = 0, 16 shared policies × 4 arms) before any decision was read.
- **H27** (corrected-phase continuation gate) is implemented and tested
  (`gate_phase="aligned"`, `tests/test_h27_gate.py`) but **not run** — time.
- The session-3 H20/thin10 and H16/topk06 artifacts remain on the hub as
  superseded/provenance. No slot has been spent on anything; all numbers above
  are spatial-holdout proxies, not leaderboard scores.
- **Provenance fixed:** the workspace's shallow clone broke both approvals
  ("validation code changed or missing"); `git fetch --unshallow origin` +
  PR-ref fetch restored every pinned blob, and both workflows now use
  `fetch-depth: 0`.
- H21 catalogue differencing stays closed (labels = the public catalogue);
  the emission-density question is still open (see P0).

## P0: run H27 (only unexecuted preregistered candidate)

H27 is fully implemented: `gate_phase="aligned"` flips the continuation gate to
the physically intended phase (the as-built gate passes lineaments *perpendicular*
to strike — measured in REVIEW.md), an H27 arm exists in `validate_candidate.py`,
`--gate-phase` exists in `build_continuation.py`, tests are green. The one run:

```
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
  scripts/validate_candidate.py --hypothesis H27 --policy-set v4 \
  --extra data/external/dem10/dem10_channels.f32.npy \
  --incumbent-report reports/h20_blocked.json --incumbent-arm H20 \
  --report reports/h27_blocked.json --work scratch/h27
```

(~45 min, 4 arms). Eligibility identical to H24/H25: beat the external incumbent
(H20/thin10 — or, if the incumbent rule is updated, the *then-current* holdout
best) and every same-run arm on development mean **and** confirmation. Note the
updated holdout best for future candidates: **H25/ridge15, dev 0.23663,
conf 0.25179** — a future candidate must beat *that* (protocol item 1 defines the
incumbent as the named report's arm/policy; decide at preregistration whether
H25 replaces H20/thin10 as the incumbent reference and record it *before* scoring).

## P1: candidate work, ranked

1. **Submission decision (user's call):** the H25/ridge15 artifact is the
   holdout-validated best and is one click on the hub. Whether it spends a
   weekly slot is the account holder's decision (SUBMISSION_GUIDE.md); the
   repository will not upload automatically.
2. **H25 follow-ups, preregistered before any run:** (a) diagnose fold 0 — the
   only fold where H25 loses to H20 under ridge15 (context may be adding noise
   where the confirmation geography differs); (b) wider context windows
   (31×31 ≈ 3 km) or a second-order interaction set; (c) a density-aware
   emission *sweep over the ridge family* extending
   `reports/budget_density_sweep.json` (now that ridge15 is the released policy,
   the thin-family sweep results are historical).
3. **H22 seismicity lineament coherence** (USGS ComCat via the runner;
   `scripts/ext/` pattern) — source descriptor verified, transport job still
   to be written; ranked below the H25 follow-ups.
4. **H15 paleo-discharge residual** — point archives in hand (`ext/catalogue-*`);
   needs an observation model that respects resource-biased sampling.
5. **H20/H25 external-data scale-ups:** 1 m lidar stage (login-gated
   `1m_DEM_links.csv`, blocked with a named path), multi-scale 10 m channels.
6. **H18 / H17 / H14** remain untested backups (session-2 register).

## P0-bis: git history (resolved — the lesson now ships in CI)

Sessions 2–3 pins live in pushed PR history, and this workspace started shallow
(depth 1), which made `build_site.py` refuse both approvals until
`git fetch --unshallow origin` + `git fetch origin '+refs/pull/*/head:refs/remotes/pr/*'`
restored 32 commits (pinned blobs `b00f5fa0`, `8dddfa4e`, `e2877ce0`,
`1756ba1e`, …). Both workflows now check out with `fetch-depth: 0`.
**If a gate ever says "validation code changed or missing" again, unshallow first,
then reassess — never re-pin reports to silence it.**

**Session-4 end state (BLOCKER for the next session):** the GitHub token expired
mid-session (`gh auth status` → token in GH_TOKEN no longer valid). Commits
`4ac76f8` and `ea78b79` (all session-4 results, releases and docs) are **local
only** on `arena/01a0e4b8-gemsdoe10`. First action next session: reconnect
GitHub in Arena, `git push origin arena/01a0e4b8-gemsdoe10`, open the PR, merge
to main — the site workflow and evidence gates then run on the pushed history.

## P2: hygiene

- `data/external/` products are ignored; re-restore with the tag commands in
  KNOWLEDGE.md §8. Tags are immutable evidence — do not delete them.
- The external-data workflow is push-triggered via `scripts/ext/JOBS`
  (dispatch is 403 for this token). Set JOBS to the job you need *before*
  editing anything under `scripts/ext/`, or both jobs run.
- Long local runs: never wrap in `| tee` (it masks OOM kills as exit 0 —
  session-4 lesson), use `MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=2
  OPENBLAS_NUM_THREADS=2` on this 4 GB box.
- `wellspringdata.gdb` layer listing is still unfixed (`ogrinfo -so -q` parsing)
  if H15 needs wells/springs.
- Keep the three-pass discipline and the PR-then-merge rule every session.

## What must not be claimed

- No leaderboard forecast from any local delta (four correlated stripe folds,
  re-used confirmation geography, development-selected policy; the H25
  development margin over same-run H20 is within fold noise).
- No geothermal-resource claim; the target is fault pixels.
- No slot spent, no upload made, no credentials touched by this repository.
- The H27 phase correction is *implemented*, not *validated* — never present it
  as a result until its report exists.
