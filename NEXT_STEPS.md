# Next steps — session 2 (2026-09-27)

Read README.md (including the preserved project prompt), HYPOTHESES.md (session-2
register + preregistration + results), REVIEW.md, KNOWLEDGE.md first.

## Current state

- **H16 ELIGIBLE** under the preregistered rule (`reports/h16_blocked.json`):
  dev mean Δ +0.003734, confirmation Δ +0.000704 vs baseline107 AND
  baseline107_discovery, all arms selected topk06_binary. `promotion_allowed`
  stays false until the final-training binding exists.
- **H13 NOT ELIGIBLE** (`reports/h13_blocked.json`): three positive development
  folds (+0.003638/+0.003252/+0.003250) but confirmation −0.001450. Measured
  negative; no slot spent.
- **H12 rejected** earlier (`reports/h12_blocked.json`); **H15 blocked** by
  in-sandbox TLS on the GDR 1391 binaries (official index verified, CC BY 4.0).
- A hash-consistent clean H16 re-run supersedes the first report for the
  release binding (a reporting-only edit to `validate_candidate.py` landed
  after that run started).

## P0: finish the H16 release chain (in order)

1. Confirm the clean H16 re-run reproduces the decision exactly (fixed seeds,
   same inputs).
2. `python scripts/train_final.py --extra data/features_continuation.npy
   --hypothesis H16 --bind-to reports/h16_blocked.json --work-dir final_out`
   — refuses if the decision is not eligible; writes the training manifest and
   sets `final_prediction` + `promotion_allowed` in the report.
3. `python scripts/build_submission.py --prob final_out/prob_final.npy
   --policy topk06_binary --name gems10-h16-continuation
   --validation reports/h16_blocked.json --note "..."` — one-click download
   appears on the hub only after every gate passes.
4. `python scripts/build_site.py`; re-run the full test suite; commit; open PR
   to main and merge (session requirement).

## P1: what this session must not claim

- No leaderboard forecast from +0.0037/+0.0007 (four correlated stripe folds,
  re-used confirmation geography, dev-selected policy).
- No geothermal-resource claim; the target is fault pixels.
- No slot spent, no upload made, no credentials touched.

## P2: remaining scientific work (ranked)

1. **H18 cross-field edge coincidence** (tmi + iso_gravity edge orientation
   coherence) and **H17 geodetic strain-rate coherence** — next untested
   candidates if more signal is needed; preregister before implementation.
2. **H15 unblock:** fetch the GDR 1391 archives via an unrestricted-network
   path (GitHub Actions runner or the user), hash + attribute them, then
   implement the paleo-discharge residual.
3. **INGENIOUS Quaternary Faults v2** (2023-06-27, same GDR): audit whether it
   contains traces absent from the provided label raster before any use.
4. **Baseline normalization ablation:** Frangi tile-local scale → global or
   train-calibrated normalization (changes the 107 baseline; new preregistered
   run required).
5. **1 m DEM / U-Net comparison** per the official reference solution
   (needs storage + compute and its own frozen protocol).

## Session-1 carry-over (still open)

- Fold-0 random-control anomaly diagnosis (feature/label geography shift).
- `tc`/depth band metadata semantics vs official releases.
- Entity-level upload ledger coordination (user-side).
