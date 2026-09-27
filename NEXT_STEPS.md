# Next session — measured improvements, not more of the same

Read README.md (including preserved requirements), REVIEW.md and HYPOTHESES.md first.

## Completed this session

- Restored feature raster automatically; all three inherited byte pins match. CPU training works.
- Audited ten sibling artifacts at immutable commits; exact and catalogue-only duplicates identified.
- Read official leaderboard, submission/metric rules and staff masking/new-geometry/cadence/hidden-source posts first-hand.
- Preregistered four physical hypotheses. Tested H12 against baseline107 and its discovery variant on four buffered spatial stripes; confirmation failed in the initial run. A completed same-configuration reproducibility run matched every fold score and decision while checking persisted OOF serialization; the final report is `reports/h12_blocked.json`.
- Added semantic duplicate, raw probability, spatial evidence and training-provenance release gates. Rebuilt site with no misleading download recommendation.
- Added official-feed refresh, stale/failure labels, and daily Pages schedule. No competition slot used.

## P0: carry forward the negative result

1. **Do not promote or retune H12 on the same confirmation fold.** Initial development delta +0.001218; confirmation delta −0.002171. Use the completed report for full precision. A positive development mean did not transfer.
2. **Retain the baseline and random controls.** Fold 0's HGB and H12 underperform the budget-matched random field. Diagnose feature/label distribution shifts and geographic support before interpreting tiny average gains as discovery.
3. **Keep saved-array readback checks.** An initial OOF integrity check found unexpected finite zeros in the first raster row of earlier saved NPYs. The fail-closed rerun with buffered serialization, readback checks and per-file SHA reproduced every metric exactly; all final saved files passed subsequent checks. Never silently repair predictions to make a file publishable. The initial in-memory rejection decision is not permission to trust an unverified saved grid.
4. **Coordinate one entity-level upload ledger.** Actual history and slots are not available here. Multiple repos are not proof of misconduct. No automatic uploads or alternate registrations.

## P1: next distinct experiment

5. **H13: lateral displacement of magnetic texture.** Implement opposing-strip correlation with nonzero lag advantage; use independent strip agreement and synthetic displaced-contact tests. Require a no-shift control and lithologic-contact confound control. No new external source is needed. Preserve trial accounting; fold 3 has now been inspected and is not a pristine future lockbox. Use a preregistered nested spatial evaluation or newly held-out independent labels before claiming confirmation.
6. **Fix baseline Frangi tile normalization in a separately recorded ablation.** Current c uses each tile's maximum, so changing tile_rows changes features. Choose a deterministic calibration from training data or a fixed physical normalization, test tile invariance, rerun all paired controls. Do not overwrite historic reports.
7. **Resolve band semantics from official releases.** Compare `tc` to GeoDAWN total-count data and clarify conductive-base versus basement-depth metadata before assigning physical meaning. Preserve raw tags and checksum evidence.
8. **Improve new-fault validation rather than multiplying transforms.** Seek independent expert-reviewed traces with licenses and provenance, audited against all supplied catalogue variants. A geographic catalogue holdout is necessary but insufficient to establish new-fault performance.

## P2: external work / publication

9. **H15 stays BLOCKED.** GDR 1391 page and archive URLs are known, but binary downloads failed TLS here. Retry on GitHub Actions/unrestricted runner, record hash/license, inspect content and measure footprint coverage before adding it to a model. Page availability is not evidence of downloaded usable data.
10. **H14 potential-field continuation** is a separate buried-structure experiment, not another Gaussian-blur sweep. Test boundary padding/source ambiguities and 100 m localization before evaluation.
11. If a candidate eventually wins: implement its correct full-data training recipe, store final model/probability hashes and input/code/config binding, pass format + semantic novelty + publication rechecks, then expose one-click TIFF/ZIP plus a unique note. The old `train_final.py` does not train H12.
12. GPU segmentation/1 m DEM pilot remains optional future work needing compute/storage. Eligibility, disclosure and final platform submission must be done through the authorized participant account; no credentials in chat or Git.

## Reproduce / inspect

`python scripts/verify_project.py --with-features` runs tests, pins, site generation and local link checks.
`python scripts/audit_submissions.py` refreshes immutable artifact comparisons.
`python scripts/refresh_sources.py` refreshes official text availability and public leaderboard with honest stale states.
`python scripts/validate_hypothesis.py` is an offline experiment, not an upload job.
