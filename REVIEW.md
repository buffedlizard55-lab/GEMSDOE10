# Repository review and evidence corrections — 2026-09-27

## Executive decision

Do not use a weekly slot merely to rename or re-encode an existing prediction. Do not infer scientific novelty or independent methodology from a different repository name. Do not equate a local catalogue proxy with the private new-fault target. The new H12 experiment must improve on the matched spatial baseline **and** its discovery variant before release is even considered. The live result is `reports/h12_blocked.json`; a failed experiment is evidence, not a deliverable to upload.

## Duplicate investigation: actual files, not rounded scores

Reproduce with `python scripts/audit_submissions.py`. Each source is resolved to a commit, fetched through GitHub's API, read with rasterio, and compared on the template footprint. Evidence, complete SHA-256 values and immutable source links are in [reports/submission_audit.json](reports/submission_audit.json).

| Comparison with GEMSDOE1 | Measured differences | Interpretation |
|---|---:|---|
| 5GEMSDOE adopted artifact | **0 pixels**; same file SHA-256 `7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15` | Exact copy. Renaming the downloaded TIF does not change the prediction. |
| GEMSDOE2 recall arm | **0 pixels**, same SHA | Another copy; this is not necessarily the arm associated with the user-reported 0.1560. |
| GEMSDOE2 currently advertised dual-family union | 10,668 changed pixels, 9,430 outside catalogue | Similar, but NOT identical; positive-pixel IoU 0.9419. |
| 8GEMSDOE currently advertised hedge | 54,533 changed pixels, **0 outside catalogue** | A catalogue-only union, not additional discovery. Container SHA differs. Do not call this a new detector. |
| GEMSDOE3/4/6 | Nonzero noncatalogue differences (full pairwise matrix in report) | They are distinct fields. Poorer reported scores do not prove copying. |

**What remains unknown:** the actual authenticated upload history. User scores are recorded as user-reported. A current website may have changed since upload; in particular 8GEMSDOE still describes its hedge as unuploaded. We verified published artifacts, not the identity of every file submitted to DrivenData. Different predictions can legitimately tie when rounded to four decimals, or differ only outside evaluated pixels. Unique scores cannot be guaranteed, nor should noise be added just to force uniqueness.

## Findings requiring correction / review

| Severity | Finding | Evidence / action |
|---|---|---|
| Critical | Existing publication gate did not require any holdout improvement. | `build_submission.py` previously accepted any NPY plus policy. Now fails closed without compatible evidence and final-prediction binding; experimental files forced to ignored scratch, never site downloads. |
| High | Byte-only duplicate test is defeated by TIFF compression, tags or new filenames. | Added canonical float32 field and noncatalogue field hashes, geometry/support included. Tests cover masked-only changes and signed zeros. Refuses copies in the audited registry and prior local sidecars. |
| High | Existing `run_cv.py` uses interleaved 8-connected components, not spatial blocks. | New paired runner uses existing `cv.make_folds(4,4)` geography: four west–east stripe folds, with 40-pixel training exclusion. Component holdout remains supplemental. Raster components are not established geological fault systems. |
| High | FAR10 was justified as if hidden fault distances were known. | [Staff 11527/7](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527/7) explicitly withholds test sources/types/coverage. [11536/2](https://community.drivendata.org/t/where-do-you-draw-the-line/11536/2) allows new geometry of existing systems. No verified median-distance statistic is available. FAR10/FAR20 are sensitivity subsets only. |
| High | Baseline/discovery ranking reverses with population choice. | Persisted ALL masked means at top2: baseline 0.0928845; discovery 0.1039580; selftrain 0.0896811. Historical FAR10 favored baseline. Never silently select one column after inspecting results. New runner fixes ALL spatial-catalogue score and compares baseline plus discovery. |
| High | Format error was asserted to prove NaN-inside-footprint. | Out-of-range finite values, infinities, NaNs, or footprint/alignment errors are possible causes. Without the rejected file and backend trace, exact original cause is unknown. Gate checks all of these; raw probability input is no longer silently repaired. |
| Medium | Data placement/GPU blocker was stale. | Restored 418,912,844-byte feature raster automatically through immutable public team mirror; all three inherited file pins match. HGB runs on CPU. Hash agreement demonstrates integrity with team pins, **not independent official authenticity**; DrivenData data page still redirects to login. |
| Medium | Forum facts described as inaccessible. | This session read staff posts 11516/2, 11524/2, 11536/2 and 11527/7 without account credentials using the research fetch tool. Command-line HTTPS to several domains fails TLS; these are different access routes. |
| Medium | Site advertised results without a current publication manifest and used obsolete test counts. | Site now generated from checked reports. Downloads appear only after revalidation; no training job is invented from a JSON report's old status. |
| Medium | Feature metadata has unresolved semantics. | Band 6 is tagged `tc`/magnetic tilt; sibling 8 claims it matches radiometric total count. Band 15's TIFF tag says basement depth while the official problem describes depth to conductive base. Preserve raw names, avoid physical interpretation of these channels until checked against the original releases. H12 uses band 12 only. |
| Medium | Known-mask implementation was called exact. | Staff confirms exclusion, not public backend masking order. `metric.components(fp_ignore_mask=...)` removes FP charges but allows masked predictions to contribute to nearby TP. That is a surrogate assumption. New blocked comparison has no training catalogue pixels inside the scored region and does not rely on this assumption. |
| Medium | Frangi values depend on tile-local maximum; derived stack is not invariant to tile height. | Existing `frangi_vesselness` chooses c from tile max. New experiment fixes tile_rows=128 for both arms and records feature file hashes. Do not compare its exact scores to historical builds with a different tile size. Fix global/train-calibrated normalization in a separate, preregistered ablation. |
| Medium | Multiple repos were treated as proof of competition-rule violations on a sibling site. | Repos are not registrations. [Rules §3.4–3.6](https://docs.nlr.gov/docs/fy26osti/96647.pdf) limit submissions per participating entity and final selections. We cannot verify account/entity usage. Keep one shared slot ledger; do not accuse the group of violations from repo count alone. |
| Medium | PDF Appendix A.1 contains general 5 p.m. ET / document-format language, while task-specific §3.2/3.5 requires GeoTIFF and §1.2 directs dates to the website. | [Official PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf). Flag wording inconsistency; do not replace the competition TIF with a report PDF or wait until the final hour. Current platform timeline remains the operational reference. |

## Session-2 review additions (2026-09-27)

| Severity | Finding | Evidence / action |
|---|---|---|
| Critical (found pre-holdout-contact) | First-draft H13 statistic was miscalibrated: `adv = max(NCC_ℓ − NCC_0)` (window 8 px, lags 1–6) measured mean **+0.35** on a white-noise null — max-selection bias, not signal. | Synthetic-null audit before any label contact. Recalibrated (window 16, lags 2/4/6, pooled zero-lag-dip contrast `adv = mean_d(mean_ℓ NCC_ℓ − NCC_0)`); post-fix null mean +0.009. Documented in HYPOTHESES.md calibration addendum; enforced by `tests/test_alignment.py`. |
| High | 90° strip branch compared a column with itself row-shifted instead of adjacent columns (inconsistent with the 0° convention). | Fixed in `alignment.strip_triples`; covered by `test_ninety_degree_offset_detected`. |
| High | Full-grid float64 offset build OOM-killed on the 4 GB host. | `alignment.offset_channels_tiled` (bit-identical to the full-grid path, verified by `test_tiled_matches_full_grid`); builder uses it. |
| High | H13 candidate failed its own preregistered rule: dev mean +0.00338 across three positive folds, confirmation **−0.00145**. | `reports/h13_blocked.json`. No slot spent; recorded as a measured negative. |
| Medium | H16 passes the rule (dev mean +0.003734, confirmation +0.000704, all arms topk06_binary) but on four correlated stripe folds with a re-used confirmation geography and a dev-selected policy. | `reports/h16_blocked.json`. The margin is a proxy, not a forecast; final binding + format gates still required. A hash-consistent clean re-run supersedes the first report (a reporting-only edit landed after that run started). |
| Medium | Release gates were H12-hardcoded (protocol string, arm name, scalar scores). | `src/gems10/release.py` now protocol-aware (v1 + `policy-sweep-v2`), candidate from the report, per-arm selected policy; v1 path regression-tested; v2 path checked against both session-2 reports (pass/fail/wrong-policy cases). |
| Medium | Baseline107 Frangi tile-local normalization, `tc`/depth-band metadata semantics (session 1) remain unresolved. | No new claim made; H13/H16 add label-free and system-derived channels but inherit the 107 baseline unchanged. Fixing baseline normalization requires a new preregistered run. |

## Three review passes

1. **Implementation:** restored input; re-read official rules/forum; fetched all ten published artifacts; preregistered four hypotheses; implemented H12 and paired buffered spatial runner; added content novelty and holdout release gates; updated site and dated source feed.
2. **Adversarial review:** tests for planar-ramp false response, step/valley distinction, missing-data halos, tile equivalence, catalogue-only duplicates, nonfinite input, empty/incomplete/tied/failed confirmation, and policy changes. Check existing code for misleading scientific or access claims. Fail closed on incomplete audit or evidence.
3. **End-to-end:** rerun tests, real-data format gate and data pins; build site from reports; check local links and download selection; review actual H12 outcome before any release; CI and Pages deployment verification recorded in session handoff. See `reports/verification_run.json` for completed commands, not this checklist alone.

## Scope boundary

A verified source supports the statement beside it, not every inference made from that statement. These are candidate *fault* detectors, not confirmed geothermal vents. The hidden truth, account eligibility and private upload history are unavailable. No claim is made that this work beats the leaderboard, verifies every external dataset line, or removes the need for the account holder's legal attestation.

## Completed H12 result and second-pass integrity finding

Both training runs returned identical fold metric dictionaries and the same rejection decision:

| Geography | Baseline / discovery DTI | H12 DTI | H12 delta |
|---|---:|---:|---:|
| Fold 0 development | 0.117128 | 0.117402 | +0.000275 |
| Fold 1 development | 0.143223 | 0.141076 | −0.002147 |
| Fold 2 development | 0.170685 | 0.176211 | +0.005526 |
| Fold 3 confirmation | 0.187368 | 0.185197 | **−0.002171** |

Development mean gain +0.001218 did **not** confirm. Baseline and discovery coincide on these buffered score regions; catalogue rays add no useful scored-region signal here. H12 features expose step/even contrast and polarity to HGB, not a mandatory hard geological classification rule. The experiment does not test high-resolution lidar or prove the general scarp hypothesis false.

A post-run check detected unexpected finite zeros in the first row of some initially saved NPY probability files. We did **not** repair those files or publish them. The runner now uses buffered serialization, immediate bitwise/NaN-aware readback, explicit OOF-mask checks and per-file SHA. The entire paired experiment was retrained with the same fixed settings; all in-memory scores reproduced exactly. The final saved files pass later hash/mask checks. The underlying cause of the initial storage anomaly is not established. Evidence: [h12_reproduction.json](reports/h12_reproduction.json).

A unique hash-verified OOF diagnostic was subsequently converted to a float32 GeoTIFF and ZIP and passed the format gate. It remains in ignored scratch, labelled **EXPERIMENT ONLY — DO NOT SUBMIT**, and is not on the site's download list. [Diagnostic checks](reports/diagnostic_validation.json). This verifies file generation without spending a slot or disguising a failed experiment as a final model.

Chromium installation failed TLS, so no visual-browser test is claimed. Generated HTML internal links and local HTTP routes were checked. GitHub Pages settings update returned HTTP 403 (integration permission); root redirect pages provide compatibility with the existing legacy main/root Pages configuration. Workflow deployment outcome must be checked on GitHub, not assumed.
