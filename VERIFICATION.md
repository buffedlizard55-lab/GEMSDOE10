# Verification ledger — every claim, its evidence, its link

Re-verified 2026-09-27 by the GEMSDOE10 agent session. "Measured" = computed in this session from the official bytes; "quoted" = read from the official source named.

## Official sources (all re-checked 2026-09-27)

| # | Claim | Source | Status |
|---|---|---|---|
| 1 | Prize $300k; Initial 5×$10k; Final $100k/$70k/$40k/$25k/$15k; ends Dec 3 2026 23:59 UTC | [competition home](https://www.drivendata.org/competitions/306/competition-doe-gems/) | quoted ✓ |
| 2 | Metric = distance-weighted Tversky, α=0.2, β=0.8, triangular kernel R=300 m; worked example TP=3.00 FP=1.89 FN=2.00 → 0.60 | [problem p.967 §metric](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) | quoted ✓, example reproduced in `tests/test_metric.py` ✓ |
| 3 | Submission: single-band float32 GeoTIFF in [0,1], EPSG:32611, 100 m, same bounds, NaN outside | [problem p.967 §submission](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) | quoted ✓, enforced by `src/gems10/raster.py` ✓ |
| 4 | Known USGS/INGENIOUS pixels masked from evaluation, both rounds | [forum 11516 staff post](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516) (via sibling-session record; re-verify manually before final submission) | reported, needs account-holder re-check |
| 5 | New-fault GT pixels can lie within 300 m of known traces (corrections intended) | [forum 11536 staff post](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527) (via sibling record) | reported, needs re-check |
| 6 | 3 scored submissions / rolling 7 days | forum 11524 staff post (via sibling record) | reported, needs re-check |
| 7 | Reference solution: U-Net MC CV, Tversky loss α=0.2/β=0.8 | [reference repo](https://github.com/drivendataorg/gems-prize-reference-solution) | quoted ✓ |
| 8 | Data tab login-gated (redirects to /accounts/login/) | [data tab](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) | confirmed by sibling sessions; our data came via verified bridge |

## Byte-level measurements (this session)

| # | Claim | Evidence |
|---|---|---|
| 9 | training_features.tif 418,912,844 B, sha256 `4371c82e…` | `scripts/fetch_data.py` → ALL OK |
| 10 | labels.tif 425,830 B `7ba308cc…`; sample_submission.tif 1,599,597 B `2176d08e…` | same |
| 11 | Grid 3730×3292, EPSG:32611, 100 m, origin (243350, 4508550); footprint 5,167,373; positives 60,988 | EDA 2026-09-27 + `spec.py` |
| 12 | Sample submission = labels in submission format (60,988 ones), NOT all-zero | EDA 2026-09-27 |
| 13 | 3,199 fault systems; fold px [12180, 12151, 12379, 12151, 12127] | `run_cv.py` log |
| 14 | 71/71 tests pass | `pytest tests/` |

## Model/harness findings (this session)

| # | Claim | Evidence |
|---|---|---|
| 15 | Proximity-to-train channel memorises (train 0.98 / held-out = background 0.056; DTI 0.012) → removed by default | smoke1 vs smoke2 |
| 16 | Geophysics-only HGB fold 0: topk02_binary masked 0.0895; binary > soft; 2% > 3% | `/tmp/smoke2.json` |
| 17 | Lidar ex_max top-2% ridge alone: masked 0.0254 | external probe |
| 18 | SGMC-gap: 0.0000 by construction → excluded; QF-gap 100 px → negligible | external probe |
| 19 | Gabor orientation convention fixed to along-ridge (strike); HGB-invariant sign flip on 2 channels | `tests/test_features_new.py` |

## Claims that did NOT survive checking

1. "Predict near catalogue for free" — only true if the masking clarification holds; the harness reports unmasked numbers alongside, and the final policy is chosen to be robust under both.
2. "More channels are always better" — r6 measured agreement channels negative; we dropped all 17 and added 19 lineament channels instead (Frangi/Gabor/coherence), whose value is decided by the harness, not asserted.
3. "SGMC-gap traces are discovery gold" — false (see #18 above).
