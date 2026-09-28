# Executive submission guide

## 1. Start with release status, not a file browser

Open the [submission hub](https://buffedlizard55-lab.github.io/GEMSDOE10/).
If it says **NO APPROVED SUBMISSION**, stop: an experiment has not satisfied the evidence gates.
A format-valid TIFF alone is not a reason to spend one of the three weekly slots.

No candidate is approved by default. Decisions so far: **H16 ELIGIBLE and
released** (session 2, `reports/h16_blocked.json`, protocol `…-policy-sweep-v2`);
**H20 ELIGIBLE** (session 3, `reports/h20_blocked.json`, protocol
`…-policy-sweep-v3`: 3DEP 10 m scarp channels on top of H16, selected policy
`thin10_binary`, beat the H16 incumbent on the development mean **and** the
confirmation fold); **H24 ELIGIBLE but not released** (session 4,
`reports/h24_blocked.json`, protocol `…-policy-sweep-v4`: the preregistered
`ridge15_binary` emission on the H20 field, dev 0.23553 / conf 0.22525 — beaten
the same session by H25); **H25 ELIGIBLE and released** (session 4,
`reports/h25_blocked.json`, protocol `…-policy-sweep-v4`: +36 DEM-context
channels, 166 features, selected policy `ridge15_binary`, dev mean **0.23663**,
confirmation **0.25179**, beating the same-run H20 arm and the H20/thin10
incumbent on both); **H13 and H19 NOT ELIGIBLE** (confirmation regressed);
H12 rejected. Both session-4 runs reproduced the session-3 report exactly
(max |ΔDTI| = 0) before any decision was read. When more than one artifact is approved, the hub lists the
**newest first and marks it RECOMMENDED** — each later release had to beat the
earlier one under the same frozen protocol; older approved artifacts stay
downloadable for provenance. The website is static: CPU scripts generate
files; approved files are then one-click downloads. It does not secretly train
a model in the browser or upload to DrivenData.

**What the recommendation is and is not.** It is the artifact with the best
measured catalogue-generalisation proxy on four spatially blocked folds. It is
not a leaderboard forecast. Two known open questions are recorded in
NEXT_STEPS.md: the emission budget was selected on folds whose truth density
is the catalogue's (the hidden new-fault truth is sparser), and the confirmation
fold geography is re-used across sessions.

## 2. When a candidate is approved

1. Click **Download .tif** at the top of the hub (or its single-member ZIP). Do not upload this report, an HTML page, or a PDF.
2. Confirm the filename, hash, hypothesis and validation report shown on its card. The filename contains a UTC timestamp and prediction hash prefix. The note includes strategy, policy and artifact hash.
3. Open [DrivenData submissions](https://www.drivendata.org/competitions/306/competition-doe-gems/submissions/) using the team's registered account.
4. Choose **New submission → File to submit**, select the downloaded `.tif`, and paste the card's suggested short note into **Note (optional)**.
5. Check the account's remaining allowance, then submit. Record submission ID, time, exact file SHA, note and returned score in the shared ledger. A score without its file hash cannot establish provenance.
6. Choose one final submission for both prize rounds before the deadline; do not make the choice using unknown private scores.

Steps 3–6 require the authorized account holder and legal eligibility. This environment does not possess or request DrivenData credentials, certify eligibility, or bypass account limits. GitHub repository access is separate.

## 3. Format contract and the previous range error

Official [format and metric](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/): single-band float32 GeoTIFF, confidence in [0,1], UTM 11N/EPSG:32611, 100 m pixels, template bounds.
Measured template: **3,730 rows × 3,292 columns**, transform `(100,0,243350,0,-100,4508550)`, **5,167,373** scored cells.
Use finite [0,1] inside the template footprint, NaN outside and NaN nodata metadata.

**“Predicted values must be in range [0, 1]” does not uniquely diagnose the rejected file.**
Possible causes include out-of-range values, infinity, NaN inside the footprint, or an alignment/footprint mismatch.
We cannot prove which caused the original rejection without that exact file.
Do not silently clip or replace bad model outputs: investigate them. The publisher now rejects malformed raw probabilities before applying an emission policy.

```bash
python scripts/validate_submission.py path/to/approved.tif
```

The local gate follows the published specification and pinned template, not private backend source code.
A local pass is not a guarantee of backend acceptance or a competitive score.

## 4. Build and release

See README.md for data → features → blocked comparison. **Do not run old baseline final training and label it a passing candidate.**
For H16 the bound pipeline is: `build_continuation.py` (all-systems 10-channel grid) → `train_final.py --extra data/features_continuation.npy --hypothesis H16 --bind-to reports/h16_blocked.json` (refuses to bind unless the report's decision is eligible; writes the training manifest, sets `promotion_allowed`, and records `final_prediction` in the report) → `build_submission.py --prob ... --policy topk06_binary --validation reports/h16_blocked.json`.
For H20 (session 3): `fetch_external.py --tag ext/dem10-36326816737` → `build_dem10_grid.py` → `build_continuation.py` → `train_final.py --extra data/features_continuation.npy,data/external/dem10/dem10_channels.f32.npy --hypothesis H20 --bind-to reports/h20_blocked.json --work-dir final_out_h20` → `build_submission.py --prob final_out_h20/prob_final.npy --policy thin10_binary --validation reports/h20_blocked.json`. Column order `[107 | H16 10 | dem10 13]` must match the validated arm.

For H25 (session 4, current): same restore/build steps plus
`build_dem10_context.py`, then `train_final.py --extra data/features_continuation.npy,data/external/dem10/dem10_channels.f32.npy,data/external/dem10/dem10_context.f32.npy --hypothesis H25 --bind-to reports/h25_blocked.json --work-dir final_out_h25` → `build_submission.py --prob final_out_h25/prob_final.npy --policy ridge15_binary --validation reports/h25_blocked.json`. Column order `[107 | H16 10 | dem10 13 | context 36]` must match the validated arm. Manifests are keyed by the validation report (`final_manifest_h25.json`), so a later release never overwrites an earlier release's binding.
The final-prediction manifest must tie the generated NPY hash to its training recipe and the compatible completed validation report.
`build_submission.py --validation ...` refuses missing, non-improving, incomplete, mismatched-policy or unbound evidence (fail-closed gates in `src/gems10/release.py`, protocol-aware for v1, v2 and v3 reports; v3 additionally requires every recorded incumbent comparison to pass; validated code hashes may live in git history when the working tree has moved on).
`--experiment` writes diagnostics only into `scratch/experiments/`, never the download hub.

The canonical field hash detects renamed/recompressed copies; the noncatalogue hash catches catalogue-only changes.
A unique file is not guaranteed a unique rounded DTI, and changing noise to force novelty is prohibited by our methodology.

## 5. Rules, disclosure and shared slot ledger

[Official rules §3.2, §3.4–3.6](https://docs.nlr.gov/docs/fy26osti/96647.pdf): up to three submissions per week per participating entity, one final choice, AI-use narrative, reproducible solution assets.
[Staff cadence clarification](https://community.drivendata.org/t/weekly-submissions/11524/2): rolling window.
Multiple repositories do not by themselves establish multiple registrations or a rule violation. The group must use a shared entity-level ledger.

| Submission ID | UTC | Filename / SHA-256 | Note | Score | Source |
|---|---|---|---|---|---|
| No submission made by this session | — | — | Offline validation only | — | Local scripts do not upload |

Historical group scores live in `reports/submission_audit.json` labelled user-reported; they do not reveal current remaining slots.
