# Submission guide — how to enter the file into the contest

## 1. Download the file

Go to the site's **Executive summary** (or this repo's `docs/downloads/`) and download the latest `gems10-*-<stamp>.tif` **or** its `.zip` (the form accepts either). Every published file has passed the hard gate below and carries a unique UTC stamp + sha256.

Current candidate: *(filled when the CV campaign lands — see `reports/`)*.

Suggested submission note (shown in the file's `.json` sidecar, field `suggested_note`):
`gems10-<name> | <policy> | sha <10 hex>` — e.g. `gems10-discovery-v1 | topk02_binary | sha 9f3ac41d2b`.

## 2. Submit on DrivenData

1. Sign in and open the competition: https://www.drivendata.org/competitions/306/competition-doe-gems/
2. Click **Submit** in the sidebar → **Make new submission**.
3. **File to submit:** choose the `.tif` (or the `.zip` containing the single GeoTIFF).
4. **Note (optional):** paste the suggested note — it identifies the strategy later.
5. Submit. The file must match CRS/shape/geotransform; ours is written from the official template so it does byte-exactly.

## 3. Rules that constrain submissions (account holder owns compliance)

- **3 scored submissions per rolling 7-day window.** Spend slot 1 on the current file, read the public score, then decide — do not burn all three on variants of one idea.
- **One submission is chosen blind for both rounds** before the deadline, without knowledge of private performance. Record the choice here when made.
- **Eligibility (§1.3):** US citizen/permanent resident (or US entity with such a captain); no federal employees. An ineligible winner is disqualified regardless of score.
- The public leaderboard shows public-test performance only and "may not be the same as the final scores on the private leaderboard".

## 4. What the gate guarantees (and what it cannot)

`scripts/validate_submission.py` enforces 9 rules: single-band float32 GeoTIFF; EPSG:32611; 100 m; 3292×3730; exact geotransform; finite values in [0,1]; **no NaN/Inf inside the footprint** (the published cause of the *"Predicted values must be in range [0, 1]"* form rejection); footprint exactly equals the official mask; nodata = NaN. A file that passes is *form-acceptable*; only the leaderboard measures *score*.
