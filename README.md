# GEMSDOE10 — evidence-first hidden-fault discovery

**Start every session here**, then read [NEXT_STEPS.md](NEXT_STEPS.md), [PROJECT_BRIEF.md](PROJECT_BRIEF.md), [REVIEW.md](REVIEW.md), and [HYPOTHESES.md](HYPOTHESES.md). The original project request is preserved below as requirements, **not verified factual claims**.

**Goal:** improve discovery of geological fault pixels in the [DOE GEMS Prize](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/), not merely redraw known traces. Fault prediction is not proof of a geothermal vent or economic resource. Aim to win; never promise a leaderboard score.

## Start here: submission status

[Submission hub](https://buffedlizard55-lab.github.io/GEMSDOE10/) · [Executive submission guide](https://buffedlizard55-lab.github.io/GEMSDOE10/executive_summary.html)

**No new submission is approved merely because it passes the TIFF format gate.** The publisher requires a completed, matched spatial holdout improvement, artifact-bound final-training evidence, and canonical prediction novelty. If those conditions fail, the site intentionally offers **no upload recommendation**. Zero competition slots are used by these scripts.

- **Duplicate cause measured:** GEMSDOE1 = 5GEMSDOE = GEMSDOE2 recall, byte-for-byte and pixel-for-pixel. Current 8GEMSDOE hedge changes only known catalogue pixels. [Immutable artifact audit](reports/submission_audit.json).
- **Data placement resolved:** automatic pinned public-team-mirror restoration works here; no manual download or GPU needed for HGB. Integrity with inherited pins is verified, independent official-file authentication is not.
- **Session 3 (2026-09-27, later): a new approved artifact.** **H20** — 13 label-free scarp channels from the official USGS 3DEP 1/3 arc-second (~10 m) DEM (built on a GitHub runner, sha256-provenanced, restored through immutable `ext/*` tags because the sandbox cannot reach USGS) on top of the H16 stack — beat the H16 incumbent under the frozen spatially blocked protocol on the development mean (**+0.0147**) *and* the confirmation fold (**+0.0141**), was fit on all data with a hash-bound manifest and released as `gems10-h20-dem10-scarp-thin-…-ffc91a1686.tif` (policy `thin10_binary`, 153,957 cells). **H19** (skeleton thinning on the H16 field) improved development folds but lost the confirmation fold (−0.021) → not released. **H21** measured that the current USGS QFFD and INGENIOUS v1/v2 catalogues contain **no** trace absent from the provided labels (1 pixel each beyond 300 m) — the labels *are* the public catalogue. The session-2 H16 report reproduced exactly (96/96 numbers). [Session-3 register + results](HYPOTHESES.md), [H20 report](reports/h20_blocked.json), [H19 report](reports/h19_blocked.json), [catalogue evidence](KNOWLEDGE.md), [open budget question](NEXT_STEPS.md).
- **Validation repaired and extended (session 2):** historical reports use interleaved raster-component holdouts, not geographic blocks. The spatial protocol (four stripe folds, 4 km training exclusion, confirmation fold, paired baseline/discovery) now also sweeps a preregistered 8-policy set on saved OOF probabilities. Results: **H12 rejected**, **H13 not eligible** (confirmation regressed), **H16 eligible** (dev mean +0.00373, confirmation +0.00070 vs both baselines under topk06_binary). [Preregistration + calibration record](HYPOTHESES.md), [H16 report](reports/h16_blocked.json), [H13 report](reports/h13_blocked.json).
- **Official evidence re-verified 2026-09-27:** public leader DARD **0.3049**; group scores remain user-reported. Staff confirms pixel-exact masking, that new geometry of existing systems qualifies, and that new ground truth may lie within 300 m of known traces; hidden data types/coverage are undisclosed. [Dated source ledger](reports/official_feed.json).
- **Durable knowledge base:** [KNOWLEDGE.md](KNOWLEDGE.md) distills the verified mechanics, sources, implementation map and dead ends for future sessions.

## Core values

**Maximize P(Win):** choose measured discovery improvement over novelty theatre, sunk costs, renamed files, or leaderboard probing. **Own the Outcome:** retain negative results, repair evidence gaps, automate checks, and explain what is still blocked.

## Reproduce on CPU

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py --download       # immutable mirror through gh; all pins checked
python scripts/audit_submissions.py           # ten published files, immutable source links
OMP_NUM_THREADS=2 python scripts/build_features.py --tile-rows 128
python scripts/build_offset.py                # H13 39-channel strip-NCC grid (tiled, low-mem)
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/validate_candidate.py --hypothesis H16 --policy-set v2
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/validate_candidate.py --hypothesis H13 --extra data/features_offset.npy --policy-set v2
# session 3: external 10 m DEM channels (built by .github/workflows/external-data.yml, restored by tag)
python scripts/fetch_external.py --tag ext/dem10-36326816737 && python scripts/build_dem10_grid.py
python scripts/validate_candidate.py --hypothesis H20 --policy-set v3 \
  --extra data/external/dem10/dem10_channels.f32.npy --incumbent-report reports/h16_blocked.json --work scratch/h20
python -m pytest -q
python scripts/build_site.py
```

Allow ~7 GB scratch for source/derived features and OOF grids; ~4 GB RAM worked in this session with bounded batches. Runtime and package versions are recorded in the report. Large data/models stay out of Git. `gh` is used as the available transport for public mirror files; never put credentials into this repository.

### Publication contract

```bash
# H16 pipeline, only AFTER the clean re-run's decision is eligible:
python scripts/build_continuation.py          # all-systems 10-channel H16 grid
python scripts/train_final.py --extra data/features_continuation.npy \
  --hypothesis H16 --bind-to reports/h16_blocked.json \
  --work-dir final_out
python scripts/build_submission.py --prob final_out/prob_final.npy \
  --policy topk06_binary --name gems10-h16-continuation \
  --validation reports/h16_blocked.json \
  --note "H16 endpoint-continuation features (tmi/det_elev/rtp), 117-feature HGB, topk06_binary"
# A diagnostic can instead use --experiment; output is forced into scratch, never Pages.
```

For H20 the bound pipeline is `build_continuation.py` → `train_final.py --extra data/features_continuation.npy,data/external/dem10/dem10_channels.f32.npy --hypothesis H20 --bind-to reports/h20_blocked.json --work-dir final_out_h20` → `build_submission.py --prob final_out_h20/prob_final.npy --policy thin10_binary --name gems10-h20-dem10-scarp-thin --validation reports/h20_blocked.json --note "..."` (done this session).

`train_final.py --bind-to` refuses to write the binding unless the validation report's decision is eligible, and `build_submission.py` re-verifies every hash, the format gate and the duplicate registry before anything reaches `docs/downloads`. A failed candidate must never trigger final training or publication. [Submission guide](SUBMISSION_GUIDE.md).

## Reusable knowledge and operations

- [Ranked hypotheses with layers, physical signatures, novelty and costs](HYPOTHESES.md)
- [Review findings and corrected assumptions](REVIEW.md)
- [Verification ledger](VERIFICATION.md) · [Limitations](LIMITATIONS.md)
- `scripts/refresh_sources.py`: official quotation/availability checks and leaderboard snapshot; failed retrievals preserve dated last-good evidence, visibly marked stale.
- Daily Pages workflow rebuilds the feed/site without auto-submitting, changing model policy, or asserting that changed source text was scientifically reviewed.
- `reports/submission_audit.json`: exact byte and canonical prediction identity, pairwise changed pixels and overlap. Different scores are not a requirement; genuinely different supported predictions are.
- [AI disclosure](AI_DISCLOSURE.md): must reflect actual work and be approved by the account holder. Eligibility and competition registration cannot be certified by code.

## Original user project request — read every session

<details>
<summary>Preserved project prompt (requirements and historical statements; see current evidence above)</summary>

Review the repo.

Here are the results from our groups submissions, separated by ....:

GEMSDOE1

https://buffedlizard55-lab.github.io/GEMSDOE/docs/index.html

GEMSDOE SCORE: 0.1563

....

https://buffedlizard55-lab.github.io/6GEMSDOE/

6GEMSDOE SCORE: 0.0286

....

GEMSDOE3

https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html

GEMSDOE3 SCORE: 0.1193

1 · SUBMIT FIRST

f347b70daa

Pindrop nodes

....

GEMSDOE2

https://buffedlizard55-lab.github.io/GEMSDOE2/docs/index.html

GEMSDOE2 SCORE: 0.1560

....

GEMSDOE3

https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html

GEMSDOE3 SCORE: 0.0830

2 · SUBMIT SECOND37f9d5b855

Pindrop catalogue-gap target SECOND SYSTEM

....

https://buffedlizard55-lab.github.io/GEMSDOE4/

GEMSDOE 4 SCORE: 0.0343

....

GEMSDOE3

https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html

GEMSDOE3 SCORE: 0.1152

3 · CONTROL · UPLOAD LAST

4e03fc9705

Pindrop dense ridge control

....

https://buffedlizard55-lab.github.io/5GEMSDOE/docs/index.html

5GEMSDOE SCORE: 0.1563

....

https://buffedlizard55-lab.github.io/8GEMSDOE/

8GEMSDOESCORE: 0.1563

....

The following is the leaderboard for the competition:

https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/

We need to figure out why we keep scoring 0.1563, are we copying the same work over and over again? we need to come up with different ideas, and not just the same idea tried a different way.

Need to figure out why 5GEMSDOE and GEMSDOE1 have the same score. We should not be generating the same score submissions, they should all be unique.

The following is the leaderboard for the competition:

https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/

Before implementing, generate 3–5 candidate geological hypotheses we haven't tried yet, each naming: the specific layer(s) involved, the physical signature being targeted (e.g., an edge-detection or curvature transform), why it should catch a fault missing from the USGS/INGENIOUS catalogue rather than one already in it, and how it differs from anything already implemented in this repo. Rank them by expected DTI improvement and implementation cost. Validate the top candidate on our spatially-blocked holdout set before touching a weekly submission slot — do not spend a submission slot on an idea that hasn't beaten the current holdout best. If a candidate can't be validated without new external data, name the specific free, official source needed and check it's obtainable before proposing the idea as viable.

Work line by line verifying from official verified trusted sources, provide links for manual review. There should be no manual input, work on your own to complete tasks. Flag any irregularities for review. No hallucinations.

Verify no hallucinations.

The goal of this project is to get a full list that follow our requirements. No hallucinations. Verify line by line.

The following is the leaderboard for the competition:

https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/

We need to quickly look at the results and our results.

We have a good understanding of how our hypothesis, methodology, calculations, analysis are done so we should be able to figure out a way to score higher on the leaderboard using previous results and scoring that we have across the sites listed above. We need to come up with distinct and unique strategies to score higher in this competition leaderboard. We need to start doing heavy and deep research into the part of the project that matters the most, which is the scientific discovery of geothermal vents. We should store all of our information and knowledge that we can gather from official verified sources. This will serve as a starting point for other projects as well. We need to think outside the box but still be grounded in proper scientific research, we are ultimately aiming for a top prize that many others are competing for. So it's important to be contrarian but be smart about it. We need to find sources of data that others are over looking or areas of the project when it comes to geothermal vents. We need to do deep research and critical thinking and come up with new hypothesis to test.

Use these sites as a starting point for understanding how our group has generated submissions in the past.

GEMSDOE1

https://buffedlizard55-lab.github.io/GEMSDOE/docs/index.html

GEMSDOE SCORE: 0.1563

....

https://buffedlizard55-lab.github.io/6GEMSDOE/

6GEMSDOE SCORE: 0.0286

....

GEMSDOE3

https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html

GEMSDOE3 SCORE: 0.1193

1 · SUBMIT FIRST

f347b70daa

Pindrop nodes

....

GEMSDOE2

https://buffedlizard55-lab.github.io/GEMSDOE2/docs/index.html

GEMSDOE2 SCORE: 0.1560

....

GEMSDOE3

https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html

GEMSDOE3 SCORE: 0.0830

2 · SUBMIT SECOND37f9d5b855

Pindrop catalogue-gap target SECOND SYSTEM

....

https://buffedlizard55-lab.github.io/GEMSDOE4/

GEMSDOE 4 SCORE: 0.0343

....

GEMSDOE3

https://buffedlizard55-lab.github.io/GEMSDOE3/docs/index.html

GEMSDOE3 SCORE: 0.1152

3 · CONTROL · UPLOAD LAST

4e03fc9705

Pindrop dense ridge control

....

https://buffedlizard55-lab.github.io/5GEMSDOE/docs/index.html

5GEMSDOE SCORE: 0.1563

....

https://buffedlizard55-lab.github.io/8GEMSDOE/

8GEMSDOESCORE: 0.1563

....

Need to figure out why 5GEMSDOE and GEMSDOE1 have the same score. We should not be generating the same score submissions, they should all be unique.

The following is the leaderboard for the competition:

https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/

0.3049 is the highest score right now so we need to design a new strategy, research, testing, analyzing, and generating submission system than the current website. It should be unique, take unique approaches to generating a submission that can score higher than .3049.

Put this prompt into the repo readme and read it everytime we work on the project as a starting point to make sure we are building what we are aiming for and have a strong base to continue building and improving on making something useful for everyday use. It should solve the problem of having to manually check everything ourselves and having an up to date current feed.

Review the repo.

The following is taken from the Arena AI team and I think it makes a good point on building a successful project, so let's keep the Core Values and Own the Outcome as a focal point when building, developing, researching, suggesting upgrades, and implementing the work.

Our Core Values

Maximize P(Win)

“Maximize the Probability of Winning”: our decision making framework. In every decision, we weigh tradeoffs, assess risk, and choose the path that maximizes the probability that Arena succeeds. We set aside our emotions and make tough decisions in order to maximize P(Win). “Maximize P(Win)” frees us from constraints and clarifies that we must put Arena first.

Own the Outcome

We own results end to end — not just our individual slice of the work. When problems arise and we have the means to act, we do so without waiting for permission or assignment. We treat failure and success as signals and use them to improve. At Arena, we stay accountable to the final outcome.

Work line by line verifying from official verified trusted sources, provide links for manual review. There should be no manual input, work on your own to complete tasks. Flag any irregularities for review. No hallucinations.

Verify no hallucinations.

The goal of this project is to get a full list that follow our requirements. No hallucinations. Verify line by line.

We need to focus on being able to generate a submission into the competition.

The site should be able to generate a TIF file that is required for submission. It should be as easy as download to click a File to submit into the competition. This needs to be in the executive summary or the very beginning of the site. it should be obvious when you visit the site.

I tried to submit the document that i downloaded from the site but it returned this error on the submission form:

"Predicted values must be in range [0, 1]"

Also we need to give it a unique name and A short comment to help you or your team tell submissions apart later e.g. clustering with k=25

Here is the submission page when i click submit file

New submission

File to submitNo file chosen

You can submit a single-band GeoTIFF (.tif) file, or a .zip file containing a single GeoTIFF, with your predictions. It must match the submission format's CRS, shape, and geotransform. You may wish to review the competition rules first.

Note (optional)

A short comment to help you or your team tell submissions apart later e.g. clustering with k=25

Create a executive summary subpage that explains exactly how to make a submission into the contest.

Work on the next steps from the previous sessions first.

The goal of this project is to place top of the leaderboard in this competition. The following is the competition:

https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/

We need to create a project that can compete and place top of the leaderboard. We need to understand the problem, collect all the data and organize it into a clean easily auditable table with official verified links for manual verification.

This is the guidelines we need to follow. https://www.drivendata.org/competitions/306/competition-doe-gems/

Get familiar with the problem through the overview and problem description, https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/. You might also want to reference additional resources available on the about page, https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/.

Download the data from the data, https://www.drivendata.org/competitions/306/competition-doe-gems/data/, tab.

Create and train your own model. This reference solution, https://github.com/drivendataorg/gems-prize-reference-solution implements a simple approach.

Use your model to generate predictions that match the submission format.

Tell me what are you limitations and what you need access to during this project. We will need to find free publicly available sources and data from official and verified sources if we are to use 3rd party or external data.

this pdf outlines how submissions must be entered into the competition.

https://docs.nlr.gov/docs/fy26osti/96647.pdf

You must be able to do your own research, deep research, scientific literature research and organize the knowledge so that we can critically think through the problem and generate a solution through scientific and free publicly available information. this must be done autonomously and must be constantly reviewed and improved upon. Provide suggestions and improvements and implement them.

❌ No DrivenData auth → cannot auto-download training_features.tif, labels.tif, sample_submission.tif, 1m_DEM_links.csv from https://www.drivendata.org/competitions/306/competition-doe-gems/data/ (verified redirect to login)

See below for links from the above site. See attached files for links from the above site.

https://gdr.openei.org/submissions/1391

Download competition data from https://www.drivendata.org/competitions/306/competition-doe-gems/data/ (requires login) to data/

See links below for competition data:

https://www.dropbox.com/scl/fi/aemhtutjgcp6tr3tint94/GEMS_96647.pdf?rlkey=rek210cj2smnmzb8n0sla1vmd&st=wz4kofki&dl=0

https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&st=8junzdyw&dl=0

https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&st=rnino7ya&dl=0

https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=0

https://www.dropbox.com/scl/fi/ig0mban712ns1atphgphe/Digital-elevation-model-links-JSON.pdf?rlkey=zm77f1vbtt2if8hlruymptnu3&st=srhhir10&dl=0

Work line by line verifying from official verified trusted sources, provide links for manual review. There should be no manual input, work on your own to complete tasks. Flag any irregularities for review. No hallucinations.

Verify no hallucinations.

The goal of this project is to get a full list that follow our requirements. No hallucinations. Verify line by line.

Site creation

Create a github page for this repo that has clean ui, user friendly, simple and easy to use. It should be organized and clean.

It should include all relevant information in an easy to read format with official verified links as sources for review. Work line by line verify everything no hallucinations.

**The single remaining blocker to training is data placement**: run `bash scripts/download_competition_data.sh` on any unrestricted machine into `data/`, then `python scripts/prepare_data.py` — after that the full train→inference→validate pipeline is ready to run (GPU needed for training; metric/losses/validation all verified working here on CPU).

> Current-session clarification: the preceding commands describe earlier work and are not scripts in this checkout. Use the current Quickstart above; the CPU data-placement blocker has been resolved.

You need to complete the above task by yourself. Work line by line verifying from official verified trusted sources, provide links for manual review. There should be no manual input, work on your own to complete tasks. Flag any irregularities for review. No hallucinations.

Verify no hallucinations.

The goal of this project is to get a full list that follow our requirements. No hallucinations. Verify line by line.

Run this task through multiple passes.

Pass 1: Implement the task completely and verify the result.

Pass 2: Review your work for bugs, missing requirements, incorrect assumptions, and edge cases. Fix everything you find.

Pass 3: Re-check the entire implementation against the original request. Improve accuracy, reliability, completeness, and code quality. Fix any remaining issues.

Do not stop after the first pass. Each pass must build on the previous one. Before finishing, verify that the final result fully satisfies the original request. Work line by line verify everything no hallucinations.

Go ahead and create a pull request and then merge the pull request onto the main. Make suggestions for what work still needs to be done and any limitations that is in the way of a successful project. It should be worked on in this next session or the next session. Work line by line verify everything no hallucinations.

</details>

## Latest completed result

**Session 3:** H20 (3DEP 10 m scarp channels + H16 + baseline107, 130-feature HGB, `thin10_binary`) is the recommended download on the hub: development folds 0.16483 / 0.18130 / 0.22339 (mean 0.18984) and confirmation 0.18680 versus the H16 incumbent's 0.15197 / 0.17242 / 0.20094 and 0.17274 (`reports/h20_blocked.json`, `reports/final_manifest_h20.json`). These are catalogue-generalisation proxies on four correlated stripe folds with a development-selected policy and a re-used confirmation geography — not a leaderboard forecast. No slot was spent; uploading is the account holder's decision (SUBMISSION_GUIDE.md).

**Session 1 (retained):** H12 was retrained with the same frozen settings during the verification pass: every fold score and the rejection decision reproduced exactly ([reproduction evidence](reports/h12_reproduction.json)). Development delta **+0.001218**, confirmation delta **−0.002171**. **Rejected; no weekly slot.** A unique out-of-fold diagnostic TIFF/ZIP was generated and format-validated in ignored scratch, with [its checks recorded](reports/diagnostic_validation.json); it is deliberately not an approved site download. It is not a full-data final model.

Use `requirements-lock.txt` to recreate this session's Python package versions (Python version is recorded in the experiment report). Browser screenshot testing was attempted but the Chromium download failed TLS; static links and local HTTP routes were tested instead.
