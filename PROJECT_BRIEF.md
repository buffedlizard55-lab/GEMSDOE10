# Project brief — read every session

> **Maximize P(Win)** — in every decision, weigh tradeoffs, assess risk, and choose the path that maximizes the probability of placing top of the leaderboard. Set aside sunk costs; follow measured evidence.
>
> **Own the Outcome** — own results end to end. When problems arise and we have the means to act, act without waiting. Treat failure and success as signals.

## Mission

Place top of the leaderboard in the [GEMS Prize Challenge](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) (currently #1 DARD **0.3049**; our group best 0.1563; gap +0.1486). Build a unique, well-tested, geologically-grounded submission system — not another copy of the same file.

## Why 0.1563 keeps repeating

Root-cause audit (sibling GEMSDOE9, 2026-09-26): GEMSDOE1, GEMSDOE2-recall and 5GEMSDOE shipped **byte-identical** files (`submission.tif` sha256 `7f00890a…`, 172,974 px at 1.0) — copied across repos without changing the pixel field, so the leaderboard returned the same DTI. Separately, 6GEMSDOE's catalogue-memorising HGB scored 0.0286 and GEMSDOE4 0.0343.

**GEMSDOE10 fixes:** every artifact is built from scratch with a unique UTC stamp + sha; `scripts/build_submission.py` **refuses** to publish a byte-duplicate of any known artifact; the submission note carries strategy + sha; and model selection uses hidden-fault simulation (system-holdout), never catalogue reproduction.

## Operating rules

- Work autonomously, line by line, from official verified sources; link every claim (see VERIFICATION.md). No hallucinations; flag irregularities.
- No manual input required: data placement, training, validation and the site build are all scripted.
- External data only from free, public, licensed sources (USGS public domain; GeoDAWN ScienceBase).
- Every submission file must pass the hard gate (`scripts/validate_submission.py`) — the form rejects NaN-inside-footprint with "Predicted values must be in range [0, 1]".

## Key links

- Competition: https://www.drivendata.org/competitions/306/competition-doe-gems/
- Problem/metric/format: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- About/task: https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/
- Data (login): https://www.drivendata.org/competitions/306/competition-doe-gems/data/
- Rules PDF: https://docs.nlr.gov/docs/fy26osti/96647.pdf
- Reference solution: https://github.com/drivendataorg/gems-prize-reference-solution
- Leaderboard: https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/
- Forum (masking 11516; new-fault ID 11527/11536; cadence 11524): https://community.drivendata.org/c/gems-prize-challenge/111
- GeoDAWN: https://doi.org/10.5066/P93LGLVQ · INGENIOUS: https://doi.org/10.15121/1881483 · QFaults: https://earthquake.usgs.gov/hazards/qfaults/
