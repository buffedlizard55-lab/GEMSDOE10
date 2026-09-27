# Next steps — ordered by expected effect on P(win)

Status 2026-09-27: CV campaign running (`reports/cv_*.json` when done). Gate + metric + harness done and tested.

## P0 — before spending any submission slot

1. **Account holder: confirm eligibility (§1.3)** — US citizen/PR or US entity with such a captain; not a federal employee. Nothing here can check it.
2. **Re-verify the three forum rulings first-hand** (masking 11516, corrections 11536, cadence 11524) and record quotes + dates in VERIFICATION.md.
3. **Read the leaderboard, spend slot 1 wisely** — upload the campaign winner, read the public score, then decide. Record slot usage in SUBMISSION_GUIDE.md.
4. **Approve the AI disclosure** (AI_DISCLOSURE.md) and prepare the Winning Model Documentation Template narrative.

## P1 — accuracy, in harness order

5. **Promote the campaign winner** (baseline vs +externals vs +discovery vs +selftrain by masked-mean DTI) to `train_final.py` + `build_submission.py`.
6. **Ablate the 19 new lineament channels** (Frangi/Gabor/coherence vs the ported 88) — keep only measured gains.
7. **Tune the discovery weights** (`--discovery-w` grid) and ray length under the harness; try lidar-ridge noisy-OR fusion as a discovery layer.
8. **Budget/policy refinement** around the winner (1–5% × binary/soft/envelope/halo2) + a masking-robustness check (masked vs unmasked ranking must agree).
9. **Seed ensemble** (3–5 seeds) if it moves the harness number.

## P2 — robustness & Phase 2

10. Per-candidate geological write-ups (strike, agreeing signals, depth proxy, confidence) for the top lineaments — Phase 2 is graded by geologists.
11. GPU U-Net comparison run with the true distance-weighted loss (needs a GPU host).
12. 1 m DEM pilot on an unrestricted host (re-derive tile list from the 3DEP bucket, not the PDF).
13. Clean-machine reproduce: fetch → features → CV → final → gate → pytest, with stable sha256 for fixed seeds.
