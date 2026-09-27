# Limitations — what this entry cannot do (yet), and what it needs

## Honest limits of the current system

1. **The harness is still a catalogue proxy.** System-holdout simulates *unmapped catalogue-like* faults; truly novel faults (new areas, new styles) cannot be validated offline. Absolute harness numbers (≈0.09) are not leaderboard predictions.
2. **No GPU here.** The model is gradient boosting over sampled pixels, not the reference U-Net. A ResNet U-Net with the true distance-weighted loss on a GPU host remains untested by this group.
3. **The 1 m DEM link list is unused.** Lidar enters only via the sibling-built 100 m scarp aggregates (12 bands, 75% coverage). Raw 1 m tiles (100s of GB) need an unrestricted host.
4. **Forum clarifications are second-hand.** Masking (11516), corrections (11536), cadence (11524) come via sibling-session records; the account holder must re-verify each on the forum before the final blind choice.
5. **Single-model family.** No CNN, no transformer, no physics inversion; the ensemble is same-family seeds at most.
6. **Phase-2 narratives unwritten.** Per-candidate geological reasoning (trend, agreeing signals, depth, confidence) and the Winning Model Documentation Template are still open.

## What we need (access / compute)

- DrivenData credentials to read the public leaderboard + submission history (slot accounting) and to re-verify the three forum rulings.
- A GPU host (or patience) for the U-Net comparison run.
- An unrestricted-egress host + ~500 GB scratch for the 1 m DEM pilot.
- The account holder's eligibility confirmation (§1.3) and AI-disclosure sign-off (§3.2).

## Known-good (do not redo)

- Metric (+ masked variant), spec pins, format gate, system-holdout harness, 107-channel streaming builder — all tested (71/71).
- Proximity-to-catalogue features: measured poison for hidden faults (kept behind `--with-proximity` for ablation only).
- SGMC-gap traces: measured 0.0000, excluded. QF-gap: negligible, excluded.
