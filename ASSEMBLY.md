# Final assembly + online iteration plan

## Decision rule (offline)

1. Rank configs by **GT_FAR10 masked @topk01/02_binary** (decision column), GT_ALL as tiebreak, GT_FAR20 as sanity.
2. Current standings (5-fold masked means):
   - baseline (107ch geophysics): FAR10 **0.0508** @topk01 (0.0445 @topk02); ALL 0.0929 @topk02
   - +discovery (rays .5 + corr .4, global): FAR10 0.0314 → **rejected global** (crowds out far detections); far-only variant pending in fuse_search
   - +selftrain: pending (fold 0: 0.0883 ALL ≈ baseline 0.0894 — early neutral)
   - +externals (lidar+rad features): pending rerun (memmap fix)
3. Promote the winner; fuse_search sets discovery weights (far-only) post-hoc.

## The file (v1)

- `train_final.py` with the winning config (full 60,988 positives + 200k negatives, seed 7) → `prob_final.npy`
- Discovery complement (far-only, weights from fuse_search) + modest lidar-ridge mass for Phase-2 EV
- Core budget 1–2% binary (harness optimum shifts 2%→1% as GT sparser/distant); complement sized so total effective mass ≈ 2%
- `build_submission.py` (gate + duplicate-refusal + zip + note) → `docs/downloads/` → `build_site.py`

## Online iteration (3 slots/week until Dec 3 ≈ 27 slots)

- Slot 1: v1 file → read public score. If ≥0.20: iterate budgets/weights around v1. If <0.12: the harness→public mapping is off — pivot to coverage/lineament-heavy variants.
- Keep a submissions log (file sha, policy, public score, date) in `reports/submissions_log.json`.
- Never spend 2 slots on the same idea; never submit an ungated file.
- The blind final choice (one file, both rounds) is made before the deadline from the log + harness + geology notes — never from private-score knowledge (there is none).
