# Assembly contract — supersedes the earlier slot-1 plan

The old plan prioritized FAR10 and proposed an upload before a matched spatial improvement. That is withdrawn. Hidden-fault distances are not public, and the user expressly requires offline improvement first.

1. Read HYPOTHESES.md, REVIEW.md and the completed `reports/h12_blocked.json`.
2. H12's initial run failed confirmation. Do not train a final H12 model, add a lidar hedge, change the budget after scoring, or upload it.
3. A future successful experiment must beat the current compatible holdout incumbents, survive confirmation, and establish new noncatalogue predictions. Keep experiment history and negative results.
4. Implement the winner's actual full-data training recipe. The existing `train_final.py` is baseline107 only. A final-training JSON manifest must contain model and probability SHA-256, hypothesis, identical validated feature/input identity, fixed config and validation protocol.
5. Bind that manifest's filename/hash and probability hash to the completed evaluation report; do not hand-label baseline outputs as a new hypothesis. `release.verify_training_binding` checks persisted files and code identity.
6. `build_submission.py --validation ...` stages the TIF, checks raw probability values, exact geometry/footprint, novelty, ZIP integrity metadata, timestamp/hash name and note. Metadata is published last. `--experiment` is forced to ignored scratch.
7. `build_site.py` rechecks hashes, training binding, format, novelty and the ZIP's actual member before rendering download links. A sidecar alone is not evidence.
8. The account holder uses the normal DrivenData submission page and records the exact artifact/score association. Code does not upload or allocate slots. One participating entity, one final choice, rolling allowance; see SUBMISSION_GUIDE.md.

No numerical public-score trigger, implied remaining-slot count, or prize guarantee is justified by our current evidence.
