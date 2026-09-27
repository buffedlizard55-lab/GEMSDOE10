# Generative-AI disclosure (draft — account holder must review & approve)

> Competition rules §3.2 require disclosure of AI assistance. This project was produced with an AI coding agent (Arena.ai Agent Mode). The statement below is a review-ready draft; the account holder must verify it and submit the final wording.

## Draft statement

"This submission was developed with the assistance of an AI coding agent (Arena.ai Agent Mode, September 2026). The agent wrote code, ran experiments, and drafted documentation under the direction of the competitor, who reviewed the approach, verified results against official sources, and made all final submission decisions. All data are the official competition rasters (hash-verified) plus public-domain USGS/GeoDAWN externals listed in data/README.md. The model is a histogram gradient-boosting classifier over physics-informed geophysical features; no AI-generated training labels were used. Test-set faults were never accessed — model selection used only system-holdout cross-validation on the public catalogue."

## Evidence per sentence

1. "developed with an AI coding agent" — this repo's full history is agent-authored commits on `arena/01a0e067-gemsdoe10`.
2. "competitor directed and decided" — *(account holder: confirm — you chose the strategy and the submitted file)*.
3. "official rasters hash-verified + public externals" — `scripts/fetch_data.py` pins; `data/README.md` sources.
4. "HGB over physics-informed features; no AI labels" — `src/gems10/modeling.py`, `scripts/run_cv.py`; pseudo-labels (if used) come from the model's own CV predictions, flagged in `reports/`.
5. "never accessed test faults" — true by construction: the private set is unreleased; all scores reported are system-holdout proxies documented in `reports/`.
