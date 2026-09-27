# Requirements matrix

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| F-01 | Single-band float32 GeoTIFF, values in [0,1] | p.967 §format | ✅ enforced | `raster.write_submission`, gate rule 6 |
| F-02 | EPSG:32611, 100 m, 3292×3730, exact geotransform | p.967 §format | ✅ enforced | gate rules 2–5 |
| F-03 | NaN only outside footprint; footprint == official mask | p.967 §format + form behavior | ✅ enforced | gate rules 7–9, `tests/test_gate.py` |
| F-04 | Score = DTI α=0.2 β=0.8 R=300 m; worked example 0.60 | p.967 §metric | ✅ implemented | `metric.py`, `tests/test_metric.py` |
| F-05 | Model the catalogue masking in selection | forum 11516 | ✅ implemented | `fp_ignore_mask`, harness reports both |
| F-06 | Select on hidden-fault simulation, not catalogue fit | competition structure | ✅ implemented | `systems.py` + `run_cv.py` |
| F-07 | No byte-duplicate of known artifacts | 0.1563 audit | ✅ enforced | `build_submission.py` refusal list |
| F-08 | Unique stamp + sha + note per artifact | brief | ✅ enforced | sidecar `.json` |
| F-09 | Reproducible from scripts (no manual steps) | brief | ✅ | fetch→features→cv→final→gate |
| F-10 | External data licensed (USGS/GeoDAWN public) | rules | ✅ | `data/README.md` sources |
| F-11 | AI disclosure drafted | rules §3.2 | ✅ draft | `AI_DISCLOSURE.md` (needs sign-off) |
| F-12 | Eligibility confirmed | rules §1.3 | ⬜ owner action | — |
| F-13 | Slot accounting (3/week) + blind final choice | rules §3.4/§3.6 | ⬜ owner action | `SUBMISSION_GUIDE.md` template |
| F-14 | Site with executive summary + downloads | brief | ✅ | `docs/` |
| F-15 | CI: tests + gate on every push | brief | ✅ | `.github/workflows/ci.yml` |
