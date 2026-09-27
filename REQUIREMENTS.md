# Requirements matrix — no blanket completion claim

| Requirement | Status | Evidence / remaining boundary |
|---|---|---|
| Review sibling duplicate submissions | Implemented and measured | Ten artifacts, immutable commits, byte/canonical hashes and changed-pixel matrix in reports/submission_audit.json; actual account associations unverified |
| 3–5 novel geological hypotheses before implementation | Four preregistered | HYPOTHESES.md: layers, signatures, rationale, novelty boundary, qualitative opportunity/cost rank, source access |
| Validate top hypothesis on spatially blocked holdout | Implemented and executed | H12 runner, four stripe folds, 4 km exclusion, paired controls, fixed top2 policy; no private labels |
| No slot without beating current holdout best | Fail-closed release gate | H12 failed confirmation; no upload. External historical arms without persistent data remain unvalidated |
| Official links and scientific support | Curated first-hand ledger | VERIFICATION.md, official_feed.json, HYPOTHESES.md; not a claim that every scientific inference is verified |
| Autonomous data placement | Resolved via pinned mirror | fetch_data.py --download; three inherited pins match. Official tab still login-gated |
| Single-band float32 TIF, exact CRS/shape/transform, finite [0,1] footprint | Enforced locally | raster.py tests and publisher raw input checks; backend acceptance not guaranteed |
| Diagnose original range error | Guarded, root cause unknown | Rejected artifact absent; checks NaN, infinity, range and alignment |
| Unique predictions/name/note | Enforced by canonical identity and metadata | Reject catalogue-only/renamed copies; distinct rounded scores cannot be guaranteed |
| One-click site download and executive guide | UI implemented, release intentionally withheld | No unproven candidate promoted. Static site exposes TIFF/ZIP only after all evidence gates |
| Current automated feed | Implemented | Daily Pages workflow checks official text/leaderboard; stale evidence remains visibly dated |
| Core values and original prompt | Preserved | README.md + AGENTS.md entry-point instruction |
| Three review passes | Commands and findings recorded | REVIEW.md and reports/verification_run.json; no blanket zero-bug assertion |
| AI disclosure and eligibility | Draft / account-holder obligation | AI_DISCLOSURE.md; cannot certify participant's legal facts |
| Top leaderboard / >.3049 | NOT achieved or promised | Requires better independently validated detector and actual platform evidence |
| External H15 usable | BLOCKED | Official GDR landing page available; binary requests fail TLS, geographic coverage unknown |
| PR, merge, Pages | Verify via GitHub records | Session handoff/final response; do not claim success until API confirms |
