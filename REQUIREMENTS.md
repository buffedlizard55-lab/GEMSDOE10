# Requirements matrix — no blanket completion claim

| Requirement | Status | Evidence / remaining boundary |
|---|---|---|
| Review sibling duplicate submissions | Implemented and measured | Ten artifacts, immutable commits, byte/canonical hashes and changed-pixel matrix in reports/submission_audit.json; actual account associations unverified |
| 3–5 novel geological hypotheses before implementation | Preregistered every session (session 5: H28–H31, ranked gain × cost, committed before holdout contact) | HYPOTHESES.md: layers, signatures, rationale, novelty boundary, qualitative opportunity/cost rank, source access |
| Validate top hypothesis on spatially blocked holdout | Implemented and executed | H12 runner, four stripe folds, 4 km exclusion, paired controls; sessions 3–5 ran H19/H20/H24/H25/H28/H29 with exact reproduction gates (session 5: byte-identical OOF grids); session 5 adds density-matched protocol v5; no private labels |
| No slot without beating current holdout best | Fail-closed release gate | H12 failed confirmation; H24/H25 (v4) and H28 (v5, recomputed fixed-incumbent win + full-density non-inferiority) beat the incumbent on dev mean AND confirmation before any release binding; H29 refused; train_final refuses non-eligible reports; no upload by this repo |
| Official links and scientific support | Curated first-hand ledger | VERIFICATION.md, official_feed.json, HYPOTHESES.md; not a claim that every scientific inference is verified |
| Autonomous data placement | Resolved via pinned mirror | fetch_data.py --download; three inherited pins match. Official tab still login-gated |
| Single-band float32 TIF, exact CRS/shape/transform, finite [0,1] footprint | Enforced locally | raster.py tests and publisher raw input checks; backend acceptance not guaranteed |
| Diagnose original range error | Guarded, root cause unknown | Rejected artifact absent; checks NaN, infinity, range and alignment |
| Unique predictions/name/note | Enforced by canonical identity and metadata | Reject catalogue-only/renamed copies; distinct rounded scores cannot be guaranteed |
| One-click site download and executive guide | UI implemented, three releases promoted | Hub exposes 3 approved downloads (H16 superseded, H20 superseded, H25 current) only after eligibility + binding + format + novelty gates (build_site: 3 approved, 0 refused) |
| Current automated feed | Implemented | Daily Pages workflow checks official text/leaderboard; stale evidence remains visibly dated |
| Core values and original prompt | Preserved | README.md + AGENTS.md entry-point instruction |
| Three review passes | Commands and findings recorded | REVIEW.md (session-4 and session-5 findings tables) and reports/verification_run.json; no blanket zero-bug assertion |
| AI disclosure and eligibility | Draft / account-holder obligation | AI_DISCLOSURE.md; cannot certify participant's legal facts |
| Top leaderboard / >0.3049 (live leader 0.3168, re-read 2026-09-27) | NOT achieved or promised | Requires better independently validated detector and actual platform evidence; local holdout gains do not forecast it |
| External H15 usable | BLOCKED | Official GDR landing page available; binary requests fail TLS, geographic coverage unknown |
| PR, merge, Pages | Verify via GitHub records | Session handoff/final response; do not claim success until API confirms |
| Explain why several sibling sites scored 0.1563 | Measured | Byte-identical artifacts (GEMSDOE1 = 5GEMSDOE = GEMSDOE2 recall) + catalogue-only 8GEMSDOE hedge; re-verified 2026-09-28 |
| Overlooked official data sources, obtainability checked | Session 5: 3DEP 1 m GeoDAWN lidar measured | 98.3 % footprint coverage, 686 tiles / 157 GB (`reports/lidar1m_inventory.json`, runner tag); derivation is next-session P0 |
| Learn from scored leaderboard history | Session 5: leaderboard probe | `reports/lb_probe.json` — density bound 0.13–0.6 %, near-catalogue emission profile; bounds, not measurements |
