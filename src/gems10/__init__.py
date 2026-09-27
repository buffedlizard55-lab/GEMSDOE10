"""gems10 — GEMS Prize (DOE/DrivenData) hidden-fault discovery toolkit.

GEMSDOE10's thesis: both prize rounds score faults MISSING from the catalogue,
with known-catalogue pixels masked from evaluation (staff clarification, forum
thread 11516). Optimising catalogue reproduction is therefore the wrong
objective — the 6GEMSDOE HGB reached 0.17 blocked-CV catalogue DTI and scored
0.0286 on the leaderboard. This package optimises *hidden-fault detection*:

  * metric.py     — distance-weighted Tversky index (official formula) plus the
                    masked variant that models the staff-clarified scoring.
  * raster.py     — submission IO + the hard format gate (NaN-inside-footprint).
  * features.py   — derived structural features (edges/curvature/lineaments).
  * placement.py  — metric-aware prediction placement (budgets, envelopes).
  * cv.py         — spatially-blocked, buffered cross-validation (geography).
  * systems.py    — fault-system labelling + SYSTEM-HOLDOUT folds. The primary
                    yardstick: whole fault systems are held out to simulate
                    unmapped faults, train systems are masked from FP exactly
                    like the official scoring masks the known catalogue.
  * discovery.py  — strike continuation, relay corridors, external-gap fusion.
  * selftrain.py  — semi-supervised second pass (verified-against-independent-
                    signals pseudo-positives).

Provenance: metric/spec/raster/features/cv/placement were adapted from the
same owner's 6GEMSDOE repo and re-verified here (tests/ + VERIFICATION.md).
systems/discovery/selftrain and the masked scoring are new in GEMSDOE10.

Every constant is either (a) quoted from an official source with a URL, or
(b) measured from the official bytes and pinned with a sha256. Nothing else.
"""

__all__ = [
    "metric",
    "raster",
    "spec",
    "features",
    "placement",
    "cv",
    "systems",
    "discovery",
    "selftrain",
]
__version__ = "10.0.0"
