#!/usr/bin/env python3
"""Re-fetch sibling artifacts at immutable commits; compare bytes AND pixels.

Uses authenticated gh transport to PUBLIC team repositories. Does not read
DrivenData accounts; scores are explicitly user-reported, not inferred from files.
"""
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems10.novelty import raster_identity
from gems10.raster import sha256_file

ARTIFACTS = [
    ("GEMSDOE1", "GEMSDOE", "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif", .1563),
    ("5GEMSDOE", "5GEMSDOE", "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif", .1563),
    ("GEMSDOE2-recall", "GEMSDOE2", "docs/gemsdoe2-recall-arm-7f00890a.tif", None),
    ("GEMSDOE2-union", "GEMSDOE2", "docs/gemsdoe2-dual-family-union-f68e590f.tif", .1560),
    ("GEMSDOE3-nodes", "GEMSDOE3", "docs/downloads/pindrop-v4-nodes-20260925T152420Z-f347b70daa.tif", .1193),
    ("GEMSDOE3-gap", "GEMSDOE3", "docs/downloads/pindrop-v4-discovery-20260925T152423Z-37f9d5b855.tif", .0830),
    ("GEMSDOE3-ridge", "GEMSDOE3", "docs/downloads/pindrop-v4-ridge-20260925T152422Z-4e03fc9705.tif", .1152),
    ("GEMSDOE4", "GEMSDOE4", "data/evidence/combined/submission.tif", .0343),
    ("6GEMSDOE", "6GEMSDOE", "downloads/gems6_hgb88-topk03_33cec71ff0.tif", .0286),
    ("8GEMSDOE-hedge-current", "8GEMSDOE", "docs/downloads/8GEMSDOE_Hedge-v2_submission.tif", .1563),
]


def gh(endpoint, out=None):
    cmd = ["gh", "api"]
    if out is not None:
        with out.open("wb") as stream:
            subprocess.run(cmd + ["-H", "Accept: application/vnd.github.raw+json", endpoint],
                           stdout=stream, check=True, timeout=180)
    else:
        return json.loads(subprocess.check_output(cmd + [endpoint], timeout=60))


def main():
    work = ROOT / "scratch/artifact_audit"
    work.mkdir(parents=True, exist_ok=True)
    entries, fields, refs = [], {}, {}
    for ident, repo, path, score in ARTIFACTS:
        entry = {"id": ident, "repository": repo, "path": path,
                 "user_reported_score": score,
                 "score_provenance": "user message; account submission/file association NOT independently verified"}
        try:
            if repo not in refs:
                refs[repo] = gh(f"repos/buffedlizard55-lab/{repo}/commits/main")["sha"]
            ref = refs[repo]
            entry["commit"] = ref
            entry["source_url"] = f"https://github.com/buffedlizard55-lab/{repo}/blob/{ref}/{path}"
            file = work / f"{ident}.tif"
            gh(f"repos/buffedlizard55-lab/{repo}/contents/{path}?ref={ref}", file)
            entry.update(raster_identity(file, ROOT / "data/sample_submission.tif", ROOT / "data/labels.tif"))
            entry["sha256"] = sha256_file(file)
            entry["bytes"] = file.stat().st_size
            entry["status"] = "measured"
            with rasterio.open(file) as ds:
                fields[ident] = ds.read(1)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            entry["status"] = "unavailable"
            entry["error"] = str(exc)
        entries.append(entry)
        print(ident, entry["status"], flush=True)
    with rasterio.open(ROOT / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    with rasterio.open(ROOT / "data/labels.tif") as ds:
        known = ds.read(1) == 1
    pairs = []
    for i, (a, x) in enumerate(fields.items()):
        for b, y in list(fields.items())[i+1:]:
            changed = (x != y) & fp
            pairs.append({"a": a, "b": b, "changed_pixels": int(changed.sum()),
                          "changed_noncatalogue_pixels": int((changed & ~known).sum()),
                          "positive_iou": float(((x > 0) & (y > 0) & fp).sum() /
                                                max(1, (((x > 0) | (y > 0)) & fp).sum()))})
    report = {"checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "artifacts": entries, "pairs": pairs,
              "caveats": ["Same rounded DTI alone does not prove identical predictions",
                          "Current published artifact may not be the file actually uploaded",
                          "Masked-only differences need not improve scoring; do not waste a slot",
                          "Scores are user-reported, not authenticated submission-history evidence"]}
    (ROOT / "reports/submission_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(e["status"] != "measured" for e in entries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
