#!/usr/bin/env python3
"""Assemble docs/ data payloads from reports/ (zero heavy deps).

Copies reports/*.json -> docs/data/, writes docs/data/site.json manifest
(latest submission sidecar, harness aggregates, build stamp). Safe to run in
CI without the big rasters.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    docs = REPO_ROOT / "docs"
    (docs / "data").mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "reports": [],
        "submissions": [],
    }
    for rep in sorted((REPO_ROOT / "reports").glob("*.json")):
        shutil.copy(rep, docs / "data" / rep.name)
        manifest["reports"].append(rep.name)
    for meta in sorted((docs / "downloads").glob("*.json")):
        try:
            payload = json.loads(meta.read_text())
        except Exception:
            continue
        manifest["submissions"].append({
            "file": payload.get("file"), "zip": payload.get("zip"),
            "sha256": payload.get("sha256"), "policy": payload.get("policy"),
            "generated_utc": payload.get("generated_utc"),
            "suggested_note": payload.get("suggested_note"),
            "positive_px": payload.get("positive_px"),
            "mass": payload.get("mass"),
        })
    (docs / "data" / "site.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"site data: {len(manifest['reports'])} reports, "
          f"{len(manifest['submissions'])} submissions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
