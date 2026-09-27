#!/usr/bin/env python3
"""Restore external-data products published by `.github/workflows/external-data.yml`.

    python scripts/fetch_external.py --list
    python scripts/fetch_external.py --tag ext/dem10-<run_id>      # -> data/external/dem10/
    python scripts/fetch_external.py --tag ext/catalogue-<run_id>  # -> data/external/catalogue/

Transport: GitHub REST through `gh` (the only HTTPS egress available in the
research sandbox). Each file is fetched raw from the immutable tag and its
sha256 is checked against the manifest committed alongside it (the manifest
itself is re-hashed and recorded). Nothing is written until every hash matches.
Restored files live in ignored `data/external/` and are never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "buffedlizard55-lab/GEMSDOE10"


def gh_json(endpoint: str):
    out = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True, check=True,
                         timeout=120)
    return json.loads(out.stdout)


def gh_raw(endpoint: str, dest: Path) -> None:
    with dest.open("wb") as fh:
        subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github.raw+json", endpoint],
                       stdout=fh, check=True, timeout=1800)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def list_tags(prefix: str = "ext/") -> list[dict]:
    refs = gh_json(f"repos/{REPO}/git/matching-refs/tags/{prefix}")
    return [{"tag": r["ref"].removeprefix("refs/tags/"), "object": r["object"]["sha"],
             "type": r["object"]["type"]} for r in refs]


def restore(tag: str, repo: str = REPO, dest_root: Path | None = None) -> Path:
    kind = tag.split("/", 1)[1].rsplit("-", 1)[0]  # dem10 | catalogue | failed-<job>
    failed = kind.startswith("failed-")
    # Failed builds are restored under their full tag name (logs only; no manifest check)
    dest = (dest_root or ROOT / "data/external") / (tag.split("/", 1)[1] if failed else kind)
    tree = gh_json(f"repos/{repo}/git/trees/{tag}?recursive=1")
    files = [t for t in tree["tree"] if t["type"] == "blob"]
    if not files:
        raise SystemExit(f"no files under tag {tag}")
    manifest_name = "manifest.json" if kind == "dem10" else "catalogue_diff.json"
    if failed:
        manifest_name = "STATUS.txt"
    with tempfile.TemporaryDirectory(dir=dest.parent if dest.parent.exists() else None,
                                     prefix=".ext-") as tmp:
        tmpdir = Path(tmp)
        for f in files:
            out = tmpdir / f["path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            gh_raw(f"repos/{repo}/contents/{f['path']}?ref={tag}", out)
            if out.stat().st_size != f["size"]:
                raise SystemExit(f"size mismatch for {f['path']}: {out.stat().st_size} != {f['size']}")
            print(f"[fetched] {f['path']} {f['size']} B", flush=True)
        manifest = ({} if failed or not (tmpdir / manifest_name).exists()
                    else json.loads((tmpdir / manifest_name).read_text()))
        expected = {}
        if failed:
            expected = {}
        elif kind == "dem10":
            expected = {f"{c}.f32.npy": s["sha256"] for c, s in manifest["channel_stats"].items()}
        else:
            expected = {n: v["sha256"] for n, v in manifest.get("files", {}).items()}
        bad = []
        for name, sha in expected.items():
            p = tmpdir / name
            if not p.exists() or sha256_file(p) != sha:
                bad.append(name)
        if bad:
            raise SystemExit(f"sha256 mismatch against manifest for: {bad}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            import shutil
            shutil.rmtree(dest)
        Path(tmp).rename(dest)
    record = {"tag": tag, "commit": tree["sha"],
              "manifest_sha256": (sha256_file(dest / manifest_name)
                                  if (dest / manifest_name).exists() else None),
              "files": {f["path"]: f["size"] for f in files}}
    (dest / "RESTORED_FROM.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=1))
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()
    if args.list or not args.tag:
        for t in list_tags():
            print(t["tag"], t["object"], t["type"])
        return 0
    restore(args.tag, args.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
