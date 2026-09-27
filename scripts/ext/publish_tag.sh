#!/usr/bin/env bash
# Publish a directory as an ORPHAN commit referenced only by an immutable tag.
# Usage: publish_tag.sh <dir> <tag> <message>
# Runs on GitHub Actions with the job's GITHUB_TOKEN (contents: write). It never
# creates or updates a branch; the sandbox reads files back through
# `gh api repos/<repo>/contents/<file>?ref=<tag>` (raw), which is the only
# GitHub transport reachable from the restricted research runtime.
set -euo pipefail
DIR="$1"; TAG="$2"; MSG="$3"
: "${GITHUB_REPOSITORY:?}" "${GITHUB_TOKEN:?}"
cd "$DIR"
# Refuse files that the contents API cannot serve raw (100 MB limit).
find . -type f -size +95M -print -exec false {} + || { echo "file >95 MB present; split it"; exit 1; }
git init -q
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git checkout -q --orphan ext-data
git add -A
git commit -q -m "$MSG"
sha=$(git rev-parse HEAD)
git tag -a "$TAG" -m "$MSG" "$sha"
git push -q "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" "refs/tags/${TAG}"
echo "published tag ${TAG} -> commit ${sha}"
{
  echo "### external data published"
  echo "- tag: \`${TAG}\`"
  echo "- commit: \`${sha}\`"
  echo "- files:"
  git ls-files | while read -r f; do echo "  - \`$f\` ($(stat -c %s "$f") B)"; done
} >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
