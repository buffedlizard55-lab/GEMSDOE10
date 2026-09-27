#!/usr/bin/env python3
"""Generate a static evidence hub. Never publish an unchecked download sidecar."""
from __future__ import annotations
import datetime
from html import escape as esc
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

import markdown

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gems10 import novelty, raster, release

NAV = [('index', 'Submission hub'), ('executive_summary', 'How to submit'),
       ('results', 'Results & audit'), ('hypotheses', 'Research'),
       ('sources', 'Official sources'), ('verification', 'Review')]
REPO_URL = 'https://github.com/buffedlizard55-lab/GEMSDOE10/blob/main/'


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def md(text):
    html = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    html = re.sub(r'href="(reports/[^"#]+)"', lambda m: 'href="data/' + m[1][8:] + '"', html)
    html = re.sub(r'href="([^":]+\.md(?:#[^"]*)?)"', lambda m: 'href="' + REPO_URL + m[1] + '"', html)
    return html


def approved_submissions(root):
    accepted, refused = [], []
    registry = read_json(root / 'reports/submission_audit.json', {}).get('artifacts', [])
    for path in sorted((root / 'docs/downloads').glob('*.json')):
        try:
            data = read_json(path)
            if data.get('status') != 'approved':
                raise ValueError('not approved')
            for key in ('file', 'zip', 'validation_file'):
                if not data.get(key) or Path(data[key]).name != data[key]:
                    raise ValueError('invalid artifact basename')
            if not registry or any(r.get('status') != 'measured' for r in registry):
                raise ValueError('sibling audit missing or incomplete')
            evidence_path = root / 'reports' / data['validation_file']
            if raster.sha256_file(evidence_path) != data.get('validation_sha256'):
                raise ValueError('validation hash mismatch')
            evidence = read_json(evidence_path)
            release.check_release(evidence, data['probability_sha256'], data['policy'])
            release.verify_training_binding(evidence, root)
            tif, zpath = path.parent / data['file'], path.parent / data['zip']
            if raster.sha256_file(tif) != data['sha256'] or raster.sha256_file(zpath) != data['zip_sha256']:
                raise ValueError('artifact hash mismatch')
            raster.assert_submittable(tif, root / 'data/sample_submission.tif')
            identity = novelty.raster_identity(tif, root / 'data/sample_submission.tif', root / 'data/labels.tif')
            for key in ('field_sha256', 'noncatalogue_sha256'):
                if identity[key] != data.get(key):
                    raise ValueError('canonical identity mismatch')
            novelty.refuse_duplicate(identity, registry + accepted)
            with zipfile.ZipFile(zpath) as z:
                if z.namelist() != [tif.name] or z.read(tif.name) != tif.read_bytes():
                    raise ValueError('ZIP does not contain exactly the verified TIFF')
            accepted.append(data)
        except Exception as exc:
            refused.append({'metadata': path.name, 'reason': str(exc)})
    return accepted, refused


def page(slug, title, content, stamp):
    nav = ''.join(f'<a href="{s}.html" {"aria-current=page" if s == slug else ""}>{t}</a>' for s, t in NAV)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Evidence-first geological fault discovery: audited predictions, spatial validation and safe submission downloads.">
<title>{esc(title)} · GEMSDOE10</title><link rel="stylesheet" href="assets/style.css"></head>
<body><a class="skip" href="#main">Skip to content</a><header><div class="wrap masthead"><a class="brand" href="index.html"><span class="mark">G/10</span><span>GEMSDOE10<small>GEOTHERMAL FAULT RESEARCH</small></span></a><span class="edition">Evidence before emission<br>DOE GEMS Prize · 2026</span></div><nav class="wrap">{nav}</nav></header>
<main id="main" class="wrap">{content}</main><footer class="wrap"><b>Maximize P(Win). Own the Outcome.</b><p>Local evidence is not a private competition score. AI-assisted research; official links and limitations are explicit.</p><p>Site built {esc(stamp)} · <a href="https://github.com/buffedlizard55-lab/GEMSDOE10">Repository</a> · <a href="data/site.json">Machine-readable manifest</a></p></footer></body></html>'''


def submission_panel(submissions):
    if not submissions:
        return '''<section class="release"><span class="eyebrow">SUBMISSION DESK / RELEASE GATE</span><h2>No approved submission</h2><p>Protect the weekly slot. No new candidate has all the evidence required for publication. A renamed TIFF or a catalogue-only change is not a new discovery.</p><a class="button" href="results.html">Review the holdout decision <span>↗</span></a><a class="text-link" href="executive_summary.html">How submission works →</a><p class="fine">Nothing is uploaded automatically. Download buttons appear here only after improvement, provenance, novelty and GeoTIFF checks pass.</p></section>'''
    cards = []
    for s in submissions:
        cards.append(f'''<section class="release"><span class="eyebrow">APPROVED / FORMAT RECHECKED</span><h2>{esc(s['name'])}</h2><a class="button" href="downloads/{esc(s['file'])}" download>Download .tif ↓</a><a class="text-link" href="downloads/{esc(s['zip'])}" download>Single-file ZIP</a><p>Note: <code>{esc(s['suggested_note'])}</code></p><p class="fine">{esc(s['file'])} · sha256 {esc(s['sha256'])}</p></section>''')
    return ''.join(cards)


def _arm_dti(row, arm, selection):
    """DTI for (fold, arm): v2 reports score per policy and select per arm on
    development folds; v1 reports carry a scalar dti per arm."""
    scores = row.get('scores', {}).get(arm)
    if scores is None:
        return None
    if isinstance(scores, dict) and 'dti' not in scores:
        pol = (selection or {}).get(arm)
        if pol and pol in scores and isinstance(scores[pol], dict):
            v = scores[pol].get('dti')
            return v if isinstance(v, (int, float)) else None
        return None
    v = scores.get('dti')
    return v if isinstance(v, (int, float)) else None


def _experiment_section(exp, filename):
    hyp = exp.get('hypothesis', 'H12')
    is_v2 = exp.get('protocol', '').endswith('policy-sweep-v2')
    selection = exp.get('policy_selection', {}) if is_v2 else {}
    decision = exp.get('decision', {})
    status = str(exp.get('status', 'unknown')).upper()
    reason = decision.get('reason', 'Evaluation has not finished; no release.')
    protocol_line = (
        'Per-arm policy selected on development folds only (preregistered 8-policy '
        'set); no confirmation-fold tuning.'
        if is_v2 else
        'Fixed top 2%, HGB 200 iterations. Four geographic stripe folds; 4 km '
        'training exclusion. Fold 3 is confirmation.')
    binding = exp.get('final_prediction') or {}
    binding_line = ''
    if binding.get('training_manifest_file'):
        binding_line = (f'<p><b>Final prediction hash-bound</b> '
                        f'({esc(binding["training_manifest_file"])}) — release '
                        f'evidence complete; approved artifact below.</p>')
    content = f'<section class="card"><h2>{esc(hyp)} spatial holdout</h2><p><strong>{esc(status)}</strong> · {esc(reason)}</p>{binding_line}<p>{protocol_line} Scores below are catalogue-generalization proxies, not new-fault leaderboard estimates.</p>'
    arms = ['baseline107', 'baseline107_discovery', hyp, 'random_budget_control']
    content += '<div class="table-scroll"><table><thead><tr><th>Fold / role</th>' + ''.join(f'<th>{esc(a)}</th>' for a in arms) + f'<th>{esc(hyp)} − baseline</th></tr></thead><tbody>'
    for f in exp.get('folds', []):
        vals = [_arm_dti(f, a, selection) for a in arms]
        base = vals[0]; cand = vals[2]
        cells = ''.join(('<td>{:.6f}</td>'.format(v)) if v is not None else '<td>—</td>' for v in vals)
        delta = f'{cand-base:+.6f}' if (base is not None and cand is not None) else '—'
        if is_v2 and selection:
            pol = selection.get(hyp, '?')
        else:
            pol = ''
        content += f'<tr><td>{f["fold"]} / {esc(f["role"])}{f" ({esc(pol)})" if pol and f.get("role")=="development" else ""}</td>{cells}<td>{delta}</td></tr>'
    content += '</tbody></table></div><p><a href="data/' + esc(filename) + '">Full experiment: inputs, code hashes, masks, weighted metric terms →</a></p></section>'
    return content


def results_html(audit, experiments):
    content = '<div class="intro"><span class="eyebrow">MEASURE / COMPARE / REJECT WHEN NECESSARY</span><h1>What actually changed?</h1><p>Scores are not identities. This audit reads the files, the geography and the predictions.</p></div>'
    for name, exp in experiments:
        if exp:
            content += _experiment_section(exp, name)
    content += f'<section class="card"><h2>Published artifact census</h2><p>Artifact snapshot: {esc(audit.get("checked_utc", "unknown"))}.</p><p>Scores supplied by the user; exact file-to-upload associations remain unverified. Files below are audited evidence, <b>not download recommendations</b>.</p><div class="table-scroll"><table><thead><tr><th>Artifact</th><th>User score</th><th>Positive cells</th><th>File SHA</th><th>Evidence</th></tr></thead><tbody>'
    for a in audit.get('artifacts', []):
        score = a.get('user_reported_score')
        score_text = f'{score:.4f}' if score is not None else 'not supplied'
        content += f'<tr><td>{esc(a["id"])}</td><td>{score_text}</td><td>{a.get("positive_pixels", "—"):,}</td><td><code>{esc(a.get("sha256", ""))[:12]}</code></td><td><a href="{esc(a.get("source_url", "#"))}">{esc(a["status"])}</a></td></tr>' if a.get('positive_pixels') is not None else f'<tr><td>{esc(a["id"])}</td><td colspan="4">Unavailable; see audit JSON</td></tr>'
    content += '</tbody></table></div><h3>Why 0.1563 repeats</h3><p>GEMSDOE1, 5GEMSDOE and GEMSDOE2 recall have identical bytes and pixels. Current 8GEMSDOE hedge changes 54,533 cells, all on the existing catalogue: zero additional noncatalogue predictions. A new file name does not fix either problem.</p><p><a href="data/submission_audit.json">All immutable sources and pairwise changed-pixel counts →</a></p></section>'
    content += '<section class="card"><h2>Historical reports are a different protocol</h2><p>Component-holdout ALL/top2 means: baseline 0.092885, discovery 0.103958, selftrain 0.089681. These are retained historical reports, not the spatial results above. FAR10 reverses the ranking and must not be substituted after seeing results. No local number here is comparable to 0.3049 as a private-score forecast.</p><a href="data/cv_baseline.json">Baseline report</a> · <a href="data/cv_discovery.json">Discovery</a> · <a href="data/cv_selftrain.json">Selftrain</a></section>'
    return content


def main():
    docs = ROOT / 'docs'; (docs / 'data').mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    submissions, refused = approved_submissions(ROOT)
    reports = []
    for p in sorted((ROOT / 'reports').glob('*.json')):
        shutil.copy2(p, docs / 'data' / p.name); reports.append(p.name)
    manifest = {'built_utc': stamp, 'reports': reports, 'submissions': submissions,
                'refused_publications': refused}
    (docs / 'data/site.json').write_text(json.dumps(manifest, indent=2) + '\n')
    audit = read_json(ROOT / 'reports/submission_audit.json', {})
    experiments = [(name, read_json(ROOT / 'reports' / name, {}))
                   for name in ('h12_blocked.json', 'h16_blocked.json',
                                'h13_blocked.json')
                   if (ROOT / 'reports' / name).exists()]
    feed = read_json(ROOT / 'reports/official_feed.json', {})
    board = feed.get('leaderboard', {})
    panel = submission_panel(submissions)
    home = '''<div class="intro"><span class="eyebrow">RESEARCH LOG / NORTHWESTERN GREAT BASIN</span><h1>Find a different fault.<br><em>Not a different filename.</em></h1><p>Scientific hypotheses, spatially separated tests and auditable predictions. Built to discover missing geological structure—not repeat the same submission.</p></div>''' + panel
    home += f'''<div class="stats"><div><span>PUBLIC LEADER SNAPSHOT</span><strong>{board.get('score', 0):.4f}</strong><small>{esc(board.get('rank1', 'Unknown'))} · {esc(board.get('last_success_date', 'unknown date'))}<br>{esc(board.get('status', 'unverified'))}</small></div><div><span>GROUP BEST / USER-REPORTED</span><strong>0.1563</strong><small>Exact duplicate identified<br>Upload history not authenticated</small></div><div><span>NEW PHYSICAL HYPOTHESES</span><strong>05</strong><small>H13/H16 on spatial-block holdouts<br>H12 rejected · H15 blocked by data access</small></div></div>'''
    home += '''<div class="grid"><section class="card"><span class="eyebrow">01 / AUDIT</span><h2>Copying is measurable.</h2><p>Ten published artifacts, immutable source commits, file hashes and canonical prediction hashes. Renamed and recompressed duplicates are refused.</p><a href="results.html">See the comparison →</a></section><section class="card"><span class="eyebrow">02 / EXPERIMENT</span><h2>Two faults, one offset.</h2><p>H16 lights rays that continue a discovered system's strike past its mapped end; H13 measures the zero-lag alignment dip of strip pairs across candidate traces. Both are tested on spatially blocked folds before any submission slot is spent. H12 (scarp polarity) was measured and rejected on the same geography.</p><a href="hypotheses.html">Read all five hypotheses →</a></section></div><section class="card"><h2>Evidence has a boundary.</h2><p>Official staff confirm that new geometry of existing systems can count. They do not disclose the hidden faults’ data sources, types or coverage. A proxy score is not a promise of first place; fault pixels are not confirmed geothermal vents.</p><a href="sources.html">Official sources and freshness →</a> · <a href="verification.html">Limitations and irregularities →</a></section>'''
    sources = '<div class="intro"><span class="eyebrow">OFFICIAL / DATED / REVIEWABLE</span><h1>Source ledger</h1><p>Daily checks preserve the last-good evidence when a page is unavailable. A failed refresh is not a new verification.</p></div>'
    for s in feed.get('sources', []):
        sources += f'<section class="card"><span class="eyebrow">{esc(s["status"])} · LAST SUCCESS {esc(s["last_success_date"])}</span><h2>{esc(s["claim"])}</h2><p>{esc(s["scope"])}</p><a href="{esc(s["url"])}">Read official source ↗</a>'
        if s.get('error'):
            sources += f'<p class="warning">Latest check: {esc(s["error"])}</p>'
        sources += '</section>'
    sources += f'<section class="card"><h2>Leaderboard snapshot</h2><p>{esc(board.get("rank1", "Unknown"))}: {board.get("score", 0):.4f}; last success {esc(board.get("last_success_date", "unknown"))}; {esc(board.get("status", "unknown"))}.</p><a href="{esc(board.get("url", "#"))}">Live official leaderboard ↗</a></section>'
    sources += '<section class="card"><h2>Scientific literature and external data</h2><p>USGS geomorphic mapping and buried-fault geophysics motivate H12–H14. GDR temperature and paleo-discharge archives motivate H15, but the binary download check failed here, so H15 is blocked. Sources support the motivation, not claimed detector accuracy.</p><a href="hypotheses.html">Layer-by-layer hypotheses and source links →</a><p><a href="data/official_feed.json">Raw source/check ledger</a></p></section>'
    pages = {
        'index': ('Submission hub', home),
        'executive_summary': ('How to submit', panel + '<article>' + md((ROOT/'SUBMISSION_GUIDE.md').read_text()) + '</article>'),
        'results': ('Results & audit', results_html(audit, experiments)),
        'hypotheses': ('Geological hypotheses', '<article>' + md((ROOT/'HYPOTHESES.md').read_text()) + '</article>'),
        'sources': ('Official sources', sources),
        'verification': ('Review & limitations', '<article>' + md((ROOT/'REVIEW.md').read_text()) + md((ROOT/'LIMITATIONS.md').read_text()) + '</article>'),
    }
    for slug, (title, content) in pages.items():
        (docs / f'{slug}.html').write_text(page(slug, title, content, stamp))
    # Preserve old deep link without duplicating stale instructions.
    (docs / 'how_to_submit.html').write_text(page('executive_summary', 'How to submit', pages['executive_summary'][1], stamp))
    print(f'{len(pages)} pages; {len(submissions)} approved downloads; {len(refused)} refused')
    if refused:
        print(json.dumps(refused, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
