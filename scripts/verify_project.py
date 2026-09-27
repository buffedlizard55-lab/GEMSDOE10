#!/usr/bin/env python3
"""Run reproducible final checks, persist command results, and rebuild the site."""
from __future__ import annotations
import argparse
import datetime
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        for key, val in attrs:
            if key in ('href', 'src') and val:
                self.links.append(val)


def check_links(docs):
    count = 0; missing = []
    for page in docs.glob('*.html'):
        parser = Links(); parser.feed(page.read_text())
        for href in parser.links:
            url = urlsplit(href)
            if url.scheme or url.netloc or not url.path:
                continue
            dest = (page.parent / unquote(url.path)).resolve()
            count += 1
            if not dest.is_relative_to(docs.resolve()) or not dest.exists():
                missing.append(f'{page.name}: {href}')
    return {'checked_internal_links': count, 'missing': missing, 'ok': not missing}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--with-features', action='store_true')
    args = ap.parse_args()
    report_path = ROOT / 'reports/verification_run.json'
    report = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'running', 'checks': [],
              'scope': 'local tests, data integrity and static site links; not all scientific claims or private backend behavior'}
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    (ROOT / 'scratch').mkdir(exist_ok=True)
    commands = [[sys.executable, 'scripts/fetch_data.py'] + ([] if args.with_features else ['--no-features']),
                [sys.executable, '-m', 'pytest', '-q', '--junitxml=scratch/test-results.xml'],
                [sys.executable, 'scripts/build_site.py']]
    for cmd in commands:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
        report['checks'].append({'command': ' '.join(cmd), 'exit_code': result.returncode,
                                 'stdout': result.stdout[-15000:], 'stderr': result.stderr[-4000:]})
        print(result.stdout)
        if result.returncode:
            print(result.stderr, file=sys.stderr)
    xml = ROOT / 'scratch/test-results.xml'
    if xml.exists():
        suites = ET.parse(xml).getroot().findall('testsuite')
        report['tests'] = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
                           for key in ('tests', 'failures', 'errors', 'skipped')}
    report['site_links'] = check_links(ROOT / 'docs')
    manifest = json.loads((ROOT / 'docs/data/site.json').read_text())
    report['approved_downloads'] = len(manifest['submissions'])
    report['refused_publications'] = manifest['refused_publications']
    ok = all(c['exit_code'] == 0 for c in report['checks']) and report['site_links']['ok']
    report['status'] = 'passed' if ok else 'failed'
    report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    # Include the finished report in the static data copy, not the running version.
    subprocess.run([sys.executable, 'scripts/build_site.py'], cwd=ROOT, check=True)
    print(json.dumps({k: report[k] for k in ('status', 'tests', 'site_links')}, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
