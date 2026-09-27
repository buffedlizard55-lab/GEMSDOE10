#!/usr/bin/env python3
"""Refresh official availability/quotation checks without inventing new facts.

Failed checks preserve the dated last-good evidence and visibly mark it stale.
Does not authenticate to DrivenData, submit files, or automatically reinterpret rules.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
from html import unescape
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def plain(html):
    return ' '.join(unescape(re.sub(r'<[^>]+>', ' ', html)).split())


def read_page(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'GEMSDOE10-evidence-monitor/1.0'})
    with urllib.request.urlopen(req, timeout=20) as response:
        if 'login' in response.url:
            raise ValueError('redirected to login; not verified')
        raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ValueError('page exceeds safety size limit')
        if response.status != 200:
            raise ValueError(f'HTTP {response.status}')
    return raw.decode('utf-8'), hashlib.sha256(raw).hexdigest()


def parse_leaderboard(html):
    for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', html, re.S | re.I):
        text = plain(row)
        if re.search(r'(?<!\d)#1(?!\d)', text):
            values = re.findall(r'(?<![\d.])0\.\d{4}(?!\d)', text)
            participants = re.findall(r'/users/([^/"?]+)/', row)
            if len(set(values)) == 1 and len(set(participants)) == 1:
                return {'rank1': unescape(participants[0]), 'score': float(values[0])}
    raise ValueError('rank-one row ambiguous or markup changed; previous snapshot retained')


def refresh(feed, getter=read_page):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    feed['last_attempt_utc'] = now
    for source in feed['sources'] + [feed['leaderboard']]:
        try:
            html, sha = getter(source['url'])
            if source is feed['leaderboard']:
                source.update(parse_leaderboard(html))
            elif source['required_excerpt'].casefold() not in plain(html).casefold():
                raise ValueError('expected official excerpt missing; needs review')
            source.update(status='verified_http', last_success_date=now[:10],
                          last_success_utc=now, response_sha256=sha)
            source.pop('error', None)
        except Exception as exc:
            source.update(status='stale_check_failed', error=f'{type(exc).__name__}: {exc}')
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', type=Path, default=ROOT / 'reports/official_feed.json')
    args = ap.parse_args()
    feed = refresh(json.loads(args.file.read_text()))
    staged = args.file.with_suffix('.tmp')
    staged.write_text(json.dumps(feed, indent=2, allow_nan=False) + '\n')
    staged.replace(args.file)
    failed = sum(s['status'] == 'stale_check_failed'
                 for s in feed['sources'] + [feed['leaderboard']])
    print(f'{failed} checks failed; previous dated evidence retained and marked stale')
    # Availability failures are data, not reason to suppress the warning website.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
