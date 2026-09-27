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
from html.parser import HTMLParser
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


class LeaderboardText(HTMLParser):
    """Visible text + profile markers, independent of table vs div layout."""
    def __init__(self):
        super().__init__(); self.parts = []; self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'head'):
            self.skip += 1
        if not self.skip and tag == 'a':
            href = dict(attrs).get('href', '')
            match = re.search(r'/users/([^/?#]+)/?$', href)
            if match:
                self.parts.append(' PROFILE:' + match[1] + ' ')
    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'head') and self.skip:
            self.skip -= 1
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def parse_leaderboard(html):
    for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', html, re.S | re.I):
        text = plain(row)
        cells = re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>', row, re.S | re.I)
        rank_cell = plain(cells[0]) if cells else ''
        # The site can split '#' and the digit across nested spans/whitespace.
        if re.fullmatch(r'#?\s*1', rank_cell) or re.search(r'(?<!\d)#\s*1(?!\d)', text):
            values = re.findall(r'(?<![\d.])0\.\d{4}(?!\d)', text)
            participants = re.findall(r'''/users/([^/"'?<>\s]+)(?:[/"'])''', row)
            if len(set(values)) == 1 and len(set(participants)) == 1:
                return {'rank1': unescape(participants[0]), 'score': float(values[0])}
    parser = LeaderboardText(); parser.feed(html)
    visible = ' '.join(' '.join(parser.parts).split())
    # DrivenData may use CSS-grid/div rows rather than HTML table rows.
    winner = re.search(r'#\s*1(?!\d)(.*?)(?=#\s*2(?!\d)|$)', visible)
    if winner:
        segment = winner[1]
        values = set(re.findall(r'(?<![\d.])0\.\d{4}(?!\d)', segment))
        participants = set(re.findall(r'PROFILE:([^\s]+)', segment))
        if len(values) == len(participants) == 1:
            return {'rank1': participants.pop(), 'score': float(values.pop())}
    raise ValueError('rank-one row ambiguous or unavailable; previous snapshot retained; '
                     + 'visible response excerpt: ' + visible[:700])



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
