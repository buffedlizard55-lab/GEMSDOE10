import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_leaderboard_parser_is_rank_scoped():
    mod = load_script('refresh_sources')
    html = '<table><tr><td>#10</td><td><a href="/users/Wrong/">Wrong</a></td><td>0.9999</td></tr><tr><td>#1</td><td><a href="/users/DARD/">DARD</a></td><td>0.3049</td></tr></table>'
    assert mod.parse_leaderboard(html) == {'rank1': 'DARD', 'score': .3049}
    with pytest.raises(ValueError):
        mod.parse_leaderboard('<h1>Sign in</h1>0.9999')


def test_failed_refresh_preserves_dated_evidence():
    mod = load_script('refresh_sources')
    feed = {'sources': [{'url': 'https://example.org', 'required_excerpt': 'known',
                         'claim': 'old claim', 'last_success_date': '2026-09-27'}],
            'leaderboard': {'url': 'https://example.org/leaderboard', 'score': .3049,
                            'rank1': 'DARD', 'last_success_date': '2026-09-27'}}
    def fail(url):
        raise OSError('TLS blocked')
    out = mod.refresh(feed, getter=fail)
    assert out['sources'][0]['claim'] == 'old claim'
    assert out['leaderboard']['score'] == .3049
    assert out['leaderboard']['last_success_date'] == '2026-09-27'
    assert out['leaderboard']['status'] == 'stale_check_failed'


def test_changed_excerpt_cannot_verify_fact():
    mod = load_script('refresh_sources')
    feed = {'sources': [{'url': 'https://example.org', 'required_excerpt': 'masked faults'}],
            'leaderboard': {'url': 'https://example.org'}}
    out = mod.refresh(feed, getter=lambda url: ('<h1>different rules</h1>', 'sha'))
    assert out['sources'][0]['status'] == 'stale_check_failed'
    assert 'last_success_date' not in out['sources'][0]


def test_experimental_sidecar_cannot_be_published(tmp_path):
    mod = load_script('build_site')
    folder = tmp_path/'docs/downloads'; folder.mkdir(parents=True)
    (folder/'experiment.json').write_text(json.dumps({'status':'experimental_do_not_submit'}))
    accepted, refused = mod.approved_submissions(tmp_path)
    assert accepted == [] and len(refused) == 1


def test_forged_approved_sidecar_without_evidence_is_refused(tmp_path):
    mod = load_script('build_site')
    folder = tmp_path/'docs/downloads'; folder.mkdir(parents=True)
    (folder/'fake.json').write_text(json.dumps({'status':'approved', 'file':'fake.tif',
                                               'zip':'fake.zip', 'validation_file':'fake.json'}))
    accepted, refused = mod.approved_submissions(tmp_path)
    assert accepted == [] and 'audit' in refused[0]['reason']


def test_submission_panel_has_no_dead_download_links():
    mod = load_script('build_site')
    html = mod.submission_panel([])
    assert 'No approved submission' in html
    assert 'href="downloads/' not in html


def test_leaderboard_rank_split_across_spans_and_profile_without_slash():
    mod = load_script('refresh_sources')
    html = '<table><tr><td><span>#</span> <strong>1</strong></td><td><a href="/users/DARD">DARD</a></td><td>0.3049</td></tr></table>'
    assert mod.parse_leaderboard(html) == {'rank1': 'DARD', 'score': .3049}


def test_leaderboard_div_layout_and_script_values():
    mod = load_script('refresh_sources')
    html = '<html><head><script>var fake = "#1 0.9999";</script></head><body><h1>Leaderboard</h1><div><span>#</span><span>1</span><a href="/users/DARD/">DARD</a><b>0.3049</b></div><div>#2<a href="/users/Second/">Second</a>0.2993</div></body></html>'
    assert mod.parse_leaderboard(html) == {'rank1': 'DARD', 'score': .3049}
    with pytest.raises(ValueError, match='unavailable'):
        mod.parse_leaderboard('<html><body>Please enable JavaScript to verify you are human</body></html>')
