import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from gems10 import novelty, release, cv


def test_field_identity_ignores_outside_nan_and_negative_zero():
    a = np.array([[0., .4], [np.nan, .7]])
    fp = np.array([[1, 1], [0, 1]], bool)
    b = a.copy(); b[0, 0] = -0.; b[1, 0] = 0
    assert novelty.field_hash(a, fp) == novelty.field_hash(b, fp)
    b[1, 1] = .8
    assert novelty.field_hash(a, fp) != novelty.field_hash(b, fp)


def test_catalogue_only_change_is_blocked():
    fp = np.ones((2, 2), bool)
    known = np.array([[1, 0], [0, 0]], bool)
    a = np.zeros((2, 2)); b = a.copy(); b[0, 0] = 1
    assert novelty.field_hash(a, fp) != novelty.field_hash(b, fp)
    identity = {'noncatalogue_sha256': novelty.field_hash(a, fp, excluded=known)}
    other = {'id': 'renamed-TIFF', 'noncatalogue_sha256': novelty.field_hash(b, fp, excluded=known)}
    with pytest.raises(ValueError, match='duplicate'):
        novelty.refuse_duplicate(identity, [other])


@pytest.mark.parametrize('v', [np.nan, np.inf, -1., 1.01])
def test_invalid_values_never_canonicalized(v):
    with pytest.raises(ValueError):
        novelty.field_hash(np.array([[v]]), np.ones((1, 1), bool))


def evidence():
    return {'status': 'completed', 'protocol': 'spatial-4x4-strided-v1-buffer40-top2',
            'policy': 'topk02_binary', 'promotion_allowed': True,
            'final_prediction': {'sha256': 'pred', 'training_manifest_sha256': 'manifest'},
            'folds': [{'fold': k, 'scores': {'H12': {'dti': .2},
                                            'baseline107': {'dti': .1},
                                            'baseline107_discovery': {'dti': .11}}}
                      for k in range(4)]}


def test_release_requires_positive_confirmation_and_binding():
    d = evidence()
    release.check_release(d, 'pred', 'topk02_binary')
    for mutated in [dict(d, status='running'), dict(d, promotion_allowed=False),
                    dict(d, final_prediction={}), dict(d, folds=d['folds'][:3])]:
        with pytest.raises(ValueError):
            release.check_release(mutated, 'pred', 'topk02_binary')
    d['folds'][3]['scores']['H12']['dti'] = .1
    with pytest.raises(ValueError, match='confirmation'):
        release.check_release(d, 'pred', 'topk02_binary')


def test_release_rejects_nan_and_policy_switch():
    d = evidence()
    with pytest.raises(ValueError, match='policy'):
        release.check_release(d, 'pred', 'topk03_binary')
    d['folds'][0]['scores']['H12']['dti'] = float('nan')
    with pytest.raises(ValueError, match='invalid'):
        release.check_release(d, 'pred', 'topk02_binary')


def test_spatial_buffer_and_unique_confirmation():
    from scipy.ndimage import distance_transform_edt
    folds = cv.make_folds(n_blocks=4, n_folds=4, buffer_px=40)
    coverage = np.zeros((400, 400), int)
    for k in range(4):
        train, test = folds.train_test_masks(coverage.shape, k)
        assert not (train & test).any()
        assert distance_transform_edt(~test)[train].min() > 40
        coverage += test
    assert np.all(coverage == 1)


def test_decision_matches_release_logic():
    script = Path(__file__).parents[1] / 'scripts/validate_hypothesis.py'
    spec = importlib.util.spec_from_file_location('hypothesis', script)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    d = evidence()
    assert mod.paired_decision(d['folds'])['eligible']
    bad = copy.deepcopy(d['folds'])
    bad[3]['scores']['H12']['dti'] = .09
    assert not mod.paired_decision(bad)['eligible']
    assert not mod.paired_decision(bad[:3])['eligible']


def test_training_binding_requires_actual_manifest(tmp_path):
    with pytest.raises(ValueError, match='manifest'):
        release.verify_training_binding(evidence(), tmp_path)


def test_publisher_refuses_missing_evidence_before_creating_download(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location('publisher', root/'scripts/build_submission.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    import sys
    from gems10 import spec as grid
    raw = tmp_path/'raw.npy'
    np.save(raw, np.zeros((grid.HEIGHT, grid.WIDTH), dtype=np.uint8))
    out = tmp_path/'downloads'
    monkeypatch.setattr(sys, 'argv', ['build_submission.py', '--prob', str(raw), '--out-dir', str(out)])
    with pytest.raises(ValueError, match='holdout evidence required'):
        mod.main()
    assert not out.exists()


def test_publisher_rejects_nonfinite_probabilities(tmp_path, monkeypatch):
    import sys
    import rasterio
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location('publisher_invalid', root/'scripts/build_submission.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    with rasterio.open(root/'data/sample_submission.tif') as ds:
        p = ds.read(1)
    p[np.isfinite(p)] = np.nan
    raw = tmp_path/'bad.npy'; np.save(raw, p)
    monkeypatch.setattr(sys, 'argv', ['build_submission.py', '--prob', str(raw), '--experiment'])
    with pytest.raises(ValueError, match='no silent repair'):
        mod.main()


def evidence_v3(candidate='H20'):
    """Session-3 protocol: policy dicts per arm, explicit arms, incumbent comparisons."""
    def scores(cand_dti, base_dti, inc_dti):
        return {'baseline107': {'topk06_binary': {'dti': base_dti}},
                'baseline107_discovery': {'topk06_binary': {'dti': base_dti + .001}},
                'H16': {'topk06_binary': {'dti': inc_dti}},
                candidate: {'thin08_binary': {'dti': cand_dti}}}
    folds = [{'fold': k, 'role': 'development' if k < 3 else 'confirmation',
              'scores': scores(.20, .15, .17)} for k in range(4)]
    return {'status': 'completed', 'hypothesis': candidate,
            'protocol': 'spatial-4x4-strided-v1-buffer40-policy-sweep-v3',
            'arms': ['baseline107', 'H16', candidate],
            'policy_selection': {'baseline107': 'topk06_binary',
                                 'baseline107_discovery': 'topk06_binary',
                                 'H16': 'topk06_binary', candidate: 'thin08_binary'},
            'decision': {'eligible': True, 'candidate_policy': 'thin08_binary',
                         'comparisons': {
                             'baseline107': {'passes': True},
                             'baseline107_discovery': {'passes': True},
                             'H16': {'passes': True},
                             'incumbent_report:H16': {'passes': True,
                                                      'delta_by_fold': [.03, .03, .03, .02]}}},
            'promotion_allowed': True,
            'final_prediction': {'sha256': 'pred', 'training_manifest_sha256': 'manifest'},
            'folds': folds}


def test_release_v3_requires_every_incumbent_comparison():
    d = evidence_v3()
    release.check_release(d, 'pred', 'thin08_binary')
    with pytest.raises(ValueError, match='policy'):
        release.check_release(d, 'pred', 'topk06_binary')
    # same-run H16 incumbent arm beats the candidate on the confirmation fold
    bad = copy.deepcopy(d)
    bad['folds'][3]['scores']['H16']['topk06_binary']['dti'] = .25
    with pytest.raises(ValueError, match='H16'):
        release.check_release(bad, 'pred', 'thin08_binary')
    # external incumbent report comparison recorded as failing
    bad = copy.deepcopy(d)
    bad['decision']['comparisons']['incumbent_report:H16']['passes'] = False
    with pytest.raises(ValueError, match='incumbent'):
        release.check_release(bad, 'pred', 'thin08_binary')
    # external incumbent deltas negative on confirmation even if flagged passing
    bad = copy.deepcopy(d)
    bad['decision']['comparisons']['incumbent_report:H16']['delta_by_fold'] = [.03, .03, .03, -.01]
    with pytest.raises(ValueError, match='confirmation'):
        release.check_release(bad, 'pred', 'thin08_binary')
    bad = copy.deepcopy(d)
    bad['decision']['eligible'] = False
    with pytest.raises(ValueError):
        release.check_release(bad, 'pred', 'thin08_binary')


def test_release_v3_refuses_the_real_h19_report():
    """The measured session-3 H19 report must never pass the gate."""
    import json
    path = Path(__file__).resolve().parents[1] / 'reports' / 'h19_blocked.json'
    if not path.exists():
        pytest.skip('report not present')
    d = json.loads(path.read_text())
    d = dict(d, promotion_allowed=True,
             final_prediction={'sha256': 'pred', 'training_manifest_sha256': 'm'})
    with pytest.raises(ValueError):
        release.check_release(d, 'pred', d['policy_selection']['H16'])
