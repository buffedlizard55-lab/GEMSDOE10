"""Session 5: along-strike dotting (H28), exact binary metric, line support (H29),
density-matched protocol v5 gate."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gems10 import metric, modeling, placement, release  # noqa: E402


def test_parse_dot_policies():
    assert modeling.parse_policy("ridge15_d2") == (0.15, "ridgedot2")
    assert modeling.parse_policy("ridge40_d3") == (0.40, "ridgedot3")
    assert modeling.parse_policy("ridge15_binary") == (0.15, "ridge")
    with pytest.raises(AssertionError):
        modeling.parse_policy("ridge15_d4")


def test_dot_offsets_counts():
    assert len(placement.dot_offsets(2)[0]) == 8
    assert len(placement.dot_offsets(3)[0]) == 24
    with pytest.raises(ValueError):
        placement.dot_offsets(1)


@pytest.mark.parametrize("radius", [2, 3])
def test_dot_nms_spacing_subset_and_determinism(radius):
    m = np.zeros((7, 40), bool)
    m[3, 2:35] = True
    p = np.zeros(m.shape)
    p[3, 2:35] = np.linspace(0.1, 0.9, 33)
    out = placement.dot_nms(m, p, radius)
    xs = np.nonzero(out[3])[0]
    assert set(np.diff(xs)) == {radius}
    assert not (out & ~m).any()
    assert np.array_equal(out, placement.dot_nms(m, p, radius))
    # minimum pairwise distance >= radius on a random thin field
    rng = np.random.default_rng(3)
    mm = rng.random((50, 50)) < 0.15
    pp = rng.random((50, 50))
    o = placement.dot_nms(mm, pp, radius)
    ys, xs = np.nonzero(o)
    d = np.hypot(ys[:, None] - ys[None], xs[:, None] - xs[None])
    np.fill_diagonal(d, np.inf)
    assert d.min() >= radius


def test_dot_policy_is_subset_of_ridge():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:80, 0:80]
    prob = np.exp(-((yy - 0.6 * xx - 10) ** 2) / 20.0) + 0.05 * rng.random((80, 80))
    fp = np.ones_like(prob, bool)
    r = modeling.apply_policy(prob, fp, "ridge15_binary") > 0
    d2 = modeling.apply_policy(prob, fp, "ridge15_d2") > 0
    d3 = modeling.apply_policy(prob, fp, "ridge15_d3") > 0
    assert not (d2 & ~r).any() and not (d3 & ~r).any()
    assert d3.sum() < d2.sum() < r.sum()


def test_binary_components_exact_all_semantics():
    rng = np.random.default_rng(11)
    for _ in range(10):
        pred = (rng.random((40, 45)) < 0.1).astype(float)
        gt = np.zeros((40, 45), bool)
        gt[rng.integers(0, 40), 5:40] = True
        gt |= rng.random((40, 45)) < 0.01
        ign = (rng.random((40, 45)) < 0.05) & ~gt
        pairs = [(metric.components(pred, gt), metric.binary_components(pred, gt)),
                 (metric.components(pred, gt, fp_ignore_mask=ign),
                  metric.binary_components(pred, gt, ignore=ign, zero_ignored=False)),
                 (metric.components(pred * ~ign, gt, fp_ignore_mask=ign),
                  metric.binary_components(pred, gt, ignore=ign, zero_ignored=True))]
        for a, b in pairs:
            assert a.as_dict() == b.as_dict()


def test_binary_components_rejects_soft_and_overlap():
    gt = np.zeros((5, 5), bool)
    gt[2, 2] = True
    with pytest.raises(ValueError):
        metric.binary_components(np.full((5, 5), 0.5), gt)
    with pytest.raises(ValueError):
        metric.binary_components(np.zeros((5, 5)), gt, ignore=gt)


def test_line_kernel_zero_sum_and_response():
    import build_linesupport as B
    for L in B.LENGTHS:
        for i in range(B.N_THETA):
            K = B.line_kernel(L, np.pi * i / B.N_THETA)
            assert abs(K.sum()) < 1e-12 and K.shape[0] % 2 == 1
    u = np.zeros((120, 120))
    u[60, 10:110] = 1.0  # horizontal line
    res = B.oriented_support(u, lengths=(21,), n_theta=12)
    rmax, aniso = res[21]
    assert rmax[60, 60] > 0.9 and rmax[20, 60] < 0.05 and aniso[60, 60] > 0.5


def test_rank_normalise_bounds():
    import build_linesupport as B
    x = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    valid = np.array([[True, True, True], [True, True, False]])
    u = B.rank_normalise(x, valid)
    assert u.min() == -0.5 and u.max() == 0.5 and u[0, 2] == 0 and u[1, 2] == 0


def test_split_truth_partition_and_nesting():
    import evaluate_density_matched as E
    gt = np.zeros((30, 30), bool)
    for r in range(2, 28, 3):
        gt[r, 3:25] = True
    t5, k5 = E.split_truth(gt, fold=1, draw=0, f=0.5)
    t25, k25 = E.split_truth(gt, fold=1, draw=0, f=0.25)
    assert not (t5 & k5).any() and np.array_equal(t5 | k5, gt)
    assert not (t25 & ~t5).any()  # same seed: f=0.25 truth nested in f=0.5 truth


def _v5_report(cand_dti, inc_dti, cand_full, inc_full, margin=0.01):
    rows = []
    for k in range(4):
        rows.append({"fold": k, "role": "confirmation" if k == 3 else "development",
                     "scores": {"H28": {"ridge15_d2": {"dti": cand_dti[k], "dti_full": cand_full[k]}},
                                "H25": {"ridge15_binary": {"dti": inc_dti[k], "dti_full": inc_full[k]}}}})
    comps = {"incumbent_fixed:H25@ridge15_binary": {"passes": True},
             "full_density_noninferiority": {"passes": True}}
    return {"decision": {"comparisons": comps}, "noninferiority_margin_full_density": margin}, rows


def test_check_v5_recomputes_and_fails_closed():
    rep, rows = _v5_report([.2, .2, .2, .2], [.1, .1, .1, .1], [.25] * 4, [.25] * 4)
    release._check_v5(rep, rows, "H28", "ridge15_d2")  # passes
    rep, rows = _v5_report([.2, .2, .2, .05], [.1, .1, .1, .1], [.25] * 4, [.25] * 4)
    with pytest.raises(ValueError):  # confirmation loss despite recorded "passes"
        release._check_v5(rep, rows, "H28", "ridge15_d2")
    rep, rows = _v5_report([.2] * 4, [.1] * 4, [.20] * 4, [.25] * 4)
    with pytest.raises(ValueError):  # full-density loss 0.05 > margin
        release._check_v5(rep, rows, "H28", "ridge15_d2")
    rep, rows = _v5_report([.2] * 4, [.1] * 4, [.25] * 4, [.25] * 4, margin=0.05)
    with pytest.raises(ValueError):  # margin looser than the hard cap
        release._check_v5(rep, rows, "H28", "ridge15_d2")


def test_anchor_protocol_is_never_releasable():
    import validate_candidate as V
    assert V.POLICY_SETS["anchor"] == ["ridge15_binary"]
    assert V.PROTOCOL["anchor"] not in release.PROTOCOLS
    assert "H29" in V.STATIC_EXTRA and "H29" in V.NEEDS_H16
    import evaluate_density_matched as E
    assert E.PROTOCOL_V5 == release.PROTOCOLS[4]
    assert E.INCUMBENT_POLICY in E.V5_POLICIES and len(E.V5_POLICIES) == 16
    assert release.V5_INCUMBENT == (E.INCUMBENT_ARM, E.INCUMBENT_POLICY)


def test_committed_h28_report_passes_gate_and_wrong_inputs_fail():
    import json
    rep = json.loads((ROOT / "reports/h28_blocked.json").read_text())
    sha = rep["final_prediction"]["sha256"]
    release.check_release(rep, sha, "ridge20_d3")          # the released policy
    release.verify_training_binding(rep, ROOT)
    with pytest.raises(ValueError):
        release.check_release(rep, sha, "ridge15_d2")      # not the selected policy
    with pytest.raises(ValueError):
        release.check_release(rep, "0" * 64, "ridge20_d3")  # unbound probability grid
    r29 = json.loads((ROOT / "reports/h29_blocked.json").read_text())
    assert r29["decision"]["eligible"] is False
    with pytest.raises(ValueError):
        release.check_release(r29, "0" * 64, r29["decision"]["candidate_policy"])


def test_evaluator_reason_text():
    import evaluate_density_matched as E
    ok = E._reason(True, {"a": {"passes": True}})
    bad = E._reason(False, {"H25": {"passes": False, "development_mean_delta": 0.0013,
                                    "confirmation_delta": -0.006},
                            "x": {"passes": True}})
    assert ok.startswith("all preregistered v5") and "H25 (dev +0.0013, conf -0.0060)" in bad
    assert "x (" not in bad
