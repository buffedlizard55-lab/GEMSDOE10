"""Placement policies + evaluation harness."""

import numpy as np
import pytest

from gems10 import modeling


def _prob_footprint():
    rng = np.random.default_rng(1)
    prob = rng.random((50, 60))
    fp = np.ones((50, 60), dtype=bool)
    fp[0, :] = False  # a nodata frame row
    return prob, fp


def test_parse_policy_budgets():
    assert modeling.parse_policy("topk03_binary") == (0.03, "binary")
    assert modeling.parse_policy("topk02_envelope") == (0.02, "envelope")
    with pytest.raises(AssertionError):
        modeling.parse_policy("topk03_skeleton")


def test_binary_policy_emits_exact_budget():
    prob, fp = _prob_footprint()
    out = modeling.apply_policy(prob, fp, "topk02_binary")
    assert out.shape == prob.shape
    assert bool((~np.isfinite(out[~fp])).all())  # NaN outside footprint
    inside = out[fp]
    assert set(np.unique(inside).tolist()) <= {0.0, 1.0}
    expect = round(0.02 * fp.sum())
    assert int((inside == 1.0).sum()) == pytest.approx(expect, abs=2)


def test_envelope_adds_a_soft_ring():
    prob, fp = _prob_footprint()
    core = modeling.apply_policy(prob, fp, "topk02_binary")
    env = modeling.apply_policy(prob, fp, "topk02_envelope")
    assert set(np.unique(env[fp]).tolist()) <= {0.0, 0.5, 1.0}
    assert int((env == 1.0).sum()) == int((core == 1.0).sum())
    assert int((env == 0.5).sum()) > 0  # ring exists


def test_halo2_matches_metric_kernel_weights():
    prob, fp = _prob_footprint()
    out = modeling.apply_policy(prob, fp, "topk02_halo2")
    vals = set(np.unique(out[fp]).tolist())
    assert vals <= {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}
    assert 2.0 / 3.0 in vals  # ring1 uses k(1)


def test_evaluate_reports_masked_and_unmasked():
    prob, fp = _prob_footprint()
    gt = np.zeros_like(fp)
    gt[10, 10] = True
    gt[40, 50] = True
    ign = np.zeros_like(fp)
    ign[25, 25] = True
    res = modeling.evaluate_policies(prob, fp, gt, ign, ["topk02_binary"])
    row = res["topk02_binary"]
    assert row["dti_masked"] >= row["dti_unmasked"]  # masking can only help
    assert row["tp_w"] + row["fn_w"] == pytest.approx(2.0)


def test_thin_policy_is_a_subset_skeleton_of_topk():
    """H19 (session 3): thinNN_binary = Zhang-Suen skeleton of the top-NN% mask."""
    assert modeling.parse_policy("thin06_binary") == (0.06, "thin")
    with pytest.raises(AssertionError):
        modeling.parse_policy("thin06_soft")
    prob, fp = _prob_footprint()
    # A thick diagonal ridge: top-k selects a band several pixels wide.
    rr, cc = np.mgrid[:prob.shape[0], :prob.shape[1]]
    prob = np.where(np.abs(rr - cc) <= 2, 1.0, prob * 0.5)
    core = modeling.apply_policy(prob, fp, "topk04_binary")
    thin = modeling.apply_policy(prob, fp, "thin04_binary")
    assert thin.shape == prob.shape
    assert bool((~np.isfinite(thin[~fp])).all())
    assert set(np.unique(thin[fp]).tolist()) <= {0.0, 1.0}
    assert bool(((thin == 1.0) <= (core == 1.0)).all())   # subset of the top-k core
    assert 0 < int((thin == 1.0).sum()) < int((core == 1.0).sum())
