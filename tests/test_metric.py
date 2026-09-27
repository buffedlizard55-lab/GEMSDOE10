"""The metric must match the published formula exactly.

Reference: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
section "Scoring example":
    TP_w = 3.00, FP_w = 1.89, FN_w = 2.00
    TI_w(a=0.2, b=0.8) = 3.00 / (3.00 + 0.2*1.89 + 0.8*2.00) = 0.60
"""

import numpy as np
import pytest

from gems10 import metric


def _grid():
    return np.zeros((9, 9), dtype=np.float32)


def test_official_worked_example_arithmetic():
    tp, fp, fn = 3.00, 1.89, 2.00
    value = tp / (tp + metric.ALPHA * fp + metric.BETA * fn)
    assert value == pytest.approx(0.6026516673, abs=1e-9)
    assert round(value, 2) == 0.60  # exactly what the page prints


def test_kernel_discrete_values_match_the_100m_grid():
    _, _, k = metric.kernel_offsets()
    got = sorted(set(np.round(k, 6)), reverse=True)
    for expected in (1.0, 2 / 3, 0.528595, 1 / 3, 0.254644, 0.057191, 0.0):
        assert any(abs(g - expected) < 1e-6 for g in got), (expected, got)


def test_radius_is_exactly_three_pixels():
    assert metric.RADIUS_PX == 3.0
    assert metric.RADIUS_M == 300.0
    assert metric.PIXEL_SIZE_M == 100.0


def test_perfect_prediction_scores_one():
    g = _grid()
    g[4, 4] = 1
    c = metric.components(g.copy(), g)
    assert c.dti == pytest.approx(1.0)
    assert (c.tp_w, c.fp_w, c.fn_w) == (1.0, 0.0, 0.0)


def test_empty_prediction_scores_zero():
    g = _grid()
    g[4, 4] = 1
    c = metric.components(_grid(), g)
    assert c.dti == 0.0
    assert c.fn_w == pytest.approx(1.0)


def test_hand_computable_one_pixel_offset():
    """One GT pixel, one prediction 1 px right of it (d = 1 px, k = 2/3).

    TP_w = 2/3, FN_w = 1/3, FP_w = p*(1-k) = 1/3
    DTI  = (2/3) / ((2/3) + 0.2*(1/3) + 0.8*(1/3)) = 2/3
    """
    g = _grid()
    g[4, 4] = 1
    p = _grid()
    p[4, 5] = 1
    c = metric.components(p, g)
    assert c.tp_w == pytest.approx(2 / 3)
    assert c.fn_w == pytest.approx(1 / 3)
    assert c.fp_w == pytest.approx(1 / 3)
    assert c.dti == pytest.approx(2 / 3)


def test_two_pixel_offset_uses_diagonal_kernel_value():
    """Distance sqrt(5) px gives k = 1 - sqrt(5)/3 = 0.254644."""
    g = _grid()
    g[4, 4] = 1
    p = _grid()
    p[6, 5] = 1  # dy=2, dx=1 -> sqrt(5)
    c = metric.components(p, g)
    assert c.tp_w == pytest.approx(1 - np.sqrt(5) / 3)
    assert c.fn_w == pytest.approx(np.sqrt(5) / 3)


def test_beyond_radius_contributes_nothing_to_tp():
    g = _grid()
    g[4, 4] = 1
    p = _grid()
    p[4, 8] = 1  # 4 px away > R
    c = metric.components(p, g)
    assert c.tp_w == 0.0
    assert c.fn_w == pytest.approx(1.0)
    assert c.fp_w == pytest.approx(1.0)  # k(d)=0 there, so the full penalty applies


def test_no_ground_truth_is_not_a_nan_score():
    c = metric.components(np.zeros((5, 5)), np.zeros((5, 5)))
    assert c.dti == 0.0 and not np.isnan(c.dti)


def test_empty_prediction_over_empty_gt_is_not_nan():
    c = metric.components(np.zeros((5, 5)), np.zeros((5, 5)))
    assert np.isfinite(c.dti)


def test_probability_weights_are_linear_in_tp():
    g = _grid()
    g[4, 4] = 1
    p = _grid()
    p[4, 4] = 0.5
    c = metric.components(p, g)
    assert c.tp_w == pytest.approx(0.5)
    assert c.fn_w == pytest.approx(0.5)
    assert c.fp_w == pytest.approx(0.0)  # k=1 at d=0 removes the FP penalty


def test_all_ones_matches_brute_force_definition():
    """Brute-force check of the published sums on a small grid."""
    rng = np.random.default_rng(0)
    g = (rng.random((11, 11)) > 0.8)
    p = rng.random((11, 11)).astype(np.float64)
    c = metric.components(p, g)
    R = 3.0
    gs = np.argwhere(g)
    tp = fp = fn = 0.0
    for gy, gx in gs:
        best = 0.0
        for py in range(11):
            for px in range(11):
                d = np.hypot(py - gy, px - gx)
                if d <= R:
                    best = max(best, p[py, px] * max(1 - d / R, 0.0))
        tp += best
        fn += 1 - best
    for py in range(11):
        for px in range(11):
            if p[py, px] <= 0:
                continue
            dmin = min((np.hypot(py - gy, px - gx) for gy, gx in gs), default=np.inf)
            k = max(1 - dmin / R, 0.0) if np.isfinite(dmin) else 0.0
            fp += p[py, px] * (1 - k)
    assert c.tp_w == pytest.approx(tp, rel=1e-9)
    assert c.fn_w == pytest.approx(fn, rel=1e-9)
    assert c.fp_w == pytest.approx(fp, rel=1e-9)


def test_analytic_everywhere_matches_closed_form():
    c = 0.011803
    assert metric.analytic_everywhere_score(c) == pytest.approx(c / (c + 0.2 * (1 - c)))


def test_tp_plus_fn_equals_the_ground_truth_count():
    """Identity used by the placement algebra: FN_w = n_gt - TP_w.

    Follows from the published definitions
        TP_w = sum_{g in G} max_{x : d<=R} p(x) k(d)
        FN_w = sum_{g in G} [1 - max_{x : d<=R} p(x) k(d)]
    so their sum is exactly |G| for any prediction. The reduced form of the
    metric, DTI = TP_w / (beta*n_gt + alpha*FP_w + (1-beta)*TP_w), depends on
    it, so it is pinned rather than assumed.
    """
    rng = np.random.default_rng(7)
    g = (rng.random((23, 19)) > 0.85)
    p = rng.random((23, 19)).astype(np.float64) * (rng.random((23, 19)) > 0.4)
    c = metric.components(p, g)
    assert c.tp_w + c.fn_w == pytest.approx(float(g.sum()), rel=1e-9)


def test_reduced_form_matches_the_published_formula():
    """DTI = TP/(TP + a FP + b FN) == TP/(b*n_gt + a FP + (1-b) TP)."""
    rng = np.random.default_rng(11)
    g = (rng.random((17, 21)) > 0.8)
    p = rng.random((17, 21)).astype(np.float64) * (rng.random((17, 21)) > 0.5)
    c = metric.components(p, g)
    lhs = c.tp_w / (c.tp_w + metric.ALPHA * c.fp_w + metric.BETA * c.fn_w)
    rhs = c.tp_w / (metric.BETA * c.n_gt + metric.ALPHA * c.fp_w
                    + (1.0 - metric.BETA) * c.tp_w)
    assert lhs == pytest.approx(rhs, rel=1e-12)
    assert c.dti == pytest.approx(lhs, rel=1e-12)


def test_scaling_all_predictions_up_never_lowers_the_score():
    """p -> lambda*p is monotone increasing in DTI while lambda*p <= 1.

    This is the reason a submission should carry 1.0 on the pixels it selects
    rather than the model's fractional probability: the beta*n_gt term in the
    denominator does not scale with p.
    """
    rng = np.random.default_rng(3)
    g = (rng.random((15, 15)) > 0.82)
    p = rng.random((15, 15)).astype(np.float64) * 0.5
    prev = -1.0
    for lam in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        dti = metric.distance_weighted_tversky(p * lam, g)
        assert dti > prev
        prev = dti


def test_scoring_on_a_bbox_crop_equals_scoring_on_the_full_grid():
    """The optimisation scripts/experiment.py relies on.

    Scoring is done on the bounding box of the held-out block dilated by the
    kernel radius rather than on the full 3730x3292 grid. That is only valid
    because the ground truth and the prediction are both zero outside the scored
    block: then no term of TP_w / FP_w / FN_w can involve a pixel outside the
    crop. This pins that argument instead of leaving it as a comment.
    """
    rng = np.random.default_rng(21)
    g = np.zeros((60, 70), dtype=bool)
    g[20:40, 25:45] = rng.random((20, 20)) > 0.75      # GT inside a sub-window
    p = (rng.random((60, 70)) > 0.6).astype(np.float64) * rng.random((60, 70))
    p[~g] *= 0.3
    # zero the prediction outside a block that contains all the GT
    block = np.zeros((60, 70), dtype=bool)
    block[15:45, 20:50] = True
    p = np.where(block, p, 0.0)

    full = metric.components(p, g)
    ys, xs = np.nonzero(block)
    margin = int(np.ceil(metric.RADIUS_PX)) + 1
    sl = (slice(max(0, ys.min() - margin), min(60, ys.max() + margin + 1)),
          slice(max(0, xs.min() - margin), min(70, xs.max() + margin + 1)))
    crop = metric.components(p[sl], g[sl])
    assert crop.tp_w == pytest.approx(full.tp_w, rel=1e-9)
    assert crop.fp_w == pytest.approx(full.fp_w, rel=1e-9)
    assert crop.fn_w == pytest.approx(full.fn_w, rel=1e-9)
    assert crop.dti == pytest.approx(full.dti, rel=1e-9)
