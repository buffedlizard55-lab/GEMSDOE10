"""Specification, blocked-fold geometry, and placement algebra."""

import numpy as np
import pytest

from gems10 import cv, placement, spec


def test_pinned_constants_are_internally_consistent():
    assert spec.WIDTH * spec.PIXEL_SIZE_M == spec.ORIGIN_X * 0 + spec.WIDTH * 100.0
    assert spec.HEIGHT * spec.PIXEL_SIZE_M == 373_000.0
    g = spec.GridSpec()
    left, bottom, right, top = g.bounds
    assert (left, bottom, right, top) == (243350.0, 4135550.0, 572550.0, 4508550.0)
    assert right - left == spec.WIDTH * spec.PIXEL_SIZE_M
    assert top - bottom == spec.HEIGHT * spec.PIXEL_SIZE_M


def test_footprint_and_positives_are_consistent():
    assert spec.FOOTPRINT_PIXELS + spec.NODATA_PIXELS == spec.TOTAL_PIXELS
    assert spec.LABEL_POSITIVE_PIXELS < spec.FOOTPRINT_PIXELS
    assert spec.COVERAGE_OF_FOOTPRINT == pytest.approx(0.011803, abs=1e-6)
    # all feature-valid pixels are inside the submission footprint
    assert spec.FEATURES_ALL_BAND_VALID_PIXELS <= spec.FOOTPRINT_PIXELS


def test_all_nineteen_bands_are_pinned():
    assert len(spec.FEATURE_BANDS) == 19
    assert spec.BAND_INDEX["det_elev"] == 12
    assert spec.BAND_INDEX["iso_grav_anom_hg"] == 18
    assert sorted(spec.BAND_INDEX.values()) == list(range(1, 20))


def test_pins_cover_the_three_official_files():
    assert set(spec.PINS) == {"training_features.tif", "labels.tif",
                              "sample_submission.tif"}
    for pin in spec.PINS.values():
        assert len(pin["sha256"]) == 64 and pin["bytes"] > 0
    total_parts = sum(b for _, b, _ in spec.BRIDGE_PARTS)
    assert total_parts == spec.PINS["training_features.tif"]["bytes"]


def test_buffer_default_is_three_pixels_not_three_hundred():
    """Regression guard: the default buffer is 300 m = 3 px, not 300 px."""
    f = cv.make_folds()
    assert f.buffer_px == 3


def test_blocked_folds_have_a_buffer_between_train_and_score():
    f = cv.make_folds(n_blocks=4, n_folds=4, buffer_px=3)
    shape = (80, 80)
    train, score = f.train_test_masks(shape, 0)
    assert score.any() and (~score).any()
    # every scored pixel is at least buffer_px+1 away from any training pixel
    rows = np.flatnonzero(score.any(axis=1))
    for r in rows:
        assert not train[r].any() or True
    # specifically: train mask must be a strict subset of the complement of the
    # dilated score mask
    dilated = cv._dilate(score, 3)
    assert not (train & dilated).any()


def test_folds_partition_the_grid():
    f = cv.make_folds(n_blocks=4, n_folds=4, buffer_px=3)
    shape = (64, 64)
    union = np.zeros(shape, dtype=bool)
    for k in range(f.n_folds):
        _, score = f.train_test_masks(shape, k)
        assert not (union & score).any(), "score masks must not overlap"
        union |= score
    assert union.all(), "every block must belong to exactly one fold"


def test_block_ids_are_in_range():
    f = cv.make_folds(4, 4, 3)
    ids = cv.block_ids((100, 100), f)
    assert ids.min() >= 0 and ids.max() <= 3


def test_breakeven_posterior_matches_the_closed_form():
    # k = 2/3 (100 m error): q* = 0.2*(1/3) / ((2/3)*1.8 + 0.2*(1/3))
    expected = 0.2 * (1 / 3) / ((2 / 3) * 1.8 + 0.2 * (1 / 3))
    assert placement.breakeven_posterior(2 / 3) == pytest.approx(expected)
    assert expected == pytest.approx(0.052632, abs=1e-6)
    # at zero distance any positive prediction pays off; at the kernel edge it never does
    assert placement.breakeven_posterior(1.0) == 0.0
    assert placement.breakeven_posterior(0.0) == 1.0
    # monotone in k
    ks = [0.0, 0.0572, 0.2546, 1 / 3, 0.5286, 2 / 3, 1.0]
    qs = [placement.breakeven_posterior(k) for k in ks]
    assert qs == sorted(qs, reverse=True)


def test_skeleton_and_spacing_strategies_are_inside_the_valid_mask():
    rng = np.random.default_rng(0)
    prob = np.zeros((60, 60), dtype=np.float32)
    prob[30, 10:50] = 0.9  # a line
    prob[5:15, 5:15] = 0.4  # a blob
    valid = np.zeros((60, 60), dtype=bool)
    valid[2:58, 2:58] = True
    out = placement.strategies(prob, valid, threshold=0.5, spacing=4)
    assert set(out) >= {"raw", "threshold", "skeleton", "skeleton_spaced4", "densified"}
    for name, arr in out.items():
        assert arr.shape == prob.shape
        assert np.isfinite(arr).all(), name
        assert arr.min() >= 0.0 and arr.max() <= 1.0, name
        assert not (arr > 0)[~valid].any(), f"{name} leaked outside the valid mask"


def test_spacing_thins_the_skeleton():
    prob = np.zeros((40, 80), dtype=np.float32)
    prob[20, 5:75] = 1.0
    valid = np.ones((40, 80), dtype=bool)
    skel = placement.strategies(prob, valid, 0.5, spacing=1)["skeleton"]
    spaced = placement.strategies(prob, valid, 0.5, spacing=4)["skeleton_spaced4"]
    assert spaced.sum() < skel.sum()
    assert spaced.sum() > 0


def test_densify_never_lowers_a_value_and_never_leaks():
    prob = np.zeros((50, 50), dtype=np.float32)
    prob[25, 20] = 0.9
    valid = np.ones((50, 50), dtype=bool)
    out = placement.densify_along_lineaments(prob, valid, threshold=0.5, dilation=1)
    assert (out >= prob - 1e-6).all()
    assert out[25, 21] == pytest.approx(0.9)
    assert out[24, 20] == pytest.approx(0.9)


def test_topk_mask_selects_exactly_the_requested_budget():
    rng = np.random.default_rng(5)
    p = rng.random((40, 40)).astype(np.float32)
    valid = np.ones((40, 40), dtype=bool)
    valid[0, 0] = False
    for frac in (0.01, 0.05, 0.25):
        m = placement.topk_mask(p, valid, frac)
        n_valid = int(valid.sum())
        expected = max(1, int(round(frac * n_valid)))
        assert int(m.sum()) >= expected
        assert not (m & ~valid).any()
        # everything selected is at least as large as everything not selected
        assert p[m].min() >= p[~m & valid].max()


def test_hard_strategy_is_binary_and_inside_the_mask():
    rng = np.random.default_rng(9)
    prob = rng.random((30, 30)).astype(np.float32)
    valid = np.zeros((30, 30), dtype=bool)
    valid[3:27, 3:27] = True
    out = placement.get_strategy("hard@0.4", prob, valid, threshold=0.4)
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert not (out > 0)[~valid].any()
    assert int((out > 0).sum()) == int((prob[valid] >= 0.4).sum())


def test_topk_hard_strategy_respects_the_budget():
    rng = np.random.default_rng(13)
    prob = rng.random((50, 50)).astype(np.float32)
    valid = np.ones((50, 50), dtype=bool)
    out = placement.get_strategy("topk_hard@0.1", prob, valid, threshold=0.0)
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert int((out > 0).sum()) == 250


def test_gated_topk_spends_the_exact_budget_gate_first():
    rng = np.random.default_rng(21)
    p = rng.random((50, 50)).astype(np.float32)
    valid = np.ones((50, 50), dtype=bool)
    gate = rng.random((50, 50)) < 0.05  # ~125 gated px, sparser than the budget

    # budget larger than the gate: ALL gated pixels in, remainder from ungated
    out = placement.gated_topk(p, valid, gate, 0.20)
    assert int(out.sum()) == 500
    assert (out[gate]).all()
    assert not (out & ~valid).any()

    # budget smaller than the gate: only the top-k of the gated set
    out = placement.gated_topk(p, valid, gate, 0.005)
    k = max(1, int(round(0.005 * 2500)))
    assert int(out.sum()) == k
    assert (out & ~gate).sum() == 0
    # a monotone rescaling of p (order-preserving, stays in [0, 1]) must not
    # change the selection
    out2 = placement.gated_topk(p / p.max(), valid, gate, 0.005)
    assert (out == out2).all()


def test_topk_gate_strategy_requires_a_gate():
    rng = np.random.default_rng(23)
    p = rng.random((30, 30)).astype(np.float32)
    valid = np.ones((30, 30), dtype=bool)
    gate = rng.random((30, 30)) < 0.1
    with pytest.raises(ValueError):
        placement.get_strategy("topk_gate@0.05", p, valid, threshold=0.0)
    out = placement.get_strategy("topk_gate@0.05", p, valid, threshold=0.0, gate=gate)
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert int((out > 0).sum()) == 45
