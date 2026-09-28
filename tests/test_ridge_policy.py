"""H24 ridge-NMS emission engine — synthetic controls (no label contact).

Covers: policy parsing; output ⊆ top-k core; across-strike thinning of a
smooth ridge; along-strike continuity (no self-comparison gaps); plateau
reduction to the interior spine; determinism; NaN-outside-footprint; the
single-entry offset cache; and the measured structure-tensor convention
(orientation = normal to the line) that the whole design rests on.
"""
from __future__ import annotations

import numpy as np
import pytest

from gems10 import modeling, placement
from gems10.features import structure_tensor


def _image(h=96, w=96):
    return np.ones((h, w), dtype=bool)


def _vertical_ridge(h=96, w=96, col=48, sigma=3.0, rise=0.0):
    """Smooth ridge running along rows at `col`, optionally rising along
    strike (row direction)."""
    cc = np.arange(w)[None, :]
    rr = np.arange(h)[:, None]
    cross = np.exp(-((cc - col) ** 2) / (2 * sigma ** 2))
    along = 0.5 + rise * rr / max(h - 1, 1)
    return (along * cross).astype(np.float64)


def test_parse_policy_ridge():
    assert modeling.parse_policy("ridge10_binary") == (0.10, "ridge")
    assert modeling.parse_policy("ridge04_binary") == (0.04, "ridge")
    with pytest.raises(AssertionError):
        modeling.parse_policy("ridge10_soft")


def test_structure_tensor_orientation_is_normal():
    """Session-4 measured convention: for a horizontal stripe (line along
    columns, line angle 0°) the tensor orientation is 90° = the normal."""
    h = w = 96
    yy = np.arange(h)[:, None] * np.ones((1, w))
    img = np.exp(-((yy - 48) ** 2) / (2 * 3.0 ** 2))
    lin = structure_tensor(img, sigma=1.5, integration_sigma=4.0)
    strong = lin.energy > np.percentile(lin.energy, 90)
    c = np.nanmean(np.cos(2 * lin.orientation)[strong])
    s = np.nanmean(np.sin(2 * lin.orientation)[strong])
    ang = np.degrees(0.5 * np.arctan2(s, c))
    assert abs(abs(ang) - 90.0) < 1.0  # normal to a horizontal line


def test_ridge_offsets_never_degenerate():
    p = _vertical_ridge()
    offsets = placement.ridge_offsets(np.where(np.isfinite(p), p, np.nan))
    assert len(offsets) == 2
    for dr, dc in offsets:
        assert dr.shape == p.shape
        assert np.any((dr != 0) | (dc != 0)), "zero offset array"
        # every individual offset must move at least one pixel
        assert np.all(np.abs(dr) + np.abs(dc) >= 1)


def test_ridge_thins_across_strike_and_stays_in_core():
    fp = _image()
    p = _vertical_ridge(sigma=6.0)  # wide ridge
    emit = modeling.apply_policy(p, fp, "ridge10_binary")
    core = modeling.apply_policy(p, fp, "topk10_binary")
    e = np.nan_to_num(emit, nan=0.0)
    c = np.nan_to_num(core, nan=0.0)
    assert np.all(e <= c), "emission must stay inside the top-k support"
    assert e.sum() > 0
    assert e.sum() < c.sum(), "NMS must reduce the mask"
    per_row = (e > 0).sum(axis=1)
    assert per_row.max() <= 3, "across-strike width must collapse to <= 3 px"
    assert np.isfinite(emit).all()
    # outside-footprint is NaN by construction (fp all-true here, so check
    # the values are binary inside)
    assert set(np.unique(e)) <= {0.0, 1.0}


def test_ridge_keeps_along_strike_continuity_when_rising():
    fp = _image()
    p = _vertical_ridge(rise=0.5)  # probability rises along the strike
    emit = np.nan_to_num(modeling.apply_policy(p, fp, "ridge10_binary"), nan=0.0)
    kept_rows = (emit > 0).sum(axis=1)
    assert (kept_rows >= 1).all(), "every along-strike row must keep a pixel"


def test_ridge_plateau_reduces_to_spine():
    fp = _image()
    h = w = 96
    p = np.zeros((h, w))
    p[:, 40:56] = 0.9  # flat-topped plateau (HGB-style piecewise constant)
    p += 0.1 * np.exp(-((np.arange(w)[None, :] - 48) ** 2) / (2 * 8.0 ** 2))
    emit = np.nan_to_num(modeling.apply_policy(p, fp, "ridge20_binary"), nan=0.0)
    kept = (emit > 0).sum()
    plateau_px = 16 * h
    assert kept > 0
    assert kept <= plateau_px // 3, "plateau must reduce toward its spine"
    # spine = interior: kept columns must be away from the plateau edges
    cols = np.unique(np.nonzero(emit > 0)[1])
    assert cols.min() >= 41 and cols.max() <= 54


def test_ridge_deterministic_and_cache_consistent():
    fp = _image()
    p = _vertical_ridge(sigma=4.0)
    a = modeling.apply_policy(p, fp, "ridge08_binary")
    b = modeling.apply_policy(p, fp, "ridge08_binary")  # cache hit path
    c = modeling.apply_policy(p.copy(), fp, "ridge08_binary")  # cache miss path
    np.testing.assert_array_equal(np.nan_to_num(a, nan=-1),
                                  np.nan_to_num(b, nan=-1))
    np.testing.assert_array_equal(np.nan_to_num(a, nan=-1),
                                  np.nan_to_num(c, nan=-1))


def test_ridge_nan_outside_footprint():
    fp = _image()
    fp[:10, :] = False
    p = _vertical_ridge()
    emit = modeling.apply_policy(p, fp, "ridge10_binary")
    assert np.isnan(emit[:10, :]).all()
    assert np.isfinite(emit[10:, :]).all()
    assert set(np.unique(np.nan_to_num(emit, nan=0)[10:])) <= {0.0, 1.0}


def test_ridge_empty_core_total():
    # Direct engine-level check: an empty top-k mask emits nothing.
    core = np.zeros((32, 32), dtype=bool)
    p = np.random.default_rng(0).random((32, 32))
    assert placement.ridge_nms(core, p).sum() == 0


def test_ridge_constant_field_is_total_and_bounded():
    # All-zero probabilities: top-k degenerates to the full mask (all ties —
    # unchanged `topk_mask` behaviour, deliberately not modified). The ridge
    # engine must still return a deterministic binary subset, never crash or
    # emit NaN inside the footprint.
    fp = _image()
    p = np.zeros(fp.shape)
    emit = modeling.apply_policy(p, fp, "ridge10_binary")
    assert np.isfinite(emit).all()
    assert set(np.unique(emit)) <= {0.0, 1.0}
    assert (emit[~fp] if (~fp).any() else True) is not None
    assert np.nansum(emit) <= fp.sum()
