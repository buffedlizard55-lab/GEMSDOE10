"""Masked DTI: known-catalogue pixels excluded from FP (forum 11516)."""

import numpy as np
import pytest

from gems10 import metric


def test_masked_predictions_on_known_faults_cost_nothing():
    # GT = one new fault; catalogue mask = elsewhere; predict p=1 on both.
    gt = np.zeros((12, 12), dtype=bool)
    gt[6, 6] = True
    mask = np.zeros((12, 12), dtype=bool)
    mask[1, 1] = True
    pred = np.zeros((12, 12))
    pred[6, 6] = 1.0
    pred[1, 1] = 1.0  # on a masked known-fault pixel: must be free
    plain = metric.components(pred, gt)
    masked = metric.components(pred, gt, fp_ignore_mask=mask)
    assert masked.tp_w == pytest.approx(plain.tp_w)
    assert masked.fn_w == pytest.approx(plain.fn_w)
    # the (1,1) mass (far from GT, k=0) contributed 1.0 to plain FP_w
    assert plain.fp_w == pytest.approx(1.0)
    assert masked.fp_w == pytest.approx(0.0)
    assert masked.dti == pytest.approx(1.0)


def test_mask_does_not_change_tp_or_fn_identity():
    rng = np.random.default_rng(3)
    gt = np.zeros((25, 25), dtype=bool)
    gt[5, 5] = True
    gt[18, 18] = True
    mask = np.zeros((25, 25), dtype=bool)
    mask[5, 6] = True
    pred = rng.random((25, 25)) * 0.5
    c = metric.components(pred, gt, fp_ignore_mask=mask)
    assert c.tp_w + c.fn_w == pytest.approx(2.0)  # identity preserved


def test_mask_overlapping_gt_is_rejected():
    gt = np.zeros((8, 8), dtype=bool)
    gt[4, 4] = True
    mask = np.zeros((8, 8), dtype=bool)
    mask[4, 4] = True
    with pytest.raises(ValueError):
        metric.components(np.zeros((8, 8)), gt, fp_ignore_mask=mask)


def test_unmasked_default_matches_published_formula():
    # without a mask the masked path must equal the plain path exactly
    rng = np.random.default_rng(5)
    gt = rng.random((20, 20)) < 0.05
    pred = rng.random((20, 20))
    a = metric.components(pred, gt)
    b = metric.components(pred, gt, fp_ignore_mask=None)
    assert a.as_dict() == b.as_dict()
