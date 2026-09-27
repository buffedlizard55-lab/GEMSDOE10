"""Fault-system labelling + system-holdout folds (GEMSDOE10 harness core)."""

import numpy as np
import pytest

from gems10 import systems


def _toy_labels():
    # two separated horizontal systems + one singleton
    lab = np.zeros((20, 20), dtype=np.int8)
    lab[3, 2:8] = 1
    lab[12, 10:17] = 1
    lab[17, 17] = 1
    return lab


def test_label_systems_counts_components_and_sizes():
    sys_id, n, sizes = systems.label_systems(_toy_labels())
    assert n == 3
    assert sorted(sizes.tolist()) == [1, 6, 7]
    assert sys_id.dtype == np.int32
    assert set(np.unique(sys_id).tolist()) == {0, 1, 2, 3}


def test_diagonal_pixels_are_one_system_under_8_connectivity():
    lab = np.zeros((6, 6), dtype=np.int8)
    lab[1, 1] = 1
    lab[2, 2] = 1
    _, n, sizes = systems.label_systems(lab)
    assert n == 1 and sizes.tolist() == [2]


def test_fold_assignment_is_deterministic_and_balanced():
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 60, size=500)
    a = systems.assign_system_folds(sizes, n_folds=5, seed=7)
    b = systems.assign_system_folds(sizes, n_folds=5, seed=7)
    assert (a == b).all()
    assert set(np.unique(a).tolist()) == {0, 1, 2, 3, 4}
    px = [int(sizes[a == k].sum()) for k in range(5)]
    # greedy bin-packing keeps folds within one max-system of each other
    assert max(px) - min(px) <= sizes.max()


def test_fold_masks_partition_and_buffer_is_exact():
    lab = _toy_labels()
    sf = systems.make_system_folds(lab, n_folds=2, buffer_px=3, seed=7)
    assert sf.n_systems == 3
    assert sum(sf.fold_pixel_counts()) == 14
    for k in range(2):
        held = sf.heldout_mask(k)
        train_sys = sf.train_system_mask(k)
        assert not (held & train_sys).any()
        assert (held | train_sys).sum() == 14  # every fault px is in one side
        trainable = sf.trainable_mask(k)
        # no trainable pixel within 3px (Euclidean) of a held-out pixel
        from scipy import ndimage

        d = ndimage.distance_transform_edt(~held)
        assert bool((trainable & (d <= 3.0)).any()) is False
        assert bool((held & trainable).any()) is False  # held-out never trainable


def test_sample_training_pixels_keeps_all_positives_and_caps_negatives():
    lab = _toy_labels()
    sf = systems.make_system_folds(lab, n_folds=2, buffer_px=3, seed=7)
    footprint = np.ones_like(lab, dtype=bool)
    for k in range(2):
        trainable = sf.trainable_mask(k) & footprint
        train_pos = sf.train_system_mask(k)
        rows, cols, y = systems.sample_training_pixels(
            trainable, train_pos, footprint, n_neg=25, seed=11
        )
        n_pos_expect = int((trainable & train_pos).sum())
        assert int(y.sum()) == n_pos_expect
        assert int((y == 0).sum()) == 25
        # determinism
        r2, c2, y2 = systems.sample_training_pixels(
            trainable, train_pos, footprint, n_neg=25, seed=11
        )
        assert (rows == r2).all() and (y == y2).all()
