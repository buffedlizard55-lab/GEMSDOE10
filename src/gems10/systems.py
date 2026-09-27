"""Fault-system labelling and SYSTEM-HOLDOUT cross-validation.

Why this module is the heart of GEMSDOE10
----------------------------------------
Both prize rounds score faults MISSING from the catalogue, with known-catalogue
pixels masked from evaluation (staff clarification, forum thread 11516). A
model selected by how well it reproduces the catalogue is selected for the
wrong skill — 6GEMSDOE's HGB reached 0.17 blocked-CV catalogue DTI and scored
0.0286 on the leaderboard.

The honest offline simulation of the official scoring is therefore:

  * hold out WHOLE FAULT SYSTEMS (8-connected components of the label raster),
    never random pixels — the held-out systems play the role of "unmapped
    faults", exactly like the private test set;
  * train on the remaining systems;
  * score the FULL footprint with GT = held-out systems and
    fp_ignore_mask = train systems — mirroring the official masking of the
    known catalogue (see metric.components).

Design details
--------------
* Systems are 8-connected components of `labels == 1`. Tiny 1–2 px systems are
  kept (they are real catalogue entries) but folds are balanced by PIXEL count
  via greedy bin-packing so every fold holds a comparable amount of hidden
  fault length.
* A buffer of `buffer_px` (default 3 px = 300 m = one kernel support, exact via
  the Euclidean distance transform) around held-out systems is excluded from
  TRAINING, so no training pixel lies within the kernel support of a scored
  ground-truth pixel. The full footprint is still SCORED — like the platform.
* Deterministic: fold assignment depends only on (sizes, n_folds, seed).

Measured on the official labels (2026-09-27, this repo, scripts/eda_systems.py):
  3,199 systems; median 12 px; mean 19.06 px; max 360 px; 1,977 >= 10 px.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


def label_systems(
    labels: np.ndarray, connectivity: np.ndarray = CONNECTIVITY_8
) -> tuple[np.ndarray, int, np.ndarray]:
    """Label 8-connected fault systems.

    Returns (system_id, n_systems, sizes) where system_id is int32 with 0 =
    background and 1..n_systems the system ids, and sizes[i] is the pixel count
    of system i+1.
    """
    pos = np.asarray(labels) == 1
    sys_id, n = ndimage.label(pos, structure=connectivity)
    sys_id = sys_id.astype(np.int32)
    sizes = np.bincount(sys_id.ravel(), minlength=n + 1)[1:].astype(np.int64)
    return sys_id, int(n), sizes


def assign_system_folds(
    sizes: np.ndarray, n_folds: int = 5, seed: int = 7
) -> np.ndarray:
    """Assign each system to a fold, balancing total hidden pixels per fold.

    Greedy bin-packing over a seeded shuffle: systems are dealt in random order
    to the fold currently holding the fewest pixels. Deterministic in
    (sizes, n_folds, seed). Returns int array fold_of_system[i] in [0, n_folds).
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if sizes.size == 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.size)
    fold_of_system = np.full(sizes.size, -1, dtype=np.int64)
    fold_pixels = np.zeros(n_folds, dtype=np.int64)
    for idx in order:
        k = int(np.argmin(fold_pixels))
        fold_of_system[idx] = k
        fold_pixels[k] += sizes[idx]
    return fold_of_system


@dataclass
class SystemFolds:
    """System-holdout fold geometry (labels + assignment, reusable across runs)."""

    system_id: np.ndarray  # int32 grid, 0 = background
    n_systems: int
    sizes: np.ndarray  # (n_systems,) pixel counts
    fold_of_system: np.ndarray  # (n_systems,) fold ids
    n_folds: int
    buffer_px: int = 3

    @property
    def shape(self) -> tuple[int, int]:
        return (self.system_id.shape[0], self.system_id.shape[1])

    def heldout_mask(self, k: int) -> np.ndarray:
        """Boolean grid: pixels of systems held out in fold k (= the GT)."""
        ids = np.nonzero(self.fold_of_system == k)[0] + 1
        return np.isin(self.system_id, ids)

    def train_system_mask(self, k: int) -> np.ndarray:
        """Boolean grid: pixels of systems used for training in fold k."""
        ids = np.nonzero(self.fold_of_system != k)[0] + 1
        return np.isin(self.system_id, ids)

    def trainable_mask(self, k: int) -> np.ndarray:
        """Boolean grid: pixels legal for TRAINING in fold k.

        Everything except the held-out systems and their buffer (pixels within
        `buffer_px` Euclidean pixels of a held-out system pixel). Exact via the
        distance transform. The caller intersects with the footprint.
        """
        held = self.heldout_mask(k)
        if not held.any():
            return np.ones(self.shape, dtype=bool)
        d = ndimage.distance_transform_edt(~held, sampling=1.0)
        return d > float(self.buffer_px)

    def fold_pixel_counts(self) -> list[int]:
        out = []
        for k in range(self.n_folds):
            out.append(int(self.sizes[self.fold_of_system == k].sum()))
        return out


def make_system_folds(
    labels: np.ndarray,
    n_folds: int = 5,
    buffer_px: int = 3,
    seed: int = 7,
) -> SystemFolds:
    """Build the system-holdout folds from a label grid (== 1 is fault)."""
    sys_id, n, sizes = label_systems(labels)
    folds = assign_system_folds(sizes, n_folds=n_folds, seed=seed)
    return SystemFolds(
        system_id=sys_id,
        n_systems=n,
        sizes=sizes,
        fold_of_system=folds,
        n_folds=n_folds,
        buffer_px=buffer_px,
    )


def sample_training_pixels(
    trainable: np.ndarray,
    train_pos: np.ndarray,
    footprint: np.ndarray,
    n_neg: int,
    seed: int,
    pos_fraction_cap: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (rows, cols, y) for one fold's training set.

    All trainable positive pixels are kept (optionally capped to a fraction of
    the negatives via pos_fraction_cap); negatives are uniform-random without
    replacement. Deterministic in the seed.
    """
    trainable = np.asarray(trainable, dtype=bool)
    train_pos = np.asarray(train_pos, dtype=bool)
    footprint = np.asarray(footprint, dtype=bool)
    pos = trainable & train_pos & footprint
    neg = trainable & ~train_pos & footprint
    py, px = np.nonzero(pos)
    ny, nx = np.nonzero(neg)
    rng = np.random.default_rng(seed)
    if ny.size > n_neg:
        sel = rng.choice(ny.size, size=n_neg, replace=False)
        ny, nx = ny[sel], nx[sel]
    if pos_fraction_cap is not None and py.size > 0:
        cap = int(pos_fraction_cap * ny.size)
        if py.size > cap and cap > 0:
            sel = rng.choice(py.size, size=cap, replace=False)
            py, px = py[sel], px[sel]
    rows = np.concatenate([py, ny])
    cols = np.concatenate([px, nx])
    y = np.concatenate(
        [np.ones(py.size, dtype=np.int8), np.zeros(ny.size, dtype=np.int8)]
    )
    order = rng.permutation(rows.size)
    return rows[order], cols[order], y[order]
