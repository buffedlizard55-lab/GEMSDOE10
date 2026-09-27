# Provenance: adapted from buffedlizard55-lab/6GEMSDOE src/gems (same owner, re-verified here).
"""Spatially blocked, buffered cross-validation.

Why not a random pixel split: faults are spatially autocorrelated, and this
competition's scoring set is FAULTS MISSING FROM THE CATALOGUE. A random split
puts pixels of the same fault in train and test, so a model can memorise the
line and the score becomes a lookup test. The reference solution uses random
patch splits (`make_patches(..., seed=10*mc)` with `rng.permutation`), which is
exactly the leak; the task brief explicitly asks for blocked folds instead.

Design:
  * The grid is cut into `n_blocks x n_blocks` rectangular blocks.
  * Folds are formed by dealing blocks in a strided pattern so that each fold's
    blocks are spread over the region rather than clustered in one corner.
  * A BUFFER of `buffer_px` pixels around every held-out block is also held out
    of *training* (so no training pixel is within the kernel support of a scored
    pixel) and excluded from *scoring* (so the scored pixels are not influenced
    by test-block edges).
  * Because the target is unmapped faults, we also report the score with the
    catalogued faults removed from the ground truth inside the held-out blocks
    ("gap score"), which is the closest honest proxy available without the
    private labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Folds:
    fold_of_block: np.ndarray      # (n_blocks, n_blocks) int
    n_folds: int
    n_blocks: int
    buffer_px: int

    def train_test_masks(self, shape: tuple[int, int], k: int
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Boolean (train, score) masks for fold k.

        `train` excludes the held-out blocks *and* their buffer, so no training
        pixel lies within the kernel support of a scored pixel. `score` is the
        held-out blocks themselves — the buffer is not scored either, because a
        pixel next to the train/test boundary is contaminated by whichever side
        the model learned from.
        """
        h, w = shape
        block_of_row = np.minimum(np.arange(h) * self.n_blocks // h,
                                  self.n_blocks - 1)
        block_of_col = np.minimum(np.arange(w) * self.n_blocks // w,
                                  self.n_blocks - 1)
        blk = self.fold_of_block[np.ix_(block_of_row, block_of_col)]
        test = blk == k
        buf = _dilate(test, self.buffer_px)
        return (~buf), test

    def score_mask(self, shape: tuple[int, int], k: int) -> np.ndarray:
        return self.train_test_masks(shape, k)[1]


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask
    out = mask.copy()
    for _ in range(iterations):
        shifted = out.copy()
        shifted[1:, :] |= out[:-1, :]
        shifted[:-1, :] |= out[1:, :]
        shifted[:, 1:] |= out[:, :-1]
        shifted[:, :-1] |= out[:, 1:]
        out = shifted
    return out


def make_folds(n_blocks: int = 4, n_folds: int = 4, buffer_px: int = 3) -> Folds:
    """Strided block folds.

    The default buffer is 3 px = 300 m = one full kernel support, in PIXELS. (An
    earlier revision defaulted to 300 *pixels* = 30 km, which would have held out
    most of the region and made the folds meaningless — the units are stated here
    and a test pins the default.)
    """
    if n_folds > n_blocks * n_blocks:
        raise ValueError("n_folds cannot exceed the number of blocks")
    fold_of_block = np.full((n_blocks, n_blocks), -1, dtype=int)
    counter = 0
    for i in range(n_blocks):
        for j in range(n_blocks):
            fold_of_block[i, j] = counter % n_folds
            counter += 1
    return Folds(fold_of_block=fold_of_block, n_folds=n_folds, n_blocks=n_blocks,
                 buffer_px=buffer_px)


def block_ids(shape: tuple[int, int], folds: Folds) -> np.ndarray:
    """Full-grid fold id map (uint8) for bookkeeping and for the site figures."""
    h, w = shape
    b = folds.n_blocks
    block_of_row = np.minimum(np.arange(h) * b // h, b - 1)
    block_of_col = np.minimum(np.arange(w) * b // w, b - 1)
    return folds.fold_of_block[np.ix_(block_of_row, block_of_col)].astype(np.uint8)


def sample_pixels(mask: np.ndarray, n: int, seed: int = 0,
                  valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample up to n pixel coordinates from `mask` (row, col)."""
    m = mask if valid is None else (mask & valid)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return ys, xs
    rng = np.random.default_rng(seed)
    if ys.size > n:
        sel = rng.choice(ys.size, size=n, replace=False)
        ys, xs = ys[sel], xs[sel]
    return ys, xs
