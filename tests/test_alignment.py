"""Preregistered controls for the H13/H16 alignment engine (HYPOTHESES.md).

A synthetic displaced lineation MUST be detected with the correct lag; a
no-displacement control and a null field MUST NOT produce a displacement
advantage. H16 ray scoring must light only along train-system endpoint rays.
"""
import numpy as np
import pytest

from gems10 import alignment
from gems10.discovery import Strike
from gems10.features import Lineament


def _bumps(length, n_bumps=6, seed=1):
    rng = np.random.default_rng(seed)
    x = np.zeros(length)
    for _ in range(n_bumps):
        c = rng.integers(8, length - 8)
        w = rng.uniform(3, 7)
        x += np.exp(-(np.arange(length) - c) ** 2 / (2 * w * w)) * rng.uniform(0.5, 1.5)
    return x


def _square_wave(length, period=8, noise=0.15, seed=1):
    """Alternating 4-on/4-off blocks — texture whose 3 px misalignment is
    large relative to its feature scale (the displacement test needs this)."""
    x = (np.arange(length) % period) < 4
    x = x.astype(float) * 2.0 - 1.0
    x += np.random.default_rng(seed).standard_normal(length) * noise
    return x


def test_displaced_lineation_detected_with_correct_lag():
    W = 96
    shift = 3
    p = _square_wave(W)
    H = 64
    f = H // 2  # E-W fault between rows f-1 and f
    band = np.tile(p[None, :], (H, 1))
    below = np.zeros_like(band)
    for i in range(f, H):
        below[i, shift:] = band[i, :W - shift]  # p(col - shift), no wrap
    band = np.where(np.arange(H)[:, None] >= f, below, band)
    chans = alignment.offset_channels(band, angles=(0,), seps=(1, 2))
    # channels per angle: (ncc0, ncc_best, adv) + aggregate; last axis = chan
    ncc0, best, adv = chans[..., 0], chans[..., 1], chans[..., 2]
    col = W // 2
    # Strips are (row i, row i-d): the fault (between rows f-1 and f) is
    # straddled at row f. There: high off-axis alignment (ncc_best) with
    # reduced zero-lag alignment (ncc0) -> positive adv.
    assert best[f, col] > 0.4, f"best={best[f, col]}"
    assert ncc0[f, col] < 0.6, f"ncc0={ncc0[f, col]}"
    assert adv[f, col] > 0.15, f"adv={adv[f, col]}"
    # Far from the fault (both strips on the same side): no positive advantage.
    assert adv[25, col] < 0.15
    assert adv[40, col] < 0.15
    # Aggregate channel exists and is finite where the per-angle channel is.
    assert np.isfinite(chans[f, col, 3])


def test_no_displacement_control():
    W = 128
    p = _bumps(W, seed=3)
    band = np.tile(p[None, :], (64, 1))
    chans = alignment.offset_channels(band, angles=(0,), seps=(1, 2))
    adv = chans[..., 2]
    finite = adv[np.isfinite(adv)]
    assert finite.size > 100
    assert adv[30, 48] < 0.15, "zero-displacement pattern must not show advantage"
    assert np.nanmean(adv) < 0.10


def test_null_field_no_advantage():
    rng = np.random.default_rng(4)
    band = rng.standard_normal((96, 128)).astype(np.float32)
    chans = alignment.offset_channels(band, angles=(0,), seps=(1,))
    adv = chans[..., 2]
    # Calibrated engine: no systematic displacement on pure noise.
    assert abs(np.nanmean(adv)) < 0.05
    assert np.nanmedian(adv) < 0.03


def test_nan_halo_invalidated():
    rng = np.random.default_rng(5)
    band = rng.standard_normal((128, 128)).astype(np.float32)
    band[60:64, 60:64] = np.nan
    chans = alignment.offset_channels(band, angles=(0,), seps=(1,))
    assert np.all(~np.isfinite(chans[62, 62, :]))
    # A point outside the halo remains finite.
    assert np.isfinite(chans[20, 20, 0])


def test_tiled_matches_full_grid():
    rng = np.random.default_rng(11)
    band = rng.standard_normal((128, 150)).astype(np.float32)
    band[55:60, 30:40] = np.nan  # NaN patch inside
    full = alignment.offset_channels(band)
    tiled = alignment.offset_channels_tiled(band, block_rows=48)
    assert tiled.shape == full.shape
    same = np.array_equal(tiled, full, equal_nan=True)
    assert same, f"tiled differs at {(np.isnan(tiled) != np.isnan(full)).sum()} NaN-pattern cells"


def test_ninety_degree_offset_detected():
    # N-S fault between cols c-1 and c: columns are square waves in the row
    # direction; the east side is shifted +3 rows (no wrap).
    H, W = 64, 128
    c = W // 2
    shift = 3
    colprof = _square_wave(H)
    band = np.tile(colprof[:, None], (1, W))
    for j in range(c, W):
        band[shift:, j] = band[:H - shift, j]
        band[:shift, j] = 0.0
    chans = alignment.offset_channels(band, angles=(90,))
    ncc0, best, adv = chans[..., 0], chans[..., 1], chans[..., 2]
    row = H // 2
    # Straddle column c: adjacent columns offset -> positive adv.
    assert adv[row, c] > 0.15, f"adv={adv[row, c]}"
    assert best[row, c] > 0.4
    assert ncc0[row, c] < 0.6
    # Far from the fault: no positive advantage.
    assert adv[row, 20] < 0.15
    assert adv[row, W - 20] < 0.15


def test_all_angles_and_shape():
    rng = np.random.default_rng(6)
    band = rng.standard_normal((80, 72)).astype(np.float32)
    chans = alignment.offset_channels(band)
    assert chans.shape == (80, 72, 4 * 3 + 1)


def _strike(angle, end, centroid, width=2.0):
    return Strike(system=1, centroid=centroid, angle=angle, half_len=20.0,
                  width=width, n_px=40, end_a=end, end_b=(centroid[0], centroid[1]))


def test_h16_ray_lights_only_along_rays():
    H, W = 64, 96
    lin = Lineament(energy=np.full((H, W), 2.0), coherence=np.full((H, W), 0.8),
                    orientation=np.full((H, W), 0.0))  # line along columns (E-W)
    # System: E-W fault, centroid left of endpoint; ray points +col direction.
    s = _strike(angle=0.0, end=(32, 20), centroid=(32, 10))
    ncc = np.zeros((H, W))
    out, names = alignment.ray_continuation_channels(
        ncc, {"tmi": lin}, [s], ncc_band=ncc)
    assert names[:4] == ["cont_tmi_r5", "cont_tmi_r15", "cont_tmi_r30", "cont_tmi_r60"]
    # Ray points: endpoint (32,20) + t*(1,0) -> (32, 25/35/50/80); each
    # radius lights its own channel.
    for t, c in ((5, 25), (15, 35), (30, 50), (60, 80)):
        ch = out[..., names.index(f"cont_tmi_r{t}")]
        assert np.isfinite(ch[32, c]), f"ray point at r={t} should be lit"
        assert ch[32, c] > 0
    # Away from the ray: NaN (r5 channel).
    assert np.isnan(out[..., names.index("cont_tmi_r5")][10, 40])
    # Orientation gate: a line oriented 90 deg from the ray must score 0.
    lin_bad = Lineament(energy=np.full((H, W), 2.0), coherence=np.full((H, W), 0.8),
                        orientation=np.full((H, W), np.pi / 2))
    out_bad, _ = alignment.ray_continuation_channels(
        ncc, {"tmi": lin_bad}, [s], ncc_band=ncc)
    assert out_bad[32, 25, 0] == 0.0


def test_h16_ncc_continuity():
    # _ncc1d primitive controls first (preregistered null/displacement checks).
    x = np.arange(16, dtype=float)
    assert alignment._ncc1d(x, x) == pytest.approx(1.0)
    assert alignment._ncc1d(x, -x) == pytest.approx(-1.0)
    assert alignment._ncc1d(np.ones(16), np.ones(16)) is not None  # zero-variance -> nan, documented
    assert np.isnan(alignment._ncc1d(np.ones(16), np.ones(16)))
    rng = np.random.default_rng(7)
    assert abs(alignment._ncc1d(rng.standard_normal(16), rng.standard_normal(16))) < 0.6
    assert np.isnan(alignment._ncc1d(np.ones(2), np.ones(3)))
    # Ray-level: a row carrying a continuous signal must score higher
    # continuity than the same row carrying independent noise.
    H, W = 64, 160
    band = np.zeros((H, W))
    # Period 17 px = exactly the front/back separation, so the two halves
    # align in phase on a continuous row.
    band[32, :] = np.sin(2 * np.pi * np.arange(W) / 17.0) * 0.5
    band_noisy = band.copy()
    band_noisy[32, :] = np.random.default_rng(8).standard_normal(W)
    s = _strike(angle=0.0, end=(32, 40), centroid=(32, 20))
    v_cont, _ = alignment.ray_continuation_channels(
        band, {}, [s], ncc_band=band, radii=(), ncc_rays=(15,))
    v_noise, _ = alignment.ray_continuation_channels(
        band_noisy, {}, [s], ncc_band=band_noisy, radii=(), ncc_rays=(15,))
    vc = v_cont[32, 55, -1]
    vn = v_noise[32, 55, -1]
    assert np.isfinite(vc) and np.isfinite(vn)
    assert vc > vn + 0.2, f"continuous signal must beat noise: {vc} vs {vn}"


def test_h16_filters_small_systems():
    H, W = 64, 96
    lin = Lineament(energy=np.full((H, W), 2.0), coherence=np.full((H, W), 0.8),
                    orientation=np.full((H, W), 0.0))
    tiny = Strike(system=1, centroid=(32, 10), angle=0.0, half_len=2.0, width=1.0,
                  n_px=3, end_a=(32, 20), end_b=(32, 10))
    out, _ = alignment.ray_continuation_channels(np.zeros((H, W)), {"tmi": lin},
                                                 [tiny], ncc_band=np.zeros((H, W)))
    assert np.all(np.isnan(out))
