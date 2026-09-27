"""GEMSDOE10 feature additions: Frangi vesselness + Gabor orient-max."""

import numpy as np

from gems10 import features as F


def _ridge(n=64, angle_deg=45.0, width=2.0):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    th = np.radians(angle_deg)
    across = -xx * np.sin(th) + yy * np.cos(th)
    across -= across[n // 2, n // 2]
    return np.exp(-(across / width) ** 2)


def test_frangi_separates_ridge_from_background():
    ridge = _ridge()
    fr = F.frangi_vesselness(ridge, 2.0)
    on = float(np.nanmean(fr.vesselness[ridge > 0.5]))
    off = float(np.nanmean(fr.vesselness[ridge < 0.05]))
    assert on > 10 * off
    assert 0.0 <= np.nanmax(fr.vesselness) <= 1.0


def test_frangi_orientation_follows_the_ridge():
    ridge = _ridge(angle_deg=30.0)
    fr = F.frangi_vesselness(ridge, 2.0)
    o = np.nanmedian(fr.orientation[ridge > 0.7])
    # ridge direction 30 deg == -150 deg (mod pi); allow either
    assert min(abs(((o - np.radians(30)) + np.pi / 2) % np.pi - np.pi / 2),
               abs(((o - np.radians(-150)) + np.pi / 2) % np.pi - np.pi / 2)) < 0.2


def test_gabor_energy_and_strike():
    ridge = _ridge(angle_deg=135.0)
    e, o = F.gabor_max_response(ridge, size=11, n_orientations=8)
    on = float(np.nanmean(e[ridge > 0.5]))
    off = float(np.nanmean(e[ridge < 0.05]))
    assert on > 5 * off
    assert abs(float(np.nanmedian(o[ridge > 0.5])) - 3 * np.pi / 4) < 0.2


def test_nan_guard_grows_a_halo_not_the_signal():
    ridge = _ridge()
    ridge[0:8, :] = np.nan  # invalid strip at the top
    fr = F.frangi_vesselness(ridge, 2.0)
    assert bool(np.isnan(fr.vesselness[0:12, 32]).all())  # halo invalidated
    assert bool(np.isfinite(fr.vesselness[40, 32]))  # far field intact
    e, _ = F.gabor_max_response(ridge, size=11)
    assert bool(np.isnan(e[0:10, 32]).all())
    assert bool(np.isfinite(e[40, 32]))
