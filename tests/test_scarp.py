import numpy as np
from gems10 import scarp


def test_templates_annihilate_planes():
    y, x = np.mgrid[-4:5, -4:5]
    for theta in np.arange(8) * np.pi / 8:
        for kernel in scarp.profile_kernels(4, theta):
            assert abs(np.sum(kernel * (7 + 2*x - 3*y))) < 1e-12
            assert np.isclose(np.abs(kernel).sum(), 1)


def test_step_beats_symmetric_valley_at_center_and_ramp():
    y, x = np.mgrid[-20:21, -20:21]
    step = scarp.features(np.tanh(x*2))
    valley = scarp.features(x*x)
    ramp = scarp.features(2*x + 3*y + 50)
    assert step[20, 20, 0] > .1
    assert step[20, 20, 1] > .9
    assert step[20, 20, 2] > .99
    assert abs(valley[20, 20, 0]) < 1e-10
    assert np.nanmax(ramp[:, :, [0, 3]]) < 1e-10


def test_nan_edges_and_tile_equivalence():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(60, 50))
    a[25, 25] = np.nan
    full = scarp.features(a)
    assert np.isnan(full[25, 25]).all()
    assert np.isnan(full[0]).all()
    tile = scarp.features(a[10:50])
    np.testing.assert_allclose(full[20:40], tile[10:30], equal_nan=True)


def test_height_sign_does_not_change_features():
    y, x = np.mgrid[-20:21, -20:21]
    a = np.tanh(x) + .3 * np.sin(y)
    np.testing.assert_allclose(scarp.features(a), scarp.features(-a), equal_nan=True)
