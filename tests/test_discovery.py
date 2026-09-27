"""Discovery operators: strikes, rays, corridors, fusion."""

import numpy as np

from gems10 import discovery, systems


def _two_systems():
    lab = np.zeros((60, 60), dtype=np.int8)
    lab[10, 5:25] = 1      # horizontal system
    lab[30:50, 40] = 1     # vertical system
    return lab


def test_system_strikes_recover_axis_and_endpoints():
    sys_id, n, _ = systems.label_systems(_two_systems())
    assert n == 2
    strikes = discovery.system_strikes(sys_id, np.array([1, 2]))
    assert len(strikes) == 2
    horiz = [s for s in strikes if abs(s.centroid[0] - 10) < 3][0]
    vert = [s for s in strikes if abs(s.centroid[1] - 40) < 3][0]
    assert horiz.angle == __import__("pytest").approx(0.0, abs=0.15)
    assert vert.angle == __import__("pytest").approx(np.pi / 2, abs=0.15)
    assert horiz.half_len > 5 and vert.half_len > 5


def test_strike_rays_extend_outward_from_endpoints():
    sys_id, n, _ = systems.label_systems(_two_systems())
    strikes = discovery.system_strikes(sys_id, np.array([1, 2]))
    rays = discovery.strike_rays((60, 60), strikes, ray_len_px=8.0)
    assert rays.shape == (60, 60)
    # horizontal system endpoints near (10,5) and (10,24): rays reach col<5, col>24
    assert rays[10, 0:4].sum() > 0
    assert rays[10, 26:34].sum() > 0
    # vertical system endpoints near (30,40),(49,40): rays reach row<30, row>49
    assert rays[22:29, 40].sum() > 0
    assert rays[51:58, 40].sum() > 0


def test_relay_corridors_link_only_nearby_subparallel_ends():
    lab = np.zeros((40, 40), dtype=np.int8)
    lab[10, 5:12] = 1    # ends near (10,12)
    lab[10, 16:23] = 1   # ends near (10,16): gap 4px, parallel
    lab[30, 5:12] = 1    # far away: must not link
    sys_id, n, _ = systems.label_systems(lab)
    assert n == 3
    strikes = discovery.system_strikes(sys_id, np.array([1, 2, 3]))
    corr = discovery.relay_corridors((40, 40), strikes, min_gap_px=3.0,
                                     max_gap_px=15.0)
    assert corr[10, 12:17].sum() > 0  # the step-over is bridged
    assert corr[20:40, :].sum() == 0  # nothing near the far system


def test_noisy_or_fuse_is_bounded_and_monotone():
    rng = np.random.default_rng(0)
    p = rng.random((20, 20))
    rays = (rng.random((20, 20)) < 0.1).astype(float)
    f = discovery.noisy_or_fuse(p, (rays, 0.5))
    assert f.shape == p.shape
    assert bool((f >= p - 1e-12).all())  # fusion only adds
    assert bool((f <= 1.0).all())
    f0 = discovery.noisy_or_fuse(p, (rays, 0.0))
    assert np.allclose(f0, p)  # zero weight = identity
