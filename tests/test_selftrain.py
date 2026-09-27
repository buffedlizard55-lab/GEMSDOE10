"""Self-training utilities: family agreement + pseudo selection."""

import numpy as np

from gems10 import selftrain


def test_channel_families_cover_named_channels():
    channels = ["tmi", "iso_grav_anom", "geod_shearrate", "ieq_n100a15",
                "cond_surf", "det_elev", "frangi_tmi_s3", "x_geod_shearrate__x"]
    fams = selftrain.channel_families(channels)
    assert set(fams) == {"mag", "grav", "strain", "seis", "cond", "topo"}
    assert 0 in fams["mag"] and 6 in fams["mag"]
    assert 2 in fams["strain"] and 7 in fams["strain"]


def test_family_agreement_counts_hot_families():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 6))
    X[:50, 0] += 5.0  # mag hot
    X[:50, 2] += 5.0  # strain hot
    agree = selftrain.family_agreement(
        X, ["tmi", "iso_grav_anom", "geod_shearrate", "ieq_n100a15",
            "cond_surf", "det_elev"], z_thresh=2.0)
    assert float(agree[:50].mean()) >= 1.5
    assert float(agree[50:].mean()) < 0.5


def test_select_pseudo_takes_top_fraction_of_eligible():
    rng = np.random.default_rng(3)
    prob = rng.random((40, 40))
    elig = np.ones((40, 40), dtype=bool)
    elig[0:5, :] = False
    sel = selftrain.select_pseudo(prob, elig, None, top_frac=0.05)
    assert abs(sel.sum() / elig.sum() - 0.05) < 0.01
    assert not sel[0:5, :].any()
    agree = np.zeros((40, 40), dtype=np.int8)
    agree[20:, :] = 3
    sel2 = selftrain.select_pseudo(prob, elig, agree, top_frac=0.05, min_agree=2)
    assert not sel2[:20, :].any()  # agreement gate enforced
