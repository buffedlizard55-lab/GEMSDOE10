"""H27 corrected-phase continuation gate — synthetic controls.

The session-4 measurement: `structure_tensor.orientation` is the NORMAL to
the line. The as-built gate `clip(cos(2(o - strike)), 0, 1)` therefore passes
lineaments perpendicular to the system strike; the H27 'aligned' gate flips
the sign so lineaments PARALLEL to the strike pass — the preregistered
"orientation-matched" intent. Default behaviour must remain as-built.
"""
from __future__ import annotations

import numpy as np
import pytest

from gems10 import alignment
from gems10.discovery import Strike
from gems10.features import Lineament


def _system():
    return Strike(system=1, centroid=(32.0, 32.0), angle=0.0,
                  half_len=22.0, width=5.0, n_px=100,
                  end_a=(32.0, 10.0), end_b=(32.0, 54.0))


def _lineaments(orientation: float):
    shape = (64, 64)
    lin = Lineament(energy=np.full(shape, 2.0),
                    coherence=np.ones(shape),
                    orientation=np.full(shape, orientation))
    return {"tmi": lin}


def _run(orientation: float, gate_phase: str | None):
    rng = np.random.default_rng(0)
    band = rng.random((64, 64)).astype(np.float32)
    kwargs = {"gate_phase": gate_phase} if gate_phase is not None else {}
    grid, names = alignment.ray_continuation_channels(
        band, _lineaments(orientation), [_system()], ncc_band=band, **kwargs)
    return grid, names


def test_parallel_lineament_passes_only_aligned_phase():
    # strike = 0 (horizontal); a lineament PARALLEL to it has normal
    # orientation = 90 deg = pi/2.
    g_asbuilt, names = _run(np.pi / 2, None)          # default = asbuilt
    assert "cont_tmi_r5" in names
    i = names.index("cont_tmi_r5")
    assert np.nanmax(g_asbuilt[..., i]) == 0.0, "as-built must suppress parallel"
    g_aligned, _ = _run(np.pi / 2, "aligned")
    assert np.nanmax(g_aligned[..., i]) > 0.0, "aligned must pass parallel"


def test_perpendicular_lineament_passes_only_asbuilt_phase():
    # normal along the strike (orientation 0) => the lineament itself is
    # perpendicular to the system strike.
    i = alignment.ray_continuation_channels.__defaults__  # noqa: F841 (sanity)
    g_asbuilt, names = _run(0.0, "asbuilt")
    i = names.index("cont_tmi_r5")
    assert np.nanmax(g_asbuilt[..., i]) > 0.0
    g_aligned, _ = _run(0.0, "aligned")
    assert np.nanmax(g_aligned[..., i]) == 0.0


def test_default_is_asbuilt():
    g_default, _ = _run(np.pi / 3, None)
    g_asbuilt, _ = _run(np.pi / 3, "asbuilt")
    np.testing.assert_array_equal(
        np.nan_to_num(g_default, nan=-1), np.nan_to_num(g_asbuilt, nan=-1))


def test_unknown_gate_phase_fails_fast():
    with pytest.raises(ValueError, match="gate_phase"):
        _run(0.0, "rotated")
