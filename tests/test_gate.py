"""The submission gate must be HARD, and must catch the reported failure mode.

Production incident quoted in the task brief: the DrivenData form answered
"Predicted values must be in range [0, 1]" for a file whose every finite value was
inside [0, 1]. The cause is a NaN inside the scored footprint. These tests pin
that behaviour so it cannot regress.
"""

import numpy as np
import pytest
import rasterio

from gems10 import raster, spec

SAMPLE = "data/sample_submission.tif"


@pytest.fixture(scope="module")
def template_array():
    with rasterio.open(SAMPLE) as src:
        return src.profile.copy(), src.read(1)


def _write(arr, profile, path):
    p = profile.copy()
    p.update(count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype("float32"), 1)
    return path


def test_official_sample_submission_passes(tmp_path, template_array):
    rep = raster.check_submission(SAMPLE)
    assert rep.ok, rep.text()
    names = {c.name: c.ok for c in rep.checks}
    assert names["NAN-INSIDE-FOOTPRINT"] is True


def test_single_nan_inside_footprint_is_a_hard_failure(tmp_path, template_array):
    profile, base = template_array
    arr = base.copy()
    assert np.isfinite(arr[1800, 1600])
    arr[1800, 1600] = np.nan
    path = _write(arr, profile, tmp_path / "nan_inside.tif")

    rep = raster.check_submission(str(path))
    assert not rep.ok
    failed = {c.name: c for c in rep.failures}
    assert "NAN-INSIDE-FOOTPRINT" in failed
    assert rep.stats["nan_inside_footprint"] == 1
    # the finite values are all legal — this is the point of the incident
    assert {c.name: c.ok for c in rep.checks}["values-in-0-1"] is True


def test_assert_submittable_raises(tmp_path, template_array):
    profile, base = template_array
    arr = base.copy()
    # (10, 10) is outside the footprint, where NaN is legal; pick a scored pixel.
    idx = (1800, 1600)
    assert np.isfinite(arr[idx]), "fixture pixel must be inside the footprint"
    arr[idx] = np.nan
    path = _write(arr, profile, tmp_path / "nan2.tif")
    with pytest.raises(RuntimeError, match="NOT submittable"):
        raster.assert_submittable(str(path))


def test_nan_outside_footprint_is_accepted(tmp_path, template_array):
    """NaN where the template is NaN is the documented, legal case."""
    profile, base = template_array
    arr = base.copy()
    outside = tuple(int(v) for v in np.argwhere(~np.isfinite(base))[0])
    assert not np.isfinite(arr[outside])
    path = _write(arr, profile, tmp_path / "legal.tif")
    rep = raster.check_submission(str(path))
    assert rep.ok, rep.text()


def test_inf_inside_footprint_fails(tmp_path, template_array):
    profile, base = template_array
    arr = base.copy()
    arr[2000, 2000] = np.inf
    path = _write(arr, profile, tmp_path / "inf.tif")
    rep = raster.check_submission(str(path))
    assert not rep.ok
    assert rep.stats["inf_pixels"] == 1


def test_value_above_one_fails(tmp_path, template_array):
    profile, base = template_array
    arr = base.copy()
    arr[1900, 1700] = 1.2
    path = _write(arr, profile, tmp_path / "oor.tif")
    rep = raster.check_submission(str(path))
    assert not rep.ok
    assert "values-in-0-1" in {c.name for c in rep.failures}


def test_nan_only_outside_footprint_passes(tmp_path, template_array):
    profile, base = template_array
    path = _write(base, profile, tmp_path / "ok.tif")
    rep = raster.check_submission(str(path))
    assert rep.ok, rep.text()


def test_wrong_crs_is_caught(tmp_path, template_array):
    profile, base = template_array
    profile = dict(profile)
    from rasterio.crs import CRS
    profile["crs"] = CRS.from_epsg(4326)
    path = _write(base, profile, tmp_path / "wrong_crs.tif")
    rep = raster.check_submission(str(path))
    assert "crs-epsg32611" in {c.name for c in rep.failures}


def test_wrong_dtype_is_caught(tmp_path, template_array):
    profile, base = template_array
    p = dict(profile)
    p.update(count=1, dtype="float64", nodata=np.nan)
    path = tmp_path / "f64.tif"
    with rasterio.open(path, "w", **p) as dst:
        dst.write(base.astype("float64"), 1)
    rep = raster.check_submission(str(path))
    assert "dtype-float32" in {c.name for c in rep.failures}


def test_template_hash_is_verified(tmp_path, template_array):
    """A tampered template must be refused, not silently trusted."""
    profile, base = template_array
    bad_dir = tmp_path / "data"
    bad_dir.mkdir()
    arr = base.copy()
    _write(arr + 0.0, profile, bad_dir / "sample_submission.tif")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        raster.load_template_footprint(bad_dir / "sample_submission.tif")


def test_write_submission_refuses_non_finite(tmp_path, template_array):
    arr = np.zeros((spec.HEIGHT, spec.WIDTH), dtype=np.float32)
    arr[5, 5] = np.nan
    with pytest.raises(ValueError):
        raster.write_submission(arr, tmp_path / "x.tif", SAMPLE)


def test_write_submission_round_trips(tmp_path, template_array):
    arr = np.zeros((spec.HEIGHT, spec.WIDTH), dtype=np.float32)
    arr[100:110, 100:110] = 0.75
    out = raster.write_submission(arr, tmp_path / "rt.tif", SAMPLE)
    with rasterio.open(out) as src:
        back = src.read(1)
        assert src.crs.to_epsg() == spec.EPSG
        assert src.count == 1 and src.dtypes[0] == "float32"
    # written outside the footprint as 0.0 would violate the footprint rule, so
    # the writer is expected to be fed a footprint-shaped array; check the
    # values survived and the gate then reports the footprint mismatch honestly.
    assert back[105, 105] == pytest.approx(0.75)
    rep = raster.check_submission(str(out))
    assert "footprint-matches-official" in {c.name for c in rep.failures}
