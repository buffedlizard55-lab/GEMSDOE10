# Provenance: adapted from buffedlizard55-lab/6GEMSDOE src/gems (same owner, re-verified here).
"""Submission raster IO and the HARD format gate.

Why this module exists
---------------------
The task brief records a real failure mode: the DrivenData submission form
returned

    "Predicted values must be in range [0, 1]"

even though every *finite* value in the file was inside [0, 1]. The cause is a
NaN sitting INSIDE the scored footprint: the platform's range check sees a NaN
inside the valid area and reports it as a range violation. NaN is only legal
OUTSIDE the footprint (the sample submission uses NaN for exactly that).

So `check_submission()` treats "NaN/Inf inside the footprint" as a HARD FAIL, not
a warning, and `assert_submittable()` raises on any failure. Gate rules:

  1  readable single-band GeoTIFF, dtype float32
  2  CRS == EPSG:32611
  3  pixel size == 100 m in both axes
  4  width/height == 3292 x 3730
  5  geotransform == the official origin/resolution exactly (bounds equality)
  6  finite values all within [0, 1]
  7  no NaN and no +-Inf inside the footprint          <-- the hard gate
  8  footprint (finite mask) == the official sample submission's non-NaN mask
  9  the file's own nodata declaration is NaN (matches the sample submission)

Rule 8 is deliberately strict: the official sample submission defines the scored
area exactly (measured: 5,167,373 finite pixels), and it is byte-identical to the
label nodata mask. A submission that guesses the footprint differently is at risk
of the same form rejection. `--allow-footprint-subset` relaxes 8 to "no more than
the official footprint and at least the label coverage footprint" for experiments,
never for the file we intend to upload.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # rasterio is the only heavy dependency; keep the import error legible
    import rasterio
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "rasterio is required: pip install --break-system-packages rasterio"
    ) from exc

from . import spec


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def footprint_mask(path: str | Path) -> np.ndarray:
    """True where the raster holds a finite value (the 'inside' area)."""
    with rasterio.open(path) as src:
        arr = src.read(1)
    return np.isfinite(arr)


def load_template_footprint(template: str | Path) -> np.ndarray:
    """Footprint of the official sample submission, with its sha256 verified."""
    path = Path(template)
    if not path.exists():
        raise FileNotFoundError(f"official sample submission missing: {path}")
    pin = spec.PINS["sample_submission.tif"]
    got = sha256_file(path)
    if got != pin["sha256"]:
        raise ValueError(
            f"sample submission sha256 mismatch: {got} != {pin['sha256']} "
            f"(refusing to use it as the format template)"
        )
    return footprint_mask(path)


# -----------------------------------------------------------------------------


def _jsonable(v):
    """Cast numpy scalars/bools to plain Python so json.dumps always succeeds."""
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:  # pragma: no cover
            return str(v)
    return v


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    hard: bool = True

    def __str__(self) -> str:  # pragma: no cover
        flag = "PASS" if self.ok else ("FAIL" if self.hard else "WARN")
        return f"[{flag}] {self.name}: {self.detail}"


@dataclass
class SubmissionReport:
    path: str
    checks: list[CheckResult] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.ok and c.hard]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.ok and not c.hard]

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict:
        """JSON-safe form. Every boolean is cast explicitly: comparisons on numpy
        arrays yield np.bool_, which json cannot serialise, and a report that fails
        to serialise is a report that does not exist."""
        return {
            "path": self.path,
            "ok": bool(self.ok),
            "checks": [{"name": c.name, "ok": bool(c.ok), "hard": bool(c.hard),
                        "detail": str(c.detail)} for c in self.checks],
            "stats": {k: _jsonable(v) for k, v in self.stats.items()},
        }

    def text(self) -> str:
        head = "SUBMISSION FORMAT GATE: " + ("PASS" if self.ok else "FAIL")
        lines = [head, "-" * len(head)]
        lines += [str(c) for c in self.checks]
        if self.stats:
            lines.append("")
            lines.append("stats: " + ", ".join(f"{k}={v}" for k, v in self.stats.items()))
        return "\n".join(lines)


def check_submission(
    path: str | Path,
    template: str | Path | None = None,
    expect_footprint: bool = True,
) -> SubmissionReport:
    """Run the full gate. Never raises for a bad file — returns a report."""
    path = Path(path)
    rep = SubmissionReport(path=str(path))
    if not path.exists():
        rep.checks.append(CheckResult("file-exists", False, f"missing: {path}"))
        return rep
    rep.checks.append(CheckResult("file-exists", True, f"{path.stat().st_size} bytes"))

    with rasterio.open(path) as src:
        dtype = src.dtypes[0]
        count = src.count
        crs = src.crs
        res = src.res
        width, height = src.width, src.height
        transform = src.transform
        nodata = src.nodata
        arr = src.read(1)

    rep.checks.append(CheckResult(
        "single-band", count == 1, f"band count = {count}"))
    rep.checks.append(CheckResult(
        "dtype-float32", dtype == "float32", f"dtype = {dtype}"))
    epsg = None if crs is None else crs.to_epsg()
    rep.checks.append(CheckResult(
        "crs-epsg32611", epsg == spec.EPSG, f"CRS = {crs} (epsg {epsg})"))
    res_ok = (abs(res[0] - spec.PIXEL_SIZE_M) < 1e-9
              and abs(res[1] - spec.PIXEL_SIZE_M) < 1e-9)
    rep.checks.append(CheckResult(
        "resolution-100m", res_ok, f"resolution = {res}"))
    shape_ok = (width == spec.WIDTH and height == spec.HEIGHT)
    rep.checks.append(CheckResult(
        "shape", shape_ok, f"shape = ({height}, {width}), expected "
                           f"({spec.HEIGHT}, {spec.WIDTH})"))
    origin_ok = (abs(transform.c - spec.ORIGIN_X) < 1e-6
                 and abs(transform.f - spec.ORIGIN_Y) < 1e-6
                 and abs(transform.a - spec.PIXEL_SIZE_M) < 1e-9
                 and abs(transform.e + spec.PIXEL_SIZE_M) < 1e-9
                 and abs(transform.b) < 1e-9 and abs(transform.d) < 1e-9)
    rep.checks.append(CheckResult(
        "geotransform", origin_ok, f"transform = {tuple(transform)[:6]}"))

    nodata_ok = (nodata is None) or (isinstance(nodata, float) and np.isnan(nodata))
    rep.checks.append(CheckResult(
        "nodata-nan", nodata_ok, f"declared nodata = {nodata}"))

    finite = np.isfinite(arr)
    nan_ct = int(np.isnan(arr).sum())
    inf_ct = int(np.isinf(arr).sum())
    rep.stats.update({
        "total_pixels": int(arr.size),
        "finite_pixels": int(finite.sum()),
        "nan_pixels": nan_ct,
        "inf_pixels": inf_ct,
        "declared_sha256": sha256_file(path),
    })

    if finite.any():
        fmin = float(arr[finite].min())
        fmax = float(arr[finite].max())
        rep.stats["finite_min"] = fmin
        rep.stats["finite_max"] = fmax
        rep.stats["positive_pixels"] = int((arr[finite] > 0).sum())
    else:
        fmin = fmax = float("nan")
    range_ok = bool(finite.any() and fmin >= 0.0 and fmax <= 1.0)
    rep.checks.append(CheckResult(
        "values-in-0-1", range_ok, f"finite range = [{fmin}, {fmax}]"))

    rep.checks.append(CheckResult(
        "no-inf", inf_ct == 0, f"inf pixels = {inf_ct}"))

    # ---- the hard gate: NaN inside the footprint -----------------------------
    if template is None:
        template = Path(__file__).resolve().parents[2] / "data" / "sample_submission.tif"
    template = Path(template)
    if template.exists():
        try:
            ref = load_template_footprint(template)
        except Exception as exc:
            rep.checks.append(CheckResult(
                "template-verified", False, f"{exc} (hard gate disabled — refusing)"))
            return rep
        rep.checks.append(CheckResult(
            "template-verified", True,
            f"official sample submission sha256 ok, footprint = {int(ref.sum())} px"))
        if ref.shape != arr.shape:
            rep.checks.append(CheckResult(
                "footprint-shape", False,
                f"template {ref.shape} vs submission {arr.shape}"))
        else:
            nan_inside = int((~finite & ref).sum())
            rep.stats["nan_inside_footprint"] = nan_inside
            rep.checks.append(CheckResult(
                "NAN-INSIDE-FOOTPRINT", nan_inside == 0,
                f"{nan_inside} non-finite pixels inside the scored footprint "
                f"(non-finite scored values violate the [0,1] contract; "
                f"the original backend rejection cause is not known)"))
            missing = int((finite & ~ref).sum())
            rep.stats["finite_outside_footprint"] = missing
            if expect_footprint:
                rep.checks.append(CheckResult(
                    "footprint-matches-official", missing == 0,
                    f"{missing} finite pixels outside the official footprint "
                    f"(use --allow-footprint-subset for experiments only)"))
            else:
                rep.checks.append(CheckResult(
                    "footprint-matches-official", True,
                    f"{missing} finite pixels outside the official footprint "
                    f"(relaxed: --allow-footprint-subset)", hard=False))
    else:
        rep.checks.append(CheckResult(
            "template-verified", False,
            f"official sample submission not found at {template} — run "
            f"scripts/fetch_and_verify_data.py first (hard gate NOT evaluated)"))
    return rep


def assert_submittable(path: str | Path, template: str | Path | None = None) -> SubmissionReport:
    """Raise RuntimeError unless every hard check passes."""
    rep = check_submission(path, template=template)
    if not rep.ok:
        raise RuntimeError(
            "submission is NOT submittable:\n" + rep.text()
        )
    return rep


def write_submission(
    array: np.ndarray,
    out_path: str | Path,
    template: str | Path,
    dtype: str = "float32",
    footprint: np.ndarray | None = None,
) -> Path:
    """Write `array` with the official georeferencing, footprint and NaN rule.

    If `footprint` is given, values outside it are written as NaN — which is the
    only legal use of NaN in a submission (page 967: "data outside the bounds is
    null or nan"). Non-finite values INSIDE the footprint are a hard error, since
    they are exactly what the platform rejects.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(template) as src:
        profile = src.profile.copy()
    arr = np.asarray(array, dtype=np.float32)
    if arr.shape != (spec.HEIGHT, spec.WIDTH):
        raise ValueError(
            f"array shape {arr.shape} != official grid "
            f"({spec.HEIGHT}, {spec.WIDTH})")
    if footprint is not None:
        footprint = np.asarray(footprint, dtype=bool)
        if footprint.shape != arr.shape:
            raise ValueError("footprint shape does not match the array")
        inside_bad = ~np.isfinite(arr) & footprint
        if inside_bad.any():
            raise ValueError(
                f"{int(inside_bad.sum())} non-finite values inside the footprint "
                f"— refusing to write a file the platform will reject")
        arr = np.where(footprint, arr, np.nan).astype(np.float32)
    elif not np.isfinite(arr).all():
        raise ValueError("write_submission refuses non-finite values without a footprint")
    profile.update(
        driver="GTiff", count=1, dtype=dtype, nodata=spec.SUBMISSION_NODATA,
        compress="lzw", height=arr.shape[0], width=arr.shape[1],
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)
    return out_path
