"""H12: odd topographic steps, not another symmetric line/ridge detector.

Scientific motivation and preregistration: HYPOTHESES.md. This is a morphology
feature, NOT a validated fault or geothermal-resource classifier.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

RADII = (2, 4)
CHANNELS = [f"scarp_{kind}_r{r}" for r in RADII
            for kind in ("step", "odd_fraction", "polarity")]


def profile_kernels(radius: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Odd step and even valley templates with zero constant/planar response.

    theta is the cross-profile normal. All kernels have unit L1 norm. Least
    squares projection removes affine relief exactly on the discrete stencil.
    """
    if radius < 2:
        raise ValueError("radius must be >= 2")
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    u = x * np.cos(theta) + y * np.sin(theta)
    v = -x * np.sin(theta) + y * np.cos(theta)
    weight = np.exp(-(u * u + v * v) / (2 * (radius / 1.5) ** 2))
    design = np.column_stack([np.ones(x.size), x.ravel(), y.ravel()])
    kernels = []
    for template in (np.tanh(u * 2), u * u):
        raw = (weight * template).ravel()
        raw -= design @ np.linalg.lstsq(design, raw, rcond=None)[0]
        raw /= np.abs(raw).sum()
        kernels.append(raw.reshape(x.shape))
    return tuple(kernels)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    # Integer sample at (row+dy,col+dx); no wrap, no interpolation.
    return ndimage.shift(a, (-dy, -dx), order=0, mode="constant", cval=0,
                         prefilter=False)


def features(elevation: np.ndarray) -> np.ndarray:
    """Six local channels, orientation selected by strongest odd response.

    No labels, global percentiles, or dataset-dependent normalization. Invalid
    stencils and image edges yield NaN; tree models can handle missing channels.
    """
    a = np.asarray(elevation, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("elevation must be 2D")
    bad = ~np.isfinite(a)
    filled = np.where(bad, 0.0, a)
    output = []
    for radius in RADII:
        best = np.zeros(a.shape)
        even_best = np.zeros(a.shape)
        polarity_best = np.zeros(a.shape)
        for theta in np.arange(8) * np.pi / 8:
            odd, even = profile_kernels(radius, theta)
            response = ndimage.correlate(filled, odd, mode="constant")
            even_response = np.abs(ndimage.correlate(filled, even, mode="constant"))
            dy, dx = int(round(2 * np.cos(theta))), int(round(-2 * np.sin(theta)))
            r1, r2 = _shift(response, dy, dx), _shift(response, -dy, -dx)
            persistence = np.abs(response + r1 + r2) / (
                np.abs(response) + np.abs(r1) + np.abs(r2) + 1e-10)
            amplitude = np.abs(response)
            better = amplitude > best
            best[better] = amplitude[better]
            even_best[better] = even_response[better]
            polarity_best[better] = persistence[better]
        halo = radius + 2
        invalid = ndimage.maximum_filter(bad.astype(np.uint8), size=2 * halo + 1,
                                         mode="constant", cval=1).astype(bool)
        contrast = best / (best + even_best + 1e-10)
        for channel in (best, contrast, polarity_best):
            output.append(np.where(invalid, np.nan, channel).astype(np.float32))
    return np.stack(output, axis=-1)
