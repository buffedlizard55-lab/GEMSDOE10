#!/usr/bin/env python3
"""Build the GEMSDOE10 derived-feature stack — streaming, tile by tile.

    python scripts/build_features.py --data-dir /tmp/gems10data --out feat.npy [--tile-rows 512]

Stack (107 channels, float32 memmap, row-major [H, W, C]):
  [0:19]   the 19 official bands, sentinel-masked to NaN (passthrough)
  [19:48]  29 baseline structural channels (ported from 6GEMSDOE: analytic
           signal, tilt, HGM, curvature set, structure-tensor lineaments s1/s2,
           strain/conductivity/seismicity cross products)
  [48:88]  40 extended multi-scale channels (ported from 6GEMSDOE: HGM at
           s={1.5,3,6} on tmi/rtp/mag_anom/grav/elev; smoothed VG + ASA + TDR
           at s={1.5,3} on tmi/grav; curvature at s={1.5,3}; local std s3;
           lineament tensor on cond + rtp s2)
  [88:107] 19 GEMSDOE10 lineament channels (NEW — Frangi vesselness at s={1.5,3}
           on tmi/rtp/grav/elev + tmi-s3 ridge orientation; Gabor orient-max at
           k={7,11} on tmi/elev + tmi-k11 strike; structure coherence at s4 on
           tmi/grav/elev)

Deliberately NOT ported: the 17 percentile-rank "agreement" channels — r6
measured them wash-to-negative at the shipping budget (0.1670 vs 0.1698), and
they need global rank tables. The label-derived `dist_to_catalogue` channel is
NOT stored here either: it must be recomputed per CV fold from train systems
only (see scripts/run_cv.py), otherwise it leaks.

Memory: tiles of full-width rows with a 40 px halo keep peak RAM ~1 GB.
Nothing produced here is committed (see .gitignore).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gems10 import features as F  # noqa: E402
from gems10 import spec  # noqa: E402
from gems10.raster import sha256_file  # noqa: E402

HALO = 40  # px; > max stencil reach (gauss 3*6+1=19, gabor 11//2+1=6, deriv chain ~22)


def channel_plan() -> list[str]:
    names = [n for n, _ in spec.FEATURE_BANDS]
    baseline = [
        "mag_asa", "mag_tilt", "tmi_hgm_computed", "tmi_asa_computed",
        "rtp_hgm_computed", "mag_anom_hgm_computed",
        "grav_asa", "grav_tilt", "grav_hgm_computed",
        "curv_total", "curv_profile", "curv_plan", "curv_gaussian",
        "slope_computed", "slope_of_slope", "det_elev_hgm_computed",
        "lin_elev_energy_s1", "lin_elev_coherence_s1",
        "lin_elev_cos2theta_s1", "lin_elev_sin2theta_s1",
        "lin_elev_energy_s2", "lin_elev_coherence_s2",
        "lin_tmi_energy_s2", "lin_tmi_coherence_s2",
        "lin_grav_energy_s2", "lin_grav_coherence_s2",
        "x_geod_shearrate__geod_dilaterate",
        "x_cond_surf__depth_to_base_surf",
        "x_ieq_n100a15__deq_n100a15",
    ]
    extended = []
    for src in ("tmi", "rtp", "mag_anom", "iso_grav_anom", "det_elev"):
        for sigma in (1.5, 3.0, 6.0):
            extended.append(f"hgm_{src}_s{sigma:g}")
    for src in ("tmi", "iso_grav_anom"):
        for sigma in (1.5, 3.0):
            extended.append(f"vg_{src}_s{sigma:g}")
    for src in ("tmi", "iso_grav_anom"):
        for sigma in (1.5, 3.0):
            extended.append(f"asa_{src}_s{sigma:g}")
            extended.append(f"tdr_{src}_s{sigma:g}")
    for sigma in (1.5, 3.0):
        extended.append(f"curv_total_s{sigma:g}")
        extended.append(f"curv_plan_s{sigma:g}")
        extended.append(f"slope_of_slope_s{sigma:g}")
    for src in ("tmi", "det_elev", "iso_grav_anom"):
        extended.append(f"std_{src}_s3")
    extended += ["lin_cond_energy_s2", "lin_cond_coherence_s2",
                 "lin_rtp_energy_s2", "lin_rtp_coherence_s2"]
    g10 = []
    for src in ("tmi", "rtp", "iso_grav_anom", "det_elev"):
        for sigma in (1.5, 3.0):
            g10.append(f"frangi_{src}_s{sigma:g}")
    g10 += ["frangi_tmi_s3_cos2theta", "frangi_tmi_s3_sin2theta"]
    for src in ("tmi", "det_elev"):
        for k in (7, 11):
            g10.append(f"gabor_{src}_k{k}")
    g10 += ["gabor_tmi_k11_cos2theta", "gabor_tmi_k11_sin2theta"]
    for src in ("tmi", "iso_grav_anom", "det_elev"):
        g10.append(f"coh_{src}_s4")
    assert len(names) == 19 and len(baseline) == 29 and len(extended) == 40
    assert len(g10) == 19, len(g10)
    return names + baseline + extended + g10


def derived(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """All derived channels for one tile (any tile size; NaN-propagating)."""
    out: dict[str, np.ndarray] = {}

    hg = bands["tmi_hg"]; vg = bands["tmi_vg"]
    ghg = bands["iso_grav_anom_hg"]; gvg = bands["iso_grav_anom_vg"]
    out["mag_asa"] = np.sqrt(hg * hg + vg * vg)
    out["mag_tilt"] = F.tilt_angle(vg, np.abs(hg))
    out["grav_asa"] = np.sqrt(ghg * ghg + gvg * gvg)
    out["grav_tilt"] = F.tilt_angle(gvg, np.abs(ghg))

    for src, name in (("tmi", "tmi_hgm_computed"), ("rtp", "rtp_hgm_computed"),
                      ("mag_anom", "mag_anom_hgm_computed"),
                      ("iso_grav_anom", "grav_hgm_computed"),
                      ("det_elev", "det_elev_hgm_computed")):
        gx, gy = F.derivatives(bands[src])
        out[name] = np.sqrt(gx * gx + gy * gy)
        del gx, gy

    gx, gy = F.derivatives(bands["tmi"])
    out["tmi_asa_computed"] = np.sqrt(gx * gx + gy * gy)
    del gx, gy

    curv = F.curvature(bands["det_elev"])
    out["curv_total"] = curv.total
    out["curv_profile"] = curv.profile
    out["curv_plan"] = curv.plan
    out["curv_gaussian"] = curv.gaussian
    out["slope_computed"] = curv.slope
    out["slope_of_slope"] = curv.slope_of_slope
    del curv

    for src_name, arr, factor in (("elev", bands["det_elev"], 1),
                                  ("elev", bands["det_elev"], 2),
                                  ("tmi", bands["tmi"], 2),
                                  ("grav", bands["iso_grav_anom"], 2)):
        ds = F.downsample(arr, factor)
        t = F.structure_tensor(ds, sigma=1.0, integration_sigma=2.0 * factor)
        suffix = f"_s{factor}"
        out[f"lin_{src_name}_energy{suffix}"] = t.energy
        out[f"lin_{src_name}_coherence{suffix}"] = t.coherence
        if factor == 1:
            out[f"lin_{src_name}_cos2theta{suffix}"] = np.cos(2 * t.orientation)
            out[f"lin_{src_name}_sin2theta{suffix}"] = np.sin(2 * t.orientation)
        del t, ds

    out["x_geod_shearrate__geod_dilaterate"] = (
        bands["geod_shearrate"] * bands["geod_dilaterate"])
    out["x_cond_surf__depth_to_base_surf"] = (
        bands["cond_surf"] * bands["depth_to_base_surf"])
    out["x_ieq_n100a15__deq_n100a15"] = (
        bands["ieq_n100a15"] * bands["deq_n100a15"])

    for src in ("tmi", "rtp", "mag_anom", "iso_grav_anom", "det_elev"):
        for sigma in (1.5, 3.0, 6.0):
            gx, gy = F.derivatives(F.nan_gaussian(bands[src], sigma))
            out[f"hgm_{src}_s{sigma:g}"] = np.sqrt(gx * gx + gy * gy)
            del gx, gy

    for src, vg_band in (("tmi", "tmi_vg"), ("iso_grav_anom", "iso_grav_anom_vg")):
        for sigma in (1.5, 3.0):
            smooth = F.nan_gaussian(bands[src], sigma)
            gx, gy = F.derivatives(smooth)
            hgm = np.sqrt(gx * gx + gy * gy)
            del gx, gy, smooth
            v = F.nan_gaussian(bands[vg_band], sigma)
            out[f"vg_{src}_s{sigma:g}"] = v
            out[f"asa_{src}_s{sigma:g}"] = np.sqrt(hgm * hgm + v * v)
            out[f"tdr_{src}_s{sigma:g}"] = F.tilt_angle(v, np.abs(hgm))
            del hgm, v

    for sigma in (1.5, 3.0):
        c = F.curvature_at_scale(bands["det_elev"], sigma)
        out[f"curv_total_s{sigma:g}"] = c.total
        out[f"curv_plan_s{sigma:g}"] = c.plan
        out[f"slope_of_slope_s{sigma:g}"] = c.slope_of_slope
        del c

    for src in ("tmi", "det_elev", "iso_grav_anom"):
        out[f"std_{src}_s3"] = F.local_std(bands[src], 3.0)

    for src_name, src in (("cond", bands["cond_surf"]), ("rtp", bands["rtp"])):
        ds = F.downsample(src, 2)
        t = F.structure_tensor(ds, sigma=1.0, integration_sigma=4.0)
        out[f"lin_{src_name}_energy_s2"] = t.energy
        out[f"lin_{src_name}_coherence_s2"] = t.coherence
        del t, ds

    # ================= GEMSDOE10 lineament channels =================
    for src in ("tmi", "rtp", "iso_grav_anom", "det_elev"):
        for sigma in (1.5, 3.0):
            fr = F.frangi_vesselness(bands[src], sigma)
            out[f"frangi_{src}_s{sigma:g}"] = fr.vesselness.astype(np.float32)
            if src == "tmi" and sigma == 3.0:
                out["frangi_tmi_s3_cos2theta"] = np.cos(2 * fr.orientation)
                out["frangi_tmi_s3_sin2theta"] = np.sin(2 * fr.orientation)
            del fr
    for src in ("tmi", "det_elev"):
        for k in (7, 11):
            e, o = F.gabor_max_response(bands[src], size=k, n_orientations=8)
            out[f"gabor_{src}_k{k}"] = e.astype(np.float32)
            if src == "tmi" and k == 11:
                out["gabor_tmi_k11_cos2theta"] = np.cos(2 * o)
                out["gabor_tmi_k11_sin2theta"] = np.sin(2 * o)
            del e, o
    for src in ("tmi", "iso_grav_anom", "det_elev"):
        t = F.structure_tensor(bands[src], sigma=1.5, integration_sigma=4.0)
        out[f"coh_{src}_s4"] = t.coherence
        del t

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    ap.add_argument("--out", default=None,
                    help="output .npy memmap (default: <data-dir>/features107.f32.npy)")
    ap.add_argument("--tile-rows", type=int, default=256)
    ap.add_argument("--channels", default=None,
                    help="optional comma list / prefix slice for smoke runs, e.g. '0:19'")
    args = ap.parse_args()

    t0 = time.time()
    data_dir = Path(args.data_dir)
    feat_path = data_dir / "training_features.tif"
    if not feat_path.exists():
        print(f"missing {feat_path} — run scripts/fetch_data.py first", file=sys.stderr)
        return 1
    out_path = Path(args.out) if args.out else data_dir / "features107.f32.npy"
    channels = channel_plan()
    if args.channels:
        lo, hi = (int(v) if v else None for v in args.channels.split(":"))
        channels = channels[slice(lo, hi)]
        print(f"SMOKE slice: {len(channels)} channels")

    names = [n for n, _ in spec.FEATURE_BANDS]
    mm = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.float32,
                                   shape=(spec.HEIGHT, spec.WIDTH, len(channels)))
    mm[:] = np.nan
    idx = {n: i for i, n in enumerate(channels)}
    have_raw = set(names) & set(idx)

    with rasterio.open(feat_path) as src:
        for r0 in range(0, spec.HEIGHT, args.tile_rows):
            r1 = min(r0 + args.tile_rows, spec.HEIGHT)
            w0 = max(0, r0 - HALO)
            w1 = min(spec.HEIGHT, r1 + HALO)
            window = rasterio.windows.Window(0, w0, spec.WIDTH, w1 - w0)
            bands: dict[str, np.ndarray] = {}
            for n in names:
                a = src.read(spec.BAND_INDEX[n], window=window).astype(np.float32)
                bands[n] = np.where(a < spec.FEATURE_INVALID_BELOW, np.nan, a)
            # Assemble the whole tile block in RAM, then a SINGLE memmap
            # assignment: per-channel strided writes across a 5 GB file thrashed
            # the page cache on a 3.8 GB host (measured 70x slowdown, 2026-09-27).
            lo = r0 - w0
            hi = lo + (r1 - r0)
            block = np.full((r1 - r0, spec.WIDTH, len(channels)), np.nan,
                            dtype=np.float32)
            for n in have_raw:
                block[:, :, idx[n]] = bands[n][lo:hi, :]
            if len(channels) > 19 or args.channels:
                der = derived(bands)
                for n, arr in der.items():
                    if n not in idx:
                        continue
                    a = np.asarray(arr, dtype=np.float32)
                    if a.shape != bands["tmi"].shape:
                        fy = bands["tmi"].shape[0] // a.shape[0]
                        fx = bands["tmi"].shape[1] // a.shape[1]
                        if fy and fx:
                            a = np.repeat(np.repeat(a, fy, axis=0), fx, axis=1)
                        a = a[:bands["tmi"].shape[0], :bands["tmi"].shape[1]]
                        pr = bands["tmi"].shape[0] - a.shape[0]
                        pc = bands["tmi"].shape[1] - a.shape[1]
                        if pr or pc:
                            a = np.pad(a, ((0, pr), (0, pc)),
                                       constant_values=np.nan)
                    block[:, :, idx[n]] = a[lo:hi, :]
                del der
            del bands
            mm[r0:r1] = block
            del block
            print(f"[{time.time()-t0:7.1f}s] rows {r0}-{r1} "
                  f"({100.0*r1/spec.HEIGHT:.0f}%)", flush=True)
    mm.flush()
    meta = {
        "path": str(out_path), "shape": [spec.HEIGHT, spec.WIDTH, len(channels)],
        "channels": channels, "tile_rows": args.tile_rows, "halo": HALO,
        "features_sha256": sha256_file(feat_path),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - t0, 1),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "channels"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
