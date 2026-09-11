"""Lamin-envelope nucleus masks, ring test and territories (see package docstring)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

log = logging.getLogger(__name__)

PAD_VOX = 30                 # germline bounding-box pad (voxels, every axis), as in the analysis scripts
DEFAULTS = {
    "band_um": 2.0,          # watershed search band beyond the DAPI nucleus
    "smooth_um": 0.15,       # lamin smoothing for the watershed elevation
    "volume_ratio_min": 1.0, # envelope / DAPI volume gate
    "volume_ratio_max": 3.0,
    "ring_smooth_um": 0.25,  # lamin smoothing for the ring test
    "ring_shell_um": 0.4,    # shell just outside the label for the ring test
    "ring_ratio_max": 0.97,  # shell / inside above this = no ring
    "shell_over_thr_max": 0.75,  # shell below this fraction of the Otsu ring level = no ring
    "territory_dilate_um": 4.0,
    "cyto_um": 2.5,          # cytoplasm shell width outside the envelope
}


def germline_crop(labels: np.ndarray, germ_ids, pad: int = PAD_VOX) -> tuple[slice, ...]:
    """Bounding box of the germline labels padded by `pad` voxels on every axis (clipped)."""
    gn = np.isin(labels, list(germ_ids))
    objs = ndi.find_objects(gn.astype(np.uint8))
    if not objs or objs[0] is None:
        return tuple(slice(0, d) for d in labels.shape)
    return tuple(slice(max(0, o.start - pad), min(dim, o.stop + pad)) for o, dim in zip(objs[0], labels.shape))


def envelope_labels(lab_c: np.ndarray, germ_ids, lamin: np.ndarray, spacing, *, band_um=2.0, smooth_um=0.15,
                    volume_ratio_min=1.0, volume_ratio_max=3.0) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-nucleus envelope labels via a seeded watershed on the smoothed lamin channel (crop frame).
    Returns (int32 label image, per-nucleus table: nucleus_id, envelope_volume_um3, envelope_vol_ratio,
    envelope_fallback). A nucleus whose watershed region is not 1.0 to 3.0 x its DAPI volume keeps its
    DAPI label and is flagged as a fallback."""
    from skimage.segmentation import watershed

    sp = np.asarray(spacing, dtype=float)
    ids = [int(i) for i in germ_ids]
    gn = np.isin(lab_c, ids)
    dt = ndi.distance_transform_edt(~gn, sampling=tuple(sp))
    elev = ndi.gaussian_filter(lamin.astype(np.float32), sigma=smooth_um / sp)
    markers = np.where(gn, lab_c, 0).astype(np.int32)
    bgl = int(lab_c.max()) + 1
    markers[dt > band_um] = bgl
    ws = watershed(elev, markers=markers, mask=(dt <= band_um) | gn | (dt > band_um))
    lam = np.where(ws == bgl, 0, ws).astype(np.int32)
    dv = np.bincount(np.where(gn, lab_c, 0).ravel(), minlength=bgl + 1)
    lv = np.bincount(lam.ravel(), minlength=bgl + 1)
    out = np.zeros_like(lam)
    vvol = float(np.prod(sp))
    rows = []
    for nid in ids:
        ratio = (lv[nid] / dv[nid]) if dv[nid] > 0 else float("nan")
        ok = dv[nid] > 0 and volume_ratio_min <= ratio <= volume_ratio_max
        if ok:
            out[lam == nid] = nid
            vol = lv[nid]
        else:
            out[(lab_c == nid) & gn] = nid
            vol = dv[nid]
        rows.append({"nucleus_id": nid, "envelope_volume_um3": float(vol * vvol),
                     "envelope_vol_ratio": float(ratio), "envelope_fallback": not ok})
    return out, pd.DataFrame(rows, columns=["nucleus_id", "envelope_volume_um3", "envelope_vol_ratio", "envelope_fallback"])


def ring_scores(lab_c: np.ndarray, germ_ids, lamin: np.ndarray, spacing, *, ring_smooth_um=0.25,
                ring_shell_um=0.4) -> tuple[pd.DataFrame, float]:
    """Per label: ring_ratio = mean smoothed lamin in a thin shell just outside the label over the mean
    inside; shell_over_thr = that shell mean over the crop's Otsu ring level. Returns (table, threshold)."""
    from skimage.filters import threshold_otsu

    sp = np.asarray(spacing, dtype=float)
    ids = np.asarray([int(i) for i in germ_ids], dtype=np.int64)
    gn = np.isin(lab_c, ids)
    sm = ndi.gaussian_filter(lamin.astype(np.float32), sigma=ring_smooth_um / sp)
    hi = sm[sm > np.percentile(sm, 50)]
    thr = float(threshold_otsu(hi)) if hi.size and float(hi.max()) > float(hi.min()) else float("nan")
    dt_out, idx = ndi.distance_transform_edt(~gn, sampling=tuple(sp), return_indices=True)
    shell = (dt_out > 0) & (dt_out <= ring_shell_um)
    near = lab_c[idx[0], idx[1], idx[2]]
    shell_lab = np.where(shell, near, 0)
    inside = np.asarray(ndi.mean(sm, np.where(gn, lab_c, 0), ids), dtype=float)
    outside = np.asarray(ndi.mean(sm, shell_lab, ids), dtype=float)
    df = pd.DataFrame({"nucleus_id": ids, "ring_shell": outside, "ring_inside": inside,
                       "ring_ratio": outside / np.maximum(inside, 1.0),
                       "shell_over_thr": outside / thr if np.isfinite(thr) and thr > 0 else np.nan})
    return df, thr


def no_envelope_ids(scores: pd.DataFrame, *, ring_ratio_max=0.97, shell_over_thr_max=0.75) -> list[int]:
    m = (scores["ring_ratio"] > ring_ratio_max) & (scores["shell_over_thr"] < shell_over_thr_max)
    return [int(i) for i in scores.loc[m, "nucleus_id"]]


def territories(lab_c: np.ndarray, germ_ids, spacing, *, territory_dilate_um=4.0) -> tuple[pd.DataFrame, int, np.ndarray]:
    """2D territories of the germline labels (max projection dilated by `territory_dilate_um`, connected
    components): per nucleus the territory its centroid lies in. Returns (table nucleus_id/territory_id,
    n_territories, the 2D territory label map). Which territory the hand-traced axis crosses is decided
    by the staging stage."""
    sp = np.asarray(spacing, dtype=float)
    ids = np.asarray([int(i) for i in germ_ids], dtype=np.int64)
    gn = np.isin(lab_c, ids)
    mp = gn.max(0)
    r = int(round(territory_dilate_um / sp[1]))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    comp, n = ndi.label(ndi.binary_dilation(mp, structure=(yy ** 2 + xx ** 2) <= r * r))
    coms = ndi.center_of_mass(gn, lab_c, ids) if ids.size else []
    rows = []
    for i, c in zip(ids, coms):
        yi = min(max(int(round(c[1])), 0), comp.shape[0] - 1)
        xi = min(max(int(round(c[2])), 0), comp.shape[1] - 1)
        rows.append({"nucleus_id": int(i), "territory_id": int(comp[yi, xi])})
    return pd.DataFrame(rows, columns=["nucleus_id", "territory_id"]), int(n), comp


def cytoplasm_shell(env_mask: np.ndarray, spacing, cyto_um=2.5) -> tuple[np.ndarray, np.ndarray]:
    """Cytoplasm within `cyto_um` of the envelope, holes filled, envelope excluded (the region the
    partition coefficient measures in). Returns (cyto mask, distance-to-envelope map)."""
    dt = ndi.distance_transform_edt(~env_mask, sampling=tuple(float(s) for s in spacing))
    cyto = ndi.binary_fill_holes(dt <= cyto_um) & ~env_mask
    return cyto, dt


def run_envelope(labels: np.ndarray, germ_ids, lamin: np.ndarray, spacing, params: dict | None = None) -> dict:
    """The envelope stage on a full label image: computes everything in the padded germline crop and
    returns full-size envelope labels plus per-nucleus and per-image results."""
    p = {**DEFAULTS, **(params or {})}
    ids = [int(i) for i in germ_ids]
    sl = germline_crop(labels, ids)
    lab_c = labels[sl]
    lam_c = np.asarray(lamin)[sl]
    env_c, per = envelope_labels(lab_c, ids, lam_c, spacing, band_um=p["band_um"], smooth_um=p["smooth_um"],
                                 volume_ratio_min=p["volume_ratio_min"], volume_ratio_max=p["volume_ratio_max"])
    scores, thr = ring_scores(lab_c, ids, lam_c, spacing, ring_smooth_um=p["ring_smooth_um"],
                              ring_shell_um=p["ring_shell_um"])
    junk = set(no_envelope_ids(scores, ring_ratio_max=p["ring_ratio_max"], shell_over_thr_max=p["shell_over_thr_max"]))
    terr, n_terr, terr_map = territories(lab_c, ids, spacing, territory_dilate_um=p["territory_dilate_um"])
    per = per.merge(scores[["nucleus_id", "ring_ratio", "shell_over_thr"]], on="nucleus_id", how="left")
    per = per.merge(terr, on="nucleus_id", how="left")
    per["has_envelope"] = ~per["nucleus_id"].isin(junk)
    # envelope centroids in whole-image microns (the staging analysis projected THESE onto the trace)
    sp = np.asarray(spacing, dtype=float)
    ids_present = [i for i in ids if (env_c == i).any()]
    coms = ndi.center_of_mass(env_c > 0, env_c, ids_present) if ids_present else []
    cent = pd.DataFrame([{"nucleus_id": int(i),
                          "envelope_centroid_z_um": (c[0] + sl[0].start) * sp[0],
                          "envelope_centroid_y_um": (c[1] + sl[1].start) * sp[1],
                          "envelope_centroid_x_um": (c[2] + sl[2].start) * sp[2]} for i, c in zip(ids_present, coms)],
                        columns=["nucleus_id", "envelope_centroid_z_um", "envelope_centroid_y_um", "envelope_centroid_x_um"])
    per = per.merge(cent, on="nucleus_id", how="left")
    env_full = np.zeros(labels.shape, np.int32)
    env_full[sl] = env_c
    vvol = float(np.prod(sp))
    junk_vol = float(per.loc[~per["has_envelope"], "envelope_volume_um3"].sum())
    total_vol = float(per["envelope_volume_um3"].sum())
    # the script's lam_vol_ratio: total envelope volume over total DAPI volume, fallbacks counted at 1.0
    # (a fallback row's envelope volume IS its DAPI volume; otherwise DAPI = envelope / ratio)
    dapi_vol = np.where(per["envelope_fallback"], per["envelope_volume_um3"],
                        per["envelope_volume_um3"] / per["envelope_vol_ratio"].replace(0, np.nan))
    dapi_total = float(np.nansum(dapi_vol))
    summary = {
        "n_envelope_fallback": int(per["envelope_fallback"].sum()),
        "envelope_vol_ratio": (total_vol / dapi_total) if dapi_total > 0 else float("nan"),
        "n_no_envelope": int(len(junk)),
        "no_envelope_vol_frac": (junk_vol / total_vol) if total_vol > 0 else float("nan"),
        "n_territories": n_terr,
        "ring_thr": thr,
    }
    return {"envelope_labels": env_full, "crop": sl, "per_nucleus": per, "summary": summary,
            "no_envelope_ids": sorted(junk), "voxel_volume_um3": vvol, "territory_map": terr_map,
            "ring_scores": scores}
