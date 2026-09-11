"""Distance-matched partition coefficient with rotation null and z-shift floor (see package docstring)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

CYTO_UM = 2.5
DBINS_UM = np.arange(0.0, CYTO_UM + 1e-6, 0.25)
DZ_PLANES = 6
WHOLE_GATES = {"min_gran": 50, "min_outside": 500, "per_bin_min_gran": 20, "per_bin_min_outside": 50}
ZONE_GATES = {"min_gran": 30, "min_outside": 200, "per_bin_min_gran": 15, "per_bin_min_outside": 40}
PARTITION_COLS = ["region", "n_granules", "granule_voxels", "outside_voxels", "region_voxels", "bg",
                  "partition_coef", "partition_coef_rot", "partition_coef_zshift", "partition_coef_specific"]


def pc_dm(syp, gran, outside, dt, bg, *, bins=DBINS_UM, region=None, min_gran=50, min_outside=500,
          per_bin_min_gran=20, per_bin_min_outside=50):
    """Granule-voxel-weighted mean over distance bins of (SYP_gran - bg) / (SYP_outside_same_bin - bg).
    None when the region holds too few granule or outside voxels (the workers' gates)."""
    g0 = gran & region if region is not None else gran
    o0 = outside & region if region is not None else outside
    if int(g0.sum()) < min_gran or int(o0.sum()) < min_outside:
        return None
    num = wsum = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = (dt > lo) & (dt <= hi)
        g = g0 & b
        o = o0 & b
        ng, no = int(g.sum()), int(o.sum())
        if ng < per_bin_min_gran or no < per_bin_min_outside:
            continue
        den = float(syp[o].mean()) - bg
        if den <= 0:
            continue
        num += ng * ((float(syp[g].mean()) - bg) / den)
        wsum += ng
    return round(num / wsum, 4) if wsum > 0 else None


def rotation_null(gran, cyto):
    """The granule mask rotated 180 degrees in-plane: same mask, elsewhere in the cytoplasm."""
    return np.rot90(gran, 2, axes=(1, 2)) & cyto


def zshift_mask(gran, cyto, dz=DZ_PLANES):
    """The granule mask shifted `dz` planes: the same blur from the nucleus, no granule underneath."""
    gz = np.zeros_like(gran)
    gz[dz:] = gran[:-dz]
    return gz & cyto


WHOLE_MIN_CYTO_VOXELS = 5000      # pc_lamin_worker.metric_set: no whole-gonad PC on a tiny shell
WHOLE_MIN_GRANULE_OBJECTS = 40    # ... or with fewer than 40 granule objects (zone rows are not gated)


def partition_metrics(syp, gran, cyto, dt, bg, *, region=None, gates=WHOLE_GATES, dz=DZ_PLANES,
                      with_rotation=True, bins=DBINS_UM) -> dict:
    """PC, rotation null, z-shift floor and PC / z-shift for one region (None = whole shell)."""
    outside = cyto & ~gran
    dm = pc_dm(syp, gran, outside, dt, bg, bins=bins, region=region, **gates)
    rot = None
    if with_rotation:
        rg = rotation_null(gran, cyto)
        rot = pc_dm(syp, rg, cyto & ~rg, dt, bg, bins=bins, region=region, **gates)
    gz = zshift_mask(gran, cyto, dz)
    zsh = pc_dm(syp, gz, cyto & ~gz, dt, bg, bins=bins, region=region, **gates)
    spec = round(dm / zsh, 4) if (dm is not None and zsh is not None and zsh > 0) else None
    reg = region if region is not None else np.ones(gran.shape, bool)
    return {"partition_coef": dm, "partition_coef_rot": rot, "partition_coef_zshift": zsh,
            "partition_coef_specific": spec, "granule_voxels": int((gran & reg).sum()),
            "outside_voxels": int((outside & reg).sum()), "region_voxels": int(reg.sum())}


def zone_voxel_map(env_labels_c, zones: pd.DataFrame, spacing) -> np.ndarray:
    """Zone code per voxel (0 none, 1 early, 2 mid, 3 late): a voxel inherits the zone of the nucleus
    whose envelope is nearest (EDT indices), the pc_zone_worker convention."""
    env = env_labels_c > 0
    _dt, inds = ndi.distance_transform_edt(~env, sampling=tuple(float(s) for s in spacing), return_indices=True)
    nn = env_labels_c[tuple(inds)]
    maxl = int(env_labels_c.max())
    lut = np.zeros(maxl + 1, np.int8)
    code = {"early": 1, "mid": 2, "late": 3}
    for nid, z in zip(zones["nucleus_id"].astype(int), zones["zone"]):
        if 0 <= nid <= maxl and z in code:
            lut[nid] = code[z]
    return lut[nn]


def run_partition(syp_c, gran_c, env_mask_c, spacing, *, cyto_um=CYTO_UM, bg_percentile=3.0,
                  env_labels_c=None, zones: pd.DataFrame | None = None, dz=DZ_PLANES,
                  precomputed: dict | None = None, whole_min_cyto_voxels=WHOLE_MIN_CYTO_VOXELS,
                  whole_min_granule_objects=WHOLE_MIN_GRANULE_OBJECTS) -> dict:
    """The partition stage on the padded crop. Returns {table: PARTITION_COLS rows (whole + zones +
    pach when zones are given), summary: image_summary fields, bg, gated: reason or None}. The whole
    row carries the worker's gonad-level gates (shell under `whole_min_cyto_voxels` voxels or fewer than
    `whole_min_granule_objects` granule objects: no whole PC); zone rows are ungated, as in the zone
    worker. Distance bins follow the shell width (0.25 um steps up to `cyto_um`)."""
    if precomputed and "cyto" in precomputed and "dt" in precomputed:
        cyto, dt = precomputed["cyto"], precomputed["dt"]
    else:
        dt = ndi.distance_transform_edt(~env_mask_c, sampling=tuple(float(s) for s in spacing))
        cyto = ndi.binary_fill_holes(dt <= cyto_um) & ~env_mask_c
    bins = np.arange(0.0, float(cyto_um) + 1e-6, 0.25)
    syp = np.asarray(syp_c, dtype=np.float32)
    gran = np.asarray(gran_c, dtype=bool) & cyto
    bg = float(np.percentile(syp, bg_percentile))
    n_gran_objects = int(ndi.label(gran)[1])
    rows = []
    whole = partition_metrics(syp, gran, cyto, dt, bg, region=None, gates=WHOLE_GATES, dz=dz, bins=bins)
    gated = None
    if int(cyto.sum()) < whole_min_cyto_voxels:
        gated = f"cyto_voxels={int(cyto.sum())}<{whole_min_cyto_voxels}"
    elif n_gran_objects < whole_min_granule_objects:
        gated = f"granule_objects={n_gran_objects}<{whole_min_granule_objects}"
    if gated:
        for k in ("partition_coef", "partition_coef_rot", "partition_coef_zshift", "partition_coef_specific"):
            whole[k] = None
    rows.append({"region": "whole", "n_granules": n_gran_objects, "bg": bg, **whole})
    summary = {"partition_coef_whole": whole["partition_coef"], "partition_coef_rot_whole": whole["partition_coef_rot"],
               "partition_coef_zshift_whole": whole["partition_coef_zshift"],
               "partition_coef_specific_whole": whole["partition_coef_specific"], "partition_bg": bg}
    if zones is not None and env_labels_c is not None and len(zones):
        zv = zone_voxel_map(env_labels_c, zones, spacing)
        for name, sel in (("early", zv == 1), ("mid", zv == 2), ("late", zv == 3), ("pach", zv > 0)):
            m = partition_metrics(syp, gran, cyto, dt, bg, region=sel, gates=ZONE_GATES, dz=dz,
                                  with_rotation=False, bins=bins)
            rows.append({"region": name, "n_granules": int(ndi.label(gran & sel)[1]), "bg": bg, **m})
            summary[f"partition_coef_{name}"] = m["partition_coef"]
            summary[f"partition_coef_specific_{name}"] = m["partition_coef_specific"]
    table = pd.DataFrame(rows, columns=PARTITION_COLS)
    return {"table": table, "summary": summary, "bg": bg, "cyto": cyto, "dt": dt, "gated": gated}
