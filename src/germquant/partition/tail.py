"""Per-granule SYP-3 excess and the lit fraction (granule_tail.py v4, see package docstring)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .pc import CYTO_UM, DBINS_UM

TAIL_COLS = ["granule_id", "excess_n", "syp_excess", "pgl_raw", "pgl_n", "vol_um3", "dist_um", "dmin_um"]
TAIL_THRESHOLDS = (0.25, 0.5, 1.0)


def run_granule_tail(syp_c, pgl_c, env_labels_c, spacing, *, drop_ids=(), cyto_um=CYTO_UM, bins=DBINS_UM,
                     tophat_kw: dict | None = None, far_um=10.0) -> dict:
    """Rebuild the envelope set without `drop_ids` (nuclei with no lamin envelope), the cytoplasm shell
    and the tophat granules inside it, then the per-granule excess over the distance-matched cytoplasm
    in nuclear units. Returns {table: TAIL_COLS, summary: image_summary fields}."""
    from ..granule.segment import segment_granules_tophat

    sp = tuple(float(s) for s in spacing)
    lam = np.where(np.isin(env_labels_c, list(drop_ids)), 0, env_labels_c) if len(drop_ids) else env_labels_c
    env = lam > 0
    dt = ndi.distance_transform_edt(~env, sampling=sp)
    cyto = ndi.binary_fill_holes(dt <= cyto_um) & ~env
    syp = np.asarray(syp_c, dtype=np.float32)
    pgl = np.asarray(pgl_c, dtype=np.float32)
    lab2, gdf = segment_granules_tophat(pgl, cyto, spacing, **(tophat_kw or {}))
    nlab = int(lab2.max())
    empty = pd.DataFrame(columns=TAIL_COLS)
    base_summary = {"tail_n_granules": 0, "tail_n_no_envelope_dropped": int(len(drop_ids))}
    if nlab == 0:
        return {"table": empty, "summary": {**base_summary, **{f"tail_frac_excess_gt_{t}": np.nan for t in TAIL_THRESHOLDS},
                                            "tail_excess_p50": np.nan, "tail_excess_p90": np.nan, "tail_excess_p99": np.nan,
                                            "tail_nuclear_syp": np.nan}}
    gran = lab2 > 0
    far = dt > far_um
    bg = float(np.percentile(syp[far], 0.5)) if far.sum() > 1e4 else float(np.percentile(syp, 0.5))
    N = float(syp[env].mean()) - bg
    outside = cyto & ~gran
    binidx = np.clip(np.digitize(dt, bins) - 1, 0, len(bins) - 2)
    cyto_by_bin = np.array([float(syp[outside & (binidx == b)].mean()) - bg if (outside & (binidx == b)).sum() > 50 else np.nan
                            for b in range(len(bins) - 1)])
    idx = np.arange(1, nlab + 1)
    g_syp = np.asarray(ndi.mean(syp, lab2, idx)) - bg
    pbg = float(np.percentile(pgl[far], 0.5)) if far.sum() > 1e4 else float(np.percentile(pgl, 0.5))
    g_pgl = np.asarray(ndi.mean(pgl, lab2, idx)) - pbg
    g_bin = np.asarray(ndi.mean(binidx.astype(np.float32), lab2, idx))
    g_dist = np.asarray(ndi.mean(dt, lab2, idx))
    g_dmin = np.asarray(ndi.minimum(dt, lab2, idx))
    vvol = float(np.prod(sp))
    g_vol = np.bincount(lab2.ravel())[1:nlab + 1] * vvol
    g_bin_i = np.clip(np.round(g_bin).astype(int), 0, len(bins) - 2)
    local = cyto_by_bin[g_bin_i]
    excess = g_syp - local
    excess_n = excess / N if N > 0 else np.full_like(excess, np.nan)
    fin = np.isfinite(excess_n)
    pgl_scale = max(float(pgl[env].mean()) - pbg, 1.0)
    table = pd.DataFrame({"granule_id": idx[fin], "excess_n": excess_n[fin], "syp_excess": excess[fin],
                          "pgl_raw": g_pgl[fin], "pgl_n": (g_pgl / pgl_scale)[fin], "vol_um3": g_vol[fin],
                          "dist_um": g_dist[fin], "dmin_um": g_dmin[fin]}, columns=TAIL_COLS)
    e = excess_n[fin]
    summary = {**base_summary, "tail_n_granules": int(fin.sum()), "tail_nuclear_syp": N}
    for t in TAIL_THRESHOLDS:
        summary[f"tail_frac_excess_gt_{t}"] = round(float((e > t).mean()), 4) if e.size else np.nan
    for p in (50, 90, 99):
        summary[f"tail_excess_p{p}"] = round(float(np.percentile(e, p)), 4) if e.size else np.nan
    return {"table": table, "summary": summary}
