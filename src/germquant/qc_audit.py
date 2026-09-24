"""Mask audit (the `audit` stage; docs/ROADMAP_modular_pipeline.md step 7), ported from
analysis/coloc/scripts/qc_mask_audit.py. Two questions about the nuclei that feed the partition
coefficient, both answered inside the padded germline crop of the envelope stage:

  1. MISSED: lamin rings with no DAPI/Cellpose label at any depth. Nuclei are found from the lamin channel
     alone (ring -> filled interior per plane -> 3D components of 5 to 150 um3) and each is scored by the
     fraction of its voxels covered by a germline label (`cov_germ_label`), by any label, by the envelope.
  2. NO ENVELOPE: germline-flagged labels with no lamin ring (the envelope stage's ring test), listed
     with their scores.
Writes the `mask_audit` table (one row per candidate and per label) and, when render.montage is on, a
two-panel max-projection overlay ``<image_id>__mask_audit.png`` with all-depth mask edges (blue DAPI,
red lamin envelope, yellow = lamin nucleus with no label, magenta fill = label with no ring).

Named differences from qc_mask_audit.py: the script rounded coverages and ring scores to three decimals
BEFORE its gates (cov < 0.15, ring_ratio > 0.97, shell_over_thr < 0.75) and rounded the written values;
the stage gates and writes the unrounded values (nucleus_filter, which produced the published v4 and
zone numbers, was unrounded too), so a value within 0.0005 of a gate can be classed differently from
the August mask_audit CSVs. Volumes use the .nd2 voxel, not the 0.1083 constant (see envelope docs).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

log = logging.getLogger(__name__)

AUDIT_COLS = ["kind", "object_id", "vol_um3", "z_um", "y_um", "x_um", "cov_germ_label", "cov_any_label",
              "cov_envelope", "ring_shell", "ring_inside", "ring_ratio", "shell_over_thr", "no_envelope"]


def lamin_only_nuclei(lamin_c: np.ndarray, spacing, *, ring_smooth_um=0.25, close_um=0.35,
                      min_vol_um3=5.0, max_vol_um3=150.0):
    """Nuclei detected from the lamin channel alone. Returns (component labels, table of candidates)."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk

    sp = np.asarray(spacing, dtype=float)
    sm = ndi.gaussian_filter(np.asarray(lamin_c, dtype=np.float32), sigma=ring_smooth_um / sp)
    hi = sm[sm > np.percentile(sm, 50)]
    if hi.size == 0 or float(hi.max()) <= float(hi.min()):
        return np.zeros(lamin_c.shape, np.int32), pd.DataFrame(columns=["id", "vol_um3", "z_um", "y_um", "x_um"])
    thr = float(threshold_otsu(hi))
    ring = sm > thr
    se = disk(int(round(close_um / sp[1])))
    interior = np.zeros_like(ring)
    for z in range(ring.shape[0]):
        closed = ndi.binary_closing(ring[z], structure=se)
        interior[z] = ndi.binary_fill_holes(closed) & ~closed
    interior = ndi.binary_opening(interior, structure=np.ones((3, 3, 3), bool))
    cl, _ = ndi.label(interior)
    vvol = float(np.prod(sp))
    vol = np.bincount(cl.ravel()) * vvol
    keep = np.where((vol >= min_vol_um3) & (vol <= max_vol_um3))[0]
    keep = keep[keep != 0]
    rows = []
    for k in keep:
        cz, cy, cx = ndi.center_of_mass(cl == k)
        rows.append({"id": int(k), "vol_um3": float(vol[k]), "z_um": cz * sp[0], "y_um": cy * sp[1], "x_um": cx * sp[2]})
    return cl, pd.DataFrame(rows, columns=["id", "vol_um3", "z_um", "y_um", "x_um"])


def run_audit(labels: np.ndarray, germ_ids, lamin: np.ndarray, spacing, env: dict, *, ring_smooth_um=0.25,
              missed_cov_max=0.15) -> dict:
    """Build the mask_audit table from the envelope stage's result `env` (run_envelope output)."""
    sp = np.asarray(spacing, dtype=float)
    sl = env["crop"]
    lab_c = labels[sl]
    lam_c = np.asarray(lamin)[sl]
    ids = np.asarray([int(i) for i in germ_ids], dtype=np.int64)
    gn_c = np.isin(lab_c, ids)
    env_c = env["envelope_labels"][sl] > 0
    cl, cand = lamin_only_nuclei(lam_c, spacing, ring_smooth_um=ring_smooth_um)
    rows = []
    for r in cand.itertuples(index=False):
        m = cl == r.id
        rows.append({"kind": "lamin_candidate", "object_id": int(r.id), "vol_um3": r.vol_um3,
                     "z_um": r.z_um, "y_um": r.y_um, "x_um": r.x_um,
                     "cov_germ_label": float(gn_c[m].mean()), "cov_any_label": float((lab_c[m] > 0).mean()),
                     "cov_envelope": float(env_c[m].mean()),
                     "ring_shell": np.nan, "ring_inside": np.nan, "ring_ratio": np.nan, "shell_over_thr": np.nan,
                     "no_envelope": pd.NA})
    per = env["per_nucleus"]
    vvol = float(np.prod(sp))
    lvol = np.bincount(lab_c.ravel(), minlength=(ids.max() + 1) if ids.size else 1)
    coms = ndi.center_of_mass(gn_c, lab_c, ids) if ids.size else []
    sc = per.set_index("nucleus_id")
    rs = env.get("ring_scores")
    rs = rs.set_index("nucleus_id") if rs is not None and len(rs) else None
    for i, c in zip(ids, coms):
        rows.append({"kind": "germ_label", "object_id": int(i), "vol_um3": float(lvol[i] * vvol),
                     "z_um": c[0] * sp[0], "y_um": c[1] * sp[1], "x_um": c[2] * sp[2],
                     "cov_germ_label": np.nan, "cov_any_label": np.nan, "cov_envelope": np.nan,
                     "ring_shell": float(rs.at[i, "ring_shell"]) if rs is not None and i in rs.index else np.nan,
                     "ring_inside": float(rs.at[i, "ring_inside"]) if rs is not None and i in rs.index else np.nan,
                     "ring_ratio": float(sc.at[i, "ring_ratio"]) if i in sc.index else np.nan,
                     "shell_over_thr": float(sc.at[i, "shell_over_thr"]) if i in sc.index else np.nan,
                     "no_envelope": (not bool(sc.at[i, "has_envelope"])) if i in sc.index else pd.NA})
    table = pd.DataFrame(rows, columns=AUDIT_COLS)
    missed = table[(table["kind"] == "lamin_candidate") & (table["cov_germ_label"] < missed_cov_max)]
    missed_any = missed[missed["cov_any_label"] < missed_cov_max]
    summary = {"n_lamin_candidates": int(len(cand)), "n_missed_nuclei": int(len(missed)),
               "n_missed_unlabelled": int(len(missed_any))}
    return {"table": table, "summary": summary, "candidate_labels": cl,
            "missed_ids": [int(x) for x in missed["object_id"]]}


def write_audit_overlay(path, dapi_c: np.ndarray, lamin_c: np.ndarray, lab_c: np.ndarray, germ_ids,
                        env_c: np.ndarray, cand_labels: np.ndarray, missed_ids, no_env_ids, title: str = "") -> None:
    """Two-panel max-projection overlay with all-depth edges (blue DAPI, red envelope, yellow missed,
    magenta no-envelope fill)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    from .fsutil import long_path

    def norm(a):
        a = np.asarray(a, dtype=np.float32)
        lo, hi = np.percentile(a, 30), np.percentile(a, 99.7)
        return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

    ids = [int(i) for i in germ_ids]
    gn = np.isin(lab_c, ids)
    lam_mp, dapi_mp = norm(np.asarray(lamin_c).max(0)), norm(np.asarray(dapi_c).max(0))
    e_dapi = find_boundaries(gn.max(0), mode="inner")
    e_lam = find_boundaries((env_c > 0).max(0), mode="inner")
    miss = np.isin(cand_labels, list(missed_ids)).max(0) if len(missed_ids) else np.zeros(lam_mp.shape, bool)
    e_miss = find_boundaries(miss, mode="inner")
    junk = np.isin(lab_c, list(no_env_ids)).max(0) if len(no_env_ids) else np.zeros(lam_mp.shape, bool)
    jfill = junk & ~e_lam & ~e_dapi
    fig, ax = plt.subplots(1, 2, figsize=(20, 11), facecolor="black")
    for a, base, ttl in [(ax[0], lam_mp, "LMN-1 max projection"), (ax[1], dapi_mp, "DAPI max projection")]:
        rgb = np.stack([base, base, base], -1)
        rgb[e_dapi] = (0.2, 0.4, 1.0)
        rgb[e_lam] = (1.0, 0.15, 0.15)
        rgb[e_miss] = (1.0, 0.9, 0.0)
        rgb[jfill] = rgb[jfill] * 0.5 + np.array([0.5, 0.0, 0.5])
        a.imshow(rgb)
        a.axis("off")
        a.set_title(f"{ttl}: all-depth edges (blue DAPI, red lamin envelope); yellow = lamin nucleus with no "
                    f"label ({len(missed_ids)}); magenta fill = label with no lamin ring ({len(no_env_ids)})",
                    color="w", fontsize=9)
    fig.suptitle(title, color="w", fontsize=11)
    plt.tight_layout()
    fig.savefig(long_path(path), dpi=110, facecolor="black")
    plt.close(fig)
