"""Polyline projection and zone assignment (verbatim geometry of trace_pachytene.py), plus the traces
file (``<image_id>: {points_um: [[x, y], ...] in WHOLE-IMAGE microns, status, ...}``)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..fsutil import long_path

OFF_AXIS_UM = 20.0          # nuclei farther than this from the drawn line are excluded
ZONE_NAMES = ("early", "mid", "late")
ZONE_COLS = ["nucleus_id", "zone", "s_um", "r_um", "is_pachytene", "in_territory", "off_axis_cut_um"]


def project_to_polyline(pts, poly):
    """pts (n,2) and poly (k,2) in um -> (s, r, L): arc-length along poly (unclamped at the ends, so s<0
    means before the start and s>L means beyond the end), perpendicular distance, polyline length."""
    pts = np.asarray(pts, float)
    poly = np.asarray(poly, float)
    seg = np.diff(poly, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    best_r = np.full(len(pts), np.inf)
    best_s = np.zeros(len(pts))
    for i in range(len(seg)):
        d = seg[i]
        L2 = max(seglen[i] ** 2, 1e-12)
        t = ((pts - poly[i]) @ d) / L2
        tc = np.clip(t, 0.0, 1.0)
        proj = poly[i] + tc[:, None] * d
        r = np.linalg.norm(pts - proj, axis=1)
        upd = r < best_r
        s = cum[i] + tc * seglen[i]
        if i == 0:
            s = np.where(t < 0, cum[i] + t * seglen[i], s)
        if i == len(seg) - 1:
            s = np.where(t > 1, cum[i] + t * seglen[i], s)
        best_s = np.where(upd, s, best_s)
        best_r = np.where(upd, r, best_r)
    return best_s, best_r, float(cum[-1])


def assign_zones(df: pd.DataFrame, poly, off_axis_um: float = OFF_AXIS_UM, adaptive: bool = True,
                 x_col: str = "cx", y_col: str = "cy"):
    """df needs the x/y columns (um, same frame as poly). Returns (df + s_um, r_um, zone,
    off_axis_cut_um, L). adaptive: cut = min(off_axis_um, 2.5 x median r of the nuclei inside the
    traced span) when at least 20 nuclei are inside; a fixed 20 um readmitted second-arm nuclei in
    narrow male gonads while hermaphrodite tubes genuinely reach about 18 um."""
    s, r, L = project_to_polyline(df[[x_col, y_col]].to_numpy(), poly)
    cut = float(off_axis_um)
    if adaptive:
        inside0 = (r <= off_axis_um) & (s >= 0) & (s <= L)
        if inside0.sum() >= 20:
            cut = float(min(off_axis_um, 2.5 * np.median(r[inside0])))
    zone = np.full(len(df), "off_axis", dtype=object)
    on = r <= cut
    f = s / max(L, 1e-9)
    zone[on & (s < 0)] = "pre"
    zone[on & (s > L)] = "post"
    inside = on & (s >= 0) & (s <= L)
    zone[inside & (f < 1 / 3)] = "early"
    zone[inside & (f >= 1 / 3) & (f < 2 / 3)] = "mid"
    zone[inside & (f >= 2 / 3)] = "late"
    out = df.copy()
    out["s_um"], out["r_um"], out["zone"] = s, r, zone
    out["off_axis_cut_um"] = round(cut, 2)
    return out, L


def load_traces(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    with open(long_path(p), encoding="utf-8") as fh:
        return json.load(fh)


def save_trace(path: str | Path, image_id: str, points_um, status: str, off_axis_um: float = OFF_AXIS_UM,
               frame: str = "whole_image_um") -> dict:
    """Store one trace (points in whole-image microns, [x, y] pairs) and return the file's content."""
    p = Path(path)
    tr = load_traces(p)
    tr[image_id] = {"points_um": [[round(float(x), 2), round(float(y), 2)] for x, y in points_um],
                    "status": status, "n_points": len(points_um), "off_axis_um": float(off_axis_um),
                    "frame": frame, "traced_at": time.strftime("%Y-%m-%d %H:%M")}
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(long_path(p), "w", encoding="utf-8") as fh:
        json.dump(tr, fh, indent=1)
    return tr


def territory_hits(territory_map: np.ndarray, poly_um, spacing, origin_vox=(0, 0)) -> set[int]:
    """Territory ids (2D label map in the envelope crop frame) crossed by the polyline (whole-image um)."""
    from itertools import pairwise

    sp = np.asarray(spacing, dtype=float)
    pts = np.asarray(poly_um, dtype=float)
    hit: set[int] = set()
    for a, b in pairwise(pts):
        for t in np.linspace(0, 1, 200):
            x, y = a + t * (b - a)
            yi, xi = int(round(y / sp[1])) - int(origin_vox[0]), int(round(x / sp[2])) - int(origin_vox[1])
            if 0 <= yi < territory_map.shape[0] and 0 <= xi < territory_map.shape[1] and territory_map[yi, xi] > 0:
                hit.add(int(territory_map[yi, xi]))
    return hit


def run_staging(nuclei: pd.DataFrame, trace: dict, spacing, *, env_ctx: dict | None = None,
                off_axis_um: float | None = None, adaptive: bool = True) -> dict:
    """Assign zones to the (germline) nuclei table from one trace. Returns {zones: ZONE_COLS table,
    summary: {...}, nuclei: nuclei with zone columns merged}."""
    poly = np.asarray(trace["points_um"], dtype=float)
    if poly.ndim != 2 or len(poly) < 2:
        raise ValueError("a trace needs at least two points")
    cut_um = float(off_axis_um if off_axis_um is not None else trace.get("off_axis_um", OFF_AXIS_UM))
    z, L = assign_zones(nuclei, poly, off_axis_um=cut_um, adaptive=adaptive,
                        x_col="centroid_x_um", y_col="centroid_y_um")
    z["is_pachytene"] = z["zone"].isin(ZONE_NAMES)
    z["in_territory"] = pd.NA
    n_terr_hit = None
    if env_ctx is not None and "territory_map" in env_ctx and "territory_id" in z.columns:
        sl = env_ctx["crop"]
        hits = territory_hits(env_ctx["territory_map"], poly, spacing, origin_vox=(sl[1].start, sl[2].start))
        if not hits:                              # polyline missed every territory: keep the largest
            tm = env_ctx["territory_map"]
            sizes = np.bincount(tm.ravel()); sizes[0] = 0
            hits = {int(np.argmax(sizes))} if sizes.size > 1 else set()
        z["in_territory"] = z["territory_id"].isin(hits)
        n_terr_hit = len(hits)
    zones = z[ZONE_COLS].copy()
    counts = z["zone"].value_counts()
    summary = {
        "n_zoned_nuclei": int(z["is_pachytene"].sum()),
        "pachytene_length_um": float(L),
        "off_axis_cut_um": float(z["off_axis_cut_um"].iloc[0]) if len(z) else float("nan"),
        "n_early": int(counts.get("early", 0)), "n_mid": int(counts.get("mid", 0)), "n_late": int(counts.get("late", 0)),
        "n_off_axis": int(counts.get("off_axis", 0)),
        "n_territories_on_trace": n_terr_hit if n_terr_hit is not None else float("nan"),
    }
    merged = nuclei.merge(zones, on="nucleus_id", how="left")
    return {"zones": zones, "summary": summary, "nuclei": merged, "length_um": L}
