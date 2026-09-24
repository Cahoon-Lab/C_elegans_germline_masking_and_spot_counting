"""Convert the August 2026 hand traces (analysis/coloc/staging/pachytene_traces.json, points in CROP
microns: crop pixel x 0.1083 um, crop = germline bounding box of the labels padded by 30 voxels) into
whole-image microns for the pipeline's staging stage.

    python scripts/migrate_traces.py [--out analysis/coloc/staging/pachytene_traces_whole_image.json]

For each traced gonad the run folder (nuclei labels TIF + nuclei.csv) is located under the July/August
result trees, the crop offset re-derived exactly as crescent_axis.load_crops did (in_germline labels,
PAD 30), and every point mapped as: crop_px = p / 0.1083; whole_um = (crop_px + offset_px) x the real
voxel size from nuclei.csv (0.108333 um, not the 0.1083 constant the tracer displayed with).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = Path("C:/Users/ryane/coloc_analysis/staging/pachytene_traces.json")
RUN_DIRS = [Path("C:/Users/ryane/ccw77_fullres"), Path("C:/Users/ryane/ccw77_0622"), Path("C:/Users/ryane/n2_dryice_results")]
PXY_CONST = 0.1083
PAD = 30


def find_run(iid: str) -> Path | None:
    for d in RUN_DIRS:
        for cand in (d, d / iid):
            if (cand / f"{iid}__nuclei_labels.tif").is_file() and (cand / f"{iid}__nuclei.csv").is_file():
                return cand
    return None


def crop_offset(rd: Path, iid: str) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    import tifffile
    from scipy import ndimage as ndi

    nc = pd.read_csv(rd / f"{iid}__nuclei.csv")
    germ = [int(x) for x in nc[nc["in_germline"].astype(bool)]["nucleus_id"]]
    lab = tifffile.imread(rd / f"{iid}__nuclei_labels.tif")
    gn = np.isin(lab, germ)
    obj = ndi.find_objects(gn.astype(np.uint8))[0]
    sl = tuple(slice(max(0, o.start - PAD), min(dim, o.stop + PAD)) for o, dim in zip(obj, gn.shape))
    vox = (float(nc["voxel_dz_um"].iloc[0]), float(nc["voxel_dy_um"].iloc[0]), float(nc["voxel_dx_um"].iloc[0]))
    return (sl[0].start, sl[1].start, sl[2].start), vox


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(REPO / "analysis" / "coloc" / "staging" / "pachytene_traces_whole_image.json"))
    args = ap.parse_args(argv)
    src = json.loads(Path(args.src).read_text(encoding="utf-8"))
    out: dict = {}
    for iid, t in sorted(src.items()):
        if t.get("status") != "traced":
            out[iid] = {**t, "frame": "unconverted"}; continue
        rd = find_run(iid)
        if rd is None:
            print(f"{iid}: run folder not found; skipped"); continue
        (z0, y0, x0), vox = crop_offset(rd, iid)
        pts = []
        for x_um, y_um in t["points_um"]:
            px, py = x_um / PXY_CONST, y_um / PXY_CONST
            pts.append([round((px + x0) * vox[2], 3), round((py + y0) * vox[1], 3)])
        out[iid] = {"points_um": pts, "status": "traced", "n_points": len(pts), "off_axis_um": t.get("off_axis_um", 20.0),
                    "frame": "whole_image_um", "traced_at": t.get("traced_at"),
                    "migrated_from": {"frame": "crop_um_0.1083", "crop_offset_vox": [z0, y0, x0], "voxel_um": list(vox),
                                      "run_dir": str(rd)}}
        print(f"{iid[-22:]:24s} offset y{y0} x{x0}  vox {vox[1]:.6f}  first {t['points_um'][0]} -> {pts[0]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {sum(1 for v in out.values() if v.get('frame') == 'whole_image_um')} traces converted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
