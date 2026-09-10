"""Exact-equality regression diff between two germquant output trees (golden vs candidate).

    python scripts/regression_diff.py GOLDEN_DIR CANDIDATE_DIR [--atol 0] [--strict-cols] [--strict-ids]
                                      [--ignore-cols ...] [--json out.json]

Images are keyed by their folder relative to the root plus the image id, so a golden tree produced by
scripts/make_goldens.py (``<out>/<sha>/<tag>/``) may hold the same gonad under several tags (default,
``--no-coloc``, ``--no-spots``) and each tag is compared with the candidate folder of the same name.

For every golden image it compares:
  * every ``<image>__<table>.csv``: every golden column must exist in the candidate, in the same order,
    with the same row count and every cell equal (numeric cells within --atol, default 0 = bit-for-bit
    after the CSV round-trip; NA == NA). Columns that exist only in the candidate are reported as
    ADDED and do not fail the run unless --strict-cols (append-only schema growth is allowed by the
    roadmap; a dropped or reordered golden column always fails). Volatile provenance columns
    (git_sha, run_timestamp, pipeline_version, file_path) are ignored;
  * every ``<image>__*.tif`` label or mask image: same shape and dtype, arrays equal; a label image whose
    objects are identical but renumbered is reported as PERMUTED (a pass unless --strict-ids);
  * ``run_manifest.json``: config_hash equal (git_sha and timestamps are expected to differ).
Montage PNGs, parquet copies and ``__stages.json`` are ignored (the CSV is the record).

Exit codes: 0 all matched, 1 differences found, 2 no golden images found (an empty or wrong golden
folder must never pass the gate). Written for docs/ROADMAP_modular_pipeline.md step 0: every step of
that work must leave the goldens identical unless the step is explicitly number-changing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# provenance columns that legitimately differ between two runs of the same recipe
VOLATILE = {"git_sha", "run_timestamp", "pipeline_version", "file_path"}


def _images(root: Path) -> dict[str, tuple[Path, str]]:
    """{relative_dir/image_id: (dir, image_id)} for every folder holding <image_id>__*.csv files."""
    out: dict[str, tuple[Path, str]] = {}
    for f in sorted(root.rglob("*__*.csv")):
        iid = f.name.split("__")[0]
        rel = f.parent.relative_to(root)
        key = f"{rel.as_posix()}/{iid}" if str(rel) != "." else iid
        out.setdefault(key, (f.parent, iid))
    return out


def _load(p: Path) -> pd.DataFrame:
    # exact text -> double parsing; the default parser can be 1 ulp off on 17-digit values
    return pd.read_csv(p, low_memory=False, float_precision="round_trip")


def diff_table(g: pd.DataFrame, c: pd.DataFrame, atol: float, ignore: set[str],
               strict_cols: bool = False) -> tuple[list[str], list[str]]:
    """Return (problems, notes). Empty problems = identical on every golden column."""
    probs: list[str] = []
    notes: list[str] = []
    g = g.drop(columns=[x for x in g.columns if x in ignore], errors="ignore")
    c = c.drop(columns=[x for x in c.columns if x in ignore], errors="ignore")
    gc, cc = list(g.columns), list(c.columns)
    missing = [x for x in gc if x not in cc]
    added = [x for x in cc if x not in gc]
    if missing:
        probs.append(f"golden columns missing from candidate: {missing}")
    if added:
        (probs if strict_cols else notes).append(f"columns added in candidate: {added}")
    common_c = [x for x in cc if x in gc]
    common_g = [x for x in gc if x in cc]
    if common_c != common_g:
        probs.append(f"column order differs: golden {common_g} vs candidate {common_c}")
    if len(g) != len(c):
        probs.append(f"row count {len(g)} vs {len(c)}")
        return probs, notes
    for col in common_g:
        a, b = g[col], c[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            na_ok = np.array_equal(np.isnan(av), np.isnan(bv))
            both = ~np.isnan(av) & ~np.isnan(bv)
            close = np.allclose(av[both], bv[both], rtol=0.0, atol=atol) if both.any() else True
            if not (na_ok and close):
                bad = np.where(both & ~np.isclose(av, bv, rtol=0.0, atol=atol, equal_nan=True))[0]
                worst = float(np.nanmax(np.abs(av - bv))) if both.any() else float("nan")
                probs.append(f"{col}: {len(bad)} cells differ (max |diff| {worst:.6g})"
                             f"{'' if na_ok else ', NA pattern differs'}")
        else:
            a2, b2 = a.astype("string").fillna("<NA>"), b.astype("string").fillna("<NA>")
            ne = (a2 != b2).to_numpy()
            if ne.any():
                i = int(np.argmax(ne))
                probs.append(f"{col}: {int(ne.sum())} cells differ (first row {i}: {a2.iloc[i]!r} vs {b2.iloc[i]!r})")
    return probs, notes


def diff_tif(gp: Path, cp: Path) -> tuple[str, str]:
    """Return (status, detail): IDENTICAL, PERMUTED (same partition, ids differ), DIFFERENT, MISSING."""
    import tifffile

    if not cp.exists():
        return "MISSING", "candidate file missing"
    g, c = tifffile.imread(gp), tifffile.imread(cp)
    if g.shape != c.shape or g.dtype != c.dtype:
        return "DIFFERENT", f"shape/dtype {g.shape}/{g.dtype} vs {c.shape}/{c.dtype}"
    if np.array_equal(g, c):
        return "IDENTICAL", ""
    if g.dtype == np.uint8 and set(np.unique(g)) <= {0, 255}:           # binary mask
        n = int((g != c).sum())
        return "DIFFERENT", f"{n} voxels differ ({n / g.size:.2e} of volume)"
    if not np.array_equal(g > 0, c > 0):                                  # label image: foreground
        n = int(((g > 0) != (c > 0)).sum())
        return "DIFFERENT", f"foreground differs in {n} voxels; {len(np.unique(g)) - 1} vs {len(np.unique(c)) - 1} objects"
    pairs = np.unique(np.stack([g[g > 0], c[g > 0]], axis=1), axis=0)   # same partition under relabel?
    one_to_one = len(pairs) == len(np.unique(pairs[:, 0])) == len(np.unique(pairs[:, 1]))
    if one_to_one:
        return "PERMUTED", f"{len(pairs)} objects identical, ids permuted"
    return "DIFFERENT", (f"object partition differs ({len(np.unique(g)) - 1} vs {len(np.unique(c)) - 1} objects, "
                         f"{len(pairs)} id pairs)")


def compare(gdir: Path, cdir: Path, *, atol: float = 0.0, ignore: set[str] | None = None,
            strict_cols: bool = False, strict_ids: bool = False) -> tuple[dict, int]:
    """Compare two trees. Returns (report, n_fail); report[key] = {name: [...]}."""
    ignore = VOLATILE | (ignore or set())
    report: dict[str, dict] = {}
    n_fail = 0
    gimgs, cimgs = _images(gdir), _images(cdir)
    for key, (gd, iid) in gimgs.items():
        rep: dict[str, object] = {}
        if key not in cimgs:
            rep["status"] = "MISSING_IMAGE"; n_fail += 1; report[key] = rep
            print(f"[MISSING] {key}: no candidate output"); continue
        cd = cimgs[key][0]
        bad: dict[str, object] = {}
        notes: list[str] = []
        for gf in sorted(gd.glob(f"{iid}__*.csv")):
            table = gf.name.split("__")[1][:-4]
            cf = cd / gf.name
            if not cf.exists():
                rep[table] = ["candidate table missing"]; bad[table] = rep[table]; n_fail += 1; continue
            probs, nts = diff_table(_load(gf), _load(cf), atol, ignore, strict_cols)
            rep[table] = probs
            notes += [f"{table}: {n}" for n in nts]
            if probs:
                bad[table] = probs; n_fail += 1
        for gt in sorted(gd.glob(f"{iid}__*.tif")):
            status, detail = diff_tif(gt, cd / gt.name)
            name = gt.name.split("__")[1]
            rep[name] = [status, detail]
            if status in ("DIFFERENT", "MISSING") or (status == "PERMUTED" and strict_ids):
                bad[name] = rep[name]; n_fail += 1
            elif status == "PERMUTED":
                notes.append(f"{name}: ids permuted")
        gm, cm = gd / "run_manifest.json", cd / "run_manifest.json"
        if gm.exists() and cm.exists():
            gh = json.loads(gm.read_text(encoding="utf-8")).get("config_hash")
            ch = json.loads(cm.read_text(encoding="utf-8")).get("config_hash")
            rep["config_hash"] = [gh, ch]
            if gh != ch:
                rep["config_hash"] = [gh, ch, "DIFFERS"]; bad["config_hash"] = rep["config_hash"]; n_fail += 1
        rep["notes"] = notes
        report[key] = rep
        print(f"[{'OK' if not bad else 'DIFF'}] {key}" + (f"  ({'; '.join(notes)})" if notes else ""))
        for k, v in bad.items():
            print(f"      {k}: {v}")
    extra = sorted(set(cimgs) - set(gimgs))
    if extra:
        print(f"note: candidate has {len(extra)} images not in golden: {extra[:5]}{'...' if len(extra) > 5 else ''}")
    return report, n_fail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("golden")
    ap.add_argument("candidate")
    ap.add_argument("--atol", type=float, default=0.0, help="numeric tolerance (default 0 = exact)")
    ap.add_argument("--ignore-cols", nargs="*", default=[], help="extra columns to ignore")
    ap.add_argument("--strict-cols", action="store_true", help="fail on columns added in the candidate")
    ap.add_argument("--strict-ids", action="store_true", help="treat a label-id permutation as a failure")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args(argv)

    gdir, cdir = Path(args.golden), Path(args.candidate)
    if not gdir.is_dir():
        print(f"golden folder does not exist: {gdir}"); return 2
    report, n_fail = compare(gdir, cdir, atol=args.atol, ignore=set(args.ignore_cols),
                             strict_cols=args.strict_cols, strict_ids=args.strict_ids)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if not report:
        print(f"no golden images (*__*.csv) found under {gdir}"); return 2
    print(f"\n{len(report)} golden images checked, {n_fail} failing comparisons")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
