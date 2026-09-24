"""Produce frozen reference outputs ("goldens") for the regression gate of the modular-pipeline work.

    python scripts/make_goldens.py MANIFEST.json --out C:/Users/ryane/germquant_golden [--commit 8666f07] [--only TAG ...]

MANIFEST.json is a list of runs:
    [{"tag": "n2dryice_noHS_male_001", "nd2": "C:/.../x.nd2", "config": "config/config_n2dryice.yaml",
      "xy_stride": 1, "z_range": null, "flags": []}, ...]

Each run is written to <out>/<commit-or-HEAD sha>/<tag>/ through ``germquant run`` and recorded in
<out>/<sha>/golden_manifest.json (nd2 path, config path and its sha256, stride, elapsed seconds, the
run's config_hash and git_sha). Compare later runs with scripts/regression_diff.py.

--commit CHECKS OUT THAT COMMIT IN A TEMPORARY GIT WORKTREE and runs the package from there (PYTHONPATH),
so goldens can always be regenerated from the frozen commit (docs/ROADMAP_modular_pipeline.md step 0)
even after the working tree has moved on. Without --commit the current checkout is used and the
manifest records its sha (with -dirty when the tree is not clean).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def _sha(repo: Path) -> str:
    out = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                           capture_output=True, text=True, check=False).stdout.strip()
    return out.stdout.strip() + ("-dirty" if dirty else "")


def _worktree(commit: str) -> Path:
    """Detached worktree of `commit` next to the repo, with the (gitignored) models folder linked in so
    the CWD-relative cellpose_model path in config.yaml resolves there too."""
    wt = REPO.parent / f"germquant_wt_{commit}"
    if not wt.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(wt), commit], cwd=REPO, check=True)
    models_src, models_dst = REPO / "models", wt / "models"
    if models_src.exists() and not models_dst.exists():
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(models_dst), str(models_src)], check=True)
        else:
            models_dst.symlink_to(models_src, target_is_directory=True)
    return wt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", help="run the package from a worktree of this commit")
    ap.add_argument("--only", nargs="*", help="run only these tags")
    ap.add_argument("--force", action="store_true", help="re-run tags whose output folder already exists")
    args = ap.parse_args(argv)

    runs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.only:
        runs = [r for r in runs if r["tag"] in set(args.only)]
    src_repo = _worktree(args.commit) if args.commit else REPO
    sha = _sha(src_repo)
    root = Path(args.out) / sha
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(src_repo / "src"), PYTHONIOENCODING="utf-8")
    manifest_path = root / "golden_manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    for r in runs:
        tag = r["tag"]
        out_dir = root / tag
        done = (tag in records and records[tag].get("returncode") == 0
                and out_dir.exists() and any(out_dir.glob("*__nuclei.csv")))
        if done and not args.force:
            print(f"skip {tag} (complete)"); continue
        cfg = (src_repo / r["config"]) if not Path(r["config"]).is_absolute() else Path(r["config"])
        cmd = [PY, "-m", "germquant.cli", "run", r["nd2"], "--config", str(cfg), "--out", str(out_dir),
               "--xy-stride", str(r.get("xy_stride", 1))]
        if r.get("z_range"):
            cmd += ["--z-range", *map(str, r["z_range"])]
        cmd += list(r.get("flags", []))
        print(f"== {tag}: {' '.join(cmd[3:])}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=src_repo, env=env, check=False).returncode
        elapsed = time.time() - t0
        rec = {"nd2": r["nd2"], "config": str(cfg), "config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest()[:12],
               "xy_stride": r.get("xy_stride", 1), "z_range": r.get("z_range"), "flags": r.get("flags", []),
               "returncode": rc, "elapsed_s": round(elapsed, 1), "source_repo": str(src_repo), "git_sha": sha}
        rm = out_dir / "run_manifest.json"
        if rm.exists():
            m = json.loads(rm.read_text(encoding="utf-8"))
            rec["config_hash"] = m.get("config_hash"); rec["run_git_sha"] = m.get("git_sha")
        records[tag] = rec
        manifest_path.write_text(json.dumps(records, indent=1), encoding="utf-8")
        print(f"   rc={rc} {elapsed / 60:.1f} min", flush=True)
    print(f"goldens in {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
