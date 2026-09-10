"""Batch hygiene (docs/ROADMAP_modular_pipeline.md step 5): file discovery with an exclusions file,
resume keyed on the completion marker, and the collect step that stacks per-image tables.

Shared by `germquant batch`, `germquant collect` and workflow/Snakefile so the three never disagree on
which images are in a study or on what "already done" means.
"""
from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path

import pandas as pd

from . import provenance, schema
from .fsutil import long_path
from .stages import STAGES, stage_enabled

log = logging.getLogger(__name__)

# columns of batch_summary.csv (unchanged since the first release; R scripts read it)
SUMMARY_COLS = ["image_id", "n_nuclei", "n_germline", "qc_pass", "qc_flags", "out_dir"]


def load_exclusions(path: str | Path | None, base_dir: Path | None = None) -> list[str]:
    """Image-id substrings to drop from a study. Accepts a JSON list, or an object with an
    ``excluded_short_ids`` list (the format of analysis/coloc/exclusions.json) and optionally
    ``excluded_batch`` (a substring too). Missing path -> no exclusions."""
    if not path:
        return []
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    if not p.is_file():
        raise FileNotFoundError(f"qc.exclusions_file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    subs = [str(x) for x in data.get("excluded_short_ids", [])]
    if data.get("excluded_batch"):
        subs.append(str(data["excluded_batch"]))
    return subs


def discover_files(root: str | Path, glob: str = "**/*.nd2", exclude_patterns: list[str] | None = None,
                   exclusions: list[str] | None = None) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Files to process, sorted, plus the (file, reason) pairs that were dropped. ``exclude_patterns``
    are filename globs (10x overviews, largeimage); ``exclusions`` are image-id substrings from the
    study's exclusions file, matched against the file stem."""
    root = Path(root)
    keep: list[Path] = []
    dropped: list[tuple[Path, str]] = []
    for f in sorted(root.glob(glob)):
        if f.suffix.lower() != ".nd2":
            continue
        name = f.name.lower()
        pat = next((p for p in (exclude_patterns or []) if fnmatch.fnmatch(name, p.lower())), None)
        if pat is not None:
            dropped.append((f, f"exclude_pattern {pat}")); continue
        sub = next((s for s in (exclusions or []) if s in f.stem), None)
        if sub is not None:
            dropped.append((f, f"exclusions_file {sub}")); continue
        keep.append(f)
    return keep, dropped


def enabled_stage_names(cfg) -> list[str]:
    return [s.name for s in STAGES if stage_enabled(cfg, s.name)]


def is_done(out_dir: str | Path, image_id: str, cfg) -> tuple[bool, str]:
    """True when ``<image_id>__done.json`` exists in out_dir with the same config_hash and the same set
    of enabled stages as ``cfg`` (so a config edit or a stage switch forces a rerun)."""
    m = provenance.read_done_marker(out_dir, image_id)
    if not m:
        return False, "no completion marker"
    if m.get("config_hash") != cfg.hash:
        return False, f"config_hash {m.get('config_hash')} != {cfg.hash}"
    if sorted(m.get("enabled_stages", [])) != sorted(enabled_stage_names(cfg)):
        return False, "enabled stages differ"
    return True, "done"


def collect(out_root: str | Path, tables: tuple[str, ...] = tuple(schema.TABLES)) -> dict[str, int]:
    """Stack every per-image ``<image_id>__<table>.csv`` under out_root into ``batch_<table>.csv`` and
    rebuild ``batch_summary.csv`` (image_id, n_nuclei, n_germline, qc_pass, qc_flags, out_dir) from the
    image summaries and stage records. Returns {table: n_rows}. Idempotent; safe to rerun after any
    subset of images was reprocessed."""
    out_root = Path(out_root)
    counts: dict[str, int] = {}
    for t in tables:
        files = sorted(out_root.rglob(f"*__{t}.csv"))
        files = [f for f in files if not f.name.startswith("batch_")]
        frames = []
        for f in files:
            try:
                frames.append(pd.read_csv(long_path(f), low_memory=False))
            except Exception as e:  # noqa: BLE001 - one unreadable table must not block the stack
                log.warning("collect: skipping %s (%s)", f, e)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=schema.TABLES[t])
        df.to_csv(long_path(out_root / f"batch_{t}.csv"), index=False)
        counts[t] = len(df)
    # batch_summary.csv from the image summaries (same six columns as `germquant batch` writes)
    rows = []
    for f in sorted(out_root.rglob("*__image_summary.csv")):
        try:
            s = pd.read_csv(long_path(f), low_memory=False)
        except Exception as e:  # noqa: BLE001 - keep stacking the rest
            log.warning("collect: skipping %s (%s)", f, e)
            continue
        if s.empty:
            continue
        r = s.iloc[0]
        rows.append({"image_id": r.get("image_id", f.name.split("__")[0]), "n_nuclei": int(r.get("n_nuclei", 0)),
                     "n_germline": int(r.get("n_germline_nuclei", 0) if pd.notna(r.get("n_germline_nuclei", 0)) else 0),
                     "qc_pass": bool(r.get("qc_pass", False)), "qc_flags": r.get("qc_flags", "") if pd.notna(r.get("qc_flags", "")) else "",
                     "out_dir": str(f.parent)})
    pd.DataFrame(rows, columns=SUMMARY_COLS).to_csv(long_path(out_root / "batch_summary.csv"), index=False)
    counts["batch_summary"] = len(rows)
    return counts
