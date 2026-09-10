"""Batch hygiene: exclusions file, discovery, resume keyed on the completion marker, collect."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from test_end_to_end import patched_reader  # noqa: F401  (fixture)

from germquant import batch, cli, pipeline, provenance
from germquant.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_exclusions_formats(tmp_path):
    assert batch.load_exclusions(None) == []
    lst = tmp_path / "a.json"; lst.write_text('["HS_herm_14", "noHS_male_05"]')
    assert batch.load_exclusions(lst) == ["HS_herm_14", "noHS_male_05"]
    obj = tmp_path / "b.json"
    obj.write_text(json.dumps({"note": "x", "excluded_short_ids": ["HS_male_07"], "excluded_batch": "20260622"}))
    assert batch.load_exclusions("b.json", base_dir=tmp_path) == ["HS_male_07", "20260622"]
    with pytest.raises(FileNotFoundError):
        batch.load_exclusions("missing.json", base_dir=tmp_path)
    # the real study file is readable in its own format
    assert "HS_herm_14" in batch.load_exclusions(ROOT / "analysis" / "coloc" / "exclusions.json")


def test_discover_files_applies_globs_and_exclusions(tmp_path):
    for n in ["a_HS_male_07.nd2", "b_noHS_herm_01.nd2", "c_10x_overview.nd2", "d.nd2", "sub/e.nd2", "f.txt"]:
        p = tmp_path / n; p.parent.mkdir(exist_ok=True); p.write_bytes(b"")
    files, dropped = batch.discover_files(tmp_path, "**/*.nd2", ["*10x*"], ["HS_male_07"])
    assert [f.name for f in files] == ["b_noHS_herm_01.nd2", "d.nd2", "e.nd2"]
    reasons = {f.name: why for f, why in dropped}
    assert reasons["a_HS_male_07.nd2"].startswith("exclusions_file") and reasons["c_10x_overview.nd2"].startswith("exclude_pattern")


def test_is_done_requires_matching_marker(tmp_path):
    cfg = load_config(ROOT / "config" / "config.yaml")
    assert batch.is_done(tmp_path, "img", cfg) == (False, "no completion marker")
    provenance.write_done_marker(tmp_path, "img", config_hash=cfg.hash,
                                 enabled_stages=batch.enabled_stage_names(cfg), stage_hashes={})
    assert batch.is_done(tmp_path, "img", cfg)[0] is True
    cfg.set("spots.enabled", False)                       # a switch changes the hash and the stage set
    assert batch.is_done(tmp_path, "img", cfg)[0] is False


def _fake_process(calls):
    def fake(nd2, cfg, out_dir, *, xy_stride=1, z_range=None, prov=None):
        calls.append(Path(nd2).name)
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        iid = Path(nd2).stem
        pd.DataFrame([{"image_id": iid, "n_nuclei": 3, "n_germline_nuclei": 2, "qc_pass": True, "qc_flags": ""}]) \
            .to_csv(out_dir / f"{iid}__image_summary.csv", index=False)
        pd.DataFrame([{"image_id": iid, "nucleus_id": 1}]).to_csv(out_dir / f"{iid}__nuclei.csv", index=False)
        provenance.write_done_marker(out_dir, iid, config_hash=cfg.hash,
                                     enabled_stages=batch.enabled_stage_names(cfg), stage_hashes={})
        return {"image_id": iid, "n_nuclei": 3, "qc_pass": True, "qc_flags": [], "out_dir": str(out_dir),
                "tables": {"image_summary": pd.DataFrame([{"n_germline_nuclei": 2}])}, "stages": {}}
    return fake


def test_batch_resume_skips_done_images(tmp_path, monkeypatch):
    src = tmp_path / "in"; src.mkdir()
    for n in ["x1.nd2", "x2.nd2", "x_HS_male_07.nd2"]:
        (src / n).write_bytes(b"")
    excl = tmp_path / "excl.json"; excl.write_text('["HS_male_07"]')
    cfg_path = tmp_path / "config" / "config.yaml"; cfg_path.parent.mkdir()
    # a profile over the real config that adds the exclusions file (channel maps copied so relative
    # paths resolve against this tmp "repo")
    cfg_path.write_text(f"extends: {(ROOT / 'config' / 'config.yaml').as_posix()}\n"
                        f"qc:\n  exclusions_file: {excl.as_posix()}\n")
    (tmp_path / "config" / "channel_maps").mkdir()
    for cm in (ROOT / "config" / "channel_maps").glob("*.yaml"):
        (tmp_path / "config" / "channel_maps" / cm.name).write_text(cm.read_text())
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "process_image", _fake_process(calls))
    out = tmp_path / "out"
    rc = cli.main(["batch", str(src), "--config", str(cfg_path), "--out", str(out)])
    assert rc == 0 and calls == ["x1.nd2", "x2.nd2"]                     # excluded gonad never processed
    summ = pd.read_csv(out / "batch_summary.csv")
    assert list(summ.columns) == batch.SUMMARY_COLS and len(summ) == 2
    man = json.loads((out / "run_manifest.json").read_text())
    assert any("x_HS_male_07" in e for e in man["excluded"])
    # second invocation with --resume: nothing to do, summary rebuilt from disk
    calls.clear()
    rc = cli.main(["batch", str(src), "--config", str(cfg_path), "--out", str(out), "--resume"])
    assert rc == 0 and calls == []
    assert len(pd.read_csv(out / "batch_summary.csv")) == 2
    assert (out / "batch_nuclei.csv").exists()
    # a switch changes the recipe: --resume reprocesses
    rc = cli.main(["batch", str(src), "--config", str(cfg_path), "--out", str(out), "--resume", "--no-spots"])
    assert rc == 0 and calls == ["x1.nd2", "x2.nd2"]


def test_collect_stacks_real_outputs(tmp_path, patched_reader):  # noqa: F811
    cfg = load_config(ROOT / "config" / "config.yaml")
    cfg._data["segmentation"]["nuclei"]["method"] = "classical"
    root = tmp_path / "res"
    for iid in ("20251105_n2_nohs_HERM_001.nd2", "20251105_n2_hs_MALE_002.nd2"):
        pipeline.process_image(iid, cfg, root / Path(iid).stem, prov=None)
    counts = batch.collect(root)
    assert counts["image_summary"] == 2 and counts["batch_summary"] == 2 and counts["nuclei"] > 2
    stacked = pd.read_csv(root / "batch_nuclei.csv")
    assert set(stacked["image_id"]) == {"20251105_n2_nohs_HERM_001", "20251105_n2_hs_MALE_002"}
    assert cli.main(["collect", str(root)]) == 0
