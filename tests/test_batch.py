"""Batch hygiene: exclusions file, discovery, resume keyed on the completion marker, collect."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from test_end_to_end import patched_reader  # noqa: F401  (fixture)

from germquant import batch, cli, pipeline, provenance
from germquant.batch import Exclusions
from germquant.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_exclusions_formats(tmp_path):
    assert batch.load_exclusions(None).patterns == []
    lst = tmp_path / "a.json"; lst.write_text('["HS_herm_14", "noHS_male_05"]')
    e = batch.load_exclusions(lst)
    assert e.patterns == ["HS_herm_14", "noHS_male_05"] and e.batch is None
    obj = tmp_path / "b.json"
    obj.write_text(json.dumps({"note": "x", "excluded_short_ids": ["HS_male_07"], "excluded_batch": "20260622"}))
    e = batch.load_exclusions("b.json", base_dir=tmp_path)
    assert e.patterns == ["HS_male_07"] and e.batch == "20260622"
    with pytest.raises(FileNotFoundError):
        batch.load_exclusions("missing.json", base_dir=tmp_path)


def test_exclusions_match_whole_tokens_and_batch():
    e = Exclusions(["HS_herm_14", "HS_male_1"])
    assert e.matches("20260622_ccw77_IF_syp1_pgl1_LMN1_HS_herm_14") == "HS_herm_14"
    assert e.matches("20260622_ccw77_IF_syp1_pgl1_LMN1_noHS_herm_14") is None       # no fragment match
    assert e.matches("20260708_ccw77_IF_pgl1_syp3_lmn1_HS_male_10") is None          # HS_male_1 != HS_male_10
    assert e.matches("20260708_ccw77_IF_pgl1_syp3_lmn1_HS_male_1") == "HS_male_1"
    assert Exclusions(["20260708_ccw77_IF_pgl1_syp3_lmn1_HS_male_008"]).matches("20260708_ccw77_IF_pgl1_syp3_lmn1_HS_male_008")
    # the real study file: only the 22-Jun batch, only the listed ids
    real = batch.load_exclusions(ROOT / "analysis" / "coloc" / "exclusions.json")
    assert real.batch == "20260622" and "HS_male_07" in real.patterns
    assert real.matches("20260622_ccw77_IF_syp1_pgl1_LMN1_HS_male_07") == "HS_male_07"
    assert real.matches("20260622_ccw77_IF_syp1_pgl1_LMN1_HS_male_10") is None          # kept gonad
    assert real.matches("20260708_ccw77_IF_pgl1_syp3_lmn1_HS_male_007") is None         # other batch
    assert real.matches("20260622_ccw77_IF_syp1_pgl1_LMN1_noHS_herm_01") == "noHS_herm_01"


def test_discover_files_applies_globs_and_exclusions(tmp_path):
    for n in ["a_HS_male_07.nd2", "b_noHS_herm_01.nd2", "c_10x_overview.nd2", "d.nd2", "sub/e.nd2", "f.txt"]:
        p = tmp_path / n; p.parent.mkdir(exist_ok=True); p.write_bytes(b"")
    files, dropped = batch.discover_files(tmp_path, "**/*.nd2", ["*10x*"], ["HS_male_07"])
    assert [f.name for f in files] == ["b_noHS_herm_01.nd2", "d.nd2", "e.nd2"]
    reasons = {f.name: why for f, why in dropped}
    assert reasons["a_HS_male_07.nd2"].startswith("exclusions_file") and reasons["c_10x_overview.nd2"].startswith("exclude_pattern")


def _marker(out, iid, cfg, **kw):
    provenance.write_done_marker(out, iid, config_hash=cfg.hash, enabled_stages=batch.enabled_stage_names(cfg),
                                 stage_hashes={"segment": "abc"}, **kw)


def test_is_done_requires_matching_marker(tmp_path):
    cfg = load_config(ROOT / "config" / "config.yaml")
    assert batch.is_done(tmp_path, "img", cfg) == (False, "no completion marker")
    _marker(tmp_path, "img", cfg)
    assert batch.is_done(tmp_path, "img", cfg)[0] is True
    assert batch.is_done(tmp_path, "img", cfg, segment_hash="abc")[0] is True
    assert batch.is_done(tmp_path, "img", cfg, segment_hash="zzz")[0] is False          # model / cellpose changed
    assert batch.is_done(tmp_path, "img", cfg, xy_stride=4)[0] is False                 # stride differs
    _marker(tmp_path, "s4", cfg, xy_stride=4)
    assert batch.is_done(tmp_path, "s4", cfg)[0] is False and batch.is_done(tmp_path, "s4", cfg, xy_stride=4)[0]
    _marker(tmp_path, "z", cfg, z_range=[10, 20])
    assert batch.is_done(tmp_path, "z", cfg)[0] is False and batch.is_done(tmp_path, "z", cfg, z_range=(10, 20))[0]
    _marker(tmp_path, "f", cfg, failed_stages=["spots"])
    done, why = batch.is_done(tmp_path, "f", cfg)
    assert done is False and "spots" in why
    assert batch.is_done(tmp_path, "f", cfg, retry_failed=False)[0] is True
    legacy = tmp_path / "old__done.json"                                                # marker without geometry
    legacy.write_text(json.dumps({"image_id": "old", "config_hash": cfg.hash, "enabled_stages": batch.enabled_stage_names(cfg)}))
    assert batch.is_done(tmp_path, "old", cfg)[0] is False
    cfg.set("spots.enabled", False)                                                     # a switch changes hash + stages
    assert batch.is_done(tmp_path, "img", cfg)[0] is False


def _fake_process(calls, fail=()):
    def fake(nd2, cfg, out_dir, *, xy_stride=1, z_range=None, prov=None):
        calls.append(Path(nd2).name)
        iid = Path(nd2).stem
        if iid in fail:
            raise RuntimeError("boom")
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"image_id": iid, "n_nuclei": 3, "n_germline_nuclei": 2, "qc_pass": True, "qc_flags": ""}]) \
            .to_csv(out_dir / f"{iid}__image_summary.csv", index=False)
        pd.DataFrame([{"image_id": iid, "nucleus_id": 1}]).to_csv(out_dir / f"{iid}__nuclei.csv", index=False)
        provenance.write_done_marker(out_dir, iid, config_hash=cfg.hash, enabled_stages=batch.enabled_stage_names(cfg),
                                     stage_hashes={"segment": _seg_hash(cfg)}, xy_stride=xy_stride)
        return {"image_id": iid, "n_nuclei": 3, "qc_pass": True, "qc_flags": [], "out_dir": str(out_dir),
                "tables": {"image_summary": pd.DataFrame([{"n_germline_nuclei": 2}])}, "stages": {}}
    return fake


def _seg_hash(cfg):
    from germquant.stages import stage_hashes

    return stage_hashes(cfg, {}, provenance.tool_versions(), "unused")["segment"]


def _study(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    for n in ["x1.nd2", "x2.nd2", "x3.nd2", "x_HS_male_07.nd2"]:
        (src / n).write_bytes(b"")
    excl = tmp_path / "excl.json"; excl.write_text('["HS_male_07"]')
    cfg_path = tmp_path / "study.yaml"          # a profile OUTSIDE the repo, extending the shipped config
    cfg_path.write_text(f"extends: {(ROOT / 'config' / 'config.yaml').as_posix()}\n"
                        f"segmentation:\n  nuclei:\n    method: classical\n"
                        f"qc:\n  exclusions_file: {excl.as_posix()}\n")
    return src, cfg_path


def test_batch_resume_skips_done_and_retries_failed(tmp_path, monkeypatch):
    src, cfg_path = _study(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "process_image", _fake_process(calls, fail=("x3",)))
    out = tmp_path / "out"
    rc = cli.main(["batch", str(src), "--config", str(cfg_path), "--out", str(out)])
    assert rc == 0 and calls == ["x1.nd2", "x2.nd2", "x3.nd2"]           # excluded gonad never processed
    summ = pd.read_csv(out / "batch_summary.csv")
    assert list(summ.columns) == batch.SUMMARY_COLS and len(summ) == 3
    assert summ.set_index("image_id").loc["x3", "qc_flags"].startswith("EXCEPTION")
    man = json.loads((out / "run_manifest.json").read_text())
    assert any("x_HS_male_07" in e for e in man["excluded"]) and man["xy_stride"] == 1
    # --resume: x1, x2 done; x3 (no marker) is retried, fails again, and its EXCEPTION row survives
    calls.clear()
    rc = cli.main(["batch", str(src), "--config", str(cfg_path), "--out", str(out), "--resume"])
    assert rc == 0 and calls == ["x3.nd2"]
    summ = pd.read_csv(out / "batch_summary.csv").set_index("image_id")
    assert len(summ) == 3 and summ.loc["x3", "qc_flags"].startswith("EXCEPTION") and not summ.loc["x3", "qc_pass"]
    assert (out / "batch_nuclei.csv").exists() and len(pd.read_csv(out / "batch_nuclei.csv")) == 2
    # a switch changes the recipe: --resume reprocesses everything
    calls.clear()
    rc = cli.main(["batch", str(src), "--config", str(cfg_path), "--out", str(out), "--resume", "--no-spots"])
    assert rc == 0 and calls == ["x1.nd2", "x2.nd2", "x3.nd2"]
    # a stride preview never satisfies the full-resolution resume
    calls.clear()
    cli.main(["batch", str(src), "--config", str(cfg_path), "--out", str(out), "--resume", "--no-spots", "--xy-stride", "4"])
    assert calls == ["x1.nd2", "x2.nd2", "x3.nd2"]


def test_collect_marker_filter_and_extra_rows(tmp_path):
    root = tmp_path / "res"
    cfg = load_config(ROOT / "config" / "config.yaml")
    for iid, mark in (("a", True), ("b", False)):
        d = root / iid; d.mkdir(parents=True)
        pd.DataFrame([{"image_id": iid, "n_nuclei": 5, "n_germline_nuclei": 4, "qc_pass": True, "qc_flags": ""}]) \
            .to_csv(d / f"{iid}__image_summary.csv", index=False)
        pd.DataFrame([{"image_id": iid, "nucleus_id": 1}]).to_csv(d / f"{iid}__nuclei.csv", index=False)
        if mark:
            _marker(d, iid, cfg)
    counts = batch.collect(root, extra_rows=[{"image_id": "c", "n_nuclei": 0, "n_germline": 0, "qc_pass": False,
                                              "qc_flags": "EXCEPTION:x", "out_dir": "z"}])
    summ = pd.read_csv(root / "batch_summary.csv").set_index("image_id")
    assert counts["nuclei"] == 1                                       # only the folder with a marker is stacked
    assert summ.loc["a", "qc_pass"] and not summ.loc["b", "qc_pass"] and "incomplete:no_done_marker" in summ.loc["b", "qc_flags"]
    assert summ.loc["c", "qc_flags"] == "EXCEPTION:x"
    counts = batch.collect(root, image_ids={"a"})
    assert counts["batch_summary"] == 1


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


def test_framing_qc_flag_text():
    rows = [{"image_id": str(i), "n_nuclei": 10, "n_germline": g, "qc_pass": True, "qc_flags": ""} for i, g in enumerate([10, 10, 11, 30])]
    out = batch.apply_framing_qc(rows, 1.8)
    assert out[3]["qc_flags"] == "qc:germline_count_outlier_30_vs_median10_review_framing" and out[0]["qc_flags"] == ""
