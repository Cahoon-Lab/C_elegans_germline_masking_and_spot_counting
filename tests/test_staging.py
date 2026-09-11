"""Staging stage: the ported zone geometry (checked against the August zone tables), the traces file,
the pipeline wiring and `restage`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_envelope import patched_reader_lamin  # noqa: F401  (fixture)

from germquant import cli, pipeline
from germquant.config import load_config
from germquant.staging import assign_zones, load_traces, project_to_polyline, save_trace
from germquant.staging.tracer import restage

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_STAGING = Path("C:/Users/ryane/coloc_analysis/staging")      # the August per-gonad staging tables (local)


def test_projection_geometry():
    poly = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
    s, r, L = project_to_polyline([[5.0, 1.0], [-2.0, 0.0], [10.0, 12.0], [12.0, 5.0]], poly)
    assert L == 20.0
    # s is unclamped beyond the free ends (so pre / post can be told apart); r is always the distance
    # to the clamped projection, so the point 2 um before the start reads r = 2
    assert np.allclose(s, [5.0, -2.0, 22.0, 15.0]) and np.allclose(r, [1.0, 2.0, 2.0, 2.0])


def test_assign_zones_thirds_and_cutoff():
    poly = [[0.0, 0.0], [30.0, 0.0]]
    df = pd.DataFrame({"cx": [1, 12, 25, 31, -1, 15, 15], "cy": [0, 0, 0, 0, 0, 5, 40]})
    z, L = assign_zones(df, np.asarray(poly), off_axis_um=20.0, adaptive=False)
    assert L == 30 and list(z["zone"]) == ["early", "mid", "late", "post", "pre", "mid", "off_axis"]
    assert (z["off_axis_cut_um"] == 20.0).all()


@pytest.mark.skipif(not (ANALYSIS_STAGING / "zones").is_dir(), reason="August staging tables not on this machine")
def test_ported_geometry_reproduces_the_august_zone_tables():
    """The stage's geometry must reproduce the zones Ryan's re-trace produced, nucleus by nucleus."""
    traces = json.loads((ROOT / "analysis" / "coloc" / "staging" / "pachytene_traces.json").read_text(encoding="utf-8"))
    checked = 0
    for iid, t in traces.items():
        nuc = ANALYSIS_STAGING / f"{iid}_nuclei.csv"
        ref = ROOT / "analysis" / "coloc" / "staging" / "zones" / f"{iid}_zones.csv"
        if t.get("status") != "traced" or not nuc.is_file() or not ref.is_file():
            continue
        df = pd.read_csv(nuc)
        z, _L = assign_zones(df, np.asarray(t["points_um"]))
        expect = pd.read_csv(ref)
        m = z[["nucleus_id", "zone", "s_um", "r_um", "off_axis_cut_um"]].merge(
            expect[["nucleus_id", "zone", "s_um", "r_um", "off_axis_cut_um"]], on="nucleus_id", suffixes=("", "_ref"))
        assert len(m) == len(expect), iid
        assert (m["zone"] == m["zone_ref"]).all(), iid
        assert np.allclose(m["s_um"], m["s_um_ref"]) and np.allclose(m["r_um"], m["r_um_ref"]), iid
        assert np.allclose(m["off_axis_cut_um"], m["off_axis_cut_um_ref"]), iid
        checked += 1
    assert checked >= 10


def test_migrated_traces_are_whole_image_and_complete():
    mig = json.loads((ROOT / "analysis" / "coloc" / "staging" / "pachytene_traces_whole_image.json").read_text(encoding="utf-8"))
    assert len(mig) == 25 and all(v["frame"] == "whole_image_um" and v["status"] == "traced" for v in mig.values())
    v = mig["20260708_ccw77_IF_pgl1_syp3_lmn1_HS_male_008"]
    assert v["migrated_from"]["crop_offset_vox"][2] == 444 and abs(v["points_um"][0][0] - 196.786) < 1e-3


def test_save_and_load_traces(tmp_path):
    tf = tmp_path / "t.json"
    tr = save_trace(tf, "img", [(1.234, 5.678), (9.0, 9.0)], "traced")
    assert load_traces(tf)["img"]["points_um"] == [[1.23, 5.68], [9.0, 9.0]] and tr["img"]["frame"] == "whole_image_um"


def _cfg(traces_file, **switches):
    cfg = load_config("config/config_ccw77.yaml")
    cfg._data["segmentation"]["nuclei"]["method"] = "classical"
    cfg.set("coloc.enabled", False)
    cfg.set("staging.enabled", True)
    cfg.set("staging.traces_file", str(traces_file))
    for k, v in switches.items():
        cfg.set(k, v)
    return cfg


IID = "20260708_ccw77_IF_pgl1_syp3_lmn1_HS_male_008"


def test_staging_stage_with_and_without_envelope(tmp_path, patched_reader_lamin):  # noqa: F811
    tf = tmp_path / "traces.json"
    # the synthetic germline runs along x at y = 8 um (40 px x 0.2); trace from x = 6 to x = 40 um
    save_trace(tf, IID, [(6.0, 8.0), (40.0, 8.0)], "traced")
    res = pipeline.process_image(f"{IID}.nd2", _cfg(tf), tmp_path / "r", prov=None)
    assert res["stages"]["staging"]["status"] == "ran", res["stages"]["staging"]
    z = res["tables"]["zones"]
    assert set(z["zone"]) <= {"early", "mid", "late", "pre", "post", "off_axis"} and z["is_pachytene"].sum() >= 3
    isum = res["tables"]["image_summary"].iloc[0]
    assert abs(isum["pachytene_length_um"] - 34.0) < 1e-6 and isum["n_zoned_nuclei"] == z["is_pachytene"].sum()
    assert z["in_territory"].isna().all()                          # no envelope stage: territory unknown
    assert any(f.startswith("staging:zoned_n=") and f.endswith("centroid=dapi") for f in res["qc_flags"])
    nuc = pd.read_csv(next((tmp_path / "r").rglob("*__nuclei.csv")))
    assert nuc.loc[nuc["in_germline"].astype(bool), "zone"].notna().all()
    # with the envelope stage: envelope centroids are projected and the territory crossed is known
    res2 = pipeline.process_image(f"{IID}.nd2", _cfg(tf, **{"envelope.enabled": True}), tmp_path / "r2", prov=None)
    z2 = res2["tables"]["zones"]
    assert z2["in_territory"].notna().all() and z2["in_territory"].any()
    assert any(f.endswith("centroid=envelope") for f in res2["qc_flags"])
    assert res2["tables"]["image_summary"].iloc[0]["n_territories_on_trace"] >= 1
    # no trace -> skipped with a reason; a skipped trace -> skipped
    res3 = pipeline.process_image("20260708_ccw77_IF_pgl1_syp3_lmn1_HS_male_999.nd2", _cfg(tf), tmp_path / "r3", prov=None)
    assert res3["stages"]["staging"]["status"] == "skipped" and "no trace" in res3["stages"]["staging"]["reason"]
    # restage rewrites the zones table from the traces file without touching the image
    save_trace(tf, IID, [(6.0, 8.0), (20.0, 8.0)], "traced")
    done = restage(tmp_path / "r", tf, _cfg(tf))
    assert done == [IID]
    z3 = pd.read_csv(next((tmp_path / "r").rglob("*__zones.csv")))
    assert (z3["image_id"] == IID).all() and z3["is_pachytene"].sum() <= z["is_pachytene"].sum()
    assert cli.main(["restage", str(tmp_path / "r"), "--config", "config/config_ccw77.yaml", "--traces", str(tf)]) == 0
