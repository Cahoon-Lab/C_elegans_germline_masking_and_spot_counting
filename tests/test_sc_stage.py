"""The sc_trace stage wiring: off by default (empty tables, NA columns), on by config, germline-only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_end_to_end import patched_reader  # noqa: F401  (fixture)

from germquant import pipeline, schema
from germquant.config import load_config
from germquant.io.sample_metadata import expected_sc_count
from germquant.sc import skan_available


def _cfg():
    cfg = load_config("config/config.yaml")
    cfg._data["segmentation"]["nuclei"]["method"] = "classical"
    return cfg


def test_expected_sc_count():
    assert expected_sc_count("oocyte") == 6 and expected_sc_count("spermatocyte") == 5
    assert expected_sc_count("unknown") is None


def test_sc_trace_off_by_default(tmp_path, patched_reader):  # noqa: F811
    res = pipeline.process_image("20251105_n2_nohs_HERM_001.nd2", _cfg(), tmp_path / "r", prov=None)
    assert res["stages"]["sc_trace"]["status"] == "skipped" and res["stages"]["sc_trace"]["reason"] == "disabled"
    assert res["tables"]["sc_tracks"].empty and res["tables"]["sc_per_nucleus"].empty
    # the written tables are schema-conformed: declared columns present (NA) even when the stage is off
    out = tmp_path / "r"
    tracks = pd.read_csv(next(out.rglob("*__sc_tracks.csv")))
    assert list(tracks.columns[:len(schema.SC_TRACKS)]) == schema.SC_TRACKS and tracks.empty
    assert next(out.rglob("*__sc_per_nucleus.csv")).exists()
    nuc = pd.read_csv(next(out.rglob("*__nuclei.csv")))
    assert "sc_total_length_um" in nuc.columns and nuc["sc_total_length_um"].isna().all()


@pytest.mark.skipif(not skan_available(), reason="skan not installed")
def test_sc_trace_on_traces_germline_nuclei_only(tmp_path, patched_reader):  # noqa: F811
    cfg = _cfg()
    cfg.set("sc.trace.enabled", True)
    res = pipeline.process_image("20251105_n2_nohs_HERM_001.nd2", cfg, tmp_path / "r", prov=None)
    assert res["stages"]["sc_trace"]["status"] == "ran", res["stages"]["sc_trace"]
    per = res["tables"]["sc_per_nucleus"]
    nuc = res["tables"]["nuclei"]
    germ = nuc[nuc["in_germline"].astype(bool)]
    assert len(per) == len(germ)                                    # one row per germline nucleus
    assert set(per["nucleus_id"]) == set(germ["nucleus_id"])        # and none for the off-gonad junk
    assert (per["expected_n_tracks"] == 6).all()                    # HERM -> oocyte -> 6 SCs
    assert per["sc_total_length_um"].sum() > 0                      # the synthetic filaments are traced
    assert nuc.loc[nuc["in_germline"].astype(bool), "sc_total_length_um"].notna().all()
    assert nuc.loc[~nuc["in_germline"].astype(bool), "sc_total_length_um"].isna().all()
    isum = res["tables"]["image_summary"].iloc[0]
    assert np.isfinite(isum["mean_sc_total_length_um"]) and np.isfinite(isum["mean_sc_fragmentation_index"])
    assert any(f.startswith("sc_trace:n_traced=") for f in res["qc_flags"]) and "sc:uncalibrated" in res["qc_flags"]
    tracks = res["tables"]["sc_tracks"]
    assert (tracks["trace_method"] == "skan_iso").all() and set(tracks["nucleus_id"]) <= set(germ["nucleus_id"])
    # the stage does not touch spots or the RAD-51 count
    assert "n_spots" in nuc.columns and pd.api.types.is_integer_dtype(nuc["n_spots"])
    assert "spots" in res["stages"] and res["stages"]["spots"]["status"] == "ran"
