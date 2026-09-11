"""A second spots instance (COSA-1 crossover foci) on another channel role: same detector, own table,
own nuclei column, expected count per nucleus; the RAD-51 path stays untouched."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from test_end_to_end import patched_reader_4ch  # noqa: F401  (fixture: DAPI / SYP / RAD-51 / PGL-1 puncta)

from germquant import pipeline  # noqa: I001
from germquant.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _profile(tmp_path, instances: str):
    cm = tmp_path / "cosa_map.yaml"
    cm.write_text(
        "match_by: name\nroles:\n"
        "  dna: {names: ['405'], index: 0, required: true, marker: DAPI}\n"
        "  central_element: {names: ['477'], index: 1, required: true, marker: SYP}\n"
        "  foci: {names: ['545'], index: 2, required: false, marker: RAD-51}\n"
        "  crossover_foci: {names: ['647'], index: 3, required: false, marker: COSA-1}\n")
    p = tmp_path / "cosa1.yaml"
    p.write_text(f"extends: {(ROOT / 'config' / 'config.yaml').as_posix()}\n"
                 f"io:\n  channel_map: {cm.as_posix()}\n"
                 f"segmentation:\n  nuclei:\n    method: classical\n"
                 f"coloc:\n  enabled: false\n"
                 f"{instances}")
    return load_config(p)


def test_cosa1_instance_counts_on_its_own_channel(tmp_path, patched_reader_4ch):  # noqa: F811
    cfg = _profile(tmp_path, "spots:\n  instances:\n    - name: crossover\n      role: crossover_foci\n"
                             "      restrict_to_zone: late\n      expected_from_germ_cell: true\n")
    res = pipeline.process_image("20251105_n2_nohs_HERM_001.nd2", cfg, tmp_path / "r", prov=None)
    assert res["stages"]["spots"]["status"] == "ran" and res["stages"]["spots_crossover"]["status"] == "ran", res["stages"]
    nuc = res["tables"]["nuclei"]
    assert "n_spots" in nuc.columns and "n_spots_crossover" in nuc.columns
    assert pd.api.types.is_integer_dtype(nuc["n_spots_crossover"]) and nuc["n_spots_crossover"].sum() > 0
    assert (nuc.loc[~nuc["in_germline"].astype(bool), "n_spots_crossover"] == 0).all()
    xo = res["tables"]["spots_crossover"]
    assert len(xo) > 0 and (xo["marker"] == "COSA-1").all() and set(xo.columns) >= set(res["tables"]["spots"].columns)
    isum = res["tables"]["image_summary"].iloc[0]
    assert isum["crossover_expected_per_nucleus"] == 6                       # HERM -> oocyte -> 6 bivalents
    assert isum["mean_n_spots_crossover"] > 0 and "mean_n_spots_crossover_late" not in isum.index   # no staging: no late mean
    assert any(f.startswith("spots_crossover:spotmax_n=") for f in res["qc_flags"])
    assert any(f.startswith("spots:spotmax_n=") for f in res["qc_flags"])     # the RAD-51 instance is unchanged
    written = next((tmp_path / "r").rglob("*__spots_crossover.csv"))
    assert list(pd.read_csv(written).columns[:3]) == ["spot_id", "nucleus_id", "marker"]


def test_no_instances_by_default(tmp_path, patched_reader_4ch):  # noqa: F811
    cfg = _profile(tmp_path, "")
    res = pipeline.process_image("20251105_n2_nohs_HERM_001.nd2", cfg, tmp_path / "r", prov=None)
    assert not any(k.startswith("spots_") for k in res["stages"]) and "spots_crossover" not in res["tables"]
    assert "n_spots_crossover" not in res["tables"]["nuclei"].columns


def test_instance_without_its_channel_is_skipped(tmp_path, patched_reader_4ch):  # noqa: F811
    cfg = _profile(tmp_path, "spots:\n  instances:\n    - name: crossover\n      role: absent_role\n")
    res = pipeline.process_image("20251105_n2_nohs_HERM_001.nd2", cfg, tmp_path / "r", prov=None)
    assert res["stages"]["spots_crossover"]["status"] == "skipped" and "no absent_role channel" in res["stages"]["spots_crossover"]["reason"]
    assert res["tables"]["spots_crossover"].empty
