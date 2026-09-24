"""granule.method imaris_tophat (the August recipe) and granule.region cytoplasm_shell."""
from __future__ import annotations

import numpy as np
import pandas as pd
from test_envelope import patched_reader_lamin  # noqa: F401  (fixture: synthetic 4-colour gonad)

from germquant import pipeline
from germquant.config import load_config
from germquant.granule import segment_granules
from germquant.granule.segment import TOPHAT_DEFAULTS, segment_granules_tophat

SP = (0.2, 0.1083, 0.1083)


def _blobs():
    Z, Y, X = 30, 120, 120
    zz, yy, xx = np.indices((Z, Y, X)).astype(np.float32)
    img = np.zeros((Z, Y, X), np.float32)
    rng = np.random.default_rng(1)
    img += rng.normal(100, 8, img.shape).astype(np.float32)
    for (cz, cy, cx, amp) in [(15, 30, 30, 900), (15, 60, 80, 600), (15, 90, 40, 300)]:
        r2 = ((zz - cz) * SP[0]) ** 2 + ((yy - cy) * SP[1]) ** 2 + ((xx - cx) * SP[2]) ** 2
        img += amp * np.exp(-r2 / (2 * 0.25 ** 2))
    region = np.ones(img.shape, bool)
    region[:, :, 100:] = False                              # a region edge: no granule may land there
    return img, region


def test_tophat_recipe_finds_bright_puncta_and_gates_volume():
    img, region = _blobs()
    lab, df = segment_granules_tophat(img, region, SP)
    assert 2 <= len(df) <= 3 and (df["detector"] == "imaris_tophat_cc").all()
    assert lab.max() == len(df) and set(np.unique(lab)) == set(range(len(df) + 1))
    assert (df["volume_um3"] >= TOPHAT_DEFAULTS["min_volume_um3"]).all()
    assert (df["volume_um3"] <= TOPHAT_DEFAULTS["max_volume_um3"]).all()
    assert (df["x_um"] < 100 * SP[2]).all()
    # a higher k finds fewer granules; an empty region finds none
    _, fewer = segment_granules_tophat(img, region, SP, k_noise=40.0)
    assert len(fewer) <= len(df)
    lab0, df0 = segment_granules_tophat(img, np.zeros_like(region), SP)
    assert df0.empty and lab0.max() == 0
    # the default recipe on the same data still works (untouched)
    _, dtri = segment_granules(img, region, SP)
    assert isinstance(dtri, pd.DataFrame)


def _cfg(**switches):
    cfg = load_config("config/config_ccw77.yaml")
    cfg._data["segmentation"]["nuclei"]["method"] = "classical"
    cfg.set("coloc.n_random", 10)
    for k, v in switches.items():
        cfg.set(k, v)
    return cfg


def test_granule_stage_default_is_triangle_over_coloc_region(tmp_path, patched_reader_lamin):  # noqa: F811
    res = pipeline.process_image("20260708_ccw77_IF_pgl1_syp3_lmn1_HS_male_008.nd2", _cfg(), tmp_path / "r", prov=None)
    assert res["stages"]["granule"]["status"] == "ran"
    assert not any(f.startswith("granule:method=") for f in res["qc_flags"])     # default path: no new flag text
    assert (res["tables"]["granules"]["detector"] == "threshold_cc").all()


def test_granule_stage_imaris_tophat_in_cytoplasm_shell(tmp_path, patched_reader_lamin):  # noqa: F811
    cfg = _cfg(**{"envelope.enabled": True, "granule.method": "imaris_tophat", "granule.region": "cytoplasm_shell"})
    res = pipeline.process_image("20260708_ccw77_IF_pgl1_syp3_lmn1_HS_male_008.nd2", cfg, tmp_path / "r", prov=None)
    assert res["stages"]["envelope"]["status"] == "ran" and res["stages"]["granule"]["status"] == "ran", res["stages"]
    assert any(f == "granule:method=imaris_tophat,region=cytoplasm_shell" for f in res["qc_flags"])
    gr = res["tables"]["granules"]
    assert len(gr) >= 1 and (gr["detector"] == "imaris_tophat_cc").all()
    assert gr["nucleus_id"].notna().all()                                # assigned to nuclei
    assert res["stages"]["coloc"]["status"] == "ran"                      # coloc consumes the new labels
    # without the envelope stage the shell falls back to the DAPI nuclei and says so
    cfg2 = _cfg(**{"granule.method": "imaris_tophat", "granule.region": "cytoplasm_shell"})
    res2 = pipeline.process_image("20260708_ccw77_IF_pgl1_syp3_lmn1_HS_male_008.nd2", cfg2, tmp_path / "r2", prov=None)
    assert any(f == "granule:method=imaris_tophat,region=cytoplasm_shell_dapi" for f in res2["qc_flags"])
