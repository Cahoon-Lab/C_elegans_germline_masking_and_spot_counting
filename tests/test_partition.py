"""Partition coefficient and lit fraction: the ported math on planted data, and the stages end to end."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from test_envelope import patched_reader_lamin  # noqa: F401  (fixture)
from test_staging import IID

from germquant import pipeline
from germquant.config import load_config
from germquant.partition import (
    partition_metrics,
    pc_dm,
    run_granule_tail,
    run_partition,
    zone_voxel_map,
)
from germquant.partition.pc import WHOLE_GATES, ZONE_GATES
from germquant.staging import save_trace

SP = (0.2, 0.1083, 0.1083)


def _planted(enrich=2.0, seed=0):
    """A nucleus (envelope) at the centre, cytoplasm SYP that falls off with distance, granules with SYP
    = enrich x the local cytoplasm level, plus PGL puncta at the granules."""
    Z, Y, X = 40, 160, 160
    zz, yy, xx = np.indices((Z, Y, X)).astype(np.float32)
    r = np.sqrt(((zz - 20) * SP[0]) ** 2 + ((yy - 80) * SP[1]) ** 2 + ((xx - 80) * SP[2]) ** 2)
    env = r <= 3.0
    env_lab = env.astype(np.int32)
    dt = ndi.distance_transform_edt(~env, sampling=SP)
    rng = np.random.default_rng(seed)
    syp = (100.0 + 400.0 * np.exp(-dt / 1.5)).astype(np.float32) + rng.normal(0, 3, (Z, Y, X)).astype(np.float32)
    syp[env] = 2000.0
    gran = np.zeros((Z, Y, X), bool)
    pgl = rng.normal(50, 5, (Z, Y, X)).astype(np.float32)
    # granules on HALF a ring only (so the 180 degree rotation null lands on granule-free cytoplasm),
    # about 2 um apart so the tophat background blur cannot merge them
    for ang in np.linspace(0.15, np.pi - 0.15, 6):
        cy, cx = 80 + int(round(3.8 / SP[1] * np.sin(ang))), 80 + int(round(3.8 / SP[2] * np.cos(ang)))
        g = (((yy - cy) * SP[1]) ** 2 + ((xx - cx) * SP[2]) ** 2 + ((zz - 20) * SP[0]) ** 2) <= 0.35 ** 2
        gran |= g
        pgl[g] = 900.0
    syp[gran] = syp[gran] * enrich
    return syp, pgl, env_lab, gran, dt


def test_pc_dm_recovers_planted_enrichment():
    syp, _pgl, env_lab, gran, dt = _planted(enrich=2.0)
    cyto = ndi.binary_fill_holes(dt <= 2.5) & (env_lab == 0)
    bg = float(np.percentile(syp, 3))
    pc = pc_dm(syp, gran & cyto, cyto & ~gran, dt, bg, **WHOLE_GATES)
    # granules carry 2 x the local cytoplasm; after subtracting the far background the ratio sits above 2
    assert pc is not None and 1.8 < pc < 3.0, pc
    m = partition_metrics(syp, gran & cyto, cyto, dt, bg)
    assert m["partition_coef"] == pc and m["partition_coef_rot"] is not None and m["partition_coef_zshift"] is not None
    assert 0.8 < m["partition_coef_rot"] < 1.3 and m["partition_coef_specific"] > 1.3       # controls sit near 1
    assert pc_dm(syp, gran & cyto, cyto & ~gran, dt, bg, region=np.zeros_like(gran), **ZONE_GATES) is None
    flat = pc_dm(np.full_like(syp, 500.0), gran & cyto, cyto & ~gran, dt, 0.0, **WHOLE_GATES)
    assert flat is not None and abs(flat - 1.0) < 1e-6


def test_run_partition_regions_and_zone_map():
    syp, _pgl, env_lab, gran, _dt = _planted()
    zones = pd.DataFrame({"nucleus_id": [1], "zone": ["mid"]})
    res = run_partition(syp, gran, env_lab > 0, SP, env_labels_c=env_lab, zones=zones)
    t = res["table"].set_index("region")
    assert list(t.index) == ["whole", "early", "mid", "late", "pach"]
    assert t.loc["whole", "partition_coef"] > 1.5 and np.isnan(float(t.loc["early", "partition_coef"] or np.nan))
    assert abs(t.loc["mid", "partition_coef"] - t.loc["pach", "partition_coef"]) < 1e-9     # only one zoned nucleus
    assert res["summary"]["partition_coef_pach"] == t.loc["pach", "partition_coef"]
    zv = zone_voxel_map(env_lab, zones, SP)
    assert set(np.unique(zv)) == {2}                                                       # every voxel inherits "mid"


def test_run_granule_tail_lit_fraction():
    syp, pgl, env_lab, _gran, _dt = _planted(enrich=3.0)
    res = run_granule_tail(syp, pgl, env_lab, SP, drop_ids=[])
    t = res["table"]
    assert len(t) >= 4 and (t["excess_n"] > 0).mean() > 0.9
    s = res["summary"]
    assert s["tail_n_granules"] == len(t) and 0 <= s["tail_frac_excess_gt_0.25"] <= 1 and s["tail_n_no_envelope_dropped"] == 0
    res2 = run_granule_tail(syp, pgl, env_lab, SP, drop_ids=[1])              # dropping the only nucleus: no shell
    assert res2["summary"]["tail_n_granules"] == 0 and res2["summary"]["tail_n_no_envelope_dropped"] == 1


def _cfg(traces_file, **switches):
    cfg = load_config("config/config_ccw77.yaml")
    cfg._data["segmentation"]["nuclei"]["method"] = "classical"
    cfg.set("coloc.n_random", 10)
    cfg.set("sc.enabled", False)
    for k, v in {"envelope.enabled": True, "staging.enabled": True, "staging.traces_file": str(traces_file),
                 "granule.method": "imaris_tophat", "granule.region": "cytoplasm_shell",
                 "partition.enabled": True, "granule_tail.enabled": True, **switches}.items():
        cfg.set(k, v)
    return cfg


def test_partition_and_tail_stages_run(tmp_path, patched_reader_lamin):  # noqa: F811
    tf = tmp_path / "traces.json"
    save_trace(tf, IID, [(6.0, 8.0), (40.0, 8.0)], "traced")
    res = pipeline.process_image(f"{IID}.nd2", _cfg(tf), tmp_path / "r", prov=None)
    for s in ("envelope", "staging", "granule", "partition", "granule_tail"):
        assert res["stages"][s]["status"] == "ran", (s, res["stages"][s])
    part = res["tables"]["partition"]
    assert list(part["region"]) == ["whole", "early", "mid", "late", "pach"]
    assert part["region_voxels"].iloc[0] > 0 and part["bg"].notna().all()
    isum = res["tables"]["image_summary"].iloc[0]
    assert "partition_coef_whole" in isum.index and "tail_n_granules" in isum.index
    assert any(f == "partition:frame=envelope,granules=imaris_tophat" for f in res["qc_flags"])
    assert any(f.startswith("granule_tail:frame=envelope") for f in res["qc_flags"])
    assert next((tmp_path / "r").rglob("*__partition.csv")).exists() and next((tmp_path / "r").rglob("*__granule_tail.csv")).exists()
    # without envelope or staging the stages still run on the DAPI frame and the whole region only
    res2 = pipeline.process_image(f"{IID}.nd2", _cfg(tf, **{"envelope.enabled": False, "staging.enabled": False}),
                                  tmp_path / "r2", prov=None)
    assert res2["stages"]["partition"]["status"] == "ran"
    assert list(res2["tables"]["partition"]["region"]) == ["whole"]
    assert any(f == "partition:frame=dapi,granules=imaris_tophat" for f in res2["qc_flags"])


def test_ccw77_partition_profile_loads_and_switches():
    cfg = load_config("config/profiles/ccw77_partition.yaml")
    from germquant.stages import stage_enabled

    for s in ("envelope", "audit", "staging", "partition", "granule_tail", "acquisition", "coloc", "granule"):
        assert stage_enabled(cfg, s), s
    assert cfg.get("granule.method") == "imaris_tophat" and cfg.get("sc.enabled") is False
    assert (cfg.base_dir / cfg.get("staging.traces_file")).is_file()
    assert cfg.hash != load_config("config/config_ccw77.yaml").hash