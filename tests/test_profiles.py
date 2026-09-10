"""Profiles (`extends:`), the hash rule they must respect, per-stage provenance and the done marker."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_end_to_end import patched_reader  # noqa: F401  (fixture)

from germquant import pipeline, provenance
from germquant.config import _base_dir_for, load_config
from germquant.stages import STAGES, stage_enabled, stage_hashes

ROOT = Path(__file__).resolve().parents[1]


def test_plain_config_hash_is_unchanged_by_profile_support():
    """The three shipped config files must hash exactly as they did before `extends` existed
    (raw text + channel-map text): the sentinel is the synthetic golden's config_hash.txt."""
    cfg = load_config(ROOT / "config" / "config.yaml")
    sentinel = (ROOT / "tests" / "golden" / "synthetic" / "config_hash.txt").read_text().strip()
    assert cfg.hash == sentinel
    for name in ("config_ccw77.yaml", "config_n2dryice.yaml"):
        c = load_config(ROOT / "config" / name)
        assert c.hash and c.hash != sentinel and c.base_dir == ROOT


def test_base_dir_resolution_for_profiles(tmp_path):
    assert _base_dir_for(ROOT / "config" / "config.yaml") == ROOT
    assert _base_dir_for(ROOT / "config" / "profiles" / "rad51_foci.yaml") == ROOT
    stray = tmp_path / "x.yaml"
    assert _base_dir_for(stray) == tmp_path


def test_profile_overlays_base_and_hashes_differently():
    base = load_config(ROOT / "config" / "config.yaml")
    prof = load_config(ROOT / "config" / "profiles" / "rad51_foci.yaml")
    assert prof.get("coloc.enabled") is False
    assert prof.get("spots.enabled") == base.get("spots.enabled")          # inherited
    assert prof.get("segmentation.nuclei.method") == base.get("segmentation.nuclei.method")
    assert "extends" not in prof.as_dict()
    assert prof.hash != base.hash
    assert prof.channel_map.roles.keys() == base.channel_map.roles.keys()
    assert stage_enabled(prof, "granule") is False and stage_enabled(prof, "spots") is True
    seg = load_config(ROOT / "config" / "profiles" / "segmentation_only.yaml")
    assert stage_enabled(seg, "spots") is False and stage_enabled(seg, "coloc") is False
    assert seg.hash != prof.hash


def test_profile_hash_is_stable_and_override_sensitive():
    a = load_config(ROOT / "config" / "profiles" / "rad51_foci.yaml")
    b = load_config(ROOT / "config" / "profiles" / "rad51_foci.yaml")
    assert a.hash == b.hash
    b.set("spots.enabled", False)
    assert a.hash != b.hash


def test_missing_extends_target_is_an_error(tmp_path):
    p = tmp_path / "config" / "profiles" / "bad.yaml"
    p.parent.mkdir(parents=True)
    p.write_text("extends: nope.yaml\n")
    with pytest.raises(FileNotFoundError):
        load_config(p)


def test_stage_hashes_are_per_stage():
    cfg = load_config(ROOT / "config" / "config.yaml")
    roles = {"dna": 0, "central_element": 1, "foci": 2, "granule": None, "lamin": None}
    h1 = stage_hashes(cfg, roles, {"cellpose": "4.2.0"}, model_sha="abc")
    assert set(h1) == {s.name for s in STAGES}
    cfg.set("coloc.n_random", 5)
    h2 = stage_hashes(cfg, roles, {"cellpose": "4.2.0"}, model_sha="abc")
    assert h2["spots"] == h1["spots"] and h2["segment"] == h1["segment"]     # untouched stages agree
    assert h2["coloc"] != h1["coloc"] and h2["granule"] != h1["granule"]      # coloc section changed
    h3 = stage_hashes(cfg, roles, {"cellpose": "4.2.0"}, model_sha="def")
    assert h3["segment"] != h2["segment"] and h3["spots"] == h2["spots"]      # model digest is segment-only


def test_file_sha256_cache_and_absent(tmp_path):
    assert provenance.file_sha256(None) == "absent"
    assert provenance.file_sha256("cpsam") == "absent"
    assert provenance.file_sha256(tmp_path) == "absent"
    f = tmp_path / "m.bin"
    f.write_bytes(b"model")
    h = provenance.file_sha256(f)
    assert len(h) == 64 and provenance.file_sha256(f) == h


def test_process_image_writes_manifest_extras_and_done_marker(tmp_path, patched_reader):  # noqa: F811
    cfg = load_config(ROOT / "config" / "config.yaml")
    cfg._data["segmentation"]["nuclei"]["method"] = "classical"
    out = tmp_path / "r"
    res = pipeline.process_image("20251105_n2_nohs_HERM_001.nd2", cfg, out, xy_stride=2, prov=None)
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert man["xy_stride"] == 2 and man["z_range"] is None
    assert set(man["stage_hashes"]) == {s.name for s in STAGES}
    assert "segment" in man["enabled_stages"] and "spots" in man["enabled_stages"]
    assert "skan" in man["tool_versions"]
    done = provenance.read_done_marker(out, res["image_id"])
    assert done and done["config_hash"] == cfg.hash and done["stage_hashes"] == man["stage_hashes"]
    rec = json.loads(next(out.rglob("*__stages.json")).read_text(encoding="utf-8"))
    assert rec["complete"] is True and rec["stage_hashes"] == man["stage_hashes"]
    assert provenance.read_done_marker(out, "nothing") is None
