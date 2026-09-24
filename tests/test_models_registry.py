"""`germquant get-model`: download location, checksum verification, skip-if-present, failure modes."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from germquant import models_registry as R


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(payload: bytes):
    def urlopen(req, timeout=None):
        return _Resp(payload)
    return urlopen


@pytest.fixture
def small_model(monkeypatch):
    payload = b"model bytes " * 1000
    monkeypatch.setitem(R.MODELS, "tiny", {"tag": "model-tiny-v1", "asset": "tiny", "sha256": hashlib.sha256(payload).hexdigest(),
                                           "size": len(payload), "what": "test"})
    return payload


def test_asset_url_and_models_dir(tmp_path, monkeypatch):
    assert R.asset_url("germline_nuclei_combined").endswith("/releases/download/model-germline_nuclei_combined-v1/germline_nuclei_combined")
    monkeypatch.delenv("GERMQUANT_MODELS", raising=False)
    assert R.models_dir(tmp_path) == tmp_path / "models" / "models"
    monkeypatch.setenv("GERMQUANT_MODELS", str(tmp_path / "hpc"))
    assert R.models_dir(tmp_path) == tmp_path / "hpc" / "models"
    m = R.MODELS["germline_nuclei_combined"]
    assert len(m["sha256"]) == 64 and m["size"] == 1218646375


def test_get_model_downloads_verifies_and_skips(tmp_path, monkeypatch, small_model):
    import urllib.request

    monkeypatch.delenv("GERMQUANT_MODELS", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(small_model))
    dest = R.get_model("tiny", tmp_path, quiet=True)
    assert dest == tmp_path / "models" / "models" / "tiny" and dest.read_bytes() == small_model
    assert not dest.with_suffix(".part").exists()
    # second call: present and verified, no download
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download expected")))
    assert R.get_model("tiny", tmp_path, quiet=True) == dest
    # a corrupted file is re-downloaded
    dest.write_bytes(b"garbage")
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(small_model))
    assert R.get_model("tiny", tmp_path, quiet=True).read_bytes() == small_model


def test_get_model_rejects_bad_checksum_and_short_download(tmp_path, monkeypatch, small_model):
    import urllib.request

    monkeypatch.delenv("GERMQUANT_MODELS", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(small_model[:-5]))
    with pytest.raises(RuntimeError, match="ended early"):
        R.get_model("tiny", tmp_path, quiet=True)
    assert not (tmp_path / "models" / "models" / "tiny").exists()
    wrong = b"x" * len(small_model)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(wrong))
    with pytest.raises(RuntimeError, match="checksum"):
        R.get_model("tiny", tmp_path, quiet=True)
    assert not (tmp_path / "models" / "models" / "tiny.part").exists()
    with pytest.raises(KeyError):
        R.get_model("nope", tmp_path, quiet=True)


def test_cli_get_model_wiring(monkeypatch):
    from germquant import cli

    called = {}
    monkeypatch.setattr(R, "get_model", lambda name, base_dir=None, force=False, quiet=False: called.update(name=name, force=force) or Path("x"))
    assert cli.main(["get-model", "--force"]) == 0
    assert called == {"name": "germline_nuclei_combined", "force": True}
