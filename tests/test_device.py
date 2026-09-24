"""Device selection (CUDA / Apple Metal / CPU) and how the nucleus segmenter uses it."""
from __future__ import annotations

import numpy as np
import pytest

from germquant import device as D
from germquant.segment import nuclei as N


@pytest.mark.parametrize("have,want,expect", [
    ({"cuda": True, "mps": False}, None, "cuda"),
    ({"cuda": True, "mps": True}, None, "cuda"),          # CUDA first when both exist
    ({"cuda": False, "mps": True}, None, "mps"),
    ({"cuda": False, "mps": False}, None, "cpu"),
    ({"cuda": True, "mps": True}, "mps", "mps"),          # an explicit preference wins
    ({"cuda": False, "mps": True}, "cuda", "mps"),        # unavailable preference: next accelerator
    ({"cuda": False, "mps": False}, "mps", "cpu"),
    ({"cuda": True, "mps": True}, "cpu", "cpu"),
    ({"cuda": True, "mps": True}, "nonsense", "cuda"),    # unknown -> auto
])
def test_select_device_order(monkeypatch, have, want, expect):
    monkeypatch.setattr(D, "available", lambda: have)
    monkeypatch.delenv("GERMQUANT_DEVICE", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    assert D.select_device(want) == expect
    if expect == "mps":
        import os

        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_env_override_beats_config(monkeypatch):
    monkeypatch.setattr(D, "available", lambda: {"cuda": True, "mps": True})
    monkeypatch.setenv("GERMQUANT_DEVICE", "cpu")
    assert D.select_device("cuda") == "cpu"


def test_available_and_describe_do_not_raise():
    have = D.available()
    assert set(have) == {"cuda", "mps"} and all(isinstance(v, bool) for v in have.values())
    assert isinstance(D.describe(), str) and D.describe()


class _FakeModel:
    calls: "list[dict]" = []  # noqa: RUF012 - test spy, reset by the test

    def __init__(self, **kw):
        _FakeModel.calls.append(kw)

    def eval(self, dna, **kw):
        return [np.zeros(dna.shape, np.int32)]


def test_cellpose_gets_an_explicit_device_and_float32_on_mps(monkeypatch):
    import types

    fake_models = types.SimpleNamespace(CellposeModel=_FakeModel)
    monkeypatch.setitem(__import__("sys").modules, "cellpose", types.SimpleNamespace(models=fake_models))
    monkeypatch.setitem(__import__("sys").modules, "cellpose.models", fake_models)
    monkeypatch.setattr(D, "available", lambda: {"cuda": False, "mps": True})
    monkeypatch.setattr(D, "torch_device", lambda name: f"device:{name}")
    monkeypatch.delenv("GERMQUANT_DEVICE", raising=False)
    _FakeModel.calls.clear()
    dna = np.zeros((4, 8, 8), np.float32)
    N._cellpose(dna, (0.4, 0.2, 0.2), "cpsam", 3.0)
    kw = _FakeModel.calls[-1]
    assert kw["device"] == "device:mps" and kw["gpu"] is True and kw["use_bfloat16"] is False
    N._cellpose(dna, (0.4, 0.2, 0.2), "some/model/file", 3.0, device_pref="cpu")
    kw = _FakeModel.calls[-1]
    assert kw["device"] == "device:cpu" and kw["gpu"] is False and kw["pretrained_model"] == "some/model/file"


def test_accelerator_failure_retries_on_cpu(monkeypatch):
    seen = []

    def fake_cellpose(dna, spacing, model, diam, device_pref=None):
        seen.append(device_pref)
        if device_pref != "cpu":
            raise RuntimeError("MPS backend out of memory")
        return np.ones(dna.shape, np.int32)

    monkeypatch.setattr(N, "_cellpose", fake_cellpose)
    monkeypatch.setattr(D, "available", lambda: {"cuda": False, "mps": True})
    monkeypatch.delenv("GERMQUANT_DEVICE", raising=False)
    _labels, method = N.segment_nuclei(np.zeros((4, 8, 8), np.float32), (0.4, 0.2, 0.2), method="cellpose", min_volume_um3=0.0)
    assert method == "cellpose" and seen == [None, "cpu"]
