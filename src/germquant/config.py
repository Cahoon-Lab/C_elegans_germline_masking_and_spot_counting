"""Config loading. Thin wrapper over YAML with attribute access + the channel map."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .io.channel_map import ChannelMap


class Config:
    """Dict-backed config with attribute access and a resolved ChannelMap.

    Access nested keys via attributes (``cfg.segmentation.nuclei.method``) or
    ``cfg.get("segmentation.nuclei.method", default)``.
    """

    def __init__(self, data: dict, *, base_dir: Path, channel_map: ChannelMap, raw_text: str,
                 channel_map_text: str = ""):
        self._data = data
        self.base_dir = base_dir
        self.channel_map = channel_map
        self._raw_text = raw_text
        self._channel_map_text = channel_map_text
        self._overrides: dict[str, Any] = {}

    # ---- access helpers ----
    def __getattr__(self, name: str) -> Any:
        try:
            val = self._data[name]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(name) from e
        return _Node(val) if isinstance(val, dict) else val

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        """Override a (possibly nested) value at runtime — for CLI flags like ``--no-spots`` that
        flip a config switch. The override shows up in ``as_dict()`` AND changes ``hash``, so a
        ``--no-spots`` run is provenance-distinguishable from a normal one."""
        node = self._data
        parts = dotted.split(".")
        for part in parts[:-1]:
            nxt = node.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise ValueError(f"cannot set {dotted!r}: {part!r} is not a section")
            node = nxt
        node[parts[-1]] = value
        self._overrides[dotted] = value

    @property
    def hash(self) -> str:
        """Stable hash of the resolved config — goes into the provenance manifest.

        Includes the referenced channel-map file (it sets which channel is DAPI/SYP/RAD-51, as
        much an effective parameter as config.yaml) and any runtime overrides (e.g. ``--no-spots``).
        """
        payload = self._raw_text + "\x00" + self._channel_map_text
        if self._overrides:
            payload += "\x00" + json.dumps(self._overrides, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def as_dict(self) -> dict:
        return json.loads(json.dumps(self._data))


class _Node:
    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, name: str) -> Any:
        try:
            val = self._d[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return _Node(val) if isinstance(val, dict) else val

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def __repr__(self) -> str:  # pragma: no cover
        return f"_Node({self._d!r})"


def _base_dir_for(path: Path) -> Path:
    """The directory that relative paths in a config (channel map, model) resolve against: the parent
    of the nearest ancestor directory named ``config`` (so config/config.yaml and
    config/profiles/x.yaml both resolve against the repo root), else the file's own directory."""
    for d in path.resolve().parents:
        if d.name == "config":
            return d.parent
    return path.parent


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> Config:
    """Load a config file. A file may name a base with a top-level ``extends: <path>`` (relative to the
    file's directory, or to the config dir): the base is loaded first and the file's own sections are
    deep-merged over it (a "profile"). Hash rule: a file WITHOUT ``extends`` hashes exactly as it always
    has (its raw text plus the channel-map text plus runtime overrides), so existing runs keep
    reproducing; a profile hashes the base text plus its own text plus the channel-map text."""
    path = Path(path)
    raw_text = path.read_text()
    data = yaml.safe_load(raw_text) or {}
    base_dir = _base_dir_for(path)
    hash_text = raw_text

    ext = data.pop("extends", None)
    if ext:
        cand = [path.parent / ext, base_dir / "config" / ext, base_dir / ext]
        ext_path = next((c for c in cand if c.is_file()), None)
        if ext_path is None:
            raise FileNotFoundError(f"{path}: extends {ext!r} not found (tried {[str(c) for c in cand]})")
        base = load_config(ext_path)
        data = _deep_merge(base.as_dict(), data)
        hash_text = base._raw_text + "\x00extends\x00" + raw_text

    cm_rel = data.get("io", {}).get("channel_map")
    if cm_rel is None:
        raise ValueError("config.io.channel_map is required")
    cm_path = (base_dir / cm_rel) if not Path(cm_rel).is_absolute() else Path(cm_rel)
    channel_map = ChannelMap.from_yaml(cm_path)
    cm_text = cm_path.read_text() if cm_path.exists() else ""

    return Config(data, base_dir=base_dir, channel_map=channel_map, raw_text=hash_text,
                  channel_map_text=cm_text)
