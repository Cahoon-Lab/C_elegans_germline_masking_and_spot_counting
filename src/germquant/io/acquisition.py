"""Acquisition metadata from the .nd2 text description: per-channel exposure (ms) and laser power.

Ported from analysis/coloc/scripts/build_acquisition_metadata.py (the `acquisition` stage, config key
acquisition.read_exposures). The Nikon description lists one "Plane #n:" block per channel with
"Name: <channel name>", "Exposure: <ms> ms" and "ExW:<name>; Power: <value>"; values are mapped onto the
resolved channel roles by channel name, so image_summary gets ``exp_ms_<role>`` and
``laser_pow_<role>`` for every resolved role (NA when the description does not carry them).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


def parse_description(desc: str) -> dict[str, dict[str, float | None]]:
    """{channel name: {"exposure_ms": float|None, "power": float|None}} from the nd2 description text."""
    out: dict[str, dict[str, float | None]] = {}
    for pl in re.split(r"Plane #\d+:", desc or "")[1:]:
        m = re.search(r"Name:\s*(\S+)", pl)
        if not m:
            continue
        name = m.group(1)
        exp = re.search(r"Exposure:\s*([\d.]+)\s*ms", pl)
        pw = re.search(rf"ExW:{re.escape(name)}; Power:\s*([\d.]+)", pl)
        out[name] = {"exposure_ms": float(exp.group(1)) if exp else None,
                     "power": float(pw.group(1)) if pw else None}
    return out


def read_description(nd2_path: str | Path) -> str:
    import nd2

    with nd2.ND2File(str(nd2_path)) as h:
        return str(h.text_info.get("description", "") or "")


def acquisition_fields(nd2_path: str | Path, channel_names: list[str], role_to_idx: dict) -> dict[str, float]:
    """image_summary fields exp_ms_<role> / laser_pow_<role> for every resolved role."""
    try:
        per_name = parse_description(read_description(nd2_path))
    except Exception as e:  # noqa: BLE001 - a metadata read must never fail the run
        log.warning("acquisition metadata unreadable (%s); fields left NA", e)
        per_name = {}
    fields: dict[str, float] = {}
    for role, idx in role_to_idx.items():
        if idx is None:
            continue
        name = str(channel_names[idx]) if idx < len(channel_names) else ""
        rec = per_name.get(name) or next((v for k, v in per_name.items() if k and (k in name or name in k)), {})
        fields[f"exp_ms_{role}"] = rec.get("exposure_ms") if rec else float("nan")
        fields[f"laser_pow_{role}"] = rec.get("power") if rec else float("nan")
        if fields[f"exp_ms_{role}"] is None:
            fields[f"exp_ms_{role}"] = float("nan")
        if fields[f"laser_pow_{role}"] is None:
            fields[f"laser_pow_{role}"] = float("nan")
    return fields
