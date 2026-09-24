"""Where the trained models live and how to fetch them: `germquant get-model`.

The trained Cellpose nucleus model is 1.2 GB and is not in the git repository; it is published as a
GitHub release asset of this repository. `get_model` downloads it to ``models/models/<name>`` under
the repository (or under ``$GERMQUANT_MODELS`` when that is set, the HPC / container binding),
verifies the sha256 and refuses a corrupt or truncated download, so the path config.yaml already
points at (``segmentation.nuclei.cellpose_model: models/models/germline_nuclei_combined``) just
works on a new computer. A new model version is a new release tag and a new entry here.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

REPO = "ryan-eastman/C_elegans_germline_masking_and_spot_counting"

MODELS = {
    "germline_nuclei_combined": {
        "tag": "model-germline_nuclei_combined-v1",
        "asset": "germline_nuclei_combined",
        "sha256": "e4e32320cfc321a145ccbc0c52138eded50e37b87130f34bae2296c88b148f57",
        "size": 1218646375,
        "what": "Cellpose-SAM fine-tuned on real hand-labelled germline nuclei plus synthetic nuclei "
                "(June 2026): the recommended nucleus model, used by every validated run.",
    },
}


def asset_url(name: str) -> str:
    m = MODELS[name]
    return f"https://github.com/{REPO}/releases/download/{m['tag']}/{m['asset']}"


def models_dir(base_dir: str | Path | None = None) -> Path:
    """``$GERMQUANT_MODELS/models`` when the variable is set, else ``<repo>/models/models``."""
    root = os.environ.get("GERMQUANT_MODELS")
    if root:
        return Path(root) / "models"
    if base_dir is None:
        from .provenance import _repo_root

        base_dir = _repo_root() or Path.cwd()
    return Path(base_dir) / "models" / "models"


def sha256_of(path: Path, progress: bool = False) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def get_model(name: str = "germline_nuclei_combined", base_dir: str | Path | None = None, *, force: bool = False,
              quiet: bool = False) -> Path:
    """Download the model if it is not already present and correct. Returns the model path."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    m = MODELS[name]
    dest = models_dir(base_dir) / name
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    if dest.is_file() and not force:
        if dest.stat().st_size == m["size"] and sha256_of(dest) == m["sha256"]:
            say(f"{name}: already present and verified at {dest}")
            return dest
        say(f"{name}: present but wrong size or checksum; downloading again")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    url = asset_url(name)
    say(f"{name}: downloading {m['size'] / 1e9:.2f} GB from {url}")
    _download(url, tmp, m["size"], say)
    got = sha256_of(tmp)
    if got != m["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: checksum mismatch after download (got {got[:12]}..., expected "
                           f"{m['sha256'][:12]}...); the file was discarded, try again")
    os.replace(tmp, dest)
    say(f"{name}: verified and installed at {dest}")
    return dest


def _download(url: str, tmp: Path, size: int, say) -> None:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "germquant get-model"})
    t0 = time.time()
    done = 0
    last = 0
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if done - last >= (100 << 20):
                last = done
                rate = done / max(time.time() - t0, 1e-6) / 1e6
                say(f"  {done >> 20} MB of {size >> 20} MB ({rate:.0f} MB/s)")
    if size and done != size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download ended early ({done} of {size} bytes); check the connection and try again")


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="download and verify a trained model")
    ap.add_argument("name", nargs="?", default="germline_nuclei_combined", choices=sorted(MODELS))
    ap.add_argument("--force", action="store_true", help="download even if a verified copy exists")
    a = ap.parse_args(argv)
    try:
        get_model(a.name, force=a.force)
    except Exception as e:  # noqa: BLE001 - the message is the point for a lab user
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0
