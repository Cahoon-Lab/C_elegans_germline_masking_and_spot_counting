"""`germquant trace`: the pop-up polyline tool (TkAgg) for drawing the pachytene region on each gonad,
ported from analysis/coloc/scripts/trace_pachytene.py. Reads DAPI and SYP straight from the .nd2 at an
xy stride (no crop cache), shows max projections / z-slabs / single planes with CLAHE, and saves the
polyline in WHOLE-IMAGE microns to the traces file the staging stage reads.

Keys: click = add point (start at the pachytene START, end at its END); z undo; r reset; enter accept;
s skip; b back; q quit; scroll wheel = single z-planes; m or 4 max projection; 1/2/3 lower/mid/upper
z-slab; v CLAHE; up/down brightness; d SYP overlay; c nucleus centroids.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..fsutil import long_path
from ..io import read_nd2_metadata, read_stack
from .zones import OFF_AXIS_UM, load_traces, save_trace

log = logging.getLogger(__name__)


def results_images(results_root: str | Path) -> list[tuple[str, Path, Path]]:
    """(image_id, nuclei.csv, nd2 path) for every finished image under a results tree (file_path column).
    Walked through the extended-length path so deep NAS trees work."""
    from ..batch import _find

    out = []
    for f in _find(Path(results_root), "__nuclei.csv"):
        iid = f.name.split("__")[0]
        try:
            fp = pd.read_csv(long_path(f), usecols=["file_path"], nrows=1)["file_path"].iloc[0]
        except Exception as e:  # noqa: BLE001 - a table without provenance cannot be traced
            log.warning("trace: skipping %s (%s: %s)", f, type(e).__name__, e)
            continue
        out.append((iid, f, Path(str(fp))))
    return out


def load_display(nd2_path: Path, cfg, stride: int = 2):
    """(dapi, syp) (Z, Y, X) at `stride`, plus (dy, dx) of the DISPLAY pixels in um. read_stack already
    folds the stride into the spacing it returns, so it is used as is."""
    meta = read_nd2_metadata(nd2_path)
    role_to_idx, _ = cfg.channel_map.resolve(meta["channel_names"])
    st = read_stack(nd2_path, xy_stride=stride)
    dapi = st.data[role_to_idx["dna"]] if role_to_idx.get("dna") is not None else st.data[0]
    ce = role_to_idx.get("central_element")
    syp = st.data[ce] if ce is not None else np.zeros_like(dapi)
    _dz, dy, dx = st.spacing
    return np.asarray(dapi), np.asarray(syp), (float(dy), float(dx))


def run_gui(items: list[tuple[str, Path, Path]], cfg, traces_file: Path, stride: int = 2,
            off_axis_um: float = OFF_AXIS_UM) -> None:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from skimage import exposure

    state = {"idx": 0, "pts": [], "show_syp": True, "show_cent": False, "zmode": "mid", "zi": 0,
             "clahe": True, "gamma": 1.0}
    fig, ax = plt.subplots(figsize=(15, 10))
    try:
        fig.canvas.manager.set_window_title("germquant trace: pachytene tracer")
    except Exception as e:  # noqa: BLE001 - cosmetic; some backends have no window manager
        log.debug("no window title: %s", e)
    cache: dict = {"iid": None, "stack": None, "views": {}, "pix": (1.0, 1.0)}

    def load(iid, nd2):
        if cache["iid"] != iid:
            dapi, syp, pix = load_display(nd2, cfg, stride)
            cache.update(iid=iid, stack=(dapi, syp), views={}, pix=pix)
            state["zi"] = dapi.shape[0] // 2

    def view():
        D, S = cache["stack"]
        nz = D.shape[0]
        plane = state["zmode"] == "plane"
        key = ("plane", state["zi"], state["clahe"]) if plane else (state["zmode"], state["clahe"])
        if not plane and key in cache["views"]:
            return cache["views"][key]
        if plane:
            zi = int(np.clip(state["zi"], 0, nz - 1))
            d, s = D[zi].astype(np.float32), S[zi].astype(np.float32)
        else:
            zsl = {"all": slice(None), "low": slice(0, max(1, nz // 3)),
                   "mid": slice(nz // 4, max(nz // 4 + 1, 3 * nz // 4)), "high": slice(2 * nz // 3, nz)}[state["zmode"]]
            d, s = D[zsl].max(0).astype(np.float32), S[zsl].max(0).astype(np.float32)

        def stretch(a, lo, hi):
            p0, p1 = np.percentile(a, [lo, hi])
            return np.clip((a - p0) / (p1 - p0 + 1e-9), 0, 1)

        dn = stretch(d, 1.0, 99.7)
        if state["clahe"]:
            dn = exposure.equalize_adapthist(dn, kernel_size=128, clip_limit=0.01).astype(np.float32)
        v = (dn, stretch(s, 40.0, 99.7))
        if not plane:
            cache["views"][key] = v
        return v

    def draw():
        iid, nuc_csv, nd2 = items[state["idx"]]
        load(iid, nd2)
        keep_view = state.get("cur_iid") == iid and ax.images
        if keep_view:
            xl, yl = ax.get_xlim(), ax.get_ylim()
        state["cur_iid"] = iid
        dn, sn = view()
        dg = np.power(dn, state["gamma"]) if state["gamma"] != 1.0 else dn
        rgb = np.stack([dg, dg, dg], -1)
        if state["show_syp"]:
            rgb[..., 0] = np.maximum(rgb[..., 0], sn)
        ax.clear()
        h, w = dn.shape
        py, px = cache["pix"]
        ax.imshow(rgb, extent=[0, w * px, h * py, 0], interpolation="bilinear")
        if keep_view:
            ax.set_xlim(xl); ax.set_ylim(yl)
        if state["show_cent"]:
            nc = pd.read_csv(long_path(nuc_csv), usecols=["centroid_x_um", "centroid_y_um", "in_germline"])
            g = nc[nc["in_germline"].astype(bool)]
            ax.scatter(g["centroid_x_um"], g["centroid_y_um"], s=4, c="#3af", alpha=0.5)
        P = state["pts"]
        if P:
            xs, ys = zip(*P)
            ax.plot(xs, ys, "-o", color="lime", lw=2, ms=5)
            ax.annotate("START", P[0], color="lime", fontsize=11, fontweight="bold", xytext=(8, -8), textcoords="offset points")
            if len(P) > 1:
                ax.annotate("END", P[-1], color="magenta", fontsize=11, fontweight="bold", xytext=(8, -8), textcoords="offset points")
        done = sum(1 for g in load_traces(traces_file).values() if g.get("status") in ("traced", "skipped"))
        nz = cache["stack"][0].shape[0]
        vdesc = f"plane {state['zi'] + 1}/{nz}" if state["zmode"] == "plane" else state["zmode"]
        ax.set_title(f"[{state['idx'] + 1}/{len(items)}  traced:{done}]  {iid}    view: z={vdesc}  "
                     f"clahe={'on' if state['clahe'] else 'off'}  gamma={state['gamma']:.2f}\n"
                     "click: add point (start at pachytene START, end at pachytene END)   z:undo  r:reset  enter:accept  s:skip  b:back  q:quit\n"
                     "view:  SCROLL WHEEL = single z-planes   m or 4 = max projection   1/2/3 = lower/mid/upper z-slab   "
                     "v = CLAHE   up/down = brightness   d = SYP   c = centroids", fontsize=9)
        ax.set_xlabel("x (um, whole image)"); ax.set_ylabel("y (um, whole image)")
        fig.canvas.draw_idle()

    def toolbar_active():
        tb = getattr(fig.canvas.manager, "toolbar", None)
        return bool(getattr(tb, "mode", ""))

    def on_click(ev):
        if ev.inaxes != ax or toolbar_active() or ev.button != 1:
            return
        state["pts"].append((float(ev.xdata), float(ev.ydata)))
        draw()

    def nxt():
        state["pts"] = []
        if state["idx"] + 1 < len(items):
            state["idx"] += 1
            draw()
        else:
            print("all gonads done")
            plt.close(fig)

    def on_key(ev):
        k = ev.key
        iid = items[state["idx"]][0]
        if k == "z" and state["pts"]:
            state["pts"].pop(); draw()
        elif k == "r":
            state["pts"] = []; draw()
        elif k in ("enter", "n"):
            if len(state["pts"]) < 2:
                print("  need at least 2 points (start + end)"); return
            save_trace(traces_file, iid, state["pts"], "traced", off_axis_um); print(f"  saved {iid}"); nxt()
        elif k == "s":
            save_trace(traces_file, iid, state["pts"], "skipped", off_axis_um); print(f"  skipped {iid}"); nxt()
        elif k == "b" and state["idx"] > 0:
            state["idx"] -= 1; state["pts"] = []; draw()
        elif k == "d":
            state["show_syp"] = not state["show_syp"]; draw()
        elif k == "c":
            state["show_cent"] = not state["show_cent"]; draw()
        elif k in ("1", "2", "3", "4", "m"):
            state["zmode"] = {"1": "low", "2": "mid", "3": "high", "4": "all", "m": "all"}[k]; draw()
        elif k == "v":
            state["clahe"] = not state["clahe"]; draw()
        elif k == "up":
            state["gamma"] = max(0.3, state["gamma"] * 0.8); draw()
        elif k == "down":
            state["gamma"] = min(3.0, state["gamma"] * 1.25); draw()
        elif k == "q":
            plt.close(fig)

    def on_scroll(ev):
        nz = cache["stack"][0].shape[0]
        if state["zmode"] != "plane":
            state["zmode"] = "plane"
        state["zi"] = int(np.clip(state["zi"] + (1 if ev.step > 0 else -1), 0, nz - 1))
        draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    draw()
    plt.show()


def restage(results_root: str | Path, traces_file: str | Path, cfg, image_ids: list[str] | None = None) -> list[str]:
    """Recompute ``<image_id>__zones.csv`` for finished images from their nuclei table and the traces
    file (cheap geometry, no image processing). nuclei.csv is left untouched: its zone columns are from
    the original run; the zones table is the record after a re-trace."""
    from .zones import run_staging

    traces = load_traces(traces_file)
    done = []
    for iid, nuc_csv, _nd2 in results_images(results_root):
        if image_ids and iid not in image_ids:
            continue
        t = traces.get(iid)
        if not t or t.get("status") != "traced":
            continue
        nc = pd.read_csv(long_path(nuc_csv), low_memory=False)
        germ = nc[nc["in_germline"].astype(bool)].copy()
        if germ.empty:
            continue
        if str(cfg.get("staging.centroid", "envelope")) == "envelope" and "envelope_centroid_x_um" in germ.columns \
                and germ["envelope_centroid_x_um"].notna().any():
            for ax_ in ("x", "y"):
                germ[f"centroid_{ax_}_um"] = germ[f"envelope_centroid_{ax_}_um"].fillna(germ[f"centroid_{ax_}_um"])
        spacing = (float(nc["voxel_dz_um"].iloc[0]), float(nc["voxel_dy_um"].iloc[0]), float(nc["voxel_dx_um"].iloc[0]))
        res = run_staging(germ, t, spacing, off_axis_um=cfg.get("staging.off_axis_um"),
                          adaptive=bool(cfg.get("staging.adaptive_cutoff", True)))
        z = res["zones"].copy()
        z["image_id"] = iid
        z.to_csv(long_path(nuc_csv.parent / f"{iid}__zones.csv"), index=False)
        done.append(iid)
        print(f"  {iid}: L={res['length_um']:.0f} um, early/mid/late = "
              f"{res['summary']['n_early']}/{res['summary']['n_mid']}/{res['summary']['n_late']}")
    return done
