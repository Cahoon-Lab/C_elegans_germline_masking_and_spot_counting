"""Pachytene staging from a hand-traced gonad axis (the `staging` stage; docs/ROADMAP_modular_pipeline.md
step 9). Ported from analysis/coloc/scripts/trace_pachytene.py (geometry) and rezone_all.py.

Every automated pachytene stager plateaued near 5 rows of error on the ccw77 gonads, so staging is
manual: a polyline drawn from the pachytene START to its END (``germquant trace``), saved per image in a
JSON traces file in whole-image microns. The stage projects every germline nucleus centroid onto the
polyline: nuclei within the off-axis cutoff and inside the traced span get zone early / mid / late
(equal thirds of the polyline length); before the start = pre, beyond the end = post; farther than the
cutoff = off_axis. The cutoff adapts to the gonad's own tube width (min(off_axis_um, 2.5 x the median
perpendicular distance of the nuclei inside the span)).
"""
from .zones import assign_zones, load_traces, project_to_polyline, run_staging, save_trace

__all__ = ["assign_zones", "load_traces", "project_to_polyline", "run_staging", "save_trace"]
