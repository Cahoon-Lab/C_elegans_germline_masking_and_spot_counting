"""Synaptonemal-complex (SYP) signal, two independent tools:

* ``surface_sc_ribbon`` (surface.py): a binary intranuclear ribbon MASK used as the control operand
  of the coloc stage (config key ``sc.enabled``; size gate 10 voxels, no hole filling).
* ``trace_sc`` (skeleton.py): the per-nucleus SC TRACER restored verbatim from the June 2026
  upstream (config key ``sc.trace.enabled``): isotropic resample, Sato ridge, hysteresis mask with
  smoothing (size gate 30, holes under 27 voxels filled), 3D skeleton, skan; reports SC length
  (recoverable), a fragment-count LOWER BOUND and the fragmentation index. See docs/SC_TRACING.md.
The two mask cores are deliberately not unified until the tracer is calibrated against Imaris.
"""
from .skeleton import PER_NUC_COLS, TRACK_COLS, skan_available, trace_sc
from .surface import sc_ribbon_length, surface_sc_ribbon

__all__ = ["PER_NUC_COLS", "TRACK_COLS", "sc_ribbon_length", "skan_available", "surface_sc_ribbon", "trace_sc"]
