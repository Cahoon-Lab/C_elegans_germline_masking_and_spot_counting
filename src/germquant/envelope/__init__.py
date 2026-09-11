"""Lamin (LMN-1) nuclear-envelope masks: the `envelope` stage (docs/ROADMAP_modular_pipeline.md step 7).

Ported from the August 2026 ccw77 analysis (analysis/coloc/scripts/pc_lamin_worker.lamin_nuclei,
crescent_axis.lamin_labels, nucleus_filter.ring_scores / no_ring_ids / territory_ids): the validated
DAPI/Cellpose nuclei seed a 3D watershed on the smoothed lamin channel inside a band around them, so
each nucleus grows to its real envelope; a per-nucleus volume gate (1.0 to 3.0 x the DAPI volume) falls
back to the DAPI label. A ring test (lamin just outside the label vs inside, and vs the crop's Otsu ring
level) flags labels with no envelope (sperm, somatic, debris), and 2D territories separate a second gonad
arm or another worm. All of it is computed inside the germline bounding box padded by 30 voxels, exactly
like the analysis scripts, so their numbers reproduce.
"""
from .lamin import (
    cytoplasm_shell,
    envelope_labels,
    germline_crop,
    no_envelope_ids,
    ring_scores,
    run_envelope,
    territories,
)

__all__ = ["cytoplasm_shell", "envelope_labels", "germline_crop", "no_envelope_ids", "ring_scores",
           "run_envelope", "territories"]
