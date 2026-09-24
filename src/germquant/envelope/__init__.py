"""Lamin (LMN-1) nuclear-envelope masks: the `envelope` stage (docs/ROADMAP_modular_pipeline.md step 7).

Ported from the August 2026 ccw77 analysis (analysis/coloc/scripts/pc_lamin_worker.lamin_nuclei,
crescent_axis.lamin_labels, nucleus_filter.ring_scores / no_ring_ids / territory_ids): the validated
DAPI/Cellpose nuclei seed a 3D watershed on the smoothed lamin channel inside a band around them, so
each nucleus grows to its real envelope; a per-nucleus volume gate (1.0 to 3.0 x the DAPI volume) falls
back to the DAPI label. A ring test (lamin just outside the label vs inside, and vs the crop's Otsu ring
level) flags labels with no envelope (sperm, somatic, debris), and 2D territories separate a second gonad
arm or another worm. All of it is computed inside the germline bounding box padded by 30 voxels, exactly
like the analysis scripts, so their numbers reproduce.

Named differences from the scripts: the scripts used the constant voxel (0.2, 0.1083, 0.1083) um while the
stage uses the .nd2 voxel (0.2, 0.108333, 0.108333 on the Cahoon scope), so every um3 / micron column is
about 0.06 percent larger and count-based gate boundaries shift by one voxel at 5 um3, four at 15 um3 and
about 39 at 150 um3; the ring test gates the unrounded scores (as nucleus_filter did for the published v4
and zone numbers), while qc_mask_audit.py rounded them to three decimals first.
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
