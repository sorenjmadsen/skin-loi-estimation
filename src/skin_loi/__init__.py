"""Skin LOI estimation: cross-pose region-of-interest visibility classification."""

from __future__ import annotations

from . import config
from .config import Settings, add_cli_args
from .mesh_extraction import (
    MeshResult,
    MultiplePeopleDetected,
    extract_mesh,
    free_estimator_memory,
)
from .projection import pixel_to_vertex, project
from .region import n_ring
from .visibility import RegionStats, classify_region, region_verdict
from .overlay import draw_region_overlay, draw_anchor, verdict_badge_html
from .pipeline import LOIPipeline

__all__ = [
    "config",
    "Settings",
    "add_cli_args",
    "MeshResult",
    "MultiplePeopleDetected",
    "extract_mesh",
    "free_estimator_memory",
    "project",
    "pixel_to_vertex",
    "n_ring",
    "classify_region",
    "region_verdict",
    "RegionStats",
    "draw_region_overlay",
    "draw_anchor",
    "verdict_badge_html",
    "LOIPipeline",
]
