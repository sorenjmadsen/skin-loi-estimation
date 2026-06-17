"""Three-way region visibility classification (GREEN / YELLOW / RED).

For each patch vertex we combine two signals, both computed with the camera at
the origin in the render frame:
  * facing angle: cos(theta) = -(normal . viewing_ray). +1 faces the camera,
    ~0 is grazing/orthogonal, <0 points away.
  * self-occlusion: cast a ray from the camera through the vertex; if a nearer
    surface is hit, the vertex is hidden behind other geometry.

Labels (two angular bands on the facing cosine):
  GREEN  = visible and straight-on  (cos >= green_cos)
  YELLOW = visible but side/partial (side_cos <= cos < green_cos)
  RED    = occluded, back-facing, or too edge-on (cos < side_cos)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    GREEN,
    MIN_GREEN_FRAC,
    MIN_VISIBLE_FRAC,
    RED,
    YELLOW,
    GREEN_COS,
    SIDE_COS,
    VERDICT_GOOD,
    VERDICT_OBSTRUCTED,
    VERDICT_PARTIAL,
)


@dataclass
class RegionStats:
    verdict: str
    n_total: int
    n_green: int
    n_yellow: int
    n_red: int
    visible_frac: float
    green_frac: float
    mean_visible_cos: float
    ray_backend_ok: bool


def _ray_occlusion(tmesh, P: np.ndarray, vhat: np.ndarray, vertices: np.ndarray):
    """Return (occluded_bool[len(P)], backend_ok).

    A patch vertex is occluded if any surface is hit strictly nearer than the
    vertex itself along the camera->vertex ray. Uses trimesh ray-casting
    (rtree-backed); on any failure we degrade gracefully to "no occlusion".
    """
    n = len(P)
    eps = 5e-3
    d_v = np.linalg.norm(vertices[P], axis=1)
    try:
        origins = np.zeros((n, 3))
        locs, ray_i, _ = tmesh.ray.intersects_location(
            origins, vhat[P], multiple_hits=True
        )
        occluded = np.zeros(n, dtype=bool)
        if len(ray_i):
            hit_dist = np.linalg.norm(locs, axis=1)
            for k in range(n):
                sel = ray_i == k
                if sel.any():
                    occluded[k] = hit_dist[sel].min() < d_v[k] - eps
        return occluded, True
    except Exception:
        return np.zeros(n, dtype=bool), False


def classify_region(
    mesh,
    P: np.ndarray,
    green_cos: float = GREEN_COS,
    side_cos: float = SIDE_COS,
):
    """Classify each patch vertex as GREEN / YELLOW / RED.

    Args:
        green_cos: cos(theta) >= this and visible -> GREEN.
        side_cos: side_cos <= cos < green_cos and visible -> YELLOW;
            below side_cos (or occluded) -> RED.

    Returns:
        labels: (len(P),) int array of label codes.
        cos_t_P: (len(P),) facing cosine for the patch vertices.
        occluded: (len(P),) bool array of ray-occlusion.
        backend_ok: whether the ray backend ran (False -> occlusion skipped).
    """
    V = mesh.vertices
    N = mesh.vertex_normals
    P = np.asarray(P, dtype=int)

    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vhat = V / norms  # camera(origin) -> vertex ray
    cos_t = -(N * vhat).sum(axis=1)  # +1 facing, ~0 grazing, <0 away
    cos_t_P = cos_t[P]

    occluded, backend_ok = _ray_occlusion(mesh.trimesh, P, vhat, V)

    labels = np.full(len(P), RED, dtype=int)
    not_occluded = ~occluded
    labels[not_occluded & (cos_t_P >= green_cos)] = GREEN
    labels[not_occluded & (cos_t_P >= side_cos) & (cos_t_P < green_cos)] = YELLOW
    return labels, cos_t_P, occluded, backend_ok


def region_verdict(
    labels: np.ndarray,
    cos_t_P: np.ndarray,
    min_visible_frac: float = MIN_VISIBLE_FRAC,
    min_green_frac: float = MIN_GREEN_FRAC,
    backend_ok: bool = True,
) -> RegionStats:
    """Aggregate per-vertex labels into a single region verdict."""
    labels = np.asarray(labels)
    n = len(labels)
    n_green = int((labels == GREEN).sum())
    n_yellow = int((labels == YELLOW).sum())
    n_red = int((labels == RED).sum())

    visible = labels != RED
    visible_frac = float(visible.mean()) if n else 0.0
    green_frac = float((labels == GREEN).mean()) if n else 0.0
    mean_visible_cos = float(cos_t_P[visible].mean()) if visible.any() else 0.0

    if visible_frac < min_visible_frac:
        verdict = VERDICT_OBSTRUCTED
    elif green_frac >= min_green_frac:
        verdict = VERDICT_GOOD
    else:
        verdict = VERDICT_PARTIAL

    return RegionStats(
        verdict=verdict,
        n_total=n,
        n_green=n_green,
        n_yellow=n_yellow,
        n_red=n_red,
        visible_frac=visible_frac,
        green_frac=green_frac,
        mean_visible_cos=mean_visible_cos,
        ray_backend_ok=backend_ok,
    )
