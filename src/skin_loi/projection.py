"""Pinhole projection and pixel->vertex anchoring (validated in the notebook)."""

from __future__ import annotations

import numpy as np

from .config import ANCHOR_RADIUS_PX


def project(V: np.ndarray, focal: float, W: int, H: int) -> np.ndarray:
    """Project render-frame vertices to pixel (u, v). Single source of truth.

    Matches test_loi_extraction.ipynb: SAM3D body sits at Z<0 in this frame, so
    depth = -Z, and the Y term is flipped to go from Y-up to image y-down.
    """
    depth = -V[:, 2]
    u = focal * V[:, 0] / depth + W / 2
    v = -focal * V[:, 1] / depth + H / 2
    return np.stack([u, v], axis=1)


def pixel_to_vertex(mesh, poi_xy, radius_px: float = ANCHOR_RADIUS_PX):
    """Snap a clicked pixel to the frontmost mesh vertex near it.

    Args:
        mesh: MeshResult.
        poi_xy: (2,) clicked pixel (x, y).
        radius_px: snap radius; if no vertex is within it, fall back to nearest.

    Returns:
        (vstar, uv, reproj_err_px)
        vstar: int vertex index.
        uv: (V, 2) projected pixel coords of all vertices.
        reproj_err_px: distance from poi to the projected anchor vertex.
    """
    poi_xy = np.asarray(poi_xy, dtype=np.float64).reshape(2)
    uv = project(mesh.vertices, mesh.focal_length, mesh.W, mesh.H)
    d2 = ((uv - poi_xy) ** 2).sum(axis=1)

    near = np.where(d2 < radius_px**2)[0]
    if near.size > 0:
        # Frontmost = least-negative Z (closest to camera) among candidates.
        vstar = int(near[np.argmax(mesh.vertices[near, 2])])
    else:
        # Click landed off the silhouette: use the globally nearest vertex.
        vstar = int(np.argmin(d2))

    reproj_err = float(np.hypot(*(uv[vstar] - poi_xy)))
    return vstar, uv, reproj_err
