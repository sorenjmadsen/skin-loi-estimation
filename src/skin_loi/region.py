"""Grow a topological patch (region of interest) around an anchor vertex."""

from __future__ import annotations

import numpy as np


def n_ring(mesh, seed: int, rings: int = 1) -> np.ndarray:
    """Return sorted vertex indices within `rings` topological hops of `seed`.

    `mesh` is anything exposing `vertex_neighbors` (e.g. trimesh.Trimesh or a
    MeshResult, via `.trimesh`).
    """
    tmesh = getattr(mesh, "trimesh", mesh)
    neighbors = tmesh.vertex_neighbors
    selected = {int(seed)}
    for _ in range(max(0, int(rings))):
        selected |= {j for i in list(selected) for j in neighbors[i]}
    return np.array(sorted(selected), dtype=int)
