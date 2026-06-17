"""Unit tests for the 3-way visibility classifier and region verdict.

These use a tiny synthetic mesh with hand-set normals so we can predict labels
without loading the SAM 3D Body model.
"""

import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from skin_loi.config import GREEN, YELLOW, RED, VERDICT_GOOD, VERDICT_OBSTRUCTED, VERDICT_PARTIAL
from skin_loi.visibility import classify_region, region_verdict


class FakeMesh:
    """Minimal stand-in for MeshResult for classifier tests."""

    def __init__(self, vertices, normals, faces):
        self.vertices = np.asarray(vertices, dtype=float)
        self.vertex_normals = np.asarray(normals, dtype=float)
        self._faces = np.asarray(faces, dtype=int)

    @property
    def trimesh(self):
        return trimesh.Trimesh(self.vertices, self._faces, process=False)


def _single_triangle():
    # Camera at origin; body sits at z < 0 in the render frame.
    # Vertices sit near the optical axis so cos(theta) ~= the normal's z.
    vertices = [
        (-0.02, 0.0, -2.0),  # straight-on   -> GREEN  (cos ~1.0)
        (0.02, 0.0, -2.0),   # ~60 deg side  -> YELLOW (cos ~0.5)
        (0.0, 0.02, -2.0),   # back-facing   -> RED    (cos ~-1.0)
    ]
    normals = [
        (0.0, 0.0, 1.0),       # faces the camera
        (0.866, 0.0, 0.5),     # tilted ~60 deg
        (0.0, 0.0, -1.0),      # points away
    ]
    faces = [(0, 1, 2)]
    return FakeMesh(vertices, normals, faces)


def test_classify_region_three_cases():
    mesh = _single_triangle()
    P = np.array([0, 1, 2])
    labels, cos_t_P, occluded, backend_ok = classify_region(
        mesh, P, green_cos=0.7, side_cos=0.2
    )

    assert labels[0] == GREEN   # cos ~1.0
    assert labels[1] == YELLOW  # cos ~0.5
    assert labels[2] == RED     # back-facing
    # Straight-on vertex should have the highest facing cosine.
    assert cos_t_P[0] > cos_t_P[1] > cos_t_P[2]


def test_green_threshold_moves_label():
    mesh = _single_triangle()
    P = np.array([1])  # cos ~0.5
    # Lower the green cutoff below the vertex cosine -> promotes to GREEN.
    green_labels, _, _, _ = classify_region(mesh, P, green_cos=0.4, side_cos=0.2)
    yellow_labels, _, _, _ = classify_region(mesh, P, green_cos=0.7, side_cos=0.2)
    assert green_labels[0] == GREEN
    assert yellow_labels[0] == YELLOW


def test_side_threshold_pushes_to_red():
    mesh = _single_triangle()
    P = np.array([1])  # cos ~0.5
    # Raise the side cutoff above the vertex cosine -> too edge-on -> RED.
    labels, _, _, _ = classify_region(mesh, P, green_cos=0.7, side_cos=0.6)
    assert labels[0] == RED


def test_verdict_good():
    labels = np.array([GREEN, GREEN, GREEN, YELLOW])
    cos = np.array([0.9, 0.8, 0.7, 0.3])
    assert region_verdict(labels, cos).verdict == VERDICT_GOOD


def test_verdict_partial():
    labels = np.array([YELLOW, YELLOW, YELLOW, GREEN])
    cos = np.array([0.2, 0.3, 0.25, 0.6])
    assert region_verdict(labels, cos).verdict == VERDICT_PARTIAL


def test_verdict_obstructed():
    labels = np.array([RED, RED, RED, GREEN])
    cos = np.array([-0.5, -0.2, -0.9, 0.8])
    stats = region_verdict(labels, cos)
    assert stats.verdict == VERDICT_OBSTRUCTED
    assert stats.n_red == 3


def test_verdict_counts_consistent():
    labels = np.array([GREEN, YELLOW, RED, GREEN])
    cos = np.array([0.9, 0.2, -0.4, 0.7])
    stats = region_verdict(labels, cos)
    assert stats.n_green + stats.n_yellow + stats.n_red == stats.n_total == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
