"""Wrap the SAM 3D Body estimator into a clean, frame-consistent MeshResult.

The legacy notebook (test_loi_extraction.ipynb) projected vertices loaded from
the saved .ply files. Those .ply vertices are produced by
`Renderer.vertices_to_trimesh`, which applies `pred_cam_t` and then rotates the
mesh 180 degrees about the X axis. We reproduce that exact frame here so the
validated projection / visibility math carries over unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class MeshResult:
    """A single person's mesh in the validated camera/render frame."""

    vertices: np.ndarray  # (V, 3) post cam_t + 180-deg-X-rotation frame
    faces: np.ndarray  # (F, 3) shared fixed topology
    vertex_normals: np.ndarray  # (V, 3)
    focal_length: float
    cam_t: np.ndarray  # (3,) raw predicted camera translation
    image_bgr: np.ndarray  # (H, W, 3) original image, BGR
    H: int
    W: int

    @property
    def trimesh(self) -> trimesh.Trimesh:
        """Rebuild a trimesh (process=False preserves vertex ordering)."""
        return trimesh.Trimesh(
            self.vertices.copy(), self.faces.copy(), process=False
        )


def free_estimator_memory(estimator) -> None:
    """Release GPU tensors the estimator caches between runs.

    `process_one_image` clears these at the *start* of the next call, so they
    otherwise stay resident on the GPU in between and inflate peak memory when
    processing a second image. Dropping them eagerly avoids CUDA OOM on reuse.
    """
    for attr in ("batch", "image_embeddings", "output"):
        if hasattr(estimator, attr):
            setattr(estimator, attr, None)
    if hasattr(estimator, "prev_prompt"):
        estimator.prev_prompt = []

    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _to_render_frame(pred_vertices: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    """Replicate Renderer.vertices_to_trimesh: add cam_t, rotate 180 deg about X.

    rotation_matrix(180, [1,0,0]) maps (x, y, z) -> (x, -y, -z).
    """
    verts = pred_vertices.astype(np.float64) + cam_t.astype(np.float64)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    verts_h = np.c_[verts, np.ones(len(verts))]
    return (verts_h @ rot.T)[:, :3]


def extract_mesh(image_bgr: np.ndarray, estimator, person_index: int = 0) -> MeshResult | None:
    """Run the estimator on an in-memory BGR image and return a MeshResult.

    Returns None if no person is detected.
    """
    import tempfile
    import os

    import cv2

    # The estimator API consumes an image path; write to a temp file.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, image_bgr)
        outputs = estimator.process_one_image(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not outputs or person_index >= len(outputs):
        free_estimator_memory(estimator)
        return None

    out = outputs[person_index]
    cam_t = np.asarray(out["pred_cam_t"], dtype=np.float64).reshape(3)
    verts = _to_render_frame(np.asarray(out["pred_vertices"]), cam_t)
    faces = np.asarray(estimator.faces)

    mesh = trimesh.Trimesh(verts, faces, process=False)
    H, W = image_bgr.shape[:2]

    # The outputs we kept are already CPU/numpy; release the estimator's GPU
    # caches before the next extraction to avoid CUDA OOM on reuse.
    free_estimator_memory(estimator)

    return MeshResult(
        vertices=verts,
        faces=faces,
        vertex_normals=np.asarray(mesh.vertex_normals),
        focal_length=float(np.asarray(out["focal_length"]).reshape(-1)[0]),
        cam_t=cam_t,
        image_bgr=image_bgr,
        H=int(H),
        W=int(W),
    )
