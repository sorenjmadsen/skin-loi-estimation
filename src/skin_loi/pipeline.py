"""End-to-end orchestrator. Loads the heavy estimator once and exposes steps."""

from __future__ import annotations

import cv2
import numpy as np

from .config import Settings, ensure_sam3d_on_path
from .mesh_extraction import MeshResult, extract_mesh
from .overlay import draw_anchor, draw_region_overlay, verdict_badge_html
from .projection import pixel_to_vertex, project
from .region import n_ring
from .visibility import RegionStats, classify_region, region_verdict


class LOIPipeline:
    """Holds the loaded SAM 3D Body estimator and runs the LOI workflow."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._estimator = None

    @property
    def estimator(self):
        if self._estimator is None:
            ensure_sam3d_on_path()
            import torch
            from notebook.utils import setup_sam_3d_body  # noqa: E402

            device = self.settings.device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._estimator = setup_sam_3d_body(
                hf_repo_id=self.settings.hf_repo_id, device=device
            )
        return self._estimator

    # --- Step 1 & 2: extract mesh ------------------------------------------
    def extract(self, image_bgr: np.ndarray) -> MeshResult | None:
        return extract_mesh(image_bgr, self.estimator)

    def mesh_overlay(self, mesh: MeshResult) -> np.ndarray:
        """Lightweight fit-confirmation overlay (projected silhouette)."""
        img = mesh.image_bgr.copy()
        uv = project(mesh.vertices, mesh.focal_length, mesh.W, mesh.H)
        for x, y in uv.astype(int):
            cv2.circle(img, (int(x), int(y)), 1, (155, 155, 155), -1)
        return img

    # --- Step 3: select a point of interest --------------------------------
    def select_point(
        self,
        ref_mesh: MeshResult,
        poi_xy,
        rings: int | None = None,
        radius_px: float | None = None,
    ):
        """Anchor a clicked pixel to a vertex and grow the patch P.

        Returns (vstar, P, reproj_err, preview_bgr).
        """
        rings = self.settings.rings if rings is None else rings
        radius_px = self.settings.anchor_radius_px if radius_px is None else radius_px
        vstar, uv, reproj_err = pixel_to_vertex(ref_mesh, poi_xy, radius_px)
        P = n_ring(ref_mesh, vstar, rings)

        preview = ref_mesh.image_bgr.copy()
        for x, y in uv.astype(int):
            cv2.circle(preview, (int(x), int(y)), 1, (155, 155, 155), -1)
        for vi in P:
            x, y = uv[vi].astype(int)
            cv2.circle(preview, (int(x), int(y)), 3, (60, 90, 230), -1)  # terracotta
        preview = draw_anchor(preview, uv[vstar])
        return vstar, P, reproj_err, preview

    # --- Step 4 & 5: evaluate region in target pose ------------------------
    def evaluate(
        self,
        target_mesh: MeshResult,
        P,
        green_cos: float | None = None,
        side_cos: float | None = None,
        min_visible_frac: float | None = None,
        min_green_frac: float | None = None,
    ):
        """Classify the patch in the target pose and render the overlay.

        Returns (overlay_bgr, stats, badge_html).
        """
        s = self.settings
        green_cos = s.green_cos if green_cos is None else green_cos
        side_cos = s.side_cos if side_cos is None else side_cos
        min_visible_frac = s.min_visible_frac if min_visible_frac is None else min_visible_frac
        min_green_frac = s.min_green_frac if min_green_frac is None else min_green_frac

        labels, cos_t_P, _occluded, backend_ok = classify_region(
            target_mesh, P, green_cos=green_cos, side_cos=side_cos
        )
        stats = region_verdict(
            labels, cos_t_P, min_visible_frac, min_green_frac, backend_ok
        )
        overlay = draw_region_overlay(target_mesh, P, labels)
        badge = verdict_badge_html(stats)
        return overlay, stats, badge
