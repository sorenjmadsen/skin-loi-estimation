"""2D overlays (silhouette + colored region) and the HTML verdict badge."""

from __future__ import annotations

import cv2
import numpy as np

from .config import (
    ANCHOR_BGR,
    BGR_COLORS,
    HEX_COLORS,
    GREEN,
    RED,
    SILHOUETTE_BGR,
    YELLOW,
    VERDICT_GOOD,
    VERDICT_OBSTRUCTED,
    VERDICT_PARTIAL,
)
from .projection import project


def draw_region_overlay(mesh, P, labels, draw_silhouette: bool = True) -> np.ndarray:
    """Draw the projected mesh silhouette and the color-coded patch on the image.

    Returns a BGR image (copy of mesh.image_bgr).
    """
    img = mesh.image_bgr.copy()
    uv = project(mesh.vertices, mesh.focal_length, mesh.W, mesh.H)

    if draw_silhouette:
        for x, y in uv.astype(int):
            cv2.circle(img, (int(x), int(y)), 1, SILHOUETTE_BGR, -1)

    P = np.asarray(P, dtype=int)
    for vi, lab in zip(P, np.asarray(labels)):
        x, y = uv[vi].astype(int)
        cv2.circle(img, (int(x), int(y)), 3, BGR_COLORS[int(lab)], -1)
    return img


def draw_anchor(image_bgr: np.ndarray, poi_xy, color=ANCHOR_BGR, size: int = 12) -> np.ndarray:
    """Draw a crosshair marker at the clicked point. Returns a BGR copy."""
    img = image_bgr.copy()
    x, y = int(round(poi_xy[0])), int(round(poi_xy[1]))
    cv2.line(img, (x - size, y), (x + size, y), color, 2)
    cv2.line(img, (x, y - size), (x, y + size), color, 2)
    cv2.circle(img, (x, y), size, color, 2)
    return img


def verdict_badge_html(stats) -> str:
    """Build a styled HTML badge summarizing the region verdict."""
    color = HEX_COLORS[
        {VERDICT_GOOD: GREEN, VERDICT_PARTIAL: YELLOW, VERDICT_OBSTRUCTED: RED}[
            stats.verdict
        ]
    ]
    headline = {
        VERDICT_GOOD: "GOOD VIEW",
        VERDICT_PARTIAL: "PARTIAL / SIDE VIEW",
        VERDICT_OBSTRUCTED: "OBSTRUCTED",
    }[stats.verdict]

    visible_pct = round(100 * stats.visible_frac)
    green_pct = round(100 * stats.green_frac)
    warn = (
        ""
        if stats.ray_backend_ok
        else "<div style='color:#b00;font-size:12px;margin-top:6px;'>"
        "Ray backend unavailable - occlusion not evaluated (facing-angle only).</div>"
    )

    def chip(label_color, count, name):
        return (
            f"<span style='display:inline-block;margin-right:10px;'>"
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            f"background:{label_color};margin-right:4px;'></span>{name}: {count}</span>"
        )

    chips = (
        chip(HEX_COLORS[GREEN], stats.n_green, "green")
        + chip(HEX_COLORS[YELLOW], stats.n_yellow, "yellow")
        + chip(HEX_COLORS[RED], stats.n_red, "red")
    )

    return f"""
    <div style="border-left:8px solid {color};padding:14px 18px;border-radius:8px;
                background:rgba(0,0,0,0.03);font-family:system-ui,sans-serif;">
      <div style="font-size:22px;font-weight:700;color:{color};">{headline}</div>
      <div style="font-size:14px;margin-top:6px;">
        {visible_pct}% of region visible &nbsp;|&nbsp; {green_pct}% straight-on
        &nbsp;|&nbsp; mean cos&theta; {stats.mean_visible_cos:+.2f}
      </div>
      <div style="font-size:13px;margin-top:8px;color:#444;">{chips}</div>
      {warn}
    </div>
    """
