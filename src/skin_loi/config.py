"""Central configuration: model id, color constants, and default thresholds."""

from __future__ import annotations

import sys
from pathlib import Path

# --- Repo paths -------------------------------------------------------------
# config.py -> src/skin_loi/config.py, so parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM3D_ROOT = PROJECT_ROOT / "sam3d"
OUTPUT_DIR = PROJECT_ROOT / "output"


def ensure_sam3d_on_path() -> None:
    """Make `notebook.utils` and `sam_3d_body` importable (mirrors generate_mesh.py)."""
    p = str(SAM3D_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


# --- Model ------------------------------------------------------------------
HF_REPO_ID = "facebook/sam-3d-body-dinov3"

# --- Visibility label codes -------------------------------------------------
GREEN = 0  # straight-on, unoccluded
YELLOW = 1  # visible but grazing / partial / near-orthogonal
RED = 2  # occluded or back-facing

LABEL_NAMES = {GREEN: "GREEN", YELLOW: "YELLOW", RED: "RED"}

# --- Colors -----------------------------------------------------------------
# Hex (for HTML/UI) and BGR tuples (for OpenCV drawing).
HEX_COLORS = {GREEN: "#00C853", YELLOW: "#FFC107", RED: "#E53935"}
RGB_COLORS = {GREEN: (0, 200, 83), YELLOW: (255, 193, 7), RED: (229, 57, 53)}
BGR_COLORS = {label: (b, g, r) for label, (r, g, b) in RGB_COLORS.items()}

# Marker / anchor colors (BGR) used when drawing on images.
ANCHOR_BGR = (40, 225, 255)  # bright yellow crosshair for the picked point
SILHOUETTE_BGR = (155, 155, 155)  # faint full-mesh projection dots

# --- Default thresholds -----------------------------------------------------
# Two angular bands on the facing cosine cos(theta) = -(normal . view_ray):
#   cos >= GREEN_COS                -> straight-on        -> GREEN  (theta <= ~46 deg)
#   SIDE_COS <= cos < GREEN_COS     -> side / partial     -> YELLOW (~46..~78 deg)
#   cos < SIDE_COS (incl. back-facing) or occluded -> RED  (too edge-on / hidden)
GREEN_COS = 0.5  # 60 deg; below this it is no longer a "straight-on" view
SIDE_COS = 0.0  # 90 deg: exactly edge on

ANCHOR_RADIUS_PX = 10.0  # pixel radius when snapping a click to a vertex
RINGS = 2  # topological rings grown around the anchor vertex

# Region-level verdict cutoffs (fractions over the patch).
MIN_VISIBLE_FRAC = 0.3  # below this -> OBSTRUCTED
MIN_GREEN_FRAC = 0.5  # at/above this (and visible) -> GOOD, else PARTIAL

# Verdict labels.
VERDICT_GOOD = "GOOD"
VERDICT_PARTIAL = "PARTIAL"
VERDICT_OBSTRUCTED = "OBSTRUCTED"

VERDICT_TO_LABEL = {
    VERDICT_GOOD: GREEN,
    VERDICT_PARTIAL: YELLOW,
    VERDICT_OBSTRUCTED: RED,
}


# --- Tunable settings bundle ------------------------------------------------
from dataclasses import dataclass  # noqa: E402


@dataclass
class Settings:
    """All runtime-tunable knobs in one place.

    Defaults mirror the module constants above. Override via CLI args
    (see `add_cli_args` / `Settings.from_namespace`) or live in the UI panel.
    """

    # Region selection
    anchor_radius_px: float = ANCHOR_RADIUS_PX
    rings: int = RINGS

    # Per-vertex visibility bands
    green_cos: float = GREEN_COS
    side_cos: float = SIDE_COS

    # Region verdict aggregation
    min_visible_frac: float = MIN_VISIBLE_FRAC
    min_green_frac: float = MIN_GREEN_FRAC

    # Model / server
    hf_repo_id: str = HF_REPO_ID
    device: str | None = None
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False

    @staticmethod
    def from_namespace(args) -> "Settings":
        return Settings(
            anchor_radius_px=args.anchor_radius,
            rings=args.rings,
            green_cos=args.green_cos,
            side_cos=args.side_cos,
            min_visible_frac=args.min_visible_frac,
            min_green_frac=args.min_green_frac,
            hf_repo_id=args.hf_repo_id,
            device=args.device,
            host=args.host,
            port=args.port,
            share=args.share,
        )


def add_cli_args(parser) -> None:
    """Register CLI overrides for every setting on an argparse parser."""
    g = parser.add_argument_group("region selection")
    g.add_argument("--anchor-radius", type=float, default=ANCHOR_RADIUS_PX,
                   help="Pixel radius when snapping a click to a vertex.")
    g.add_argument("--rings", type=int, default=RINGS,
                   help="Topological rings grown around the anchor vertex.")

    v = parser.add_argument_group("visibility thresholds")
    v.add_argument("--green-cos", type=float, default=GREEN_COS,
                   help="cos(theta) >= this -> GREEN (straight-on).")
    v.add_argument("--side-cos", type=float, default=SIDE_COS,
                   help="cos(theta) in [side, green) -> YELLOW; below -> RED.")
    v.add_argument("--min-visible-frac", type=float, default=MIN_VISIBLE_FRAC,
                   help="Patch visible fraction below which the verdict is OBSTRUCTED.")
    v.add_argument("--min-green-frac", type=float, default=MIN_GREEN_FRAC,
                   help="Green fraction at/above which a visible region is GOOD.")

    m = parser.add_argument_group("model / server")
    m.add_argument("--hf-repo-id", default=HF_REPO_ID, help="HuggingFace model repo id.")
    m.add_argument("--device", default=None, choices=[None, "cuda", "cpu"],
                   help="Inference device (default: auto-detect).")
    m.add_argument("--host", default="0.0.0.0", help="Gradio server host.")
    m.add_argument("--port", type=int, default=7860, help="Gradio server port.")
    m.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
