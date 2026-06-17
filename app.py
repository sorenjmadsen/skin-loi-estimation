"""Skin LOI Estimation - standalone Gradio demo.

Workflow:
  1. Upload a reference pose and a target pose; extract a 3D body mesh from each.
  2. Click a point of interest on the reference image -> it snaps to a mesh
     vertex and grows a small patch (region of interest).
  3. Because SAM 3D Body produces a fixed-topology mesh, the same patch indexes
     the same anatomical region in the target pose.
  4. The patch is classified in the target view as GREEN (straight-on),
     YELLOW (side/partial), or RED (occluded / back-facing), with a verdict.

Run:  python app.py [--green-cos 0.5 --rings 2 --device cuda --port 7860 ...]
      python app.py --help   # see every tunable
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import gradio as gr
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from skin_loi.config import Settings, add_cli_args  # noqa: E402
from skin_loi.mesh_extraction import MultiplePeopleDetected  # noqa: E402
from skin_loi.pipeline import LOIPipeline  # noqa: E402

# Populated in main(); module-level so the Gradio callbacks can reach them.
PIPE: LOIPipeline
SETTINGS: Settings


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.gray,
    font=[
        gr.themes.GoogleFont("Hanken Grotesk"),
        gr.themes.GoogleFont("Inter"),
        "system-ui",
        "sans-serif",
    ],
    radius_size=gr.themes.sizes.radius_lg,
)

CSS = """
:root {
  --tyle-ink: #0c0d0c;
  --tyle-muted: #6b7280;
  --tyle-green: #1b7a46;
  --tyle-line: #ececec;
}
.gradio-container {
  background: #ffffff !important;
  font-family: 'Hanken Grotesk', 'Inter', system-ui, sans-serif !important;
  color: var(--tyle-ink);
  max-width: 1120px !important;
  margin: 0 auto !important;
}
.tyle-hero { padding: 46px 4px 26px; text-align: center; }
.tyle-kicker {
  text-transform: uppercase; letter-spacing: .18em;
  font-size: 12px; font-weight: 700; color: var(--tyle-green);
}
.tyle-title {
  font-size: 52px; line-height: 1.04; font-weight: 800;
  letter-spacing: -0.03em; margin: 12px 0 16px; color: var(--tyle-ink);
}
.tyle-title .g { color: var(--tyle-green); display: block; }
.tyle-sub {
  font-size: 18px; color: var(--tyle-muted);
  max-width: 620px; margin: 0 auto; line-height: 1.5;
}
.tyle-card {
  background: #ffffff !important;
  border: 1px solid var(--tyle-line) !important;
  border-radius: 22px !important;
  padding: 22px 24px !important;
  box-shadow: 0 10px 30px rgba(15, 23, 20, 0.04) !important;
}
.tyle-section {
  text-transform: uppercase; letter-spacing: .14em;
  font-size: 12px; font-weight: 700; color: var(--tyle-muted);
  margin: 2px 0 6px;
}
.tyle-cap {
  display: inline-block;
  background: #e7f7ee; color: #0f7a44;
  font-size: 14px; font-weight: 600;
  padding: 6px 14px; border-radius: 999px;
  margin: 2px 2px 10px; white-space: nowrap;
}
/* Keep the reference/target panels side by side; override Gradio's column
   min-width so they shrink instead of wrapping to a vertical stack. */
.tyle-pair { flex-wrap: nowrap !important; }
.tyle-pair > * { flex: 1 1 0 !important; min-width: 0 !important; }
.tyle-legend {
  display: flex; flex-wrap: wrap; gap: 22px; justify-content: center;
  margin: 16px 0 4px; font-size: 13px; color: var(--tyle-muted);
}
.tyle-legend .dot {
  display: inline-block; width: 10px; height: 10px;
  border-radius: 50%; margin-right: 6px; vertical-align: middle;
}
/* Flatten Gradio's own block chrome so we don't get box-in-box. */
.tyle-card .block,
.tyle-card .form,
.tyle-card .gap,
.tyle-card .panel {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
/* Keep a soft dropzone outline on the image inputs only. */
.tyle-card .image-container,
.tyle-card [data-testid="image"] {
  border: 1px dashed #d6dbd6 !important;
  border-radius: 14px !important;
  background: #fafbfa !important;
}
/* Tame the floating field labels (were bright pills). */
.gradio-container .block-label,
.gradio-container label > span,
.gradio-container span[data-testid="block-info"] {
  background: transparent !important;
  color: var(--tyle-muted) !important;
  border: none !important;
  font-weight: 600 !important;
}
.tyle-primary button {
  background: #0e0f0e !important; border: none !important; color: #fff !important;
  border-radius: 999px !important; font-weight: 600 !important;
  padding: 12px 26px !important; width: auto !important; min-width: 220px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.16) !important;
}
.tyle-primary button:hover { filter: brightness(1.18); }
/* Sidebar drawer styling. */
.tyle-drawer { background: #ffffff !important; border-left: 1px solid var(--tyle-line) !important; }
.tyle-banner {
  display: flex; align-items: center; gap: 10px;
  padding: 12px 16px; border-radius: 14px;
  background: #f4f6f4; border: 1px solid #e9ece9;
  color: var(--tyle-ink); font-size: 14px;
}
.tyle-spin {
  width: 16px; height: 16px; border-radius: 50%; flex: 0 0 auto;
  border: 2px solid #d6e6dc; border-top-color: var(--tyle-green);
  animation: tyle-rot .8s linear infinite;
}
@keyframes tyle-rot { to { transform: rotate(360deg); } }

/* Guarantee light surfaces even if the system/browser forces dark mode and the
   __theme=light redirect is slow or ignored. Overrides Gradio's theme vars. */
.gradio-container.dark, .dark {
  --body-background-fill: #ffffff;
  --background-fill-primary: #ffffff;
  --background-fill-secondary: #f7f8f7;
  --block-background-fill: #ffffff;
  --block-label-background-fill: #ffffff;
  --block-label-text-color: #6b7280;
  --block-title-text-color: #0c0d0c;
  --block-border-color: #ececec;
  --border-color-primary: #ececec;
  --body-text-color: #0c0d0c;
  --body-text-color-subdued: #6b7280;
  --input-background-fill: #ffffff;
  --panel-background-fill: #ffffff;
  --neutral-950: #0c0d0c;
}
body, gradio-app, .gradio-container { background: #ffffff !important; }

/* Make the Step 2 image click-only: hide just the buttons (NOT the containers,
   which also hold the displayed image). The image starts hidden until populated,
   so the empty upload prompt is never seen. */
#tyle-select button.upload-button,
#tyle-select button[aria-label="Clear"],
#tyle-select button[aria-label="Remove Image"],
#tyle-select button[aria-label="Edit"] { display: none !important; }
"""

# Force the light color scheme regardless of the viewer's system/browser setting
# (otherwise Gradio renders dark panels on our white page -> jarring).
FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""


def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def _banner(text: str, busy: bool = False) -> str:
    spin = "<span class='tyle-spin'></span>" if busy else ""
    return f"<div class='tyle-banner'>{spin}<span>{text}</span></div>"


def on_extract(ref_rgb, tgt_rgb):
    """Extract meshes from both uploads, streaming a (n/2) progress banner.

    Outputs: ref_mesh_state, tgt_mesh_state, select_img (reference, clickable),
    result_img (target), status_html.
    """
    if ref_rgb is None or tgt_rgb is None:
        yield None, None, gr.update(), gr.update(), gr.update(
            value=_banner("Please upload both a reference and a target image.")
        )
        return

    first_run = getattr(PIPE, "_estimator", None) is None
    note = " (loading the model on first run, this can take ~30-60s)" if first_run else ""

    yield None, None, gr.update(), gr.update(), gr.update(
        value=_banner(f"Extracting reference mesh (1/2)…{note}", busy=True)
    )
    try:
        ref_mesh = PIPE.extract(_to_bgr(ref_rgb))
    except MultiplePeopleDetected as e:
        yield None, None, gr.update(), gr.update(), gr.update(
            value=_banner(
                f"Detected {e.count} people in the reference image; "
                "please use a photo with a single person."
            )
        )
        return
    if ref_mesh is None:
        yield None, None, gr.update(), gr.update(), gr.update(
            value=_banner("No person detected in the reference image.")
        )
        return

    yield ref_mesh, None, gr.update(), gr.update(), gr.update(
        value=_banner("Extracting target mesh (2/2)…", busy=True)
    )
    try:
        tgt_mesh = PIPE.extract(_to_bgr(tgt_rgb))
    except MultiplePeopleDetected as e:
        yield ref_mesh, None, gr.update(), gr.update(), gr.update(
            value=_banner(
                f"Detected {e.count} people in the target image; "
                "please use a photo with a single person."
            )
        )
        return
    if tgt_mesh is None:
        yield ref_mesh, None, gr.update(), gr.update(), gr.update(
            value=_banner("No person detected in the target image.")
        )
        return

    ref_overlay = _to_rgb(PIPE.mesh_overlay(ref_mesh))
    tgt_overlay = _to_rgb(PIPE.mesh_overlay(tgt_mesh))
    yield (
        ref_mesh,
        tgt_mesh,
        gr.update(value=ref_overlay, visible=True),
        gr.update(value=tgt_overlay, visible=True),
        gr.update(
            value=_banner("Meshes ready - click a point on the reference (left) to map it onto the target (right).")
        ),
    )


def _run(ref_mesh, tgt_mesh, poi, params):
    green_cos, side_cos, rings, radius, min_vis, min_green = params
    vstar, P, err, preview = PIPE.select_point(
        ref_mesh, poi, rings=int(rings), radius_px=float(radius)
    )
    overlay, stats, badge = PIPE.evaluate(
        tgt_mesh, P,
        green_cos=float(green_cos), side_cos=float(side_cos),
        min_visible_frac=float(min_vis), min_green_frac=float(min_green),
    )
    info = (
        f"Anchored vertex `{vstar}` (reprojection error {err:.1f}px), "
        f"patch of {len(P)} vertices ({int(rings)}-ring)."
    )
    return P, _to_rgb(preview), _to_rgb(overlay), badge, info


def on_click(
    ref_mesh, tgt_mesh, green_cos, side_cos, rings, radius, min_vis, min_green,
    evt: gr.SelectData,
):
    """Handle a click on the reference image."""
    if ref_mesh is None or tgt_mesh is None:
        return None, None, None, None, gr.update(value="Extract meshes first.")
    poi = (float(evt.index[0]), float(evt.index[1]))
    params = (green_cos, side_cos, rings, radius, min_vis, min_green)
    P, preview, overlay, badge, info = _run(ref_mesh, tgt_mesh, poi, params)
    return poi, preview, overlay, badge, gr.update(value=info)


def on_param_change(
    ref_mesh, tgt_mesh, poi, green_cos, side_cos, rings, radius, min_vis, min_green
):
    """Recompute when a setting changes (if a point was already picked)."""
    if ref_mesh is None or tgt_mesh is None or poi is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    params = (green_cos, side_cos, rings, radius, min_vis, min_green)
    _P, preview, overlay, badge, info = _run(ref_mesh, tgt_mesh, poi, params)
    return preview, overlay, badge, gr.update(value=info)


def build_ui() -> gr.Blocks:
    s = SETTINGS
    with gr.Blocks(title="Skin LOI Estimation") as demo:
        ref_mesh_state = gr.State()
        tgt_mesh_state = gr.State()
        poi_state = gr.State()

        with gr.Sidebar(label="Settings", open=False, position="right", elem_classes="tyle-drawer"):
            gr.HTML("<div class='tyle-section'>Region selection</div>")
            rings_sl = gr.Slider(0, 4, value=s.rings, step=1, label="Region rings")
            radius_sl = gr.Slider(
                2, 30, value=s.anchor_radius_px, step=1, label="Anchor radius (px)"
            )
            gr.HTML("<div class='tyle-section'>Visibility thresholds</div>")
            green_sl = gr.Slider(
                0.0, 1.0, value=s.green_cos, step=0.05, label="Green cos (>= straight-on)",
            )
            side_sl = gr.Slider(
                0.0, 1.0, value=s.side_cos, step=0.05, label="Side cos (below -> red)",
            )
            gr.HTML("<div class='tyle-section'>Region verdict</div>")
            minvis_sl = gr.Slider(
                0.0, 1.0, value=s.min_visible_frac, step=0.05, label="Min visible fraction",
            )
            mingreen_sl = gr.Slider(
                0.0, 1.0, value=s.min_green_frac, step=0.05, label="Min green fraction",
            )

        gr.HTML(
            "<div class='tyle-hero'>"
            "<div class='tyle-kicker'>Skin Intelligence</div>"
            "<div class='tyle-title'>See a skin location"
            "<span class='g'>across any pose</span></div>"
            "<div class='tyle-sub'>Pick a point on a reference photo and instantly see whether "
            "that same body region is well-viewed, seen side-on, or occluded in a second pose - "
            "mapped through a fixed-topology 3D body mesh.</div>"
            "</div>"
        )

        # Order must match `_run`'s unpacking: green, side, rings, radius, min_vis, min_green.
        param_sliders = [green_sl, side_sl, rings_sl, radius_sl, minvis_sl, mingreen_sl]

        with gr.Group(elem_classes="tyle-card"):
            gr.HTML("<div class='tyle-section'>Step 1 &middot; Upload poses</div>")
            with gr.Row():
                ref_in = gr.Image(
                    label="Reference pose", type="numpy", height=300,
                    sources=["upload"], buttons=[],
                )
                tgt_in = gr.Image(
                    label="Target pose", type="numpy", height=300,
                    sources=["upload"], buttons=[],
                )
            extract_btn = gr.Button(
                "Extract meshes", variant="primary", elem_classes="tyle-primary"
            )
            status_html = gr.HTML(
                value=_banner("Upload two images and click Extract meshes.")
            )

        with gr.Group(elem_classes="tyle-card"):
            gr.HTML("<div class='tyle-section'>Step 2 &middot; Select on the reference, compare on the target</div>")
            with gr.Row(elem_classes="tyle-pair"):
                with gr.Column(min_width=0):
                    gr.HTML("<div class='tyle-cap'>Reference &mdash; click to select a point</div>")
                    # interactive=True is required for the .select click coords, but
                    # an *empty* interactive image draws the upload prompt, so we keep
                    # it hidden until extraction populates it. Label moved to the
                    # caption above so it never overlaps the mesh.
                    select_img = gr.Image(
                        type="numpy", interactive=True, sources=[], buttons=[],
                        show_label=False, height=440, elem_id="tyle-select",
                        visible=False,
                    )
                with gr.Column(min_width=0):
                    gr.HTML("<div class='tyle-cap'>Target &mdash; mapped region</div>")
                    result_img = gr.Image(
                        type="numpy", interactive=False, buttons=[], height=440,
                        show_label=False, visible=False,
                    )
            gr.HTML(
                "<div class='tyle-legend'>"
                "<span><span class='dot' style='background:#00C853'></span>green = good view</span>"
                "<span><span class='dot' style='background:#FFC107'></span>yellow = side / partial</span>"
                "<span><span class='dot' style='background:#E53935'></span>red = occluded</span>"
                "</div>"
            )
            verdict_html = gr.HTML()
            info_md = gr.Markdown()

        extract_btn.click(
            on_extract,
            inputs=[ref_in, tgt_in],
            outputs=[ref_mesh_state, tgt_mesh_state, select_img, result_img, status_html],
        )

        select_img.select(
            on_click,
            inputs=[ref_mesh_state, tgt_mesh_state, *param_sliders],
            outputs=[poi_state, select_img, result_img, verdict_html, info_md],
        )

        for sl in param_sliders:
            sl.release(
                on_param_change,
                inputs=[ref_mesh_state, tgt_mesh_state, poi_state, *param_sliders],
                outputs=[select_img, result_img, verdict_html, info_md],
            )

        # Force light mode on load (belt-and-suspenders with the CSS overrides).
        demo.load(None, None, None, js=FORCE_LIGHT_JS)

    return demo


def main():
    global PIPE, SETTINGS
    parser = argparse.ArgumentParser(description="Skin LOI Estimation Gradio demo")
    add_cli_args(parser)
    args = parser.parse_args()

    SETTINGS = Settings.from_namespace(args)
    if SETTINGS.device is None:
        SETTINGS.device = os.environ.get("SKIN_LOI_DEVICE")
    PIPE = LOIPipeline(SETTINGS)

    build_ui().launch(
        theme=THEME,
        css=CSS,
        server_name=SETTINGS.host,
        server_port=SETTINGS.port,
        share=SETTINGS.share,
    )


if __name__ == "__main__":
    main()
