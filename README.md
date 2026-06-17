# Skin LOI Estimation

Cross-pose visibility classification for a skin **location of interest (LOI)**.

Pick a point on a **reference** photo of a person, and the app tells you whether
that exact body region is well-viewed, seen side-on, or occluded in a **target**
photo of the same person in a different pose. It renders the region back onto the
target image in three colors:

- **Green** - good, straight-on view
- **Yellow** - partial / side view (surface near-orthogonal to the camera)
- **Red** - occluded (hidden behind other geometry) or back-facing

## How it works

```mermaid
flowchart LR
    refImg["Reference image"] --> refMesh["SAM 3D Body mesh A"]
    tgtImg["Target image"] --> tgtMesh["SAM 3D Body mesh B"]
    click["Click pixel on reference"] --> anchor["Snap to frontmost vertex vstar"]
    refMesh --> anchor
    anchor --> patch["Grow n-ring patch P"]
    patch --> classify["Classify P on mesh B"]
    tgtMesh --> classify
    classify --> verdict["Green / Yellow / Red verdict + overlay"]
```

The key idea: **SAM 3D Body outputs a fixed-topology body mesh** (MHR70). Every
image yields vertices with identical count and ordering plus shared faces, so the
anchor vertex and its grown patch index the *same anatomical region* across poses
with no registration step.

Per patch vertex (camera at the origin in the render frame):

- **Facing angle** `cos(theta) = -(normal . viewing_ray)` - +1 faces the camera,
  ~0 is grazing, < 0 points away.
- **Self-occlusion** - cast a ray from the camera through the vertex; if a nearer
  surface is hit, the vertex is hidden.

Labels use two angular bands on the facing cosine (defaults in
[`src/skin_loi/config.py`](src/skin_loi/config.py), tunable via CLI / UI):

- `GREEN` - visible and straight-on (`cos >= green_cos`)
- `YELLOW` - visible but side/partial (`side_cos <= cos < green_cos`)
- `RED` - occluded, back-facing, or too edge-on (`cos < side_cos`)

A region verdict aggregates the patch (visible fraction + green fraction).

## Quick Demo

Select a LOI on the reference image and `skin_loi` automatically maps  the projection onto the target image and classifies the visibility of the region. 

Step 1. Select a reference and a target image depicting  t

<table>
  <tr>
    <td align="center"><img src="results/Step1.png" alt="Step 1: select LOI on reference" width="100%"></td>
    <td align="center"><img src="results/Step2.png" alt="Step 2: classified region on target" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>Step 1</b> — select the LOI on the reference image</td>
    <td align="center"><b>Step 2</b> — mapped & classified on the target image</td>
  </tr>
</table>

## Setup

1. Install SAM 3D Body and its dependencies per [`sam3d/INSTALL.md`](sam3d/INSTALL.md),
   including PyTorch and Hugging Face access to `facebook/sam-3d-body-dinov3`.
2. Install the demo extras:

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open http://localhost:7860. Optionally set the device:

```bash
SKIN_LOI_DEVICE=cuda python app.py   # or cpu
```

### Using the app

1. Upload a **reference** and a **target** image, click **Extract meshes**.
2. Click a point on the reference image to pick the LOI; the snapped region is
   previewed.
3. The target pose shows the colored region overlay and a verdict badge. Tune the
   **Thresholds** (grazing cosine, region rings, anchor radius) to recompute live.

## Batch mesh generation (CLI)

`src/generate_mesh.py` is a standalone command-line tool for pre-extracting
SAM 3D Body meshes from images (independent of the Gradio app). For each input
image it writes a `.ply` mesh, a mesh overlay, a bbox image, and a focal-length
JSON into a per-image subfolder of the output directory. This is how the sample
`output/poseA` and `output/poseB` artifacts were produced.

```bash
# Single image
python src/generate_mesh.py --image poses/poseA.png --output-dir output

# A whole directory of images (.jpg/.jpeg/.png/.webp/.bmp)
python src/generate_mesh.py --image-dir poses --output-dir output

# Run on CPU instead of the default CUDA
python src/generate_mesh.py --image poses/poseA.png --output-dir output --device cpu
```

Arguments:

- `--image` - path to a single input image (mutually exclusive with `--image-dir`).
- `--image-dir` - directory of images to batch-process.
- `--output-dir` (required) - destination for the `.ply`, overlays, bbox images,
  and focal-length JSON.
- `--device` - `cuda` (default) or `cpu`.

Output for an image named `poseA.png` lands in `output/poseA/`:

```
output/poseA/
  poseA_mesh_000.ply
  poseA_overlay_000.png
  poseA_bbox_000.png
  poseA_focal_length.json
```

Note: this tool is not used by the web app, which extracts meshes in-memory via
`skin_loi.mesh_extraction.extract_mesh`. It is kept as a convenience for
offline/batch mesh export.

## Project layout

```
<<<<<<< HEAD
app.py                                 # Gradio web app
src/skin_loi/
  config.py                            # model id, colors, thresholds
  mesh_extraction.py                   # MeshResult + extract_mesh()
  projection.py                        # project(), pixel_to_vertex()
  region.py                            # n_ring() patch growing
  visibility.py                        # 3-way classifier + region verdict
  overlay.py                           # colored overlays + verdict badge HTML
  pipeline.py                          # LOIPipeline orchestrator (loads estimator once)
tests/test_visibility.py               # classifier threshold tests
notebooks/test_loi_extraction.ipynb    # original exploration notebook (reference)
=======
app.py                       # Gradio web app
src/
  generate_mesh.py           # standalone CLI for batch mesh export
  skin_loi/
    config.py                # model id, colors, thresholds
    mesh_extraction.py       # MeshResult + extract_mesh()
    projection.py            # project(), pixel_to_vertex()
    region.py                # n_ring() patch growing
    visibility.py            # 3-way classifier + region verdict
    overlay.py               # colored overlays + verdict badge HTML
    pipeline.py              # LOIPipeline orchestrator (loads estimator once)
tests/test_visibility.py     # classifier threshold tests
notebooks/                   # original exploration notebooks (reference)
>>>>>>> b5fcd88 (fix(readme): add docs for using the generate_mesh.py CLI tool)
```
