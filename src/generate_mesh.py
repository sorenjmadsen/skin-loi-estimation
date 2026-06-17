from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

# Resolve paths relative to this script, not the shell cwd
SCRIPT_DIR = Path(__file__).resolve().parent      # .../src
PROJECT_ROOT = SCRIPT_DIR.parent                  # .../skin-loi-estimation
SAM3D_ROOT = PROJECT_ROOT / "sam3d"

sys.path.insert(0, str(SAM3D_ROOT))

from notebook.utils import setup_sam_3d_body, save_mesh_results
from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate human mesh from image(s)")
    parser.add_argument(
        "--image",
        type=Path,
        help="Single input image",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory of images to batch-process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for .ply, overlays, bbox images, focal length JSON",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    return parser.parse_args()


def collect_images(args: argparse.Namespace) -> list[Path]:
    if args.image and args.image_dir:
        raise SystemExit("Use either --image or --image-dir, not both.")
    if args.image:
        return [args.image.resolve()]
    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        return sorted(
            p.resolve()
            for p in args.image_dir.iterdir()
            if p.suffix.lower() in exts
        )
    raise SystemExit("Provide --image or --image-dir.")

def generate_mesh_from_image(image_path: Path, estimator: SAM3DBodyEstimator = None, return_image: bool = False) -> None:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    outputs = estimator.process_one_image(str(image_path))
    if not outputs:
        return None
    if return_image:
        return outputs, img_bgr
    return outputs


def main() -> None:
    args = parse_args()
    image_paths = collect_images(args)

    estimator = setup_sam_3d_body(
        hf_repo_id="facebook/sam-3d-body-dinov3",
        device=args.device,
    )

    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        print(f"Processing {image_path}...")
        outputs, img_bgr = generate_mesh_from_image(image_path, estimator, return_image=True)
        if not outputs:
            print(f"No people detected in {image_path.name}")
            continue

        image_name = image_path.stem
        out_dir = args.output_dir / image_name
        out_dir.mkdir(parents=True, exist_ok=True)

        ply_files = save_mesh_results(
            img_bgr, outputs, estimator.faces, str(out_dir), image_name
        )
        print(f"Saved {len(ply_files)} mesh(es) to {out_dir}")


if __name__ == "__main__":
    main()