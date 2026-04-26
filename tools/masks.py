"""
masks.py  —  Dynamic Object Mask Generator
Generate binary masks for dynamic objects (drones, people, vehicles) in image
sequences so they are excluded from 3D Gaussian Splatting reconstruction.

Outputs masks as PNG images matching the input frame filenames, where:
    white (255) = object to exclude
    black (0)   = background / keep

Two strategies (selectable via --method):

  yolo  — Fast bounding-box detection with YOLOv8.
          Masks are filled rectangles. Good for quick exclusion.
          pip install ultralytics

  sam   — Precise pixel-level segmentation with Meta SAM 2.
          Uses YOLO boxes as SAM prompts for best accuracy + speed.
          pip install ultralytics torch torchvision
          pip install git+https://github.com/facebookresearch/sam2.git

Usage:
    python tools/masks.py frames_dir/ [options]
    python tools/masks.py output/scene/front/frames/ --classes drone person --method sam

Planned options:
    --classes NAMES     Space-separated class names to mask (default: person drone)
    --method METHOD     yolo | sam (default: yolo)
    --confidence FLOAT  Detection confidence threshold (default: 0.4)
    --output-dir DIR    Where to write mask PNGs (default: <input_dir>/../masks/)
    --model PATH        Custom YOLO model weights (default: yolov8n.pt auto-download)
    --sam-model PATH    SAM 2 checkpoint path
    --device DEVICE     cuda | cpu | mps (default: auto-detect)
    --dilate N          Dilate masks by N pixels to cover edges (default: 10)
    --dry-run           Report detections without writing masks
    -v, --verbose       Show per-frame detection counts

Dependencies (install when ready to use):
    pip install ultralytics>=8.0.0
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install git+https://github.com/facebookresearch/sam2.git

YOLO class names reference (COCO dataset):
    0: person   14: bird    16: dog     4: airplane
    Custom drone detection: use a fine-tuned model or class 4 (airplane) as proxy.
"""

# TODO: implement — scaffold below shows the intended structure

import argparse
import sys
from pathlib import Path

# Deferred imports — only loaded when actually running
# import cv2
# import numpy as np
# from tqdm import tqdm
# from ultralytics import YOLO


__version__ = "0.1.0-stub"

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".exr"}

DEFAULT_CLASSES = ["person", "drone"]
DEFAULT_CONFIDENCE = 0.4
DEFAULT_DILATE_PX = 10


def detect_with_yolo(
    model,         # ultralytics YOLO instance
    image_path: str,
    class_names: list[str],
    confidence: float,
):
    """
    Run YOLO on one image. Return list of (class_name, bbox_xyxy) tuples
    for all detections matching class_names above confidence.
    """
    raise NotImplementedError


def boxes_to_mask(
    detections: list,
    image_shape: tuple[int, int],
    dilate_px: int = DEFAULT_DILATE_PX,
):
    """
    Convert bounding boxes to a binary mask (numpy uint8, 0/255).
    Applies dilation to cover object edges.
    Requires cv2, numpy.
    """
    raise NotImplementedError


def refine_with_sam(
    sam_predictor,  # SAM2 predictor instance
    image,          # numpy BGR image
    boxes,          # bounding boxes as numpy array [[x1,y1,x2,y2], ...]
):
    """
    Use SAM 2 to produce precise pixel masks from bounding box prompts.
    Returns combined binary mask (numpy uint8, 0/255).
    """
    raise NotImplementedError


def process_frames_dir(
    frames_dir: str,
    output_dir: str,
    class_names: list[str],
    method: str,
    confidence: float,
    dilate_px: int,
    model_path: str | None,
    sam_model_path: str | None,
    device: str,
    dry_run: bool,
    verbose: bool,
) -> int:
    """
    Process all image frames in frames_dir, write masks to output_dir.
    Returns exit code 0/1/2.
    """
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="masks",
        description="Generate object masks for dynamic scene elements.",
    )
    p.add_argument("input", nargs="+", metavar="FRAMES_DIR",
                   help="Directory containing input image frames")
    p.add_argument("-o", "--output-dir", metavar="DIR",
                   help="Output directory for mask PNGs (default: <input>/../masks/)")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, metavar="CLASS",
                   help=f"Object classes to mask (default: {' '.join(DEFAULT_CLASSES)})")
    p.add_argument("--method", choices=["yolo", "sam"], default="yolo",
                   help="Detection method: yolo (fast bbox) or sam (precise, default: yolo)")
    p.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                   help=f"Detection confidence threshold (default: {DEFAULT_CONFIDENCE})")
    p.add_argument("--model", metavar="PATH",
                   help="YOLO model weights path (default: yolov8n.pt, auto-downloaded)")
    p.add_argument("--sam-model", metavar="PATH",
                   help="SAM 2 checkpoint path")
    p.add_argument("--device", default="auto",
                   help="Compute device: cuda | cpu | mps | auto (default: auto)")
    p.add_argument("--dilate", type=int, default=DEFAULT_DILATE_PX, metavar="N",
                   help=f"Dilate masks by N pixels (default: {DEFAULT_DILATE_PX})")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main() -> None:
    args = build_parser().parse_args()
    print(
        "masks.py is not yet implemented.\n"
        "Install dependencies first:\n"
        "  pip install ultralytics  (for --method yolo)\n"
        "  pip install torch torchvision  (for --method sam)\n"
        f"Input dirs: {args.input}"
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
