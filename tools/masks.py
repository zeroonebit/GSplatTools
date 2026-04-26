"""
masks.py  —  Dynamic Object Mask Generator
Generate binary PNG masks for dynamic objects (people, drones, vehicles)
so they are excluded from 3D Gaussian Splatting reconstruction in Lichtfeld.

Mask convention:
    white (255) = object to exclude
    black (0)   = background / keep

Two methods:
    yolo  — bounding-box fill via YOLOv8. Fast, good enough for most shots.
    sam   — pixel-precise masks via YOLO boxes as SAM 2 prompts. Slower, cleaner edges.

COCO class reference (--classes accepts names or IDs):
    person=0  bicycle=1  car=2  motorcycle=3  airplane=4  bus=5  truck=7
    Note: no native "drone" class in COCO. Use --classes airplane as proxy,
    or supply a custom YOLOv8 model via --model.

Usage:
    python tools/masks.py frames_dir/
    python tools/masks.py output/scene/front/frames/ --classes person airplane
    python tools/masks.py output/scene/front/frames/ --method sam --sam-model sam2_hiera_small.pt

Output:
    output/scene/front/masks/frame_000001.png  (white = exclude, black = keep)

Dependencies:
    pip install ultralytics opencv-python numpy tqdm
    # For --method sam:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install git+https://github.com/facebookresearch/sam2.git
"""

__version__ = "1.0.0"

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("masks")

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".webp"}
DEFAULT_CLASSES = ["person"]
DEFAULT_CONFIDENCE = 0.4
DEFAULT_DILATE_PX = 15

COCO_NAMES: dict[str, int] = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
    "airplane": 4, "bus": 5, "train": 6, "truck": 7,
    "boat": 8, "bird": 14, "cat": 15, "dog": 16,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_class_ids(class_names: list[str]) -> list[int]:
    ids: list[int] = []
    for name in class_names:
        if name.isdigit():
            ids.append(int(name))
        elif name.lower() in COCO_NAMES:
            ids.append(COCO_NAMES[name.lower()])
        else:
            known = ", ".join(COCO_NAMES.keys())
            raise ValueError(
                f"Unknown class '{name}'. Known COCO names: {known}\n"
                "Or pass a numeric class ID directly."
            )
    return ids


def dilate_mask(mask, px: int):
    import cv2
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask, kernel)


def boxes_to_mask(boxes_xyxy: list, h: int, w: int, dilate_px: int):
    import numpy as np
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes_xyxy:
        mask[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))] = 255
    return dilate_mask(mask, dilate_px)


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------

def load_yolo(model_path: str | None):
    from ultralytics import YOLO
    path = model_path or "yolov8n.pt"
    log.info("Loading YOLO model: %s", path)
    return YOLO(path)


def detect_yolo(model, image_path: str, class_ids: list[int], confidence: float) -> list:
    results = model(image_path, conf=confidence, classes=class_ids, verbose=False)
    boxes = []
    for r in results:
        if r.boxes is not None and len(r.boxes):
            boxes.extend(r.boxes.xyxy.cpu().numpy().tolist())
    return boxes


# ---------------------------------------------------------------------------
# SAM 2
# ---------------------------------------------------------------------------

def load_sam(sam_model_path: str | None, device: str):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "SAM 2 not installed.\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git"
        )
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = sam_model_path or "sam2_hiera_small.pt"
    log.info("Loading SAM 2: %s on %s", checkpoint, device)
    sam = build_sam2("sam2_hiera_s.yaml", checkpoint, device=device)
    return SAM2ImagePredictor(sam), device


def refine_with_sam(predictor, image_bgr, boxes_xyxy: list, dilate_px: int):
    import numpy as np
    import cv2
    h, w = image_bgr.shape[:2]
    if not boxes_xyxy:
        return np.zeros((h, w), dtype=np.uint8)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    boxes_np = np.array(boxes_xyxy, dtype=np.float32)
    masks_out, _, _ = predictor.predict(box=boxes_np, multimask_output=False)
    combined = np.zeros((h, w), dtype=np.uint8)
    for m in masks_out:
        arr = m[0] if m.ndim == 3 else m
        combined[arr.astype(bool)] = 255
    return dilate_mask(combined, dilate_px)


# ---------------------------------------------------------------------------
# Per-frame
# ---------------------------------------------------------------------------

def process_frame(
    image_path: Path,
    yolo_model,
    sam_predictor,
    method: str,
    class_ids: list[int],
    confidence: float,
    dilate_px: int,
) -> tuple:  # (mask_array, n_detections)
    import cv2
    import numpy as np

    boxes = detect_yolo(yolo_model, str(image_path), class_ids, confidence)
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    if not boxes:
        return np.zeros((h, w), dtype=np.uint8), 0

    if method == "sam" and sam_predictor is not None:
        mask = refine_with_sam(sam_predictor, img, boxes, dilate_px)
    else:
        mask = boxes_to_mask(boxes, h, w, dilate_px)

    return mask, len(boxes)


# ---------------------------------------------------------------------------
# Directory orchestration
# ---------------------------------------------------------------------------

def process_frames_dir(
    frames_dir: Path,
    output_dir: Path,
    class_ids: list[int],
    method: str,
    confidence: float,
    dilate_px: int,
    model_path: str | None,
    sam_model_path: str | None,
    device: str,
    dry_run: bool,
    verbose: bool,
) -> int:
    import cv2
    from tqdm import tqdm

    frames = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
    if not frames:
        log.error("No image files found in: %s", frames_dir)
        return 1

    log.info("Found %d frames in %s", len(frames), frames_dir)

    try:
        yolo = load_yolo(model_path)
    except ImportError as e:
        log.critical("%s\npip install ultralytics", e)
        return 2

    sam_predictor = None
    if method == "sam":
        try:
            sam_predictor, device = load_sam(sam_model_path, device)
        except ImportError as e:
            log.critical("%s", e)
            return 2

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    total_dets = 0
    frames_with_dets = 0

    for frame_path in tqdm(frames, unit="frame", disable=not sys.stderr.isatty()):
        if dry_run:
            boxes = detect_yolo(yolo, str(frame_path), class_ids, confidence)
            n = len(boxes)
            if verbose or n > 0:
                log.info("DRY-RUN  %s  detections=%d", frame_path.name, n)
            total_dets += n
            if n > 0:
                frames_with_dets += 1
            continue

        mask, n = process_frame(
            frame_path, yolo, sam_predictor, method, class_ids, confidence, dilate_px
        )
        cv2.imwrite(str(output_dir / frame_path.name), mask)
        total_dets += n
        if n > 0:
            frames_with_dets += 1
        if verbose:
            log.debug("%s  detections=%d", frame_path.name, n)

    log.info(
        "Done. %d/%d frames had detections (%d total).",
        frames_with_dets, len(frames), total_dets,
    )
    if not dry_run:
        log.info("Masks saved: %s", output_dir)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="masks",
        description="Generate binary masks for dynamic objects in frame sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", nargs="+", metavar="FRAMES_DIR",
                   help="Directory of PNG frames (output from sharp_frames)")
    p.add_argument("-o", "--output-dir", metavar="DIR",
                   help="Mask output dir (default: <frames_dir>/../masks/)")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, metavar="CLASS",
                   help=f"Classes to mask (default: {' '.join(DEFAULT_CLASSES)}). "
                        "Names: person airplane car motorcycle. Or numeric COCO IDs.")
    p.add_argument("--method", choices=["yolo", "sam"], default="yolo",
                   help="yolo = bbox fill (default). sam = pixel-precise via SAM 2.")
    p.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                   help=f"Detection confidence (default: {DEFAULT_CONFIDENCE})")
    p.add_argument("--model", metavar="PATH",
                   help="YOLO weights (default: yolov8n.pt, auto-downloaded)")
    p.add_argument("--sam-model", metavar="PATH",
                   help="SAM 2 checkpoint path")
    p.add_argument("--device", default="auto",
                   help="cuda | cpu | mps | auto (default: auto)")
    p.add_argument("--dilate", type=int, default=DEFAULT_DILATE_PX, metavar="PX",
                   help=f"Dilate mask edges by N pixels (default: {DEFAULT_DILATE_PX})")
    p.add_argument("--dry-run", action="store_true",
                   help="Report detections without writing masks")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        import cv2       # noqa: F401
        import numpy     # noqa: F401
        from tqdm import tqdm  # noqa: F401
    except ImportError as e:
        log.critical("Missing dependency: %s\npip install opencv-python numpy tqdm", e)
        sys.exit(2)

    try:
        class_ids = resolve_class_ids(args.classes)
    except ValueError as e:
        log.critical("%s", e)
        sys.exit(2)

    log.info("Classes: %s → IDs %s", args.classes, class_ids)

    failed = 0
    for input_str in args.input:
        frames_dir = Path(input_str)
        if not frames_dir.is_dir():
            log.error("Not a directory: %s", input_str)
            failed += 1
            continue
        output_dir = Path(args.output_dir) if args.output_dir else frames_dir.parent / "masks"
        rc = process_frames_dir(
            frames_dir=frames_dir,
            output_dir=output_dir,
            class_ids=class_ids,
            method=args.method,
            confidence=args.confidence,
            dilate_px=args.dilate,
            model_path=args.model,
            sam_model_path=args.sam_model,
            device=args.device,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if rc != 0:
            failed += 1

    if failed == len(args.input):
        sys.exit(2)
    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
