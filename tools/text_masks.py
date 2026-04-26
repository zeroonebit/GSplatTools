"""
text_masks.py  —  Text-Prompt AI Mask Generator
Type a description ("person", "drone", "tripod") and the tool finds and masks
those objects across every frame using YOLO-World detection + optional SAM 2 refinement.

This mirrors the "SAM3 text-prompt masking" approach: YOLO-World gives open-vocabulary
bounding boxes, SAM 2 turns them into pixel-precise masks.

Without SAM 2: fills the detected bounding boxes (fast, same as masks.py YOLO mode).
With SAM 2:    traces exact object edges (slower, cleaner, recommended for 360 datasets).

The equirectangular-unwrap trick from 360 Splat Pro is not needed here because this
tool operates on the already-undistorted perspective frames from eq2persp output.

Usage:
    python tools/text_masks.py output/scene/front/frames/ --text "person"
    python tools/text_masks.py output/scene/front/frames/ --text "person, drone, tripod"
    python tools/text_masks.py output/scene/front/frames/ --text "person" --sam --sam-model sam2.1_hiera_small.pt
    python tools/text_masks.py output/scene/front/frames/ --text "person" --dry-run

Dependencies:
    pip install ultralytics>=8.2.0      (YOLO-World support added in 8.x)
    # For SAM 2 refinement:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install git+https://github.com/facebookresearch/sam2.git

Output:
    output/scene/front/masks/<frame>.png   (white=exclude, black=keep)
"""

__version__ = "1.0.0"

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("text_masks")

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_YOLO_WORLD_MODEL = "yolo11x-worldv2.pt"  # best accuracy; yolo11s-worldv2.pt for speed


# ---------------------------------------------------------------------------
# YOLO-World detection
# ---------------------------------------------------------------------------

def load_yolo_world(model_path: str, text_prompts: list[str]):
    from ultralytics import YOLO
    log.info("Loading YOLO-World: %s", model_path)
    model = YOLO(model_path)
    model.set_classes(text_prompts)
    log.info("Classes set: %s", text_prompts)
    return model


def detect_text(model, image_path: str, confidence: float) -> list:
    results = model(image_path, conf=confidence, verbose=False)
    boxes = []
    for r in results:
        if r.boxes is not None and len(r.boxes):
            boxes.extend(r.boxes.xyxy.cpu().numpy().tolist())
    return boxes


# ---------------------------------------------------------------------------
# SAM 2 refinement (optional)
# ---------------------------------------------------------------------------

def _infer_sam2_config(checkpoint: str) -> str:
    name = Path(checkpoint).stem.lower()
    if "large" in name:
        return "sam2.1_hiera_l.yaml"
    if "base_plus" in name or "base+" in name:
        return "sam2.1_hiera_b+.yaml"
    if "tiny" in name:
        return "sam2.1_hiera_t.yaml"
    return "sam2.1_hiera_s.yaml"


def load_sam2(model_path: str | None, device: str):
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "SAM 2 not installed.\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git"
        )
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = model_path or "facebook/sam2.1-hiera-small"
    log.info("Loading SAM 2: %s on %s", checkpoint, device)
    if Path(checkpoint).is_file():
        from sam2.build_sam import build_sam2
        sam = build_sam2(_infer_sam2_config(checkpoint), checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam)
    else:
        predictor = SAM2ImagePredictor.from_pretrained(checkpoint)
        predictor.model.to(device)
    return predictor, device


def refine_boxes_with_sam(predictor, image_rgb, boxes_xyxy: list, device: str):
    import numpy as np
    import torch
    h, w = image_rgb.shape[:2]
    if not boxes_xyxy:
        return np.zeros((h, w), dtype=np.uint8)
    boxes_np = np.array(boxes_xyxy, dtype=np.float32)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        predictor.set_image(image_rgb)
        masks, _, _ = predictor.predict(
            box=boxes_np,
            multimask_output=False,
        )
    combined = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        arr = m[0] if m.ndim == 3 else m
        combined[arr.astype(bool)] = 255
    return combined


# ---------------------------------------------------------------------------
# Per-frame
# ---------------------------------------------------------------------------

def boxes_to_mask(boxes_xyxy: list, h: int, w: int, dilate_px: int):
    import numpy as np
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes_xyxy:
        mask[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))] = 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, k)
    return mask


def process_frame(image_path, yolo_model, sam_predictor, sam_device,
                  confidence, dilate_px, use_sam):
    import cv2
    import numpy as np

    boxes = detect_text(yolo_model, str(image_path), confidence)
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    if not boxes:
        return np.zeros((h, w), dtype=np.uint8), 0

    if use_sam and sam_predictor is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = refine_boxes_with_sam(sam_predictor, img_rgb, boxes, sam_device)
        if dilate_px > 0:
            import cv2 as _cv2
            k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
            mask = _cv2.dilate(mask, k)
    else:
        mask = boxes_to_mask(boxes, h, w, dilate_px)

    return mask, len(boxes)


# ---------------------------------------------------------------------------
# Directory orchestration
# ---------------------------------------------------------------------------

def process_frames_dir(
    frames_dir: Path,
    output_dir: Path,
    text_prompts: list[str],
    confidence: float,
    dilate_px: int,
    use_sam: bool,
    yolo_model_path: str | None,
    sam_model_path: str | None,
    device: str,
    dry_run: bool,
    verbose: bool,
) -> int:
    import cv2
    from tqdm import tqdm

    frames = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not frames:
        log.error("No frames found in %s", frames_dir)
        return 1

    log.info("%d frames | prompts: %s", len(frames), text_prompts)

    try:
        yolo = load_yolo_world(yolo_model_path or DEFAULT_YOLO_WORLD_MODEL, text_prompts)
    except Exception as e:
        log.critical("Failed to load YOLO-World: %s", e)
        return 2

    sam_predictor = None
    sam_device_resolved = device
    if use_sam:
        try:
            sam_predictor, sam_device_resolved = load_sam2(sam_model_path, device)
        except ImportError as e:
            log.warning("SAM 2 not available, falling back to bbox fill: %s", e)
            use_sam = False

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    total_dets = 0
    frames_with_dets = 0

    for frame_path in tqdm(frames, unit="frame", disable=not sys.stderr.isatty()):
        if dry_run:
            boxes = detect_text(yolo, str(frame_path), confidence)
            n = len(boxes)
            total_dets += n
            if n > 0:
                frames_with_dets += 1
            if verbose or n > 0:
                log.info("DRY-RUN  %s  detections=%d", frame_path.name, n)
            continue

        mask, n = process_frame(
            frame_path, yolo, sam_predictor, sam_device_resolved,
            confidence, dilate_px, use_sam,
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
        log.info("Masks: %s", output_dir)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="text_masks",
        description="Generate masks using text prompts + YOLO-World (+ optional SAM 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", nargs="+", metavar="FRAMES_DIR")
    p.add_argument(
        "--text", required=True, metavar="PROMPTS",
        help="Comma-separated text prompts, e.g. \"person, drone, tripod\"",
    )
    p.add_argument("-o", "--output-dir", metavar="DIR",
                   help="Mask output dir (default: <frames_dir>/../masks/)")
    p.add_argument("--confidence", type=float, default=0.25,
                   help="Detection confidence (default: 0.25 — YOLO-World works best lower than YOLO)")
    p.add_argument("--model", metavar="PATH",
                   help=f"YOLO-World model (default: {DEFAULT_YOLO_WORLD_MODEL})")
    p.add_argument("--sam", action="store_true",
                   help="Refine bbox detections with SAM 2 for pixel-precise edges")
    p.add_argument("--sam-model", metavar="PATH",
                   help="SAM 2 checkpoint path or HF model ID (default: facebook/sam2.1-hiera-small)")
    p.add_argument("--device", default="auto",
                   help="cuda | cpu | auto (default: auto)")
    p.add_argument("--dilate", type=int, default=10, metavar="PX",
                   help="Dilate mask edges by N pixels (default: 10)")
    p.add_argument("--dry-run", action="store_true")
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
        import cv2     # noqa: F401
        import numpy   # noqa: F401
        from tqdm import tqdm  # noqa: F401
    except ImportError as e:
        log.critical("Missing dependency: %s\npip install opencv-python numpy tqdm", e)
        sys.exit(2)

    prompts = [p.strip() for p in args.text.split(",") if p.strip()]
    if not prompts:
        log.critical("--text cannot be empty")
        sys.exit(2)

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
            text_prompts=prompts,
            confidence=args.confidence,
            dilate_px=args.dilate,
            use_sam=args.sam,
            yolo_model_path=args.model,
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
