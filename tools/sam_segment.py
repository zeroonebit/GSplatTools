"""
sam_segment.py  —  SAM 2 Click-Based Mask Generator
Generate masks from point prompts (positive / negative clicks) using Meta SAM 2.

Two modes:
  image  — apply prompts to every frame independently via SAM2ImagePredictor (fast, no GPU RAM hold)
  video  — propagate annotations from keyframes via SAM2VideoPredictor (accurate, requires local checkpoint)

Point prompts JSON format:
  [
    {
      "frame": "frame_000001.png",
      "points": [
        {"x": 512, "y": 300, "label": 1},
        {"x":  80, "y":  90, "label": 0}
      ]
    }
  ]
  label 1 = positive (include / mask this), label 0 = negative (exclude / do not mask).
  In image mode a single entry with "frame": "*" applies to every frame.

Usage:
  python tools/sam_segment.py output/scene/front/frames/ --prompts prompts.json
  python tools/sam_segment.py output/scene/front/frames/ --prompts prompts.json --mode video
  python tools/sam_segment.py output/scene/front/frames/ --prompts prompts.json --no-gpu

SAM 2 install:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install git+https://github.com/facebookresearch/sam2.git

Output:
  output/scene/front/masks/<frame_name>.png   (white=mask, black=keep)
"""

__version__ = "1.0.0"

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("sam_segment")

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_MODEL_HF = "facebook/sam2.1-hiera-small"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("prompts JSON must be a list of objects")
    return data


def dilate_mask(mask, px: int):
    import cv2
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask, k)


def _infer_sam2_config(checkpoint: str) -> str:
    name = Path(checkpoint).stem.lower()
    if "large" in name:
        return "sam2.1_hiera_l.yaml"
    if "base_plus" in name or "base+" in name:
        return "sam2.1_hiera_b+.yaml"
    if "tiny" in name:
        return "sam2.1_hiera_t.yaml"
    return "sam2.1_hiera_s.yaml"


def resolve_device(device: str, no_gpu: bool) -> str:
    if no_gpu:
        return "cpu"
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Image mode — SAM2ImagePredictor, one frame at a time
# ---------------------------------------------------------------------------

def _load_image_predictor(model: str, device: str):
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "SAM 2 not installed.\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git"
        )
    log.info("Loading SAM 2 image predictor: %s on %s", model, device)
    if Path(model).is_file():
        from sam2.build_sam import build_sam2
        sam = build_sam2(_infer_sam2_config(model), model, device=device)
        return SAM2ImagePredictor(sam)
    return SAM2ImagePredictor.from_pretrained(model)


def run_image_mode(
    frames_dir: Path,
    output_dir: Path,
    prompts: list[dict],
    model: str,
    device: str,
    dilate_px: int,
    dry_run: bool,
) -> int:
    import numpy as np
    import cv2
    from tqdm import tqdm

    # Build lookup: frame_name → points list.
    # A single entry with frame="*" acts as a global template for all frames.
    frame_prompts: dict[str, list[dict]] = {}
    global_pts: list[dict] = []
    for entry in prompts:
        fname = entry.get("frame", "*")
        pts = entry.get("points", [])
        if fname == "*":
            global_pts = pts
        else:
            frame_prompts[fname] = pts

    frames = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not frames:
        log.error("No frames found in %s", frames_dir)
        return 1

    if dry_run:
        log.info(
            "[DRY-RUN] image mode: %d frames, %d per-frame annotations, global=%d pts",
            len(frames), len(frame_prompts), len(global_pts),
        )
        return 0

    try:
        predictor = _load_image_predictor(model, device)
        predictor.model.to(device)
    except (ImportError, Exception) as e:
        log.critical("%s", e)
        return 2

    import torch
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_path in tqdm(frames, unit="frame"):
        pts = frame_prompts.get(frame_path.name) if frame_path.name in frame_prompts else global_pts

        img_bgr = cv2.imread(str(frame_path))
        h, w = img_bgr.shape[:2]

        if not pts:
            cv2.imwrite(str(output_dir / frame_path.name), np.zeros((h, w), np.uint8))
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        coords = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)
        labels = np.array([p["label"] for p in pts], dtype=np.int32)

        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            predictor.set_image(img_rgb)
            masks, _, _ = predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=False,
            )

        mask = (masks[0] * 255).astype(np.uint8)
        mask = dilate_mask(mask, dilate_px)
        cv2.imwrite(str(output_dir / frame_path.name), mask)

    log.info("Done. Masks: %s", output_dir)
    return 0


# ---------------------------------------------------------------------------
# Video mode — SAM2VideoPredictor, propagate from keyframes
# ---------------------------------------------------------------------------

def _load_video_predictor(model: str, device: str):
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        raise ImportError(
            "SAM 2 not installed.\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git"
        )
    if not Path(model).is_file():
        raise ValueError(
            "Video propagation mode requires a local SAM 2 checkpoint.\n"
            "Download from: https://github.com/facebookresearch/sam2/blob/main/INSTALL.md\n"
            "Example: sam2.1_hiera_small.pt\n"
            "Then pass: --model /path/to/sam2.1_hiera_small.pt"
        )
    log.info("Loading SAM 2 video predictor: %s on %s", model, device)
    return build_sam2_video_predictor(_infer_sam2_config(model), model, device=device)


def run_video_mode(
    frames_dir: Path,
    output_dir: Path,
    prompts: list[dict],
    model: str,
    device: str,
    dilate_px: int,
    dry_run: bool,
) -> int:
    import numpy as np
    import cv2

    frames = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not frames:
        log.error("No frames found in %s", frames_dir)
        return 1

    name_to_idx = {f.name: i for i, f in enumerate(frames)}

    if dry_run:
        log.info("[DRY-RUN] video mode: propagate through %d frames", len(frames))
        for entry in prompts:
            fname = entry.get("frame", "?")
            fidx = name_to_idx.get(fname, "?")
            log.info("  keyframe %-30s idx=%-4s  pts=%d",
                     fname, fidx, len(entry.get("points", [])))
        return 0

    try:
        predictor = _load_video_predictor(model, device)
    except (ImportError, ValueError) as e:
        log.critical("%s", e)
        return 2

    import torch
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frames_dir))
        predictor.reset_state(state)

        for entry in prompts:
            fname = entry.get("frame")
            if fname not in name_to_idx:
                log.warning("Frame not found in dir, skipping: %s", fname)
                continue
            pts = entry.get("points", [])
            if not pts:
                continue
            fidx = name_to_idx[fname]
            coords = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)
            labels = np.array([p["label"] for p in pts], dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=fidx,
                obj_id=1,
                points=coords,
                labels=labels,
            )
            log.info("Annotated keyframe %s (idx=%d): %d pts", fname, fidx, len(pts))

        log.info("Propagating through %d frames...", len(frames))
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            mask = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
            mask = dilate_mask(mask, dilate_px)
            cv2.imwrite(str(output_dir / frames[frame_idx].name), mask)

    log.info("Done. Masks: %s", output_dir)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sam_segment",
        description="Generate per-frame masks from click prompts using SAM 2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", metavar="FRAMES_DIR",
                   help="Directory of PNG frames (from sharp_frames output)")
    p.add_argument("--prompts", required=True, metavar="JSON",
                   help="JSON file with point annotations (see format above)")
    p.add_argument("--mode", choices=["image", "video"], default="image",
                   help="image = per-frame inference (default). video = SAM 2 propagation.")
    p.add_argument("--model", default=DEFAULT_MODEL_HF, metavar="PATH_OR_HF_ID",
                   help=f"SAM 2 model: HuggingFace ID (image mode) or local .pt (video mode). "
                        f"Default: {DEFAULT_MODEL_HF}")
    p.add_argument("--device", default="auto",
                   help="cuda | cpu | auto (default: auto)")
    p.add_argument("--no-gpu", action="store_true",
                   help="Force CPU inference")
    p.add_argument("--dilate", type=int, default=5, metavar="PX",
                   help="Dilate mask edges by N pixels (default: 5)")
    p.add_argument("-o", "--output-dir", metavar="DIR",
                   help="Mask output directory (default: <frames_dir>/../masks/)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would run without executing")
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

    frames_dir = Path(args.input)
    if not frames_dir.is_dir():
        log.critical("Not a directory: %s", args.input)
        sys.exit(2)

    output_dir = Path(args.output_dir) if args.output_dir else frames_dir.parent / "masks"

    try:
        prompts = load_prompts(args.prompts)
    except (OSError, ValueError) as e:
        log.critical("Failed to load prompts: %s", e)
        sys.exit(2)

    if not prompts:
        log.critical("Prompts file is empty — nothing to do")
        sys.exit(2)

    device = resolve_device(args.device, args.no_gpu)
    log.info("Device: %s | mode: %s | frames: %s", device, args.mode, frames_dir)

    runner = run_video_mode if args.mode == "video" else run_image_mode
    rc = runner(
        frames_dir=frames_dir,
        output_dir=output_dir,
        prompts=prompts,
        model=args.model,
        device=device,
        dilate_px=args.dilate,
        dry_run=args.dry_run,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
