"""
combine_masks.py  —  Mask Layer Combiner
Merge multiple mask directories into a single output using OR (union) logic.
White (255) = exclude, black (0) = keep — any source marking a pixel as exclude wins.

Use this to stack:
  - AI masks (YOLO / YOLO-World / SAM click)
  - Shape masks (circle, strip, rectangle)
  - Any other per-frame mask set

Usage:
    # Combine AI masks + shape masks into one output
    python tools/combine_masks.py output/scene/front/masks_ai/ output/scene/front/masks_shape/
    python tools/combine_masks.py masks_a/ masks_b/ masks_c/ -o output/scene/front/masks/

    # Built-in shape generators (use alongside OR on top of AI masks)
    python tools/combine_masks.py --shape circle   --frames output/scene/front/frames/
    python tools/combine_masks.py --shape bottom   --frames output/scene/front/frames/ --height 0.15
    python tools/combine_masks.py --shape top      --frames output/scene/front/frames/ --height 0.10

Shapes:
    circle  — filled circle inscribed in the frame (for fisheye/lens vignette masking)
    bottom  — horizontal strip at the bottom (mask tripod / rig)
    top     — horizontal strip at the top (mask sky / nadir artefacts)
    rect    — custom rectangle defined by --rect x1 y1 x2 y2 (normalised 0-1)

Dependencies:
    pip install opencv-python numpy Pillow tqdm
"""

__version__ = "1.0.0"

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("combine_masks")

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# Shape generators  — return a (H, W) uint8 mask given frame dimensions
# ---------------------------------------------------------------------------

def shape_circle(h: int, w: int, radius_frac: float = 0.95) -> np.ndarray:
    """Mask everything OUTSIDE the inscribed circle (black inside, white outside)."""
    import cv2
    mask = np.ones((h, w), dtype=np.uint8) * 255
    cx, cy = w // 2, h // 2
    r = int(min(cx, cy) * radius_frac)
    cv2.circle(mask, (cx, cy), r, 0, -1)
    return mask


def shape_strip_bottom(h: int, w: int, strip_frac: float = 0.15) -> np.ndarray:
    """Mask the bottom N% of the frame (white = exclude)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    strip_h = int(h * strip_frac)
    mask[h - strip_h:, :] = 255
    return mask


def shape_strip_top(h: int, w: int, strip_frac: float = 0.10) -> np.ndarray:
    """Mask the top N% of the frame."""
    mask = np.zeros((h, w), dtype=np.uint8)
    strip_h = int(h * strip_frac)
    mask[:strip_h, :] = 255
    return mask


def shape_rect(h: int, w: int, x1n: float, y1n: float, x2n: float, y2n: float) -> np.ndarray:
    """Mask a rectangle defined by normalised coordinates (0-1)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1 = int(x1n * w), int(y1n * h)
    x2, y2 = int(x2n * w), int(y2n * h)
    mask[y1:y2, x1:x2] = 255
    return mask


def build_shape_mask(shape: str, h: int, w: int, args: argparse.Namespace) -> np.ndarray:
    if shape == "circle":
        return shape_circle(h, w)
    if shape == "bottom":
        return shape_strip_bottom(h, w, args.strip_height)
    if shape == "top":
        return shape_strip_top(h, w, args.strip_height)
    if shape == "rect":
        coords = args.rect
        if not coords or len(coords) != 4:
            raise ValueError("--rect requires exactly 4 normalised values: x1 y1 x2 y2")
        return shape_rect(h, w, *coords)
    raise ValueError(f"Unknown shape: {shape}")


# ---------------------------------------------------------------------------
# Load all mask directories and get the union frame list
# ---------------------------------------------------------------------------

def load_mask_dirs(dirs: list[Path]) -> dict[str, list[Path]]:
    """
    Returns {frame_name: [mask_path, ...]} across all source dirs.
    Frame names are the union of filenames found in any source dir.
    """
    frame_map: dict[str, list[Path]] = {}
    for d in dirs:
        if not d.is_dir():
            log.warning("Mask dir not found, skipping: %s", d)
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in SUPPORTED_EXTS:
                frame_map.setdefault(p.name, []).append(p)
    return frame_map


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

def combine_and_save(
    frames_dir: Path | None,
    mask_dirs: list[Path],
    shape: str | None,
    strip_height: float,
    rect_coords: list[float] | None,
    output_dir: Path,
    dry_run: bool,
) -> int:
    import cv2
    from tqdm import tqdm

    frame_map = load_mask_dirs(mask_dirs) if mask_dirs else {}

    # Collect all frame names from mask dirs OR frames dir
    if frames_dir and frames_dir.is_dir():
        all_names = sorted(
            p.name for p in frames_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTS
        )
    else:
        all_names = sorted(frame_map.keys())

    if not all_names:
        log.error("No frames found — provide mask dirs with images or --frames <dir>")
        return 1

    log.info(
        "Combining %d mask source(s)%s → %d frames",
        len(mask_dirs),
        f" + shape '{shape}'" if shape else "",
        len(all_names),
    )

    if dry_run:
        log.info("[DRY-RUN] Would write %d masks to %s", len(all_names), output_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    shape_cache: np.ndarray | None = None

    for fname in tqdm(all_names, unit="frame"):
        combined: np.ndarray | None = None

        # OR all source masks
        for src_path in frame_map.get(fname, []):
            m = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            _, m_bin = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
            combined = m_bin if combined is None else np.maximum(combined, m_bin)

        # OR in the shape mask
        if shape is not None:
            if combined is not None:
                h, w = combined.shape
            elif frames_dir:
                ref = cv2.imread(str(frames_dir / fname))
                if ref is None:
                    continue
                h, w = ref.shape[:2]
            else:
                log.warning("Cannot determine frame size for %s — skipping shape", fname)
                if combined is None:
                    continue
                h, w = 0, 0

            if h > 0 and w > 0:
                if shape_cache is None or shape_cache.shape != (h, w):
                    ns = argparse.Namespace(strip_height=strip_height, rect=rect_coords)
                    shape_cache = build_shape_mask(shape, h, w, ns)
                combined = shape_cache if combined is None else np.maximum(combined, shape_cache)

        if combined is None:
            log.warning("No mask data for %s — writing blank", fname)
            # Need a size — skip if we can't determine
            continue

        cv2.imwrite(str(output_dir / fname), combined)

    log.info("Combined masks saved: %s", output_dir)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="combine_masks",
        description="Merge mask directories into one output using OR (union) logic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", nargs="*", metavar="MASK_DIR",
                   help="One or more mask directories to combine (OR logic)")
    p.add_argument("-o", "--output-dir", metavar="DIR",
                   help="Combined output directory (default: first input dir + _combined)")
    p.add_argument("--frames", metavar="DIR",
                   help="Frames dir — needed for size reference when generating shapes "
                        "without existing mask dirs")

    g_shape = p.add_argument_group("shape masks — generate and add to output")
    g_shape.add_argument("--shape", choices=["circle", "bottom", "top", "rect"],
                         help="Add a built-in shape mask layer")
    g_shape.add_argument("--strip-height", type=float, default=0.15, metavar="FRAC",
                         help="Fraction of frame height for top/bottom strips (default: 0.15)")
    g_shape.add_argument("--rect", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                         help="Normalised rectangle coordinates 0-1 (e.g. 0.1 0.8 0.9 1.0)")

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
        from tqdm import tqdm  # noqa: F401
    except ImportError as e:
        log.critical("Missing dependency: %s\npip install opencv-python tqdm", e)
        sys.exit(2)

    if not args.input and not args.shape:
        parser.error("Provide at least one MASK_DIR or --shape")

    mask_dirs = [Path(d) for d in args.input]
    frames_dir = Path(args.frames) if args.frames else None

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif mask_dirs:
        output_dir = mask_dirs[0].parent / "masks_combined"
    else:
        output_dir = (frames_dir.parent / "masks_combined") if frames_dir else Path("masks_combined")

    rc = combine_and_save(
        frames_dir=frames_dir,
        mask_dirs=mask_dirs,
        shape=args.shape,
        strip_height=args.strip_height,
        rect_coords=args.rect,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
