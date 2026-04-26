"""
sharp_frames.py  —  Sharp Frame Extractor
Select the sharpest, least-blurry frames from a video clip for use in
3D reconstruction pipelines (COLMAP, Gaussian Splatting).

Algorithm:
    Blur score = variance of the Laplacian of a grayscale frame.
    High variance = sharp edges present = sharp frame.
    Low variance = flat image = blurry frame.

Usage:
    python tools/sharp_frames.py input.mp4 [options]
    python tools/sharp_frames.py output/scene/front/scene_front.mp4 --threshold 80 --every 5

Planned options:
    --threshold FLOAT   Minimum sharpness score to keep a frame (default: 100)
    --every N           Sample every N frames instead of all frames (default: 1)
    --max-frames N      Keep at most N sharpest frames total
    --output-dir DIR    Where to write PNG frames (default: ./output/<stem>/frames/)
    --no-subdir         Write all frames flat, no per-view subfolder
    -v, --verbose       Show per-frame scores
    --dry-run           Report which frames would be saved, without writing

Dependencies:
    pip install opencv-python numpy tqdm
"""

# TODO: implement — scaffold below shows the intended structure

import argparse
import sys
from pathlib import Path

# Deferred heavy import — only loaded when actually running, not at import time
# import cv2
# import numpy as np
# from tqdm import tqdm


__version__ = "0.1.0-stub"


def blur_score(frame_gray) -> float:
    """
    Return the variance of the Laplacian for a grayscale frame.
    Higher = sharper. Typical thresholds: <50 blurry, >100 sharp.

    Requires cv2 and numpy.
    """
    # import cv2, numpy as np
    # return float(cv2.Laplacian(frame_gray, cv2.CV_64F).var())
    raise NotImplementedError("sharp_frames is not yet implemented")


def extract_sharp_frames(
    input_path: str,
    output_dir: str,
    threshold: float = 100.0,
    every: int = 1,
    max_frames: int | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> list[str]:
    """
    Extract frames above `threshold` sharpness from `input_path`.
    Returns list of written file paths (empty on dry_run).

    Algorithm:
        1. Open video with cv2.VideoCapture
        2. For each sampled frame, compute blur_score on grayscale
        3. If score >= threshold, save as PNG with zero-padded name
        4. If max_frames set, keep only the top-N by score
    """
    raise NotImplementedError("sharp_frames is not yet implemented")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sharp_frames",
        description="Extract sharpest frames from a video clip.",
    )
    p.add_argument("input", nargs="+", metavar="INPUT",
                   help="Input video file(s)")
    p.add_argument("-o", "--output-dir", default="output", metavar="DIR")
    p.add_argument("--threshold", type=float, default=100.0,
                   help="Minimum Laplacian variance to keep a frame (default: 100)")
    p.add_argument("--every", type=int, default=1, metavar="N",
                   help="Sample every N frames (default: 1 = all frames)")
    p.add_argument("--max-frames", type=int, metavar="N",
                   help="Keep at most N sharpest frames")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main() -> None:
    args = build_parser().parse_args()
    print(
        "sharp_frames is not yet implemented.\n"
        "Install dependencies first: pip install opencv-python numpy tqdm\n"
        f"Input files: {args.input}"
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
