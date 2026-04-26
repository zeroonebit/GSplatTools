"""
sharp_frames.py  —  Sharp Frame Extractor
Select the sharpest frames from a video clip for use as input to
3D reconstruction pipelines (fed into Lichtfeld for Gaussian Splatting training).

Algorithm:
    Sharpness = variance of the Laplacian of a grayscale frame.
    High variance = strong edges = sharp. Low variance = blurry.

    Two selection modes:
      --threshold   keep every frame scoring above a fixed value
      --top         keep the top-N% sharpest frames (scene-adaptive, recommended)

    Both modes support --max-frames to cap total output for large videos.

Usage:
    python tools/sharp_frames.py clip.mp4
    python tools/sharp_frames.py clip.mp4 --top 20 --max-frames 300
    python tools/sharp_frames.py clip.mp4 --threshold 80 --every 3 -v
    python tools/sharp_frames.py "output/scene/*/frames/*.mp4" --top 15

Output:
    output/<stem>/frames/frame_000001.png
                         frame_000002.png
                         ...
    output/<stem>/sharp_frames_scores.csv   (frame index, score, kept)

Dependencies:
    pip install opencv-python numpy tqdm
"""

__version__ = "1.0.0"

import argparse
import csv
import logging
import sys
from pathlib import Path

log = logging.getLogger("sharp_frames")


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def laplacian_variance(frame_gray) -> float:
    """Variance of the Laplacian — the standard blur detection metric."""
    import cv2
    return float(cv2.Laplacian(frame_gray, cv2.CV_64F).var())


def score_video(
    cap,
    every: int,
    verbose: bool,
) -> list[tuple[int, float]]:
    """
    Decode every `every`-th frame from an open VideoCapture.
    Returns list of (frame_index, sharpness_score) for all sampled frames.
    """
    import cv2
    from tqdm import tqdm

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log.debug("Video: %d frames @ %.2f fps, sampling every %d", total, fps, every)

    scores: list[tuple[int, float]] = []
    frame_idx = 0

    bar = tqdm(total=total // every, unit="frame", disable=not sys.stderr.isatty())
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = laplacian_variance(gray)
            scores.append((frame_idx, score))
            if verbose:
                log.debug("frame %06d  score=%.1f", frame_idx, score)
            bar.update(1)
        frame_idx += 1
    bar.close()

    return scores


def select_frames(
    scores: list[tuple[int, float]],
    threshold: float | None,
    top_percent: float | None,
    max_frames: int | None,
) -> list[tuple[int, float]]:
    """
    Apply threshold or top-% filter, then cap at max_frames.
    Returns selected (frame_index, score) pairs sorted by frame_index.
    """
    if not scores:
        return []

    if top_percent is not None:
        # Sort by score descending, keep top N%
        sorted_by_score = sorted(scores, key=lambda x: x[1], reverse=True)
        keep_n = max(1, int(len(sorted_by_score) * top_percent / 100.0))
        selected = sorted_by_score[:keep_n]
    elif threshold is not None:
        selected = [(i, s) for i, s in scores if s >= threshold]
    else:
        selected = list(scores)

    if max_frames and len(selected) > max_frames:
        # Among candidates, keep the top-N by score
        selected = sorted(selected, key=lambda x: x[1], reverse=True)[:max_frames]

    # Return in temporal order
    return sorted(selected, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def extract_and_save(
    input_path: str,
    selected: list[tuple[int, float]],
    all_scores: list[tuple[int, float]],
    output_dir: Path,
    dry_run: bool,
) -> list[str]:
    """
    Re-seek the video and write selected frames as PNG.
    Returns list of written file paths.
    """
    import cv2

    selected_set = {idx for idx, _ in selected}
    written: list[str] = []

    if dry_run:
        for out_n, (idx, score) in enumerate(selected, 1):
            path = str(output_dir / f"frame_{out_n:06d}.png")
            log.info("DRY-RUN  %s  (frame %d, score=%.1f)", path, idx, score)
            written.append(path)
        return written

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    frame_idx = 0
    out_n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in selected_set:
            out_n += 1
            path = output_dir / f"frame_{out_n:06d}.png"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            written.append(str(path))
        frame_idx += 1

    cap.release()
    return written


def write_scores_csv(
    csv_path: Path,
    all_scores: list[tuple[int, float]],
    selected_set: set[int],
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "sharpness_score", "kept"])
        for idx, score in all_scores:
            writer.writerow([idx, f"{score:.2f}", "1" if idx in selected_set else "0"])


# ---------------------------------------------------------------------------
# Per-file orchestration
# ---------------------------------------------------------------------------

def process_file(
    input_path: str,
    args: argparse.Namespace,
    log: logging.Logger,
) -> int:
    """Process one video file. Returns 0 on success, 1 on failure."""
    import cv2

    # If -o is given, treat it as the direct parent dir (frames go inside it).
    # If omitted, put frames adjacent to the input clip — the natural location
    # when clips are already in their view subdir (output/<stem>/<view>/).
    if args.output_dir:
        frames_base = Path(args.output_dir)
    else:
        frames_base = Path(input_path).parent
    output_dir = frames_base / "frames"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", input_path)
        return 1

    log.info("Scoring: %s", input_path)
    scores = score_video(cap, args.every, args.verbose)
    cap.release()

    if not scores:
        log.error("No frames decoded from: %s", input_path)
        return 1

    score_values = [s for _, s in scores]
    log.info(
        "Sampled %d frames — min=%.1f  mean=%.1f  max=%.1f",
        len(scores),
        min(score_values),
        sum(score_values) / len(score_values),
        max(score_values),
    )

    selected = select_frames(
        scores,
        threshold=args.threshold,
        top_percent=args.top,
        max_frames=args.max_frames,
    )

    if not selected:
        log.warning(
            "No frames passed the filter for '%s'. "
            "Try lowering --threshold or raising --top.",
            input_path,
        )
        return 1

    log.info("Selected %d / %d frames (%.1f%%)",
             len(selected), len(scores), 100 * len(selected) / len(scores))

    written = extract_and_save(input_path, selected, scores, output_dir, args.dry_run)

    if not args.dry_run:
        selected_set = {idx for idx, _ in selected}
        csv_path = frames_base / "sharp_frames_scores.csv"
        write_scores_csv(csv_path, scores, selected_set)
        log.info("Scores saved: %s", csv_path)
        log.info("Frames saved: %s", output_dir)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sharp_frames",
        description="Extract sharpest frames from video clips for 3DGS dataset preparation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "input", nargs="+", metavar="INPUT",
        help="Input video file(s) or glob patterns",
    )
    p.add_argument(
        "-o", "--output-dir", default=None, metavar="DIR",
        help="Output parent directory. Frames go into <DIR>/frames/. "
             "Defaults to the clip's own directory.",
    )

    sel = p.add_mutually_exclusive_group()
    sel.add_argument(
        "--top", type=float, metavar="PCT",
        help="Keep the top PCT%% sharpest sampled frames (e.g. --top 20). "
             "Scene-adaptive — recommended over --threshold.",
    )
    sel.add_argument(
        "--threshold", type=float, metavar="SCORE",
        help="Keep frames with Laplacian variance >= SCORE (e.g. --threshold 80). "
             "Use --dry-run -v first to find a good value for your footage.",
    )

    p.add_argument(
        "--max-frames", type=int, metavar="N",
        help="Cap output at N frames (keeps the sharpest N among candidates)",
    )
    p.add_argument(
        "--every", type=int, default=1, metavar="N",
        help="Sample every N-th frame to reduce processing time (default: 1)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print which frames would be saved without writing anything",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show per-frame sharpness scores",
    )
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

    # Default selection: top 20% if neither flag given
    if args.top is None and args.threshold is None:
        args.top = 20.0
        log.info("No selection mode specified — defaulting to --top 20")

    try:
        import cv2
        import numpy  # noqa: F401
        from tqdm import tqdm  # noqa: F401
    except ImportError as e:
        log.critical(
            "Missing dependency: %s\n"
            "Install with: pip install opencv-python numpy tqdm", e
        )
        sys.exit(2)

    import glob as _glob
    input_paths: list[str] = []
    for pattern in args.input:
        matches = _glob.glob(pattern, recursive=True)
        if matches:
            input_paths.extend(matches)
        elif Path(pattern).is_file():
            input_paths.append(pattern)
        else:
            log.warning("No files matched: %s", pattern)

    if not input_paths:
        log.critical("No valid input files found.")
        sys.exit(2)

    failed = 0
    for path in input_paths:
        rc = process_file(path, args, log)
        if rc != 0:
            failed += 1

    if failed == len(input_paths):
        sys.exit(2)
    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
