"""
pipeline.py  —  End-to-end Dataset Preparation Pipeline
Chains eq2persp → sharp_frames → masks on one or more 360° video files.

Steps:
    1. eq2persp   — slice 360° video into N perspective view clips
    2. sharp_frames — extract sharpest frames from each clip
    3. masks        — generate object masks for each frame set (optional)

Usage:
    python pipeline.py footage.mp4
    python pipeline.py footage.mp4 --skip-masks
    python pipeline.py footage.mp4 --views 6 --top 15 --max-frames 300
    python pipeline.py "*.mp4" --ffmpeg-path "C:/ffmpeg/bin/ffmpeg.exe"
    python pipeline.py footage.mp4 --dry-run

Output structure:
    output/
      <stem>/
        front/
          <stem>_front.mp4        ← eq2persp output
          frames/                 ← sharp_frames output
            frame_000001.png
            ...
          masks/                  ← masks output
            frame_000001.png
            ...
          sharp_frames_scores.csv
        right/ back/ left/ ...
"""

__version__ = "1.0.0"

import argparse
import glob as _glob
import logging
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).parent / "tools"
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def run_step(name: str, cmd: list[str], dry_run: bool) -> int:
    cmd_str = " ".join(f'"{a}"' if " " in a else a for a in cmd)
    if dry_run:
        log.info("[DRY-RUN] %s: %s", name, cmd_str)
        return 0

    log.info("▶  %s", name)
    log.debug("CMD: %s", cmd_str)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log.info("    %s", line)
    proc.wait()

    if proc.returncode != 0:
        log.error("✗  %s failed (exit %d)", name, proc.returncode)
    else:
        log.info("✓  %s done", name)

    return proc.returncode


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

VIEW_NAMES_4 = ["front", "right", "back", "left"]
VIEW_NAMES_6 = VIEW_NAMES_4 + ["top", "bottom"]


def process_file(input_path: str, args: argparse.Namespace) -> int:
    stem = Path(input_path).stem
    output_base = Path(args.output_dir) / stem
    n_failures = 0

    # ------------------------------------------------------------------
    # Step 1: eq2persp
    # ------------------------------------------------------------------
    cmd_eq = [
        sys.executable, str(TOOLS / "eq2persp.py"),
        input_path,
        "-o", args.output_dir,
        "--views", str(args.views),
        "--fov",    str(args.fov),
        "--width",  str(args.width),
        "--height", str(args.height),
        "--crf",    str(args.crf),
        "--preset", args.preset,
    ]
    if args.ffmpeg_path:
        cmd_eq += ["--ffmpeg-path", args.ffmpeg_path]
    if args.overwrite:
        cmd_eq += ["--overwrite"]

    rc = run_step(f"eq2persp [{stem}]", cmd_eq, args.dry_run)
    if rc != 0:
        log.error("Aborting pipeline for '%s' — eq2persp failed.", input_path)
        return rc

    # ------------------------------------------------------------------
    # Step 2: sharp_frames — one run per view clip
    # ------------------------------------------------------------------
    view_names = VIEW_NAMES_6 if args.views == 6 else VIEW_NAMES_4

    for view in view_names:
        clip = output_base / view / f"{stem}_{view}.mp4"
        if not clip.exists() and not args.dry_run:
            log.warning("Clip not found, skipping sharp_frames: %s", clip)
            continue

        cmd_sf = [
            sys.executable, str(TOOLS / "sharp_frames.py"),
            str(clip),
            "-o", str(output_base / view),
        ]
        if args.top is not None:
            cmd_sf += ["--top", str(args.top)]
        elif args.threshold is not None:
            cmd_sf += ["--threshold", str(args.threshold)]
        else:
            cmd_sf += ["--top", "20"]

        if args.max_frames:
            cmd_sf += ["--max-frames", str(args.max_frames)]
        if args.every > 1:
            cmd_sf += ["--every", str(args.every)]

        rc = run_step(f"sharp_frames [{stem}/{view}]", cmd_sf, args.dry_run)
        if rc not in (0, 1):
            n_failures += 1

    # ------------------------------------------------------------------
    # Step 3: masks — one run per view's frames dir
    # ------------------------------------------------------------------
    if not args.skip_masks:
        for view in view_names:
            frames_dir = output_base / view / "frames"
            if not frames_dir.exists() and not args.dry_run:
                log.warning("Frames dir not found, skipping masks: %s", frames_dir)
                continue

            cmd_mask = [
                sys.executable, str(TOOLS / "masks.py"),
                str(frames_dir),
                "--classes", *args.classes,
                "--method",     args.mask_method,
                "--confidence", str(args.confidence),
                "--dilate",     str(args.dilate),
            ]
            if args.yolo_model:
                cmd_mask += ["--model", args.yolo_model]
            if args.sam_model:
                cmd_mask += ["--sam-model", args.sam_model]
            if args.device != "auto":
                cmd_mask += ["--device", args.device]

            rc = run_step(f"masks [{stem}/{view}]", cmd_mask, args.dry_run)
            if rc not in (0, 1):
                n_failures += 1

    return 1 if n_failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="End-to-end 360° video → dataset pipeline: eq2persp + sharp_frames + masks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", nargs="+", metavar="INPUT",
                   help="360° video file(s) or glob patterns")

    g_out = p.add_argument_group("output")
    g_out.add_argument("-o", "--output-dir", default="output", metavar="DIR")
    g_out.add_argument("--overwrite", action="store_true")

    g_eq = p.add_argument_group("eq2persp")
    g_eq.add_argument("--views",  type=int, choices=[4, 6], default=4)
    g_eq.add_argument("--fov",    type=float, default=90.0)
    g_eq.add_argument("--width",  type=int, default=1920)
    g_eq.add_argument("--height", type=int, default=1080)
    g_eq.add_argument("--crf",    type=int, default=18)
    g_eq.add_argument("--preset", default="slow",
                      choices=["ultrafast","superfast","veryfast","faster","fast",
                               "medium","slow","slower","veryslow"])
    g_eq.add_argument("--ffmpeg-path", metavar="PATH")

    g_sf = p.add_argument_group("sharp_frames")
    sf_sel = g_sf.add_mutually_exclusive_group()
    sf_sel.add_argument("--top",       type=float, metavar="PCT",
                        help="Keep top PCT%% sharpest frames (default: 20)")
    sf_sel.add_argument("--threshold", type=float, metavar="SCORE")
    g_sf.add_argument("--max-frames",  type=int, metavar="N")
    g_sf.add_argument("--every",       type=int, default=1, metavar="N")

    g_mask = p.add_argument_group("masks")
    g_mask.add_argument("--skip-masks", action="store_true",
                        help="Skip the masks step entirely")
    g_mask.add_argument("--classes", nargs="+", default=["person"], metavar="CLASS")
    g_mask.add_argument("--mask-method", choices=["yolo", "sam"], default="yolo",
                        dest="mask_method")
    g_mask.add_argument("--confidence", type=float, default=0.4)
    g_mask.add_argument("--dilate",     type=int, default=15, metavar="PX")
    g_mask.add_argument("--yolo-model", metavar="PATH")
    g_mask.add_argument("--sam-model",  metavar="PATH")
    g_mask.add_argument("--device",     default="auto")

    p.add_argument("--dry-run", action="store_true",
                   help="Print all commands without executing")
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

    # Expand globs
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

    log.info("Pipeline: %d file(s) | views=%d | skip_masks=%s",
             len(input_paths), args.views, args.skip_masks)

    total_failed = 0
    for path in input_paths:
        log.info("=" * 60)
        log.info("FILE: %s", path)
        log.info("=" * 60)
        rc = process_file(path, args)
        if rc != 0:
            total_failed += 1

    log.info("=" * 60)
    ok = len(input_paths) - total_failed
    log.info("Pipeline complete. %d/%d files succeeded.", ok, len(input_paths))

    if total_failed == len(input_paths):
        sys.exit(2)
    if total_failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
