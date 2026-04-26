"""
eq2persp.py  —  Equirectangular → Perspective Views Converter
Convert 360° equirectangular video (e.g. DJI Osmo 360) into multiple
undistorted rectilinear (perspective) view clips using FFmpeg's v360 filter.

Usage:
    python eq2persp.py input.mp4 [input2.mp4 ...] [options]
    python eq2persp.py "*.mp4" --views 6 --fov 90 -o ./output

Requirements:
    FFmpeg >= 4.3 with v360 filter support.
    Download: https://www.gyan.dev/ffmpeg/builds/
"""

__version__ = "1.0.0"

import argparse
import concurrent.futures
import glob as _glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

DEFAULT_FOV = 90.0
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_CRF = 18
DEFAULT_PRESET = "slow"
DEFAULT_PIX_FMT = "yuv420p"
PIX_FMT_10BIT = "yuv420p10le"
INTERP = "cubic"
MIN_FFMPEG_VERSION = (4, 3, 0)

_COMMON_FFMPEG_PATHS = [
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    r"C:\tools\ffmpeg\bin\ffmpeg.exe",
    # Scoop
    os.path.expanduser(r"~\scoop\apps\ffmpeg\current\bin\ffmpeg.exe"),
    # Chocolatey
    r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
]
_COMMON_FFPROBE_PATHS = [p.replace("ffmpeg.exe", "ffprobe.exe") for p in _COMMON_FFMPEG_PATHS]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CameraView:
    name: str
    yaw: float
    pitch: float
    roll: float = 0.0
    h_fov: float | None = None  # None → inherit global --fov
    v_fov: float | None = None  # None → inherit global --fov


DEFAULT_VIEWS_4: list[CameraView] = [
    CameraView("front",  yaw=0,   pitch=0),
    CameraView("right",  yaw=90,  pitch=0),
    CameraView("back",   yaw=180, pitch=0),
    CameraView("left",   yaw=-90, pitch=0),
]

DEFAULT_VIEWS_6: list[CameraView] = DEFAULT_VIEWS_4 + [
    CameraView("top",    yaw=0, pitch=90),
    CameraView("bottom", yaw=0, pitch=-90),
]


@dataclass
class Job:
    view: CameraView
    input_path: str
    output_path: str
    cmd: list[str]
    skipped: bool = False


# ---------------------------------------------------------------------------
# FFmpeg discovery & version check
# ---------------------------------------------------------------------------

def find_ffmpeg(hint: str | None = None) -> tuple[str, str]:
    """Return (ffmpeg_path, ffprobe_path). ffprobe_path may be '' if not found."""
    candidates_ff = []
    candidates_fp = []

    if hint:
        candidates_ff.append(hint)
        candidates_fp.append(str(Path(hint).parent / "ffprobe.exe"))
        candidates_fp.append(str(Path(hint).parent / "ffprobe"))

    env_path = os.environ.get("FFMPEG_PATH")
    if env_path:
        candidates_ff.append(env_path)
        candidates_fp.append(str(Path(env_path).parent / "ffprobe.exe"))
        candidates_fp.append(str(Path(env_path).parent / "ffprobe"))

    which_ff = shutil.which("ffmpeg")
    if which_ff:
        candidates_ff.append(which_ff)
        which_fp = shutil.which("ffprobe")
        if which_fp:
            candidates_fp.append(which_fp)

    candidates_ff.extend(_COMMON_FFMPEG_PATHS)
    candidates_fp.extend(_COMMON_FFPROBE_PATHS)

    ffmpeg = next((p for p in candidates_ff if Path(p).is_file()), None)
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg not found. Install a build with v360 filter support:\n"
            "  https://www.gyan.dev/ffmpeg/builds/\n"
            "Then re-run with: --ffmpeg-path \"C:/path/to/ffmpeg.exe\"\n"
            "Or set the FFMPEG_PATH environment variable."
        )

    ffprobe = next((p for p in candidates_fp if Path(p).is_file()), "") or ""
    return ffmpeg, ffprobe


def check_ffmpeg_version(ffmpeg: str) -> tuple[int, int, int]:
    """Verify FFmpeg >= 4.3. Raises RuntimeError with download hint if not."""
    try:
        result = subprocess.run(
            [ffmpeg, "-version"],
            capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"Failed to run ffmpeg at '{ffmpeg}': {e}") from e

    output = result.stdout + result.stderr
    # Git/nightly builds report "ffmpeg version 2026-04-22-git-..." — treat as current
    if re.search(r"ffmpeg version \d{4}-\d{2}-\d{2}-git", output):
        return (99, 0, 0)
    match = re.search(r"ffmpeg version (\d+)\.(\d+)(?:\.(\d+))?", output)
    if not match:
        raise RuntimeError(f"Could not parse FFmpeg version from:\n{output[:300]}")

    major, minor, patch = int(match[1]), int(match[2]), int(match[3] or 0)
    ver = (major, minor, patch)
    min_ver = MIN_FFMPEG_VERSION

    if ver < min_ver:
        raise RuntimeError(
            f"FFmpeg {major}.{minor}.{patch} found at '{ffmpeg}'.\n"
            f"The v360 filter requires FFmpeg >= {min_ver[0]}.{min_ver[1]}.\n"
            "Download a current build at: https://www.gyan.dev/ffmpeg/builds/\n"
            "Then re-run with: --ffmpeg-path \"C:/path/to/new/ffmpeg.exe\""
        )
    return ver


# ---------------------------------------------------------------------------
# Stream probing
# ---------------------------------------------------------------------------

def probe_pixel_format(ffprobe: str, input_path: str) -> str:
    """Detect pix_fmt of first video stream. Falls back to DEFAULT_PIX_FMT."""
    if not ffprobe:
        return DEFAULT_PIX_FMT

    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-select_streams", "v:0",
                input_path,
            ],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(result.stdout)
        pix_fmt = data["streams"][0].get("pix_fmt", DEFAULT_PIX_FMT)
        return pix_fmt if pix_fmt else DEFAULT_PIX_FMT
    except Exception:
        return DEFAULT_PIX_FMT


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def load_config(path: str) -> tuple[list[CameraView], dict]:
    """
    Parse a JSON camera-rig config. Returns (views, raw_config).
    raw_config contains top-level keys like fov, width, height, crf, preset.
    Raises ValueError with a descriptive message on schema violations.
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Cannot read config '{path}': {e}") from e

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a JSON object, got {type(raw).__name__}")

    raw_views = raw.get("views")
    if not raw_views or not isinstance(raw_views, list):
        raise ValueError("Config must contain a non-empty 'views' list")

    views: list[CameraView] = []
    seen_names: set[str] = set()

    for i, v in enumerate(raw_views):
        if not isinstance(v, dict):
            raise ValueError(f"views[{i}] must be an object")

        name = v.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"views[{i}] missing 'name' string")
        if not _VALID_NAME_RE.match(name):
            raise ValueError(
                f"views[{i}] name '{name}' contains invalid characters. "
                "Use only letters, digits, underscore, hyphen, dot."
            )
        if name in seen_names:
            raise ValueError(f"Duplicate view name '{name}' in config")
        seen_names.add(name)

        for key in ("yaw", "pitch"):
            if key not in v or not isinstance(v[key], (int, float)):
                raise ValueError(f"views[{i}] ('{name}') missing numeric '{key}'")

        views.append(CameraView(
            name=name,
            yaw=float(v["yaw"]),
            pitch=float(v["pitch"]),
            roll=float(v.get("roll", 0.0)),
            h_fov=float(v["h_fov"]) if "h_fov" in v else None,
            v_fov=float(v["v_fov"]) if "v_fov" in v else None,
        ))

    return views, raw


# ---------------------------------------------------------------------------
# FFmpeg command builder
# ---------------------------------------------------------------------------

def build_ffmpeg_cmd(
    ffmpeg: str,
    input_path: str,
    output_path: str,
    view: CameraView,
    fov: float,
    width: int,
    height: int,
    crf: int,
    preset: str,
    pix_fmt: str,
    overwrite: bool,
    gpu: bool = False,
    ss: str | None = None,
    to: str | None = None,
    duration: str | None = None,
) -> list[str]:
    """Assemble the complete FFmpeg argument list for one view. No I/O."""
    h_fov = view.h_fov if view.h_fov is not None else fov
    v_fov = view.v_fov if view.v_fov is not None else fov

    v360_filter = (
        f"v360=e:rectilinear"
        f":h_fov={h_fov}:v_fov={v_fov}"
        f":yaw={view.yaw}:pitch={view.pitch}:roll={view.roll}"
        f":w={width}:h={height}"
        f":interp={INTERP}"
    )

    cmd = [ffmpeg, "-hide_banner"]

    if overwrite:
        cmd += ["-y"]

    if gpu:
        # Upload decoded frames to GPU for NVENC — v360 filter still runs on CPU
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

    if ss:
        cmd += ["-ss", ss]

    cmd += ["-i", input_path]

    if duration:
        cmd += ["-t", duration]
    elif to:
        cmd += ["-to", to]

    if gpu:
        # NVENC path: download back from GPU after v360, then encode
        cmd += [
            "-vf", f"{v360_filter},hwupload_cuda",
            "-c:v", "hevc_nvenc",
            "-cq", str(crf),       # same 0-51 range as CRF
            "-preset", "p4",       # NVENC preset: p1(fast)..p7(slow), p4=balanced
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            output_path,
        ]
    else:
        cmd += [
            "-vf", v360_filter,
            "-c:v", "libx265",
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", pix_fmt,
            "-c:a", "copy",
            output_path,
        ]

    return cmd


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------

def run_job(job: Job, dry_run: bool, log: logging.Logger) -> tuple[Job, int]:
    """Execute a single FFmpeg job. Never raises — returns (job, returncode)."""
    if job.skipped:
        log.info("SKIP  %s  (already exists)", job.output_path)
        return job, 0

    cmd_str = " ".join(
        (f'"{a}"' if " " in a else a) for a in job.cmd
    )

    if dry_run:
        print(cmd_str)
        return job, 0

    log.debug("RUN   %s", cmd_str)
    Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.Popen(
            job.cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stderr_lines: list[str] = []
        for line in proc.stderr:
            line = line.rstrip()
            if not line:
                continue
            stderr_lines.append(line)
            # Stream FFmpeg progress lines so the caller can see activity
            if any(k in line for k in ("frame=", "fps=", "time=", "speed=", "Error", "error", "Invalid")):
                log.info("[%s] %s", job.view.name, line)
        proc.wait()
        result_rc = proc.returncode
    except OSError as e:
        log.error("Failed to launch FFmpeg for view '%s': %s", job.view.name, e)
        return job, 1

    if result_rc != 0:
        tail = "\n".join(stderr_lines[-20:])
        log.error(
            "FFmpeg failed (rc=%d) for view '%s' of '%s':\n%s",
            result_rc, job.view.name, job.input_path, tail
        )
    else:
        log.info("OK    %s", job.output_path)

    return job, result_rc


def run_parallel(
    jobs: list[Job],
    workers: int,
    dry_run: bool,
    log: logging.Logger,
) -> list[tuple[Job, int]]:
    """Run jobs with up to `workers` threads. workers=1 → sequential."""
    if workers <= 1:
        return [run_job(j, dry_run, log) for j in jobs]

    results: list[tuple[Job, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, j, dry_run, log): j for j in jobs}
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    return results


# ---------------------------------------------------------------------------
# Per-file orchestration
# ---------------------------------------------------------------------------

def process_file(
    input_path: str,
    views: list[CameraView],
    args: argparse.Namespace,
    ffmpeg: str,
    ffprobe: str,
    log: logging.Logger,
) -> list[tuple[Job, int]]:
    """Full pipeline for one input file: probe → build jobs → run → return results."""
    stem = Path(input_path).stem
    output_base = Path(args.output_dir) / stem

    pix_fmt = probe_pixel_format(ffprobe, input_path)
    log.debug("Detected pix_fmt for '%s': %s", input_path, pix_fmt)

    jobs: list[Job] = []
    for view in views:
        out_dir = output_base / view.name
        out_path = str(out_dir / f"{stem}_{view.name}.mp4")

        skip = Path(out_path).exists() and not args.overwrite

        cmd = [] if skip else build_ffmpeg_cmd(
            ffmpeg=ffmpeg,
            input_path=input_path,
            output_path=out_path,
            view=view,
            fov=args.fov,
            width=args.width,
            height=args.height,
            crf=args.crf,
            preset=args.preset,
            pix_fmt=pix_fmt,
            overwrite=args.overwrite,
            gpu=getattr(args, "gpu", False),
            ss=getattr(args, "ss", None),
            to=getattr(args, "to", None),
            duration=getattr(args, "duration", None),
        )

        jobs.append(Job(
            view=view,
            input_path=input_path,
            output_path=out_path,
            cmd=cmd,
            skipped=skip,
        ))

    return run_parallel(jobs, args.parallel, args.dry_run, log)


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def process_batch(
    input_paths: list[str],
    views: list[CameraView],
    args: argparse.Namespace,
    ffmpeg: str,
    ffprobe: str,
    log: logging.Logger,
) -> int:
    """Process all files sequentially. Returns exit code 0/1/2."""
    total_jobs = 0
    failed_jobs = 0

    for input_path in input_paths:
        log.info("Processing: %s", input_path)
        results = process_file(input_path, views, args, ffmpeg, ffprobe, log)
        for _, rc in results:
            total_jobs += 1
            if rc != 0:
                failed_jobs += 1

    if total_jobs == 0:
        log.error("No jobs were run.")
        return 2

    ok = total_jobs - failed_jobs
    log.info("Done. %d/%d views succeeded.", ok, total_jobs)

    if failed_jobs == 0:
        return 0
    if ok == 0:
        return 2
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eq2persp",
        description="Convert equirectangular (360°) video into perspective view clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "input", nargs="+", metavar="INPUT",
        help="Input MP4 file(s) or glob patterns (e.g. '*.mp4')"
    )
    p.add_argument(
        "-o", "--output-dir", default="output", metavar="DIR",
        help="Base output directory (default: ./output)"
    )
    p.add_argument(
        "--views", type=int, choices=[4, 6], default=4,
        help="Number of standard views: 4 (FRLB) or 6 (+ top/bottom) (default: 4)"
    )
    p.add_argument(
        "--fov", type=float, default=DEFAULT_FOV,
        help=f"Horizontal field of view in degrees (default: {DEFAULT_FOV})"
    )
    p.add_argument(
        "--width", type=int, default=DEFAULT_WIDTH,
        help=f"Output width in pixels (default: {DEFAULT_WIDTH})"
    )
    p.add_argument(
        "--height", type=int, default=DEFAULT_HEIGHT,
        help=f"Output height in pixels (default: {DEFAULT_HEIGHT})"
    )
    p.add_argument(
        "--crf", type=int, default=DEFAULT_CRF,
        help=f"libx265 CRF quality (0=lossless, 51=worst; default: {DEFAULT_CRF})"
    )
    p.add_argument(
        "--preset", default=DEFAULT_PRESET,
        choices=["ultrafast","superfast","veryfast","faster","fast",
                 "medium","slow","slower","veryslow"],
        help=f"libx265 encoding preset (default: {DEFAULT_PRESET})"
    )
    p.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Process N views concurrently within each file (default: 1)"
    )
    p.add_argument(
        "--config", metavar="PATH",
        help="JSON camera rig config file (overrides --views)"
    )
    p.add_argument(
        "--ffmpeg-path", metavar="PATH",
        help="Path to ffmpeg binary (also: FFMPEG_PATH env var)"
    )
    p.add_argument(
        "--gpu", action="store_true",
        help="Use NVIDIA NVENC (hevc_nvenc) for encoding — 5-10x faster than libx265"
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print FFmpeg commands without executing"
    )
    p.add_argument(
        "--ss", metavar="TIME",
        help="Start time for all clips (e.g. 00:01:30 or 90)"
    )
    p.add_argument(
        "--to", metavar="TIME",
        help="End time for all clips"
    )
    p.add_argument(
        "--duration", "-t", metavar="SECONDS",
        help="Clip duration in seconds (alternative to --to)"
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging"
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Logging setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s"
    log_datefmt = "%H:%M:%S"
    logging.basicConfig(level=log_level, format=log_fmt, datefmt=log_datefmt)
    log = logging.getLogger("eq2persp")

    # File handler (skip in dry-run)
    if not args.dry_run:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(args.output_dir) / "eq2persp.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(log_fmt, datefmt=log_datefmt))
        logging.getLogger().addHandler(fh)

    # FFmpeg discovery & version check
    try:
        ffmpeg, ffprobe = find_ffmpeg(args.ffmpeg_path)
        ver = check_ffmpeg_version(ffmpeg)
        log.info("FFmpeg %d.%d.%d at '%s'", *ver, ffmpeg)
        if ffprobe:
            log.debug("ffprobe at '%s'", ffprobe)
        else:
            log.debug("ffprobe not found; pixel format detection disabled")
    except RuntimeError as e:
        log.critical("%s", e)
        sys.exit(2)

    # Expand glob patterns (Windows shell doesn't do this)
    input_paths: list[str] = []
    for pattern in args.input:
        matches = _glob.glob(pattern, recursive=True)
        if matches:
            input_paths.extend(matches)
        elif Path(pattern).is_file():
            input_paths.append(pattern)
        else:
            log.warning("No files matched: %s", pattern)

    input_paths = [str(Path(p).resolve()) for p in input_paths]

    if not input_paths:
        log.critical("No valid input files found. Check your file paths or glob patterns.")
        sys.exit(2)

    log.info("Input files: %d", len(input_paths))

    # Load camera rig
    config_fov = None
    if args.config:
        try:
            views, raw_config = load_config(args.config)
        except ValueError as e:
            log.critical("%s", e)
            sys.exit(2)
        # Apply config-level defaults for any args still at their parser defaults
        if args.fov == DEFAULT_FOV and "fov" in raw_config:
            args.fov = float(raw_config["fov"])
        if args.width == DEFAULT_WIDTH and "width" in raw_config:
            args.width = int(raw_config["width"])
        if args.height == DEFAULT_HEIGHT and "height" in raw_config:
            args.height = int(raw_config["height"])
        if args.crf == DEFAULT_CRF and "crf" in raw_config:
            args.crf = int(raw_config["crf"])
        if args.preset == DEFAULT_PRESET and "preset" in raw_config:
            args.preset = raw_config["preset"]
        log.info("Camera rig: %d views from config '%s'", len(views), args.config)
    else:
        views = DEFAULT_VIEWS_6 if args.views == 6 else DEFAULT_VIEWS_4
        log.info("Camera rig: %d standard views", len(views))

    rc = process_batch(input_paths, views, args, ffmpeg, ffprobe, log)
    sys.exit(rc)


if __name__ == "__main__":
    main()
