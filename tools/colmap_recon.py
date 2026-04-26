"""
colmap_recon.py  —  COLMAP Sparse Reconstruction
Gathers perspective frames from all views produced by eq2persp + sharp_frames,
then runs COLMAP feature extraction, matching, and mapping to produce a sparse
camera model for 3D Gaussian Splatting training in Lichtfeld.

All views are combined into one reconstruction so COLMAP can resolve the full
camera geometry across the 360° orbit.

Usage:
    python tools/colmap_recon.py output/scene/
    python tools/colmap_recon.py output/scene/ --matcher exhaustive
    python tools/colmap_recon.py output/scene/ --colmap-path "C:/COLMAP/colmap.exe"
    python tools/colmap_recon.py output/scene/ --no-gpu --dry-run

Output:
    output/scene/colmap/
        images/          <- all frames from all views, prefixed by view name
        masks/           <- corresponding masks (if available)
        database.db      <- COLMAP feature database
        sparse/
            0/
                cameras.bin
                images.bin
                points3D.bin

COLMAP install:
    https://github.com/colmap/colmap/releases
    Set --colmap-path or COLMAP_PATH env var if not on system PATH.
"""

__version__ = "1.0.0"

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("colmap_recon")

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}

_COMMON_COLMAP_PATHS = [
    r"C:\Program Files\COLMAP\colmap.exe",
    r"C:\COLMAP\colmap.exe",
    r"C:\Program Files\COLMAP\COLMAP.bat",
    r"C:\COLMAP\COLMAP.bat",
    os.path.expanduser(r"~\scoop\apps\colmap\current\bin\colmap.exe"),
    r"C:\ProgramData\chocolatey\bin\colmap.exe",
]


# ---------------------------------------------------------------------------
# COLMAP discovery
# ---------------------------------------------------------------------------

def find_colmap(hint: str | None = None) -> str:
    candidates: list[str] = []
    if hint:
        candidates.append(hint)
    env = os.environ.get("COLMAP_PATH")
    if env:
        candidates.append(env)
    which = shutil.which("colmap")
    if which:
        candidates.append(which)
    candidates.extend(_COMMON_COLMAP_PATHS)

    found = next((p for p in candidates if Path(p).is_file()), None)
    if not found:
        raise RuntimeError(
            "COLMAP not found.\n"
            "Download at: https://github.com/colmap/colmap/releases\n"
            "Then re-run with: --colmap-path \"C:/COLMAP/colmap.exe\"\n"
            "Or set the COLMAP_PATH environment variable."
        )
    return found


# ---------------------------------------------------------------------------
# Frame / mask gathering
# ---------------------------------------------------------------------------

def gather_images(
    output_base: Path,
) -> tuple[dict[str, Path], dict[str, Path], list[str]]:
    """
    Scan output_base/*/frames/ for images and output_base/*/masks/ for masks.

    Returns:
        images_map  — {dest_filename: source_path}
        masks_map   — {dest_filename + ".png": source_path}  (COLMAP naming)
        view_names  — ordered list of discovered view names
    """
    images: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    views_found: list[str] = []

    for view_dir in sorted(output_base.iterdir()):
        if not view_dir.is_dir() or view_dir.name == "colmap":
            continue
        frames_dir = view_dir / "frames"
        if not frames_dir.is_dir():
            continue

        masks_dir = view_dir / "masks"
        view_name = view_dir.name
        views_found.append(view_name)

        frame_files = sorted(
            p for p in frames_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTS
        )
        for frame_path in frame_files:
            dest_name = f"{view_name}_{frame_path.name}"
            images[dest_name] = frame_path
            mask_src = masks_dir / frame_path.name
            if masks_dir.is_dir() and mask_src.exists():
                # COLMAP mask naming: <image_name>.png appended
                masks[dest_name + ".png"] = mask_src

    return images, masks, views_found


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def run_colmap_step(name: str, cmd: list[str], dry_run: bool) -> int:
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
# Reconstruction orchestration
# ---------------------------------------------------------------------------

def run_reconstruction(
    output_base: Path,
    colmap: str,
    matcher: str,
    use_gpu: bool,
    overwrite: bool,
    dry_run: bool,
) -> int:
    log.info("Scanning views in: %s", output_base)
    images_map, masks_map, view_names = gather_images(output_base)

    if not images_map:
        log.error(
            "No frames found under %s\n"
            "Run sharp_frames first to populate the frames/ subdirectories.",
            output_base,
        )
        return 1

    log.info(
        "Found %d frames across %d view(s): %s",
        len(images_map), len(view_names), ", ".join(view_names),
    )
    if masks_map:
        log.info("Masks found: %d", len(masks_map))
    else:
        log.info("No masks found — proceeding without masking")

    colmap_dir = output_base / "colmap"
    images_dir = colmap_dir / "images"
    masks_dir  = colmap_dir / "masks"
    db_path    = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"

    if not dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        if masks_map:
            masks_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Copy frames + masks into colmap/images/ and colmap/masks/
    # ------------------------------------------------------------------
    if dry_run:
        log.info(
            "[DRY-RUN] Would copy %d images and %d masks into %s",
            len(images_map), len(masks_map), colmap_dir,
        )
    else:
        log.info("Copying %d images into %s ...", len(images_map), images_dir)
        for dest_name, src in images_map.items():
            dest = images_dir / dest_name
            if not dest.exists() or overwrite:
                shutil.copy2(src, dest)

        if masks_map:
            log.info("Copying %d masks ...", len(masks_map))
            for dest_name, src in masks_map.items():
                dest = masks_dir / dest_name
                if not dest.exists() or overwrite:
                    shutil.copy2(src, dest)

    gpu_flag = "1" if use_gpu else "0"

    # ------------------------------------------------------------------
    # Step 1: Feature extraction
    # ------------------------------------------------------------------
    cmd_feat = [
        colmap, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path",    str(images_dir),
        "--SiftExtraction.use_gpu", gpu_flag,
    ]
    if masks_map and not dry_run:
        cmd_feat += ["--ImageReader.mask_path", str(masks_dir)]

    rc = run_colmap_step("feature_extractor", cmd_feat, dry_run)
    if rc != 0:
        return rc

    # ------------------------------------------------------------------
    # Step 2: Matching
    # ------------------------------------------------------------------
    cmd_match = [
        colmap, f"{matcher}_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", gpu_flag,
    ]

    rc = run_colmap_step(f"{matcher}_matcher", cmd_match, dry_run)
    if rc != 0:
        return rc

    # ------------------------------------------------------------------
    # Step 3: Sparse reconstruction (mapper)
    # ------------------------------------------------------------------
    cmd_mapper = [
        colmap, "mapper",
        "--database_path", str(db_path),
        "--image_path",    str(images_dir),
        "--output_path",   str(sparse_dir),
    ]

    rc = run_colmap_step("mapper", cmd_mapper, dry_run)
    if rc != 0:
        return rc

    if not dry_run:
        models = sorted(sparse_dir.iterdir()) if sparse_dir.exists() else []
        if models:
            log.info("Sparse model(s): %s", ", ".join(str(m) for m in models))
            log.info("Ready for Lichtfeld: --colmap_dir %s", colmap_dir)
        else:
            log.warning("Mapper ran but no sparse model was produced. Check the log.")
            return 1

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="colmap_recon",
        description="Run COLMAP sparse reconstruction on all views from eq2persp output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "input", nargs="+", metavar="OUTPUT_BASE",
        help="eq2persp output directory for a scene (e.g. output/scene/)",
    )
    p.add_argument(
        "--colmap-path", metavar="PATH",
        help="Path to colmap binary (also: COLMAP_PATH env var)",
    )
    p.add_argument(
        "--matcher",
        choices=["exhaustive", "sequential", "vocab_tree"],
        default="exhaustive",
        help=(
            "exhaustive = try all pairs, best quality (default).\n"
            "sequential = match consecutive frames only, faster.\n"
            "vocab_tree = scalable for large datasets (needs vocab tree file)."
        ),
    )
    p.add_argument(
        "--no-gpu", action="store_true",
        help="Disable GPU for SIFT extraction and matching (use CPU only)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-copy images/masks even if they already exist",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
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
        colmap = find_colmap(args.colmap_path)
        log.info("COLMAP: %s", colmap)
    except RuntimeError as e:
        log.critical("%s", e)
        sys.exit(2)

    failed = 0
    for input_str in args.input:
        output_base = Path(input_str)
        if not output_base.is_dir():
            log.error("Not a directory: %s", input_str)
            failed += 1
            continue

        log.info("=" * 60)
        log.info("SCENE: %s", output_base)
        log.info("=" * 60)

        rc = run_reconstruction(
            output_base=output_base,
            colmap=colmap,
            matcher=args.matcher,
            use_gpu=not args.no_gpu,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
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
