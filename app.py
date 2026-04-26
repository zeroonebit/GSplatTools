"""
GSplatTools — Web UI
Run with: streamlit run app.py
"""

import subprocess
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="GSplatTools",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

TOOLS_DIR = Path(__file__).parent / "tools"
PIPELINE  = Path(__file__).parent / "pipeline.py"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ GSplatTools")
    st.caption("Dataset prep for Gaussian Splatting")
    st.divider()

    st.subheader("Global settings")
    ffmpeg_path = st.text_input(
        "FFmpeg path",
        value=st.session_state.get("ffmpeg_path", r"C:\Users\thiag\ffmpeg-full_build\bin\ffmpeg.exe"),
        help="Leave blank to use system PATH.",
    )
    st.session_state["ffmpeg_path"] = ffmpeg_path

    output_dir = st.text_input(
        "Output base directory",
        value=st.session_state.get("output_dir", r"H:\Projects\GSplatTools\output"),
    )
    st.session_state["output_dir"] = output_dir or r"H:\Projects\GSplatTools\output"

    st.divider()
    st.caption(f"Python {sys.version.split()[0]}")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_ffmpeg_progress(line: str) -> float | None:
    """Extract time= from FFmpeg output, return seconds or None."""
    import re
    m = re.search(r"time=(\d+):(\d+):([\d.]+)", line)
    if m:
        return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    return None


def run_command(cmd: list[str], log_container, duration_s: float = 0.0) -> int:
    """
    Run a subprocess and stream output into log_container.
    If duration_s > 0, shows a progress bar based on FFmpeg time= output.
    Returns exit code.
    """
    import os, re

    if cmd[0] == sys.executable:
        cmd = [cmd[0], "-u"] + cmd[1:]

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log_lines: list[str] = []
    log_box = log_container.code("starting…", language="")
    progress_bar = log_container.progress(0) if duration_s > 0 else None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            stripped = line.rstrip()
            if not stripped:
                continue
            log_lines.append(stripped)
            log_box.code("\n".join(log_lines[-150:]), language="")
            if progress_bar:
                t = _parse_ffmpeg_progress(stripped)
                if t is not None:
                    progress_bar.progress(min(t / duration_s, 1.0))
        proc.wait()
        if progress_bar:
            progress_bar.progress(1.0)
        return proc.returncode
    except FileNotFoundError as e:
        log_container.error(f"Could not launch process: {e}")
        return 2


def status_badge(rc: int):
    if rc == 0:
        st.success("✅ Done")
    elif rc == 1:
        st.warning("⚠️ Partial failure — check log")
    else:
        st.error("❌ Failed — check log")


def path_input(key: str, label: str, placeholder: str = "") -> str:
    return st.text_input(label, value=st.session_state.get(key, ""),
                         placeholder=placeholder, key=key)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_eq, tab_sharp, tab_mask, tab_pipe, tab_about = st.tabs([
    "🎥  360 → Views",
    "🔍  Sharp Frames",
    "🎭  Masks",
    "🚀  Pipeline",
    "ℹ️  About",
])


# ============================================================
# TAB 1 — eq2persp
# ============================================================
with tab_eq:
    st.header("Equirectangular → Perspective Views")
    st.caption("Slices a 360° video into undistorted perspective clips via FFmpeg v360.")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.subheader("Input")
        eq_input = path_input("eq_input", "360° video file path",
                              r"H:\footage\scene.mp4")

        st.subheader("Camera rig")
        eq_rig_mode = st.radio("Rig", ["Standard views", "Custom JSON config"],
                               horizontal=True, key="eq_rig_mode")
        if eq_rig_mode == "Standard views":
            eq_views = st.radio(
                "Views", [4, 6],
                format_func=lambda v: "4 — Front / Right / Back / Left" if v == 4
                                      else "6 — + Top / Bottom",
                horizontal=True, key="eq_views",
            )
            eq_config = None
        else:
            eq_config = st.text_input("Config JSON path",
                                      placeholder="configs/cameras_6view.json",
                                      key="eq_config")
            eq_views = None

        st.subheader("Field of view")
        eq_fov = st.slider("Horizontal FOV (°)", 60, 150, 90, step=5, key="eq_fov")

    with col_r:
        st.subheader("Output resolution")
        res_preset = st.selectbox("Preset",
                                  ["1920 × 1080", "2560 × 1440", "3840 × 2160", "Custom"],
                                  key="eq_res_preset")
        if res_preset == "Custom":
            eq_w = st.number_input("Width",  value=1920, step=2, key="eq_w")
            eq_h = st.number_input("Height", value=1080, step=2, key="eq_h")
        else:
            eq_w, eq_h = map(int, res_preset.replace(" ", "").split("×"))
            st.caption(f"{eq_w} × {eq_h} px")

        st.subheader("Encoding")
        eq_crf = st.slider("CRF (lower = better quality)", 0, 51, 18, key="eq_crf",
                           help="18 = near-lossless · 23 = balanced · 28 = fast")
        eq_preset = st.select_slider(
            "Preset (speed ↔ compression)",
            ["ultrafast","superfast","veryfast","faster","fast",
             "medium","slow","slower","veryslow"],
            value="slow", key="eq_preset",
        )

        st.subheader("Options")
        eq_parallel = st.number_input("Parallel views", 1, 8, 1, key="eq_parallel",
                                      help="1 = sequential (safe for 8K)")
        eq_overwrite = st.checkbox("Overwrite existing", key="eq_overwrite")
        eq_dry_run   = st.checkbox("Dry-run", key="eq_dry_run")

    st.divider()
    if st.button("▶  Run eq2persp", type="primary", use_container_width=True, key="eq_run"):
        if not eq_input:
            st.warning("Enter a video file path.")
        else:
            cmd = [sys.executable, str(TOOLS_DIR / "eq2persp.py"), eq_input,
                   "-o", st.session_state["output_dir"],
                   "--width", str(int(eq_w)), "--height", str(int(eq_h)),
                   "--fov", str(eq_fov), "--crf", str(eq_crf),
                   "--preset", eq_preset, "--parallel", str(int(eq_parallel))]
            if eq_rig_mode == "Custom JSON config" and eq_config:
                cmd += ["--config", eq_config]
            else:
                cmd += ["--views", str(eq_views)]
            if ffmpeg_path:
                cmd += ["--ffmpeg-path", ffmpeg_path]
            if eq_overwrite: cmd += ["--overwrite"]
            if eq_dry_run:   cmd += ["--dry-run"]

            st.caption("`" + " ".join(cmd) + "`")
            log_area = st.empty()
            # Probe duration for progress bar
            dur = 0.0
            try:
                import json, subprocess as _sp
                ffprobe = (Path(ffmpeg_path).parent / "ffprobe.exe") if ffmpeg_path else Path("ffprobe")
                r = _sp.run([str(ffprobe), "-v", "quiet", "-print_format", "json",
                             "-show_format", eq_input], capture_output=True, text=True, timeout=10)
                dur = float(json.loads(r.stdout).get("format", {}).get("duration", 0))
            except Exception:
                pass
            with st.spinner("Running eq2persp…"):
                rc = run_command(cmd, log_area, duration_s=dur * eq_views if eq_views else dur * 4)
            status_badge(rc)


# ============================================================
# TAB 2 — sharp_frames
# ============================================================
with tab_sharp:
    st.header("Sharp Frame Extractor")
    st.caption("Selects the least-blurry frames using Laplacian variance scoring.")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.subheader("Input")
        sf_input = path_input("sf_input", "Video file path",
                              r"output\scene\front\scene_front.mp4")

        st.subheader("Selection mode")
        sf_mode = st.radio("Mode",
                           ["Top % (adaptive — recommended)", "Fixed threshold"],
                           horizontal=True, key="sf_mode")
        if sf_mode.startswith("Top"):
            sf_top = st.slider("Keep top % of frames", 1, 100, 20, key="sf_top",
                               help="20% = good default. Lower = more selective.")
            sf_threshold = None
        else:
            sf_threshold = st.number_input("Min Laplacian score", 0.0, value=80.0,
                                           step=5.0, key="sf_threshold",
                                           help="Use dry-run + verbose to calibrate.")
            sf_top = None

    with col_r:
        st.subheader("Limits")
        sf_max = st.number_input("Max frames (0 = no limit)", 0, value=0, step=50,
                                 key="sf_max",
                                 help="300–500 is typical for COLMAP input.")
        st.subheader("Sampling")
        sf_every = st.number_input("Sample every N frames", 1, value=1,
                                   key="sf_every",
                                   help="Use 2–5 to speed up long clips.")
        st.subheader("Options")
        sf_dry_run = st.checkbox("Dry-run", key="sf_dry_run")
        sf_verbose  = st.checkbox("Verbose (show per-frame scores)", key="sf_verbose")

    st.divider()
    if st.button("▶  Run sharp_frames", type="primary", use_container_width=True, key="sf_run"):
        if not sf_input:
            st.warning("Enter a video file path.")
        else:
            cmd = [sys.executable, str(TOOLS_DIR / "sharp_frames.py"), sf_input,
                   "-o", st.session_state["output_dir"],
                   "--every", str(int(sf_every))]
            if sf_mode.startswith("Top"):
                cmd += ["--top", str(sf_top)]
            else:
                cmd += ["--threshold", str(sf_threshold)]
            if sf_max > 0:   cmd += ["--max-frames", str(int(sf_max))]
            if sf_dry_run:   cmd += ["--dry-run"]
            if sf_verbose:   cmd += ["-v"]

            st.caption("`" + " ".join(cmd) + "`")
            log_area = st.empty()
            with st.spinner("Running sharp_frames…"):
                rc = run_command(cmd, log_area)
            status_badge(rc)

            stem = Path(sf_input).stem
            csv_path = Path(st.session_state["output_dir"]) / stem / "sharp_frames_scores.csv"
            if csv_path.exists() and not sf_dry_run:
                import pandas as pd
                df = pd.read_csv(csv_path)
                st.subheader("Sharpness distribution")
                st.bar_chart(df.set_index("frame_index")["sharpness_score"],
                             use_container_width=True)
                kept = df[df["kept"] == 1]
                c1, c2, c3 = st.columns(3)
                c1.metric("Sampled", len(df))
                c2.metric("Kept", len(kept))
                c3.metric("Avg score (kept)", f"{kept['sharpness_score'].mean():.1f}")


# ============================================================
# TAB 3 — Masks
# ============================================================
with tab_mask:
    st.header("Dynamic Object Masks")
    st.caption(
        "Generates binary masks (white = exclude) for people, drones, and other "
        "dynamic objects using YOLOv8 detection. Optional SAM 2 for pixel-precise edges."
    )

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.subheader("Input")
        mk_input = path_input("mk_input", "Frames directory",
                              r"output\scene\front\frames")
        st.caption("Point at the `frames/` folder from sharp_frames output.")

        st.subheader("Classes to mask")
        mk_person   = st.checkbox("person", value=True, key="mk_person")
        mk_airplane = st.checkbox("airplane  (drone proxy)", value=False, key="mk_airplane")
        mk_car      = st.checkbox("car", value=False, key="mk_car")
        mk_custom   = st.text_input("Custom class names (space-separated)",
                                    placeholder="motorcycle bicycle",
                                    key="mk_custom")

        st.subheader("Detection")
        mk_confidence = st.slider("Confidence threshold", 0.1, 1.0, 0.4, step=0.05,
                                  key="mk_confidence")
        mk_dilate = st.slider("Mask dilation (px)", 0, 50, 15, key="mk_dilate",
                              help="Expands masks outward to cover object edges.")

    with col_r:
        st.subheader("Method")
        mk_method = st.radio(
            "Segmentation method",
            ["yolo — fast bbox fill", "sam — pixel-precise (needs SAM 2)"],
            key="mk_method",
            help="YOLO fills the bounding box. SAM 2 traces the exact object outline.",
        )

        st.subheader("Models")
        mk_model = st.text_input("YOLO model path",
                                 placeholder="yolov8n.pt  (auto-downloaded)",
                                 key="mk_model")
        sam_visible = mk_method.startswith("sam")
        if sam_visible:
            mk_sam_model = st.text_input("SAM 2 checkpoint path",
                                         placeholder="sam2_hiera_small.pt",
                                         key="mk_sam_model")
            mk_device = st.selectbox("Device", ["auto", "cuda", "cpu", "mps"],
                                     key="mk_device")
        else:
            mk_sam_model = ""
            mk_device = "auto"

        st.subheader("Options")
        mk_dry_run = st.checkbox("Dry-run (count detections only)", key="mk_dry_run")
        mk_verbose = st.checkbox("Verbose", key="mk_verbose")

    st.divider()
    if st.button("▶  Run masks", type="primary", use_container_width=True, key="mk_run"):
        if not mk_input:
            st.warning("Enter a frames directory path.")
        else:
            classes = []
            if mk_person:   classes.append("person")
            if mk_airplane: classes.append("airplane")
            if mk_car:      classes.append("car")
            if mk_custom:   classes.extend(mk_custom.split())
            if not classes:
                st.warning("Select at least one class to mask.")
            else:
                method_key = "sam" if mk_method.startswith("sam") else "yolo"
                cmd = [sys.executable, str(TOOLS_DIR / "masks.py"), mk_input,
                       "--classes", *classes,
                       "--method", method_key,
                       "--confidence", str(mk_confidence),
                       "--dilate", str(mk_dilate)]
                if mk_model:     cmd += ["--model", mk_model]
                if mk_sam_model: cmd += ["--sam-model", mk_sam_model]
                if mk_device != "auto": cmd += ["--device", mk_device]
                if mk_dry_run:   cmd += ["--dry-run"]
                if mk_verbose:   cmd += ["-v"]

                st.caption("`" + " ".join(cmd) + "`")
                log_area = st.empty()
                with st.spinner("Running masks…"):
                    rc = run_command(cmd, log_area)
                status_badge(rc)


# ============================================================
# TAB 4 — Pipeline
# ============================================================
with tab_pipe:
    st.header("Full Pipeline")
    st.caption(
        "Runs eq2persp → sharp_frames → masks in one go on one or more 360° videos."
    )

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.subheader("Input")
        pipe_input = path_input("pipe_input", "360° video file path(s) or glob",
                                r"H:\footage\*.mp4")

        st.subheader("eq2persp settings")
        pipe_views = st.radio("Views", [4, 6], horizontal=True, key="pipe_views",
                              format_func=lambda v: "4" if v == 4 else "6 (+ top/bottom)")
        pipe_fov = st.slider("FOV (°)", 60, 150, 90, step=5, key="pipe_fov")
        pipe_crf = st.slider("CRF", 0, 51, 18, key="pipe_crf")
        pipe_preset = st.select_slider(
            "Preset", ["ultrafast","superfast","veryfast","faster","fast",
                       "medium","slow","slower","veryslow"],
            value="slow", key="pipe_preset",
        )

        st.subheader("sharp_frames settings")
        pipe_top = st.slider("Keep top % of frames", 1, 100, 20, key="pipe_top")
        pipe_max = st.number_input("Max frames per view (0 = no limit)",
                                   0, value=0, step=50, key="pipe_max")

    with col_r:
        st.subheader("Masks settings")
        pipe_skip_masks = st.checkbox("Skip masks step", value=False,
                                      key="pipe_skip_masks")

        if not pipe_skip_masks:
            pipe_mk_person   = st.checkbox("Mask: person",   value=True,  key="pipe_mk_person")
            pipe_mk_airplane = st.checkbox("Mask: airplane", value=False, key="pipe_mk_airplane")
            pipe_mk_conf = st.slider("Mask confidence", 0.1, 1.0, 0.4, step=0.05,
                                     key="pipe_mk_conf")

        st.subheader("Options")
        pipe_overwrite = st.checkbox("Overwrite existing", key="pipe_overwrite")
        pipe_dry_run   = st.checkbox("Dry-run (preview all commands)", key="pipe_dry_run")
        pipe_verbose   = st.checkbox("Verbose", key="pipe_verbose")

    st.divider()
    if st.button("▶  Run full pipeline", type="primary", use_container_width=True,
                 key="pipe_run"):
        if not pipe_input:
            st.warning("Enter a video file path or glob pattern.")
        else:
            cmd = [sys.executable, str(PIPELINE), pipe_input,
                   "-o", st.session_state["output_dir"],
                   "--views",  str(pipe_views),
                   "--fov",    str(pipe_fov),
                   "--crf",    str(pipe_crf),
                   "--preset", pipe_preset,
                   "--top",    str(pipe_top)]

            if pipe_max > 0:
                cmd += ["--max-frames", str(int(pipe_max))]
            if ffmpeg_path:
                cmd += ["--ffmpeg-path", ffmpeg_path]
            if pipe_overwrite:
                cmd += ["--overwrite"]

            if pipe_skip_masks:
                cmd += ["--skip-masks"]
            else:
                classes = []
                if pipe_mk_person:   classes.append("person")
                if pipe_mk_airplane: classes.append("airplane")
                if classes:
                    cmd += ["--classes"] + classes
                cmd += ["--confidence", str(pipe_mk_conf)]

            if pipe_dry_run: cmd += ["--dry-run"]
            if pipe_verbose: cmd += ["-v"]

            st.caption("`" + " ".join(cmd) + "`")
            log_area = st.empty()
            with st.spinner("Running pipeline…"):
                rc = run_command(cmd, log_area)
            status_badge(rc)


# ============================================================
# TAB 5 — About
# ============================================================
with tab_about:
    st.header("GSplatTools")
    st.markdown("""
**Preprocessing pipeline for 3D Gaussian Splatting datasets.**
Turns DJI Osmo 360 footage into clean, undistorted frame sets for training in **Lichtfeld**.

### Pipeline
```
360° video
  ↓  eq2persp     — slice into perspective view clips (FFmpeg v360)
  ↓  sharp_frames — keep only sharpest frames (Laplacian variance)
  ↓  masks        — exclude dynamic objects (YOLO + SAM)
  ↓
Clean dataset  →  Lichtfeld  →  Gaussian Splat
```

### Tool status
| Tool | Status |
|---|---|
| `eq2persp` | ✅ Ready |
| `sharp_frames` | ✅ Ready |
| `masks` | ✅ Ready (needs `pip install ultralytics`) |
| `pipeline` | ✅ Ready |

### Setup
```bash
# 1. Install Python dependencies
pip install streamlit opencv-python numpy tqdm pandas ultralytics

# 2. Install FFmpeg >= 4.3
# https://www.gyan.dev/ffmpeg/builds/

# 3. Run
streamlit run app.py
```

---
[github.com/zeroonebit/GSplatTools](https://github.com/zeroonebit/GSplatTools)
""")
