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
    import re
    m = re.search(r"time=(\d+):(\d+):([\d.]+)", line)
    if m:
        return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    return None


def run_command(cmd: list[str], log_container, duration_s: float = 0.0,
                stop_key: str = "") -> int:
    """
    Run a subprocess, stream output into log_container.
    stop_key: session_state key — if True mid-run, process is killed.
    Returns exit code (or -1 if stopped by user).
    """
    import os

    if cmd[0] == sys.executable:
        cmd = [cmd[0], "-u"] + cmd[1:]

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log_lines: list[str] = []
    log_box      = log_container.code("starting…", language="")
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
        # Store pid so the Stop button can kill it
        if stop_key:
            st.session_state[f"{stop_key}_pid"] = proc.pid

        for line in proc.stdout:
            # Check stop flag
            if stop_key and st.session_state.get(f"{stop_key}_stop"):
                proc.kill()
                log_lines.append("— stopped by user —")
                log_box.code("\n".join(log_lines[-150:]), language="")
                st.session_state[f"{stop_key}_stop"] = False
                return -1

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


def path_input(key: str, label: str, placeholder: str = "", default: str = "") -> str:
    return st.text_input(label, value=st.session_state.get(key, default),
                         placeholder=placeholder, key=key)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_eq, tab_sharp, tab_mask, tab_sam, tab_colmap, tab_pipe, tab_about = st.tabs([
    "🎥  360 → Views",
    "🔍  Sharp Frames",
    "🎭  Masks",
    "🖱️  SAM Click",
    "🗺️  COLMAP",
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
                              default=r"H:\Projects\GSplatTools\test_vid\0426.mp4")

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
        eq_gpu      = st.checkbox("GPU encode (NVENC) — RTX 4090 5-10x faster",
                                  value=True, key="eq_gpu")
        eq_clean     = st.checkbox("Clean output before run (delete existing folder)",
                                   key="eq_clean",
                                   help="Removes output/<stem>/ entirely before starting, "
                                        "so no partial files are reused.")
        eq_overwrite = st.checkbox("Overwrite existing", key="eq_overwrite")
        eq_dry_run   = st.checkbox("Dry-run", key="eq_dry_run")

    st.divider()
    col_run, col_stop = st.columns([4, 1])
    with col_run:
        run_clicked = st.button("▶  Run eq2persp", type="primary",
                                use_container_width=True, key="eq_run")
    with col_stop:
        if st.button("⏹  Stop", use_container_width=True, key="eq_stop_btn"):
            st.session_state["eq_stop"] = True

    if run_clicked:
        st.session_state["eq_stop"] = False
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
            if eq_gpu:       cmd += ["--gpu"]
            if eq_clean:     cmd += ["--clean"]
            if eq_overwrite: cmd += ["--overwrite"]
            if eq_dry_run:   cmd += ["--dry-run"]

            st.caption("`" + " ".join(cmd) + "`")
            log_area = st.empty()
            dur = 0.0
            try:
                import json, subprocess as _sp
                ffprobe = (Path(ffmpeg_path).parent / "ffprobe.exe") if ffmpeg_path else Path("ffprobe")
                r = _sp.run([str(ffprobe), "-v", "quiet", "-print_format", "json",
                             "-show_format", eq_input], capture_output=True, text=True, timeout=10)
                dur = float(json.loads(r.stdout).get("format", {}).get("duration", 0))
            except Exception:
                pass
            n_views = eq_views if eq_views else 4
            with st.spinner("Running eq2persp…"):
                rc = run_command(cmd, log_area, duration_s=dur * n_views, stop_key="eq")
            if rc == -1:
                st.warning("⏹ Stopped by user")
            else:
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
            # No -o: sharp_frames places frames/ adjacent to the clip (correct layout)
            cmd = [sys.executable, str(TOOLS_DIR / "sharp_frames.py"), sf_input,
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

            csv_path = Path(sf_input).parent / "sharp_frames_scores.csv"
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

    mask_mode = st.radio(
        "Detection mode",
        ["YOLO — class IDs (fast)", "Text prompt — YOLO-World (open vocabulary)"],
        horizontal=True, key="mask_mode",
        help="YOLO uses fixed COCO class IDs. Text prompt lets you type any description.",
    )

    st.divider()

    mk_input = path_input("mk_input", "Frames directory",
                          placeholder=r"output\scene\front\frames")
    st.caption("Point at the `frames/` folder from sharp_frames output.")

    if mask_mode.startswith("YOLO"):
        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            st.subheader("Classes")
            mk_person   = st.checkbox("person", value=True, key="mk_person")
            mk_airplane = st.checkbox("airplane  (drone proxy)", value=False, key="mk_airplane")
            mk_car      = st.checkbox("car", value=False, key="mk_car")
            mk_custom   = st.text_input("Custom class names (space-separated)",
                                        placeholder="motorcycle bicycle", key="mk_custom")
            mk_confidence = st.slider("Confidence", 0.1, 1.0, 0.4, step=0.05, key="mk_confidence")
            mk_dilate     = st.slider("Dilation (px)", 0, 50, 15, key="mk_dilate")

        with col_r:
            st.subheader("Method")
            mk_method = st.radio(
                "Segmentation",
                ["yolo — fast bbox fill", "sam — pixel-precise (needs SAM 2)"],
                key="mk_method",
            )
            mk_model = st.text_input("YOLO model", placeholder="yolov8n.pt  (auto-downloaded)",
                                     key="mk_model")
            if mk_method.startswith("sam"):
                mk_sam_model = st.text_input("SAM 2 checkpoint",
                                             placeholder="sam2_hiera_small.pt", key="mk_sam_model")
                mk_device = st.selectbox("Device", ["auto", "cuda", "cpu"], key="mk_device")
            else:
                mk_sam_model, mk_device = "", "auto"
            mk_dry_run = st.checkbox("Dry-run", key="mk_dry_run")
            mk_verbose  = st.checkbox("Verbose", key="mk_verbose")

        if st.button("▶  Run YOLO masks", type="primary", use_container_width=True, key="mk_run"):
            if not mk_input:
                st.warning("Enter a frames directory path.")
            else:
                classes = []
                if mk_person:   classes.append("person")
                if mk_airplane: classes.append("airplane")
                if mk_car:      classes.append("car")
                if mk_custom:   classes.extend(mk_custom.split())
                if not classes:
                    st.warning("Select at least one class.")
                else:
                    method_key = "sam" if mk_method.startswith("sam") else "yolo"
                    cmd = [sys.executable, str(TOOLS_DIR / "masks.py"), mk_input,
                           "--classes", *classes, "--method", method_key,
                           "--confidence", str(mk_confidence), "--dilate", str(mk_dilate)]
                    if mk_model:            cmd += ["--model", mk_model]
                    if mk_sam_model:        cmd += ["--sam-model", mk_sam_model]
                    if mk_device != "auto": cmd += ["--device", mk_device]
                    if mk_dry_run:          cmd += ["--dry-run"]
                    if mk_verbose:          cmd += ["-v"]
                    st.caption("`" + " ".join(cmd) + "`")
                    log_area = st.empty()
                    with st.spinner("Running YOLO masks…"):
                        rc = run_command(cmd, log_area)
                    status_badge(rc)

    else:
        # Text-prompt mode (YOLO-World)
        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            st.subheader("Text prompts")
            tm_text = st.text_input(
                "Objects to mask (comma-separated)",
                value="person, drone",
                key="tm_text",
                help="Type any description. Examples: person · drone · tripod · car · umbrella",
            )
            tm_confidence = st.slider("Confidence", 0.05, 1.0, 0.25, step=0.05, key="tm_confidence",
                                      help="YOLO-World works best at lower confidence than YOLO.")
            tm_dilate = st.slider("Dilation (px)", 0, 50, 10, key="tm_dilate")

        with col_r:
            st.subheader("Refinement")
            tm_sam = st.checkbox("SAM 2 — pixel-precise edges (recommended)", value=True,
                                 key="tm_sam")
            tm_model = st.text_input("YOLO-World model",
                                     placeholder="yolo11x-worldv2.pt  (auto-downloaded)",
                                     key="tm_model")
            if tm_sam:
                tm_sam_model = st.text_input("SAM 2 checkpoint",
                                             placeholder="facebook/sam2.1-hiera-small",
                                             key="tm_sam_model")
                tm_device = st.selectbox("Device", ["auto", "cuda", "cpu"], key="tm_device")
            else:
                tm_sam_model, tm_device = "", "auto"
            tm_dry_run = st.checkbox("Dry-run", key="tm_dry_run")
            tm_verbose  = st.checkbox("Verbose", key="tm_verbose")

        if st.button("▶  Run text masks", type="primary", use_container_width=True, key="tm_run"):
            if not mk_input:
                st.warning("Enter a frames directory path.")
            elif not tm_text.strip():
                st.warning("Enter at least one text prompt.")
            else:
                cmd = [sys.executable, str(TOOLS_DIR / "text_masks.py"), mk_input,
                       "--text", tm_text,
                       "--confidence", str(tm_confidence),
                       "--dilate", str(tm_dilate)]
                if tm_model:            cmd += ["--model", tm_model]
                if tm_sam:              cmd += ["--sam"]
                if tm_sam_model:        cmd += ["--sam-model", tm_sam_model]
                if tm_device != "auto": cmd += ["--device", tm_device]
                if tm_dry_run:          cmd += ["--dry-run"]
                if tm_verbose:          cmd += ["-v"]
                st.caption("`" + " ".join(cmd) + "`")
                log_area = st.empty()
                with st.spinner("Running text masks…"):
                    rc = run_command(cmd, log_area)
                status_badge(rc)

    # -------------------------------------------------------
    # Combine Masks section
    # -------------------------------------------------------
    st.divider()
    with st.expander("Combine mask layers", expanded=False):
        st.caption(
            "OR-combine multiple mask directories into one. "
            "Stack AI masks, shape masks, SAM click masks, etc."
        )
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            cm_dirs = st.text_area(
                "Mask directories (one per line)",
                placeholder="output\\scene\\front\\masks_ai\noutput\\scene\\front\\masks_shape",
                key="cm_dirs", height=100,
            )
            cm_output = st.text_input("Output directory",
                                      placeholder="output\\scene\\front\\masks",
                                      key="cm_output")
        with col_b:
            st.markdown("**Add shape layer**")
            cm_shape = st.selectbox("Shape", ["none", "circle", "bottom", "top", "rect"],
                                    key="cm_shape")
            cm_frames = st.text_input("Frames dir (for shape size reference)",
                                      placeholder=r"output\scene\front\frames",
                                      key="cm_frames")
            cm_strip_h = st.slider("Strip height (frac)", 0.05, 0.40, 0.15, step=0.01,
                                   key="cm_strip_h",
                                   help="Used for top/bottom strip shapes.")
            cm_dry = st.checkbox("Dry-run", key="cm_dry")

        if st.button("▶  Combine masks", type="primary",
                     use_container_width=True, key="cm_run"):
            dirs = [d.strip() for d in cm_dirs.splitlines() if d.strip()]
            cmd = [sys.executable, str(TOOLS_DIR / "combine_masks.py")] + dirs
            if cm_output:             cmd += ["-o", cm_output]
            if cm_shape != "none":    cmd += ["--shape", cm_shape]
            if cm_frames:             cmd += ["--frames", cm_frames]
            if cm_shape in ("bottom", "top"):
                cmd += ["--strip-height", str(cm_strip_h)]
            if cm_dry:                cmd += ["--dry-run"]
            st.caption("`" + " ".join(cmd) + "`")
            log_area = st.empty()
            with st.spinner("Combining masks…"):
                rc = run_command(cmd, log_area)
            status_badge(rc)


# ============================================================
# TAB 4 — SAM Click
# ============================================================
with tab_sam:
    st.header("SAM 2 Click Segmentation")
    st.caption(
        "Click on frames to mark objects as positive (mask) or negative (keep). "
        "SAM 2 generates masks for all frames — either per-frame or by propagating "
        "from a few keyframes through the whole sequence."
    )

    try:
        from streamlit_image_coordinates import streamlit_image_coordinates as _img_coords
        _sam_ui_ok = True
    except ImportError:
        st.warning(
            "`streamlit-image-coordinates` not installed.  \n"
            "Install with: `pip install streamlit-image-coordinates`"
        )
        _sam_ui_ok = False

    if _sam_ui_ok:
        col_l, col_r = st.columns([3, 2], gap="large")

        with col_l:
            st.subheader("Frames directory")
            sam_frames_dir = path_input(
                "sam_frames_dir", "Frames directory",
                placeholder=r"output\scene\front\frames",
            )

            sam_frames: list = []
            if sam_frames_dir and Path(sam_frames_dir).is_dir():
                exts = {".png", ".jpg", ".jpeg"}
                sam_frames = sorted(
                    p for p in Path(sam_frames_dir).iterdir()
                    if p.suffix.lower() in exts
                )

            if sam_frames:
                st.caption(f"{len(sam_frames)} frames found")
                frame_idx = st.slider(
                    "Frame to annotate", 0, len(sam_frames) - 1, 0,
                    key="sam_frame_idx",
                )
                frame_path = sam_frames[frame_idx]

                # Render frame with accumulated click overlays
                from PIL import Image, ImageDraw
                import json as _json

                annotations: list[dict] = st.session_state.get("sam_annotations", [])
                frame_annots = next(
                    (a["points"] for a in annotations if a["frame"] == frame_path.name),
                    [],
                )

                img = Image.open(frame_path).convert("RGB")
                orig_w, orig_h = img.size
                display_w = 700
                scale = display_w / orig_w
                img_display = img.resize((display_w, int(orig_h * scale)))

                if frame_annots:
                    draw = ImageDraw.Draw(img_display)
                    r = 8
                    for pt in frame_annots:
                        dx, dy = int(pt["x"] * scale), int(pt["y"] * scale)
                        color = (0, 220, 80) if pt["label"] == 1 else (220, 40, 40)
                        draw.ellipse([dx - r, dy - r, dx + r, dy + r],
                                     fill=color, outline="white", width=2)

                click_mode = st.radio(
                    "Click mode",
                    ["Positive (mask this)", "Negative (exclude)"],
                    horizontal=True, key="sam_click_mode",
                )
                label_val = 1 if click_mode.startswith("Positive") else 0
                st.caption("Click on the frame below to add a point.")

                click = _img_coords(img_display, key=f"sam_click_{frame_idx}", width=display_w)
                if click is not None:
                    rx = int(click["x"] / scale)
                    ry = int(click["y"] / scale)
                    if "sam_annotations" not in st.session_state:
                        st.session_state["sam_annotations"] = []
                    annots = st.session_state["sam_annotations"]
                    entry = next((a for a in annots if a["frame"] == frame_path.name), None)
                    if entry is None:
                        annots.append({"frame": frame_path.name, "points": []})
                        entry = annots[-1]
                    entry["points"].append({"x": rx, "y": ry, "label": label_val})
                    st.rerun()

            else:
                st.info("Enter a valid frames directory to start annotating.")

        with col_r:
            st.subheader("Annotations")
            annotations = st.session_state.get("sam_annotations", [])
            total_pts = sum(len(a["points"]) for a in annotations)
            st.caption(f"{len(annotations)} annotated frame(s) · {total_pts} point(s) total")

            if annotations:
                for ann in annotations:
                    pos = sum(1 for p in ann["points"] if p["label"] == 1)
                    neg = len(ann["points"]) - pos
                    st.markdown(f"**{ann['frame']}** — {pos} ✅ {neg} ❌")

                col_clr, col_save = st.columns(2)
                with col_clr:
                    if st.button("Clear all", key="sam_clear"):
                        st.session_state["sam_annotations"] = []
                        st.rerun()
                with col_save:
                    st.download_button(
                        "Download prompts.json",
                        data=_json.dumps(annotations, indent=2),
                        file_name="prompts.json",
                        mime="application/json",
                        key="sam_dl",
                    )

            st.subheader("SAM 2 settings")
            sam_mode = st.radio(
                "Mode",
                ["image — per-frame (fast)", "video — propagation (accurate)"],
                key="sam_mode",
                help=(
                    "**image**: runs SAM 2 on each frame independently using the annotated "
                    "frames as prompt templates applied to all frames.  \n"
                    "**video**: SAM 2 video propagation — annotate a few keyframes and SAM "
                    "tracks the objects through the whole sequence. Requires local checkpoint."
                ),
            )
            sam_model = path_input(
                "sam_model_path", "Model (HuggingFace ID or local .pt)",
                default="facebook/sam2.1-hiera-small",
            )
            sam_no_gpu = st.checkbox("Force CPU", key="sam_no_gpu")
            sam_dilate = st.slider("Mask dilation (px)", 0, 30, 5, key="sam_dilate")
            sam_dry    = st.checkbox("Dry-run", key="sam_dry")

            st.divider()

            # Prompts JSON path — write to disk before running
            sam_prompts_path = str(
                Path(sam_frames_dir).parent / "sam_prompts.json"
                if sam_frames_dir and Path(sam_frames_dir).is_dir()
                else Path("sam_prompts.json")
            )

            if st.button("▶  Generate masks", type="primary",
                         use_container_width=True, key="sam_run"):
                cur_annotations = st.session_state.get("sam_annotations", [])
                if not cur_annotations:
                    st.warning("Add at least one annotation click first.")
                elif not sam_frames_dir:
                    st.warning("Enter a frames directory.")
                else:
                    import json as _j
                    Path(sam_prompts_path).write_text(
                        _j.dumps(cur_annotations, indent=2), encoding="utf-8"
                    )
                    mode_key = "video" if sam_mode.startswith("video") else "image"
                    cmd = [
                        sys.executable, str(TOOLS_DIR / "sam_segment.py"),
                        sam_frames_dir,
                        "--prompts", sam_prompts_path,
                        "--mode", mode_key,
                        "--model", sam_model,
                        "--dilate", str(sam_dilate),
                    ]
                    if sam_no_gpu: cmd += ["--no-gpu"]
                    if sam_dry:    cmd += ["--dry-run"]
                    st.caption("`" + " ".join(cmd) + "`")
                    log_area = st.empty()
                    with st.spinner("Running SAM 2…"):
                        rc = run_command(cmd, log_area)
                    status_badge(rc)


# ============================================================
# TAB 5 — COLMAP
# ============================================================
with tab_colmap:
    st.header("COLMAP Sparse Reconstruction")
    st.caption(
        "Combines all view frames into one dataset and runs COLMAP feature extraction, "
        "matching, and mapping. Produces the sparse camera model that Lichtfeld needs."
    )

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.subheader("Input")
        cl_input = path_input(
            "cl_input", "Scene output base directory",
            default=str(Path(st.session_state.get("output_dir",
                             r"H:\Projects\GSplatTools\output")) / "<stem>"),
            placeholder=r"output\scene",
        )
        st.caption(
            "Point at the scene folder that eq2persp created — the one containing "
            "`front/`, `right/`, etc.  "
            "Frames from all views will be combined automatically."
        )

        st.subheader("COLMAP binary")
        cl_colmap_path = path_input(
            "cl_colmap_path", "COLMAP path",
            placeholder=r"C:\COLMAP\colmap.exe  (auto-detected if on PATH)",
        )

        st.subheader("Matching strategy")
        cl_matcher = st.radio(
            "Matcher",
            ["exhaustive", "sequential", "vocab_tree"],
            index=0,
            horizontal=True,
            key="cl_matcher",
            help=(
                "**exhaustive** — tries all frame pairs, best quality "
                "(recommended for <1000 frames).  \n"
                "**sequential** — matches consecutive frames only, much faster "
                "but works best for single-view sequences.  \n"
                "**vocab_tree** — scalable for large sets (needs vocab tree file)."
            ),
        )

    with col_r:
        st.subheader("GPU")
        cl_gpu = st.checkbox(
            "GPU acceleration (CUDA) — RTX 4090",
            value=True, key="cl_gpu",
            help="Speeds up SIFT extraction and matching significantly.",
        )

        st.subheader("Options")
        cl_overwrite = st.checkbox("Re-copy images/masks even if they exist",
                                   key="cl_overwrite")
        cl_dry_run   = st.checkbox("Dry-run", key="cl_dry_run")
        cl_verbose   = st.checkbox("Verbose", key="cl_verbose")

        st.subheader("Output")
        st.markdown("""
After a successful run you'll find:
```
<scene>/colmap/
  images/       ← all frames, view-prefixed
  masks/        ← masks (if generated)
  database.db   ← COLMAP feature DB
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```
Point Lichtfeld at `<scene>/colmap/` as the COLMAP directory.
""")

    st.divider()
    col_run, col_stop = st.columns([4, 1])
    with col_run:
        cl_run_clicked = st.button("▶  Run COLMAP", type="primary",
                                   use_container_width=True, key="cl_run")
    with col_stop:
        if st.button("⏹  Stop", use_container_width=True, key="cl_stop_btn"):
            st.session_state["cl_stop"] = True

    if cl_run_clicked:
        st.session_state["cl_stop"] = False
        if not cl_input:
            st.warning("Enter the scene output directory.")
        else:
            cmd = [sys.executable, str(TOOLS_DIR / "colmap_recon.py"), cl_input,
                   "--matcher", cl_matcher]
            if cl_colmap_path:
                cmd += ["--colmap-path", cl_colmap_path]
            if not cl_gpu:
                cmd += ["--no-gpu"]
            if cl_overwrite:
                cmd += ["--overwrite"]
            if cl_dry_run:
                cmd += ["--dry-run"]
            if cl_verbose:
                cmd += ["-v"]

            st.caption("`" + " ".join(cmd) + "`")
            log_area = st.empty()
            with st.spinner("Running COLMAP…"):
                rc = run_command(cmd, log_area, stop_key="cl")
            if rc == -1:
                st.warning("⏹ Stopped by user")
            else:
                status_badge(rc)


# ============================================================
# TAB 6 — Pipeline
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
# TAB 7 — About
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
| `colmap_recon` | ✅ Ready (needs COLMAP installed separately) |
| `sam_segment` | ✅ Ready (needs SAM 2 + torch) |
| `pipeline` | ✅ Ready |

### Setup
```bash
# 1. Install Python dependencies
pip install streamlit opencv-python numpy tqdm pandas ultralytics

# 2. Install FFmpeg >= 4.3
# https://www.gyan.dev/ffmpeg/builds/

# 3. Install COLMAP
# https://github.com/colmap/colmap/releases

# 4. Run
streamlit run app.py
```

---
[github.com/zeroonebit/GSplatTools](https://github.com/zeroonebit/GSplatTools)
""")
