"""
GSplatTools — Web UI
Run with: streamlit run app.py
"""

import subprocess
import sys
import threading
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GSplatTools",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

TOOLS_DIR = Path(__file__).parent / "tools"

# ---------------------------------------------------------------------------
# Sidebar — global settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ GSplatTools")
    st.caption("Dataset prep for Gaussian Splatting")
    st.divider()

    st.subheader("Global settings")

    ffmpeg_path = st.text_input(
        "FFmpeg path",
        value=st.session_state.get("ffmpeg_path", ""),
        placeholder="C:/ffmpeg/bin/ffmpeg.exe",
        help="Path to ffmpeg binary. Leave blank to use system PATH.",
    )
    st.session_state["ffmpeg_path"] = ffmpeg_path

    output_dir = st.text_input(
        "Output base directory",
        value=st.session_state.get("output_dir", "output"),
        placeholder="output",
        help="All tools write under this folder.",
    )
    st.session_state["output_dir"] = output_dir or "output"

    st.divider()
    st.caption(f"Python {sys.version.split()[0]}")
    st.caption("Tools: tools/eq2persp.py · tools/sharp_frames.py")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_command(cmd: list[str], log_container) -> int:
    """
    Run a subprocess and stream stdout+stderr line-by-line into log_container.
    Returns the exit code.
    """
    log_lines: list[str] = []
    log_box = log_container.code("", language="")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            log_lines.append(line.rstrip())
            log_box.code("\n".join(log_lines[-120:]), language="")
        proc.wait()
        return proc.returncode
    except FileNotFoundError as e:
        log_container.error(f"Could not launch process: {e}")
        return 2


def status_badge(rc: int):
    if rc == 0:
        st.success("✅ Done")
    elif rc == 1:
        st.warning("⚠️ Partial failure — check log above")
    else:
        st.error("❌ Failed — check log above")


def video_path_input(key: str, label: str = "Video file path") -> str:
    return st.text_input(
        label,
        value=st.session_state.get(key, ""),
        placeholder=r"\\NAS\footage\scene.mp4  or  H:\clips\scene.mp4",
        key=key,
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_eq, tab_sharp, tab_about = st.tabs([
    "🎥  360 → Views  (eq2persp)",
    "🔍  Sharp Frames",
    "ℹ️  About",
])

# ============================================================
# TAB 1 — eq2persp
# ============================================================
with tab_eq:
    st.header("Equirectangular → Perspective Views")
    st.caption(
        "Slices a 360° equirectangular video into multiple undistorted "
        "perspective clips using FFmpeg's v360 filter."
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Input")
        eq_input = video_path_input("eq_input", "360° video file path")

        st.subheader("Camera rig")
        eq_rig_mode = st.radio(
            "Rig",
            ["Standard views", "Custom JSON config"],
            horizontal=True,
            key="eq_rig_mode",
        )

        if eq_rig_mode == "Standard views":
            eq_views = st.radio(
                "Number of views",
                [4, 6],
                format_func=lambda v: f"{v} views — {'Front/Right/Back/Left' if v == 4 else '+ Top/Bottom'}",
                horizontal=True,
                key="eq_views",
            )
            eq_config = None
        else:
            eq_config = st.text_input(
                "Config JSON path",
                placeholder="configs/cameras_6view.json",
                key="eq_config",
            )
            eq_views = None

        st.subheader("Field of view")
        eq_fov = st.slider("Horizontal FOV (degrees)", 60, 150, 90, step=5, key="eq_fov")

    with col_right:
        st.subheader("Output resolution")
        res_preset = st.selectbox(
            "Preset",
            ["1920 × 1080", "2560 × 1440", "3840 × 2160", "Custom"],
            key="eq_res_preset",
        )
        if res_preset == "Custom":
            eq_w = st.number_input("Width",  value=1920, step=2, key="eq_w")
            eq_h = st.number_input("Height", value=1080, step=2, key="eq_h")
        else:
            eq_w, eq_h = map(int, res_preset.replace(" ", "").split("×"))
            st.caption(f"{eq_w} × {eq_h} px")

        st.subheader("Encoding")
        eq_crf = st.slider(
            "CRF (quality — lower = better)",
            0, 51, 18,
            help="18 = near-lossless. 23 = good balance. 28 = fast/smaller.",
            key="eq_crf",
        )
        eq_preset = st.select_slider(
            "Encoding preset (speed vs file size)",
            ["ultrafast", "superfast", "veryfast", "faster", "fast",
             "medium", "slow", "slower", "veryslow"],
            value="slow",
            key="eq_preset",
        )

        st.subheader("Options")
        eq_parallel = st.number_input(
            "Parallel views", min_value=1, max_value=8, value=1,
            help="Process N views simultaneously. 1 = sequential (safe for 8K).",
            key="eq_parallel",
        )
        eq_overwrite = st.checkbox("Overwrite existing outputs", key="eq_overwrite")
        eq_dry_run   = st.checkbox("Dry-run (print commands only)", key="eq_dry_run")

    st.divider()

    eq_run = st.button("▶  Run eq2persp", type="primary", use_container_width=True)

    eq_log_area = st.container()

    if eq_run:
        if not eq_input:
            st.warning("Enter the path to a 360° video file.")
        else:
            cmd = [sys.executable, str(TOOLS_DIR / "eq2persp.py"), eq_input]
            cmd += ["-o", st.session_state["output_dir"]]
            cmd += ["--width", str(int(eq_w)), "--height", str(int(eq_h))]
            cmd += ["--fov", str(eq_fov)]
            cmd += ["--crf", str(eq_crf)]
            cmd += ["--preset", eq_preset]
            cmd += ["--parallel", str(int(eq_parallel))]

            if eq_rig_mode == "Custom JSON config" and eq_config:
                cmd += ["--config", eq_config]
            else:
                cmd += ["--views", str(eq_views)]

            if ffmpeg_path:
                cmd += ["--ffmpeg-path", ffmpeg_path]
            if eq_overwrite:
                cmd += ["--overwrite"]
            if eq_dry_run:
                cmd += ["--dry-run"]

            with eq_log_area:
                st.subheader("Output")
                st.caption("Command: `" + " ".join(cmd) + "`")
                log_container = st.empty()
                rc = run_command(cmd, log_container)
                status_badge(rc)


# ============================================================
# TAB 2 — sharp_frames
# ============================================================
with tab_sharp:
    st.header("Sharp Frame Extractor")
    st.caption(
        "Selects the sharpest, least-blurry frames from a video clip "
        "using Laplacian variance blur detection."
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Input")
        sf_input = video_path_input("sf_input", "Video file path")
        st.caption(
            "Tip: point at a perspective clip from eq2persp, "
            "e.g. `output/scene/front/scene_front.mp4`"
        )

        st.subheader("Frame selection mode")
        sf_mode = st.radio(
            "Mode",
            ["Top % (adaptive — recommended)", "Fixed threshold"],
            key="sf_mode",
            horizontal=True,
        )

        if sf_mode.startswith("Top"):
            sf_top = st.slider(
                "Keep top % of sharpest frames",
                min_value=1, max_value=100, value=20, step=1,
                key="sf_top",
                help="20% is a good default. Lower = more selective.",
            )
            sf_threshold = None
        else:
            sf_threshold = st.number_input(
                "Minimum Laplacian variance score",
                min_value=0.0, value=80.0, step=5.0,
                key="sf_threshold",
                help="Use dry-run + verbose to explore scores for your footage first.",
            )
            sf_top = None

    with col_right:
        st.subheader("Limits")
        sf_max_frames = st.number_input(
            "Max frames to keep (0 = no limit)",
            min_value=0, value=0, step=50,
            key="sf_max_frames",
            help="Caps output count — keeps the sharpest N among candidates. "
                 "300–500 is typical for COLMAP.",
        )

        st.subheader("Sampling")
        sf_every = st.number_input(
            "Sample every N frames",
            min_value=1, value=1, step=1,
            key="sf_every",
            help="Use 2–5 to speed up scoring on long clips without missing much.",
        )

        st.subheader("Options")
        sf_dry_run = st.checkbox("Dry-run (report without writing)", key="sf_dry_run")
        sf_verbose  = st.checkbox("Verbose (show per-frame scores)", key="sf_verbose")

    st.divider()

    sf_run = st.button("▶  Run sharp_frames", type="primary", use_container_width=True)

    sf_log_area = st.container()

    if sf_run:
        if not sf_input:
            st.warning("Enter the path to a video file.")
        else:
            cmd = [sys.executable, str(TOOLS_DIR / "sharp_frames.py"), sf_input]
            cmd += ["-o", st.session_state["output_dir"]]
            cmd += ["--every", str(int(sf_every))]

            if sf_mode.startswith("Top"):
                cmd += ["--top", str(sf_top)]
            else:
                cmd += ["--threshold", str(sf_threshold)]

            if sf_max_frames and sf_max_frames > 0:
                cmd += ["--max-frames", str(int(sf_max_frames))]
            if sf_dry_run:
                cmd += ["--dry-run"]
            if sf_verbose:
                cmd += ["-v"]

            with sf_log_area:
                st.subheader("Output")
                st.caption("Command: `" + " ".join(cmd) + "`")
                log_container = st.empty()
                rc = run_command(cmd, log_container)
                status_badge(rc)

                # Show scores CSV if it exists
                stem = Path(sf_input).stem
                csv_path = Path(st.session_state["output_dir"]) / stem / "sharp_frames_scores.csv"
                if csv_path.exists() and not sf_dry_run:
                    import pandas as pd
                    st.subheader("Sharpness score distribution")
                    df = pd.read_csv(csv_path)
                    st.bar_chart(
                        df.set_index("frame_index")["sharpness_score"],
                        use_container_width=True,
                    )
                    kept = df[df["kept"] == 1]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total sampled", len(df))
                    col2.metric("Kept", len(kept))
                    col3.metric("Avg score (kept)", f"{kept['sharpness_score'].mean():.1f}")


# ============================================================
# TAB 3 — About
# ============================================================
with tab_about:
    st.header("GSplatTools")
    st.markdown("""
**Dataset preparation pipeline for 3D Gaussian Splatting.**

This tool preprocesses DJI Osmo 360 footage into clean image datasets
for training in **Lichtfeld** (or any other 3DGS trainer).

### Pipeline
```
360° video
    ↓  eq2persp   — slice into N perspective view clips
    ↓  sharp_frames — select sharpest frames per view
    ↓  masks      — mask dynamic objects (coming soon)
    ↓
Clean image dataset → Lichtfeld → Gaussian Splat
```

### Tools
| Tool | Status | Description |
|---|---|---|
| `eq2persp` | ✅ Ready | Equirectangular → perspective clips via FFmpeg v360 |
| `sharp_frames` | ✅ Ready | Blur detection + frame selection (Laplacian variance) |
| `masks` | 🚧 Soon | Drone / person masking via YOLO + SAM |

### Requirements
- **FFmpeg ≥ 4.3** — [download here](https://www.gyan.dev/ffmpeg/builds/)
- `pip install streamlit opencv-python numpy tqdm pandas`

### Camera
Tested with **DJI Osmo 360** — 8K H.265 equirectangular, 10-bit.
Requires DJI Studio stitching before use.

---
[GitHub — zeroonebit/GSplatTools](https://github.com/zeroonebit/GSplatTools)
""")
