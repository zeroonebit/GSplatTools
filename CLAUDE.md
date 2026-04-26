# GSplatTools

Dataset preparation pipeline for 3D Gaussian Splatting.
Converts raw DJI Osmo 360 footage into clean frame datasets for training in **Lichtfeld**.

**Preprocessing only — no 3DGS training code lives here.**

```
360° video
  ↓  eq2persp      — slice into perspective clips   (FFmpeg v360)
  ↓  sharp_frames  — keep only sharpest frames       (Laplacian variance)
  ↓  masks         — exclude dynamic objects         (YOLO / SAM 2)
  ↓  sam_segment   — click-based SAM 2 masks         (PyTorch / SAM 2)
  ↓  colmap_recon  — sparse camera model             (COLMAP SfM)
  ↓
Clean dataset → Lichtfeld → Gaussian Splat
```

## Project structure

```
tools/
  eq2persp.py       — equirectangular → perspective clips  (FFmpeg v360)
  sharp_frames.py   — Laplacian blur detection + frame extraction
  masks.py          — YOLO bounding-box or SAM 2 pixel masks
  sam_segment.py    — interactive SAM 2 masks from click prompts
  colmap_recon.py   — COLMAP feature extraction + matching + mapping
configs/
  cameras_6view.json        — standard 6-view cube rig
  cameras_drone_orbit.json  — 8-view orbit rig, 15° down-tilt
app.py            — Streamlit web UI (all tools in one place)
pipeline.py       — CLI orchestrator: chains all steps on one or more videos
```

## Run the UI

```bash
streamlit run app.py
```

Local network access (phone / tablet on same WiFi):

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
# Then open http://<PC-IP>:8501 on the phone
# Find PC IP with: ipconfig (look for IPv4 on Wi-Fi or Ethernet)
```

## Quick CLI reference

```bash
# Full pipeline on one video
python pipeline.py footage.mp4 --views 4 --gpu --skip-masks

# Step 1: extract views
python tools/eq2persp.py footage.mp4 --views 4 --gpu \
  --ffmpeg-path "C:/Users/thiag/ffmpeg-full_build/bin/ffmpeg.exe"

# Step 2: sharp frames (outputs adjacent to clip — no -o needed)
python tools/sharp_frames.py output/footage/front/footage_front.mp4 --top 20

# Step 3a: YOLO auto masks
python tools/masks.py output/footage/front/frames/ --classes person airplane

# Step 3b: SAM 2 click masks (image mode — apply annotations to all frames)
python tools/sam_segment.py output/footage/front/frames/ --prompts prompts.json

# Step 3c: SAM 2 click masks (video propagation — requires local checkpoint)
python tools/sam_segment.py output/footage/front/frames/ \
  --prompts prompts.json --mode video --model sam2.1_hiera_small.pt

# Step 4: COLMAP reconstruction
python tools/colmap_recon.py output/footage/ \
  --colmap-path "C:/COLMAP/colmap.exe"
```

## Output layout

```
output/
  <stem>/
    front/
      <stem>_front.mp4       ← eq2persp
      frames/                ← sharp_frames
        frame_000001.png
      masks/                 ← masks or sam_segment
        frame_000001.png
      sharp_frames_scores.csv
    right/ back/ left/ ...
    colmap/
      images/                ← all views combined, prefixed
      masks/
      database.db
      sparse/0/
        cameras.bin
        images.bin
        points3D.bin
```

## Dependencies

```bash
# Core (always needed)
pip install streamlit streamlit-image-coordinates opencv-python numpy Pillow tqdm pandas

# YOLO masks
pip install ultralytics

# SAM 2 (click segmentation + precise masks)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/facebookresearch/sam2.git

# External binaries
# FFmpeg >= 4.3:  https://www.gyan.dev/ffmpeg/builds/
# COLMAP >= 3.8:  https://github.com/colmap/colmap/releases
```

## Machine

- GPU: RTX 4090 — use `--gpu` (eq2persp NVENC) and CUDA for SAM 2 / YOLO
- FFmpeg: `C:\Users\thiag\ffmpeg-full_build\bin\ffmpeg.exe`
- Default output: `H:\Projects\GSplatTools\output`
- Test video: `H:\Projects\GSplatTools\test_vid\0426.mp4`

## Development conventions

- Each tool is a standalone CLI: `if __name__ == "__main__": main()`
- No mandatory heavy imports at module level — torch/ultralytics loaded inside functions
- All tools: `--dry-run` (no writes), `-v`/`--verbose` (debug logging)
- Exit codes: 0 = success, 1 = partial failure, 2 = fatal / misconfiguration
- Frame filenames: zero-padded 6 digits — `frame_000001.png`
- Mask convention: white (255) = exclude, black (0) = keep
- `sharp_frames -o <dir>` puts frames in `<dir>/frames/` and CSV in `<dir>/`
- COLMAP view naming: frames get `<view>_` prefix (`front_frame_000001.png`)

## Camera: DJI Osmo 360

- Output after DJI Studio: 8K H.265 equirectangular MP4
- Pixel format: `yuv420p10le` (10-bit)
- FFmpeg yaw range: [-180, 180] only — use -90 instead of 270
- Camera often captures operator/drone → masking required

## Git

```bash
# Commit (PowerShell — use here-string, no apostrophes in message)
git add <files>
git commit -m @'
Commit message here
'@
git push
```
